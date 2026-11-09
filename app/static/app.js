/**
 * ClaimScope - Frontend Application Controller
 * Handles image upload validation, OpenCV signal processing requests,
 * and actuarial collision repair estimate rendering.
 */

// Application State encapsulation
const AppState = {
  currentFile: null,
  currentAnalysis: null,
  currentEstimate: null,
  originalImageSrc: null,
  activeTab: 'annotated', // 'annotated' | 'mask' | 'original'
  isInspecting: false,
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initUI();
  initEventListeners();
});

function initUI() {
  const slider = document.getElementById('sensitivity-slider');
  const sliderVal = document.getElementById('sensitivity-value');
  if (slider && sliderVal) {
    slider.addEventListener('input', (e) => {
      sliderVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
}

function initEventListeners() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        await handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', async (e) => {
      if (e.target.files && e.target.files[0]) {
        await handleFileSelected(e.target.files[0]);
      }
    });
  }

  // 1-Click Samples
  const btnSampleBumper = document.getElementById('btn-sample-bumper');
  if (btnSampleBumper) {
    btnSampleBumper.addEventListener('click', () => {
      loadSamplePreset('/sample_images/sample_bumper_scratch.jpg', 'FRONT_BUMPER', 'Bumper Scratch');
    });
  }

  const btnSampleDoor = document.getElementById('btn-sample-door');
  if (btnSampleDoor) {
    btnSampleDoor.addEventListener('click', () => {
      loadSamplePreset('/sample_images/sample_door_dent.jpg', 'DRIVER_FRONT_DOOR', 'Door Dent');
    });
  }

  const btnSampleFender = document.getElementById('btn-sample-fender');
  if (btnSampleFender) {
    btnSampleFender.addEventListener('click', () => {
      loadSamplePreset('/sample_images/sample_fender_chip.jpg', 'FRONT_FENDER_LEFT', 'Fender Chip');
    });
  }

  // Target Panel Selection Change
  const panelSelect = document.getElementById('panel-select');
  if (panelSelect) {
    panelSelect.addEventListener('change', async () => {
      if (AppState.currentFile) {
        await handleFileSelected(AppState.currentFile);
      }
    });
  }

  // Execute Inspection Button
  const btnAnalyze = document.getElementById('btn-analyze');
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      if (!AppState.currentFile) {
        showToast('Please upload a car photograph or select a 1-click test sample first.', 'warning');
        return;
      }
      executeInspection();
    });
  }

  // View Studio Tabs
  const tabAnnotated = document.getElementById('tab-annotated');
  if (tabAnnotated) tabAnnotated.addEventListener('click', () => switchTab('annotated'));

  const tabMask = document.getElementById('tab-mask');
  if (tabMask) tabMask.addEventListener('click', () => switchTab('mask'));

  const tabOriginal = document.getElementById('tab-original');
  if (tabOriginal) tabOriginal.addEventListener('click', () => switchTab('original'));

  // Export PDF Button
  const btnExportPdf = document.getElementById('btn-export-pdf');
  if (btnExportPdf) {
    btnExportPdf.addEventListener('click', downloadAppraisalPDF);
  }

  // About Modal Listeners
  const aboutModal = document.getElementById('about-modal');
  const btnAbout = document.getElementById('btn-about');
  const btnCloseAbout = document.getElementById('btn-close-about');
  const btnAboutOk = document.getElementById('btn-about-ok');

  const openAbout = () => {
    if (aboutModal) {
      aboutModal.classList.remove('hidden');
      if (window.lucide) lucide.createIcons();
    }
  };
  const closeAbout = () => {
    if (aboutModal) aboutModal.classList.add('hidden');
  };

  if (btnAbout) btnAbout.addEventListener('click', openAbout);
  if (btnCloseAbout) btnCloseAbout.addEventListener('click', closeAbout);
  if (btnAboutOk) btnAboutOk.addEventListener('click', closeAbout);
  if (aboutModal) {
    aboutModal.addEventListener('click', (e) => {
      if (e.target === aboutModal) closeAbout();
    });
  }
}

/**
 * Validates uploaded photo using the pre-flight classical CV filter endpoint.
 */
async function handleFileSelected(file) {
  const spinner = document.getElementById('validation-spinner');
  const errorBanner = document.getElementById('validation-error-banner');
  const errorMsg = document.getElementById('validation-error-message');
  const successBanner = document.getElementById('validation-success-banner');
  const successDetails = document.getElementById('validation-success-details');
  const previewInfo = document.getElementById('file-preview-info');
  const displayImg = document.getElementById('img-display');
  const emptyState = document.getElementById('empty-state');
  const fileInput = document.getElementById('file-input');

  // Reset feedback state
  if (errorBanner) errorBanner.classList.add('hidden');
  if (successBanner) successBanner.classList.add('hidden');
  if (previewInfo) previewInfo.classList.add('hidden');
  if (spinner) spinner.classList.remove('hidden');

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('panel_id', document.getElementById('panel-select')?.value || 'FRONT_BUMPER');

    const res = await fetch('/api/validate-image', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error('Image validation request failed on server.');
    }

    const validation = await res.json();

    if (!validation.is_valid) {
      // REJECT: Incompatible or non-automotive input
      AppState.currentFile = null;
      AppState.originalImageSrc = null;
      if (fileInput) fileInput.value = '';
      if (displayImg) {
        displayImg.classList.add('hidden');
        displayImg.src = '';
      }
      if (emptyState) emptyState.classList.remove('hidden');

      if (errorMsg) {
        errorMsg.textContent = validation.rejection_reason || 'Image is not a recognized vehicle panel.';
      }
      if (errorBanner) errorBanner.classList.remove('hidden');
      if (window.lucide) lucide.createIcons();
      return false;
    }

    // ACCEPT: Verified vehicle panel
    AppState.currentFile = file;
    const nameLabel = document.getElementById('file-name-label');
    if (nameLabel) {
      nameLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    }
    if (previewInfo) previewInfo.classList.remove('hidden');

    const panelName = validation.diagnostics?.target_panel || 'Vehicle Body Panel';
    const surfaceType = validation.diagnostics?.surface_type || 'Automotive Painted Surface';
    const defectInfo = validation.diagnostics?.damage_signal || `${validation.defect_count} defect contours detected`;
    
    if (successDetails) {
      successDetails.textContent = `${panelName} approved: ${surfaceType}. ${defectInfo}.`;
    }
    if (successBanner) successBanner.classList.remove('hidden');

    const reader = new FileReader();
    reader.onload = (e) => {
      AppState.originalImageSrc = e.target.result;
      if (displayImg) {
        displayImg.src = AppState.originalImageSrc;
        displayImg.classList.remove('hidden');
      }
      if (emptyState) emptyState.classList.add('hidden');
    };
    reader.readAsDataURL(file);

    if (window.lucide) lucide.createIcons();
    return true;
  } catch (err) {
    console.error('Validation error:', err);
    if (errorMsg) errorMsg.textContent = `Verification error: ${err.message}`;
    if (errorBanner) errorBanner.classList.remove('hidden');
    AppState.currentFile = null;
    if (fileInput) fileInput.value = '';
    return false;
  } finally {
    if (spinner) spinner.classList.add('hidden');
  }
}

/**
 * Loads one of the built-in 1-click test scenario presets.
 */
async function loadSamplePreset(url, panelId, label) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load preset at ${url}`);
    const blob = await res.blob();
    const file = new File([blob], `${label.toLowerCase().replace(/\s+/g, '_')}.jpg`, { type: 'image/jpeg' });
    
    const panelSelect = document.getElementById('panel-select');
    if (panelSelect) panelSelect.value = panelId;

    const isValid = await handleFileSelected(file);
    if (isValid) {
      setTimeout(() => executeInspection(), 150);
    }
  } catch (err) {
    console.error('Failed to load sample image:', err);
    showToast(`Failed to load preset: ${err.message}`, 'error');
  }
}

/**
 * Executes the full damage inspection and appraisal calculation pipeline.
 */
async function executeInspection() {
  if (AppState.isInspecting || !AppState.currentFile) return;

  const loader = document.getElementById('loader');
  const btnAnalyze = document.getElementById('btn-analyze');
  AppState.isInspecting = true;

  if (loader) loader.classList.remove('hidden');
  if (btnAnalyze) {
    btnAnalyze.disabled = true;
    btnAnalyze.classList.add('opacity-75', 'cursor-not-allowed');
  }

  const formData = new FormData();
  formData.append('file', AppState.currentFile);
  formData.append('panel_id', document.getElementById('panel-select')?.value || 'FRONT_BUMPER');
  formData.append('sensitivity', document.getElementById('sensitivity-slider')?.value || '0.55');
  formData.append('detect_scratches', document.getElementById('check-scratches')?.checked ?? true);
  formData.append('detect_dents', document.getElementById('check-dents')?.checked ?? true);
  formData.append('use_oem_parts', document.getElementById('parts-tier')?.value === 'true');
  formData.append('deductible', document.getElementById('deductible-input')?.value || '500');
  formData.append('blend_adjacent_panels', document.getElementById('blend-select')?.value === 'true');
  formData.append('zip_code', (document.getElementById('zip-input')?.value || '').trim());

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Analysis request failed' }));
      throw new Error(err.detail || 'Analysis request failed');
    }

    const data = await res.json();
    AppState.currentAnalysis = data.analysis;
    AppState.currentEstimate = data.estimate;

    renderResults(AppState.currentAnalysis, AppState.currentEstimate, data.telemetry);
    switchTab('annotated');
    showToast('Inspection appraisal completed successfully', 'success');
  } catch (err) {
    showToast(`Error running CV analysis: ${err.message}`, 'error');
  } finally {
    AppState.isInspecting = false;
    if (loader) loader.classList.add('hidden');
    if (btnAnalyze) {
      btnAnalyze.disabled = false;
      btnAnalyze.classList.remove('opacity-75', 'cursor-not-allowed');
    }
  }
}

/**
 * Renders telemetry, defect list, and actuarial invoice line items.
 */
function renderResults(analysis, estimate, telemetry) {
  // 1. Show Telemetry Bar
  const telemetryBar = document.getElementById('telemetry-bar');
  if (telemetryBar) telemetryBar.classList.remove('hidden');

  const defectsEl = document.getElementById('telemetry-defects');
  if (defectsEl) {
    defectsEl.textContent = `${analysis.items_detected.length} items`;
  }

  const areaEl = document.getElementById('telemetry-area');
  if (areaEl) {
    areaEl.textContent = `${analysis.total_damaged_area_cm2.toFixed(1)} cm² (${estimate.panel_damage_percentage}%)`;
  }
  
  const regionEl = document.getElementById('telemetry-region');
  if (regionEl) {
    regionEl.textContent = estimate.regional_market || 'US Baseline';
  }

  const latencyEl = document.getElementById('telemetry-latency');
  if (latencyEl) {
    const isHit = telemetry && telemetry.cache_hit;
    const latMs = telemetry && telemetry.total_latency_ms ? `${telemetry.total_latency_ms} ms` : '< 85 ms';
    latencyEl.textContent = `${latMs} ${isHit ? '(Cache Hit ⚡)' : '(Thread Pool)'}`;
    latencyEl.className = isHit ? 'font-bold text-sky-400 text-sm' : 'font-bold text-emerald-400 text-sm';
  }

  const severityEl = document.getElementById('telemetry-severity');
  if (severityEl) {
    severityEl.textContent = analysis.overall_severity;
    if (analysis.overall_severity === 'SEVERE') {
      severityEl.className = 'font-bold text-rose-400 text-sm';
    } else if (analysis.overall_severity === 'MODERATE') {
      severityEl.className = 'font-bold text-amber-400 text-sm';
    } else {
      severityEl.className = 'font-bold text-emerald-400 text-sm';
    }
  }

  // 2. Render Detected Defects List
  const defectsContainer = document.getElementById('defects-container');
  const defectsList = document.getElementById('defects-list');
  if (defectsList) {
    defectsList.innerHTML = '';

    if (analysis.items_detected.length === 0) {
      defectsList.innerHTML = `
        <div class="p-3 rounded-xl bg-slate-950 text-xs text-slate-400 flex items-center gap-2">
          <i data-lucide="check-circle" class="w-4 h-4 text-emerald-400"></i>
          <span>No severe surface fractures or dents detected at this threshold.</span>
        </div>`;
    } else {
      analysis.items_detected.forEach((item) => {
        const badgeColor = item.severity === 'SEVERE' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
          : item.severity === 'MODERATE' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
          : 'bg-yellow-500/10 text-yellow-300 border-yellow-500/20';

        const div = document.createElement('div');
        div.className = 'p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex items-center justify-between text-xs';
        div.innerHTML = `
          <div class="flex items-center space-x-3">
            <span class="font-mono text-slate-400 font-semibold">#${item.id}</span>
            <div>
              <div class="font-semibold text-slate-200">${item.details}</div>
              <div class="text-[11px] text-slate-500">Box: [x:${item.bounding_box.x}, y:${item.bounding_box.y}, w:${item.bounding_box.width}, h:${item.bounding_box.height}]</div>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded text-[11px] font-semibold border ${badgeColor}">
              ${item.severity}
            </span>
            <span class="text-slate-400 text-[11px] font-mono">Conf: ${(item.confidence_score * 100).toFixed(0)}%</span>
          </div>
        `;
        defectsList.appendChild(div);
      });
    }
  }
  if (defectsContainer) defectsContainer.classList.remove('hidden');

  // 3. Render Insurance Estimate Section
  const estCard = document.getElementById('estimate-card');
  const claimIdBadge = document.getElementById('claim-id-badge');
  if (claimIdBadge) claimIdBadge.textContent = estimate.claim_id;

  const actionTitle = document.getElementById('action-title');
  const actionReason = document.getElementById('action-reason');
  const actionBanner = document.getElementById('action-banner');

  if (estimate.recommended_action === 'REPLACE_PANEL') {
    if (actionTitle) {
      actionTitle.textContent = 'REPLACE PANEL (Total Replacement)';
      actionTitle.className = 'text-xl font-bold text-rose-400';
    }
    if (actionBanner) actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-rose-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  } else if (estimate.recommended_action === 'REPAIR_AND_REFINISH') {
    if (actionTitle) {
      actionTitle.textContent = 'REPAIR & REFINISH (Standard Collision)';
      actionTitle.className = 'text-xl font-bold text-emerald-400';
    }
    if (actionBanner) actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-emerald-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  } else {
    if (actionTitle) {
      actionTitle.textContent = 'BUFF & POLISH ONLY (Minor Cosmetic)';
      actionTitle.className = 'text-xl font-bold text-sky-400';
    }
    if (actionBanner) actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-sky-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  }
  if (actionReason) actionReason.textContent = estimate.action_reason;

  // Render Line Items Table
  const tbody = document.getElementById('line-items-body');
  if (tbody) {
    tbody.innerHTML = '';
    estimate.line_items.forEach((item) => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/30 transition';
      tr.innerHTML = `
        <td class="py-2.5 px-3 font-mono text-[11px] text-slate-400">${item.category}</td>
        <td class="py-2.5 px-3 font-medium text-slate-200">${item.description}</td>
        <td class="py-2.5 px-3 text-center text-slate-400">${item.quantity_or_hours.toFixed(1)}</td>
        <td class="py-2.5 px-3 text-right text-slate-400">$${item.rate.toFixed(2)}</td>
        <td class="py-2.5 px-3 text-right font-semibold text-slate-200">$${item.subtotal.toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Financial Totals
  const sumParts = document.getElementById('sum-parts');
  if (sumParts) sumParts.textContent = `$${estimate.parts_subtotal.toFixed(2)}`;

  const sumLabor = document.getElementById('sum-labor');
  if (sumLabor) sumLabor.textContent = `$${(estimate.body_labor_subtotal + estimate.paint_labor_subtotal).toFixed(2)} (${estimate.total_labor_hours} hrs)`;

  const sumSupplies = document.getElementById('sum-supplies');
  if (sumSupplies) sumSupplies.textContent = `$${(estimate.paint_materials_subtotal + estimate.fees_subtotal).toFixed(2)}`;

  const sumTax = document.getElementById('sum-tax');
  if (sumTax) sumTax.textContent = `$${estimate.sales_tax.toFixed(2)}`;

  const sumGross = document.getElementById('sum-gross');
  if (sumGross) sumGross.textContent = `$${estimate.gross_estimate_total.toFixed(2)}`;

  const sumDeductible = document.getElementById('sum-deductible');
  if (sumDeductible) sumDeductible.textContent = `- $${estimate.deductible.toFixed(2)}`;

  const sumNet = document.getElementById('sum-net');
  if (sumNet) sumNet.textContent = `$${estimate.net_claim_payout.toFixed(2)}`;

  if (estCard) estCard.classList.remove('hidden');
  if (window.lucide) lucide.createIcons();
}

/**
 * Switches between Annotated Inspection, Mask, and Original Image tabs.
 */
function switchTab(tab) {
  AppState.activeTab = tab;
  const btnAnnotated = document.getElementById('tab-annotated');
  const btnMask = document.getElementById('tab-mask');
  const btnOriginal = document.getElementById('tab-original');
  const displayImg = document.getElementById('img-display');

  [btnAnnotated, btnMask, btnOriginal].forEach((b) => {
    if (b) b.className = 'px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition';
  });

  if (tab === 'annotated') {
    if (btnAnnotated) btnAnnotated.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-500 text-white transition';
    if (displayImg && AppState.currentAnalysis) {
      displayImg.src = AppState.currentAnalysis.annotated_image_base64;
    }
  } else if (tab === 'mask') {
    if (btnMask) btnMask.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white transition';
    if (displayImg && AppState.currentAnalysis) {
      displayImg.src = AppState.currentAnalysis.mask_image_base64;
    }
  } else {
    if (btnOriginal) btnOriginal.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 text-white transition';
    if (displayImg && AppState.originalImageSrc) {
      displayImg.src = AppState.originalImageSrc;
    }
  }
}

/**
 * Requests and downloads the insurance PDF appraisal report.
 */
async function downloadAppraisalPDF() {
  if (!AppState.currentAnalysis || !AppState.currentEstimate) {
    showToast('Please execute an inspection first.', 'warning');
    return;
  }

  const btn = document.getElementById('btn-export-pdf');
  const originalHtml = btn ? btn.innerHTML : '';
  if (btn) {
    btn.innerHTML = `<span class="animate-spin mr-1">⌛</span> Generating PDF...`;
    btn.disabled = true;
  }

  try {
    const res = await fetch('/api/export-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis: AppState.currentAnalysis,
        estimate: AppState.currentEstimate,
      }),
    });

    if (!res.ok) throw new Error('PDF generation request failed.');

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ClaimScope_Appraisal_${AppState.currentEstimate.claim_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    showToast('Official PDF appraisal downloaded successfully', 'success');
  } catch (err) {
    showToast(`Failed to download PDF: ${err.message}`, 'error');
  } finally {
    if (btn) {
      btn.innerHTML = originalHtml;
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }
  }
}

/**
 * Displays a clean, modern floating toast notification.
 */
function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'fixed bottom-5 right-5 z-50 flex flex-col space-y-2 pointer-events-none max-w-sm';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  const bg = type === 'success' ? 'bg-emerald-950/90 border-emerald-700/80 text-emerald-200'
    : type === 'error' ? 'bg-rose-950/90 border-rose-700/80 text-rose-200'
    : type === 'warning' ? 'bg-amber-950/90 border-amber-700/80 text-amber-200'
    : 'bg-slate-900/90 border-slate-700 text-slate-200';

  const iconName = type === 'success' ? 'check-circle' : type === 'error' ? 'alert-triangle' : 'info';

  toast.className = `p-3 rounded-xl border shadow-2xl backdrop-blur-md flex items-center space-x-2.5 text-xs font-medium pointer-events-auto transition-all duration-300 transform translate-y-2 opacity-0 ${bg}`;
  toast.innerHTML = `
    <i data-lucide="${iconName}" class="w-4 h-4 shrink-0"></i>
    <span class="flex-1">${message}</span>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  // Animate in
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  // Auto remove after 3.5s
  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
