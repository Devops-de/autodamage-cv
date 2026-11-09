/**
 * AutoDamage CV - Frontend Controller
 * Communicates with FastAPI backend for Classical CV detection & actuarial cost calculation.
 */

let currentFile = null;
let currentAnalysis = null;
let currentEstimate = null;
let originalImageSrc = null;
let activeTab = 'annotated'; // 'annotated' | 'mask' | 'original'

document.addEventListener('DOMContentLoaded', () => {
  initUI();
  initEventListeners();
});

function initUI() {
  const slider = document.getElementById('sensitivity-slider');
  const sliderVal = document.getElementById('sensitivity-value');
  slider.addEventListener('input', (e) => {
    sliderVal.textContent = parseFloat(e.target.value).toFixed(2);
  });
}

function initEventListeners() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

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

  // 1-Click Samples
  document.getElementById('btn-sample-bumper').addEventListener('click', () => {
    loadSamplePreset('/sample_images/sample_bumper_scratch.jpg', 'FRONT_BUMPER', 'Bumper Scratch');
  });

  document.getElementById('btn-sample-door').addEventListener('click', () => {
    loadSamplePreset('/sample_images/sample_door_dent.jpg', 'DRIVER_FRONT_DOOR', 'Door Dent');
  });

  document.getElementById('btn-sample-fender').addEventListener('click', () => {
    loadSamplePreset('/sample_images/sample_fender_chip.jpg', 'FRONT_FENDER_LEFT', 'Fender Chip');
  });

  // Execute Inspection Button
  document.getElementById('btn-analyze').addEventListener('click', () => {
    if (!currentFile) {
      alert('Please upload a car photograph or select one of the 1-click test samples first.');
      return;
    }
    executeInspection();
  });

  // View Studio Tabs
  document.getElementById('tab-annotated').addEventListener('click', () => switchTab('annotated'));
  document.getElementById('tab-mask').addEventListener('click', () => switchTab('mask'));
  document.getElementById('tab-original').addEventListener('click', () => switchTab('original'));

  // Export PDF Button
  document.getElementById('btn-export-pdf').addEventListener('click', downloadAppraisalPDF);

  // About Modal Listeners
  const aboutModal = document.getElementById('about-modal');
  const btnAbout = document.getElementById('btn-about');
  const btnCloseAbout = document.getElementById('btn-close-about');
  const btnAboutOk = document.getElementById('btn-about-ok');

  const openAbout = () => {
    aboutModal.classList.remove('hidden');
    lucide.createIcons();
  };
  const closeAbout = () => aboutModal.classList.add('hidden');

  if (btnAbout) btnAbout.addEventListener('click', openAbout);
  if (btnCloseAbout) btnCloseAbout.addEventListener('click', closeAbout);
  if (btnAboutOk) btnAboutOk.addEventListener('click', closeAbout);
  if (aboutModal) {
    aboutModal.addEventListener('click', (e) => {
      if (e.target === aboutModal) closeAbout();
    });
  }
}

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
  errorBanner.classList.add('hidden');
  successBanner.classList.add('hidden');
  previewInfo.classList.add('hidden');
  spinner.classList.remove('hidden');

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/validate-image', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error('Image validation pre-check failed on server.');
    }

    const validation = await res.json();

    if (!validation.is_valid) {
      // REJECT: Do not load or display image into canvas
      currentFile = null;
      originalImageSrc = null;
      fileInput.value = '';
      displayImg.classList.add('hidden');
      displayImg.src = '';
      emptyState.classList.remove('hidden');

      errorMsg.textContent = validation.rejection_reason || 'Image is not a recognized vehicle panel.';
      errorBanner.classList.remove('hidden');
      if (window.lucide) lucide.createIcons();
      return false;
    }

    // ACCEPT: Verified vehicle panel
    currentFile = file;
    document.getElementById('file-name-label').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    previewInfo.classList.remove('hidden');

    if (validation.diagnostics && validation.diagnostics.focus_quality) {
      successDetails.textContent = `Approved: ${validation.diagnostics.focus_quality} focus, automotive surface verified.`;
    } else {
      successDetails.textContent = 'Surface texture, focus, and panel geometry approved for inspection.';
    }
    successBanner.classList.remove('hidden');

    const reader = new FileReader();
    reader.onload = (e) => {
      originalImageSrc = e.target.result;
      displayImg.src = originalImageSrc;
      displayImg.classList.remove('hidden');
      emptyState.classList.add('hidden');
    };
    reader.readAsDataURL(file);

    if (window.lucide) lucide.createIcons();
    return true;
  } catch (err) {
    console.error('Validation error:', err);
    errorMsg.textContent = `Verification error: ${err.message}`;
    errorBanner.classList.remove('hidden');
    currentFile = null;
    fileInput.value = '';
    return false;
  } finally {
    spinner.classList.add('hidden');
  }
}

async function loadSamplePreset(url, panelId, label) {
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    const file = new File([blob], `${label.toLowerCase().replace(/\s+/g, '_')}.jpg`, { type: 'image/jpeg' });
    
    document.getElementById('panel-select').value = panelId;
    const isValid = await handleFileSelected(file);
    if (isValid) {
      // Auto-execute inspection for smooth 1-click test experience
      setTimeout(() => executeInspection(), 150);
    }
  } catch (err) {
    console.error('Failed to load sample image:', err);
    alert('Failed to load sample image.');
  }
}

async function executeInspection() {
  const loader = document.getElementById('loader');
  loader.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('panel_id', document.getElementById('panel-select').value);
  formData.append('sensitivity', document.getElementById('sensitivity-slider').value);
  formData.append('detect_scratches', document.getElementById('check-scratches').checked);
  formData.append('detect_dents', document.getElementById('check-dents').checked);
  formData.append('use_oem_parts', document.getElementById('parts-tier').value === 'true');
  formData.append('deductible', document.getElementById('deductible-input').value);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis request failed');
    }

    const data = await res.json();
    currentAnalysis = data.analysis;
    currentEstimate = data.estimate;

    renderResults(currentAnalysis, currentEstimate);
    switchTab('annotated');
  } catch (err) {
    alert(`Error running CV analysis: ${err.message}`);
  } finally {
    loader.classList.add('hidden');
  }
}

function renderResults(analysis, estimate) {
  // 1. Show Telemetry Bar
  document.getElementById('telemetry-bar').classList.remove('hidden');
  document.getElementById('telemetry-defects').textContent = `${analysis.items_detected.length} items`;
  document.getElementById('telemetry-area').textContent = `${analysis.total_damaged_area_cm2.toFixed(1)} cm² (${estimate.panel_damage_percentage}%)`;

  const severityEl = document.getElementById('telemetry-severity');
  severityEl.textContent = analysis.overall_severity;
  if (analysis.overall_severity === 'SEVERE') {
    severityEl.className = 'font-bold text-rose-400 text-sm';
  } else if (analysis.overall_severity === 'MODERATE') {
    severityEl.className = 'font-bold text-amber-400 text-sm';
  } else {
    severityEl.className = 'font-bold text-emerald-400 text-sm';
  }

  // 2. Render Detected Defects List
  const defectsContainer = document.getElementById('defects-container');
  const defectsList = document.getElementById('defects-list');
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
  defectsContainer.classList.remove('hidden');

  // 3. Render Insurance Estimate Section
  const estCard = document.getElementById('estimate-card');
  document.getElementById('claim-id-badge').textContent = estimate.claim_id;

  const actionTitle = document.getElementById('action-title');
  const actionReason = document.getElementById('action-reason');
  const actionBanner = document.getElementById('action-banner');

  if (estimate.recommended_action === 'REPLACE_PANEL') {
    actionTitle.textContent = 'REPLACE PANEL (Total Replacement)';
    actionTitle.className = 'text-xl font-bold text-rose-400';
    actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-rose-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  } else if (estimate.recommended_action === 'REPAIR_AND_REFINISH') {
    actionTitle.textContent = 'REPAIR & REFINISH (Standard Collision)';
    actionTitle.className = 'text-xl font-bold text-emerald-400';
    actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-emerald-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  } else {
    actionTitle.textContent = 'BUFF & POLISH ONLY (Minor Cosmetic)';
    actionTitle.className = 'text-xl font-bold text-sky-400';
    actionBanner.className = 'p-5 border-b border-slate-800 bg-gradient-to-r from-sky-950/40 to-slate-900 flex flex-col sm:flex-row sm:items-center justify-between gap-3';
  }
  actionReason.textContent = estimate.action_reason;

  // Render Line Items Table
  const tbody = document.getElementById('line-items-body');
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

  // Financial Totals
  document.getElementById('sum-parts').textContent = `$${estimate.parts_subtotal.toFixed(2)}`;
  document.getElementById('sum-labor').textContent = `$${(estimate.body_labor_subtotal + estimate.paint_labor_subtotal).toFixed(2)} (${estimate.total_labor_hours} hrs)`;
  document.getElementById('sum-supplies').textContent = `$${(estimate.paint_materials_subtotal + estimate.fees_subtotal).toFixed(2)}`;
  document.getElementById('sum-tax').textContent = `$${estimate.sales_tax.toFixed(2)}`;
  document.getElementById('sum-gross').textContent = `$${estimate.gross_estimate_total.toFixed(2)}`;
  document.getElementById('sum-deductible').textContent = `- $${estimate.deductible.toFixed(2)}`;
  document.getElementById('sum-net').textContent = `$${estimate.net_claim_payout.toFixed(2)}`;

  estCard.classList.remove('hidden');
  lucide.createIcons();
}

function switchTab(tab) {
  activeTab = tab;
  const btnAnnotated = document.getElementById('tab-annotated');
  const btnMask = document.getElementById('tab-mask');
  const btnOriginal = document.getElementById('tab-original');
  const displayImg = document.getElementById('img-display');

  // Reset tab button styling
  [btnAnnotated, btnMask, btnOriginal].forEach((b) => {
    b.className = 'px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition';
  });

  if (tab === 'annotated') {
    btnAnnotated.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-500 text-white transition';
    if (currentAnalysis) {
      displayImg.src = currentAnalysis.annotated_image_base64;
    }
  } else if (tab === 'mask') {
    btnMask.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white transition';
    if (currentAnalysis) {
      displayImg.src = currentAnalysis.mask_image_base64;
    }
  } else {
    btnOriginal.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 text-white transition';
    if (originalImageSrc) {
      displayImg.src = originalImageSrc;
    }
  }
}

async function downloadAppraisalPDF() {
  if (!currentAnalysis || !currentEstimate) {
    alert('Please execute an inspection first.');
    return;
  }

  const btn = document.getElementById('btn-export-pdf');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<span class="animate-spin mr-1">⌛</span> Generating PDF...`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/export-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis: currentAnalysis,
        estimate: currentEstimate,
      }),
    });

    if (!res.ok) throw new Error('PDF generation request failed.');

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Damage_Appraisal_${currentEstimate.claim_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Failed to download PDF: ${err.message}`);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
    lucide.createIcons();
  }
}
