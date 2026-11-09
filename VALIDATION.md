# ClaimScope Validation & Actuarial Calibration

This document outlines the operational design domain (ODD), mathematical heuristics, calibration sources, and known limitations of the ClaimScope computer vision and collision estimating engine.

---

## 1. Operational Design Domain (ODD)

ClaimScope is designed as an automated **First Notice of Loss (FNOL) triage and preliminary appraisal tool** for vehicle body panels. It operates in real-time (< 90ms) on standard CPU hardware without external GPU servers or closed-source commercial APIs.

### Scope of Intended Operation
* **Target Panels:** Exterior automotive body panels including Front Bumper, Rear Bumper, Hood, Doors, Fenders, Quarter Panels, Trunk/Tailgate, and Windshield.
* **Target Anomalies:** Surface scratches (clear-coat and base-coat fractures), sheet metal dents, plastic bumper deformation, paint abrasions, and rock chips.
* **Lighting Conditions:** Diffuse daylight, open shade, or standard automotive shop inspection lighting (EV 10 to EV 15).
* **Input Formats:** JPEG, PNG, and WebP images with a minimum resolution of 150×150 pixels and a maximum resolution of 4096×4096 pixels (up to 16 MP and 15 MB).

---

## 2. Computer Vision Heuristics & Limitations

ClaimScope uses a 100% Classical Computer Vision pipeline (OpenCV) rather than black-box deep learning. This guarantees deterministic, reproducible results, zero model weight downloads, and explainable bounding boxes. However, classical signal processing has inherent physical boundaries.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Classical CV Detection Pipeline                       │
├─────────────┬───────────────────────────┬───────────────────────────────────┤
│ Stage       │ Algorithm                 │ Physical Principle                │
├─────────────┼───────────────────────────┼───────────────────────────────────┤
│ Smoothing   │ Bilateral Filter          │ Preserves sharp scratch edges     │
│             │ (d=9, σ_color=75, σ_space=75) while smoothing automotive paint│
│ Contrast    │ CLAHE on LAB L-channel    │ Balances uneven illumination and  │
│             │ (clipLimit=2.5, tile=8x8) │ reveals faint scrapes in shadows  │
│ Edges       │ Directional Canny + Sobel │ Measures steep spatial gradients  │
│             │ (hysteresis thresholds)   │ where paint coat is broken        │
│ Morphology  │ Morphological Close/Open  │ Bridges fractured scratch paths   │
│             │ (elliptical kernels)      │ into continuous defect contours   │
│ Geometry    │ Green's Theorem Contours  │ Computes physical surface area    │
│             │ (cv2.findContours)        │ in cm² calibrated by panel bounds │
└─────────────┴───────────────────────────┴───────────────────────────────────┘
```

### Documented Heuristic Limits

1. **Specular Highlights & Direct Glare:**
   * *Phenomenon:* Direct sunlight or high-intensity spotlights hitting metallic or pearlescent clear-coat can produce localized high-gradient white glare spots.
   * *Engine Handling:* The pre-flight validator checks luminance distribution and rejects blown-out images (brightness mean > 252.0). Bilateral filtering smooths mild glare, but extreme hot-spot reflections may trigger false positive edge contours.
   * *Mitigation:* Capture photos in shaded or indirect lighting.

2. **Environmental Contamination (Dirt, Salt, Water Droplets):**
   * *Phenomenon:* Dried mud streaks, road salt spray, or rain beads on a vehicle panel present high-frequency spatial gradients that can mimic scratch patterns.
   * *Engine Handling:* Pre-flight validation checks for automotive paint uniformity, but cannot distinguish dry salt crusts from paint abrasions.
   * *Operational Prerequisite:* Panels should be wiped clean of surface dirt and road debris prior to inspection photography.

3. **Subsurface & Hidden Structural Damage:**
   * *Phenomenon:* Minor external bumper fascia deformation can conceal cracked reinforcement bars, warped crash boxes, punctured radiators, or severed ADAS radar/ultrasonic sensor wiring.
   * *Engine Boundary:* Single-angle monocular RGB photography cannot inspect internal substructure or measure sub-millimeter frame alignment.
   * *Disclaimer:* ClaimScope estimates reflect visible exterior body and paint repair operations only.

4. **Monocular Scale Approximation:**
   * *Phenomenon:* Converting image pixels to square centimeters requires reference scaling.
   * *Engine Handling:* ClaimScope maps detected defect pixel bounding boxes proportionally against standard OEM panel surface area definitions (`app/estimator/parts_catalog.py`). If a photo is an extreme close-up showing only a 5 cm patch of a 1,100 cm² door panel, the area estimation heuristic uses normalized focal scaling (`10 px/cm` default).

---

## 3. Actuarial Calibration & Repair Standards

ClaimScope’s pricing engine is calibrated against standard North American collision repair formulas derived from **Mitchell International, CCC ONE, and Audatex** estimating guides.

### Key Actuarial Constants & Formulas

| Metric | Calibrated Value | Industry Basis & Formula |
| :--- | :--- | :--- |
| **National Baseline Labor Rate** | $65.00 / hr | Standard US collision repair body/paint labor rate. |
| **Paint Materials Allowance** | $38.00 / refinish hr | Mitchell standard rate for paint toner, clearcoat, solvent, and spray booth consumables. |
| **The 28% Replacement Rule** | 28.0% panel area (or $\ge 250\text{ cm}^2$) | When defect area $A_{\text{defect}} \ge 0.28 \times A_{\text{panel}}$ (or $\ge 250\text{ cm}^2$), repair labor exceeds practical replacement value; replacement is recommended. |
| **Mitchell Adjacent Blend Rule** | 50% refinish time | When blending adjacent panels for metallic color match, Mitchell assigns 50% of standard paint refinish hours. |
| **Structural Aluminum Surcharge** | 1.25× (25% surcharge) | Specialized clean-room tooling, dedicated welders, and explosion-proof dust extraction for aluminum hoods/doors. |
| **High-Strength Steel Surcharge** | 1.15× (15% surcharge) | Boron steel / ultra-high-strength steel requiring specialized unibody clamping and laser welding. |
| **Windshield Glass Safety Rule** | 100% Replacement | Structural ADAS windshields are always replaced rather than cosmetically filled to ensure optical and safety compliance. |

### Regional Labor Rate Indexing

Body and paint labor rates are dynamically indexed against 5 US prevailing wage tiers in `app/estimator/regional_rates.py`:

| Market Code | Name | Body & Paint Rate | Representative Metro ZIPs |
| :--- | :--- | :--- | :--- |
| `PACIFIC_METRO` | Pacific Metro (CA / WA / OR) | $91.00 / hr (1.40×) | 94103 (San Francisco), 98101 (Seattle), 90210 (Los Angeles) |
| `NORTHEAST_METRO` | Northeast Metro (NY / MA / NJ) | $87.75 / hr (1.35×) | 10001 (New York), 02108 (Boston) |
| `SOUTHEAST_CENTRAL` | South / Southeast (TX / FL / GA) | $74.75 / hr (1.15×) | 30301 (Atlanta), 33101 (Miami), 75201 (Dallas) |
| `MIDWEST_MOUNTAIN` | Midwest / Mountain (IL / OH / CO) | $68.25 / hr (1.05×) | 60601 (Chicago), 48226 (Detroit) |
| `NATIONAL_BASELINE` | US National Baseline | $65.00 / hr (1.00×) | Standard national benchmark / fallback jurisdictions |

---

## 4. Benchmark Evaluation Suite

The test suite in [tests/test_validation_benchmarks.py](tests/test_validation_benchmarks.py) verifies ClaimScope against labeled reference scenarios:

1. **Clean Baseline:** Pristine painted panel produces zero defects ($0 damage area) and triggers pre-flight validation rejection.
2. **Superficial Clear-Coat Scratch:** Isolated minor damage ($< 5\text{ cm}^2$) triggers `BUFF_AND_POLISH_ONLY` at 0.8h labor and $0 parts cost.
3. **Moderate Collision Dent:** Damage area between 15% and 28% triggers `REPAIR_AND_REFINISH` with body labor, paint labor, and paint supply line items.
4. **Severe Extensive Deformation:** Damage area $\ge$ 28% triggers `REPLACE_PANEL`, OEM replacement price, and R&I labor.
5. **Safety-Critical Glass:** Windshield damage triggers mandatory replacement regardless of minor defect size.
6. **Mathematical Determinism:** Repeated executions on the same image yield identical defect counts, bounding boxes, contour areas, and dollar amounts ($0.0000 variance).
7. **Collision-Proof Deductive Precision:** Validates that deductibles ($500.00 vs $500.50) and sensitivities produce non-colliding cache keys.

---

## 5. Regulatory & Compliance Notice

> **IMPORTANT DISCLAIMER**  
> ClaimScope is designed as an automated decision-support tool for preliminary First Notice of Loss (FNOL) triage, vehicle owner estimation, and repair shop initial screening. It does not replace a physical tear-down inspection by a licensed, I-CAR or ASE-certified automotive collision appraiser. ClaimScope makes no legal warranty regarding insurance policy claim acceptance.
