# AutoDamage CV

A fast, transparent vehicle damage inspection tool and repair cost estimator. Upload a photo of a damaged car panel, and AutoDamage CV detects scratches, dents, and paint abrasions using OpenCV, measures the damaged area, and generates an itemized repair appraisal based on standard collision repair rates.

No GPU required, no external API keys, and no black-box predictions.

---

## Why I Built This

If you have ever been in a minor car accident or scraped a bumper in a parking lot, you know how frustrating the repair process can be. You take photos, wait days for an insurance adjuster or body shop to get back to you, and when you finally receive an estimate, it is rarely clear how they calculated the numbers.

Most tech demos in this space try to solve this by immediately throwing a massive deep-learning model at the problem. But in the insurance and collision repair industry, **explainability matters just as much as detection**:
- If an adjuster marks a door panel for a \$700 replacement instead of a \$250 repair, they need to justify why (e.g. damage exceeds 28% of the surface area or involves structural tearing).
- If a customer asks how paint labor was estimated, the shop needs to point to standardized refinish formulas, not a neural network confidence score.

**AutoDamage CV** bridges that gap:
1. It uses classical computer vision (edge detection, color space thresholding, and contour analysis) to find visible surface defects in **under 90ms on standard CPUs**.
2. It feeds those measurements directly into an actuarial pricing engine based on standard US collision repair guidelines (Mitchell / CCC ONE), producing a transparent line-item invoice and downloadable PDF appraisal report.

---

## How It Works

```
  ┌─────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
  │  Vehicle Photo  │ ────> │  OpenCV Vision Engine  │ ────> │ Actuarial Cost Engine  │
  │  (JPEG / PNG)   │       │  • Canny Edge Calculus │       │  • Labor & Parts Rate  │
  └─────────────────┘       │  • LAB Shadow Gradients│       │  • Repair vs. Replace  │
                            │  • Contour Area (cm²)  │       │  • Deductible Math     │
                            └────────────────────────┘       └────────────────────────┘
                                                                          │
                                                                          ▼
                                                             ┌────────────────────────┐
                                                             │  Output & Appraisal    │
                                                             │  • Visual Mask Studio  │
                                                             │  • Itemized Claim Sheet│
                                                             │  • Official PDF Report │
                                                             └────────────────────────┘
```

### 1. Detection Pipeline (OpenCV)
* **Contrast Normalization:** Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in CIE LAB color space to reveal faint scrapes under harsh sunlight or shadows.
* **Scratch Detection:** Bilateral filtering smooths flat car paint while preserving sharp surface fractures. Directional Canny edge detection and morphological kernels then connect broken scratch strokes into continuous lines.
* **Dent & Deformity Detection:** Uses Difference of Gaussians (DoG) to identify localized concave shadows and specular reflection shifts caused by dented metal or plastic.
* **Metric Extraction:** `cv2.findContours` maps boundary polygons and computes calibrated surface area in square centimeters (cm²), assigning severity tiers (Minor, Moderate, Severe).

### 2. Repair Estimating Engine
The pricing engine uses standardized US collision repair rules:
* **The 28% Rule (Repair vs. Replace):** If the damaged area covers 28% or more of the panel's total surface area (or if severe tearing is detected), the engine recommends a complete panel replacement. Otherwise, it recommends sheet metal repair and refinishing.
* **Glass Exception:** Damaged windshields are always marked for replacement to comply with automotive safety standards.
* **Itemized Costs:** Calculates body repair labor ($65/hr), paint refinish spray hours ($65/hr), paint supplies ($38/hr), EPA hazardous waste disposal fees, and state sales tax on parts.
* **Net Insurance Payout:** Deducts the policyholder's deductible to show the exact expected insurance settlement.

---

## Key Features

- ⚡ **Lightweight & Fast:** Runs full-resolution image analysis on standard CPUs in < 90ms. No CUDA setup, no GPU hosting costs.
- 🎯 **100% Explainable:** Every dollar on the estimate links to measurable contour coordinates, calibrated surface area, and standardized hourly rates.
- 🖥️ **Interactive Web Studio:** Clean, responsive web dashboard with 1-click test scenarios (Bumper Scratch, Door Dent, Fender Chip), real-time sensitivity sliders, and side-by-side inspection views.
- 💻 **Terminal CLI Tool:** Built-in command-line interface (`cli.py`) for adjusters who prefer terminal workflows or need to script batch inspections.
- 📑 **Formal PDF Generation:** Exports a formatted Motor Vehicle Damage Appraisal Report matching US insurance adjuster standards.
- 🔌 **Pluggable Architecture:** Implements the Strategy Pattern with an abstract `BaseDamageDetector` interface, making it easy to drop in a fine-tuned YOLO or segmentation model in the future without changing the pricing engine or UI.
- 🧪 **Thoroughly Tested:** Automated `pytest` test suite with 8 tests covering edge detection reproducibility, cost thresholds, deductible arithmetic, and PDF compilation.

---

## Project Structure

```
autodamage-cv/
├── app/
│   ├── main.py                  # FastAPI web server and REST endpoints
│   ├── core/
│   │   └── config.py            # Hourly rates, tax rates, and thresholds
│   ├── cv_engine/
│   │   ├── base.py              # Abstract detector interface & Pydantic schemas
│   │   ├── classical_detector.py# OpenCV edge detection & color segmentation
│   │   └── future_ai_detector.py# Extension stub for ML models
│   ├── estimator/
│   │   ├── parts_catalog.py     # OEM vs aftermarket parts catalog
│   │   └── cost_engine.py       # Actuarial repair calculation engine
│   ├── reports/
│   │   └── pdf_generator.py     # PDF appraisal report generator (fpdf2)
│   └── static/                  # Web dashboard (HTML, Tailwind CSS, JS)
├── tests/                       # Pytest test suite
│   ├── conftest.py              # Test path setup
│   ├── test_cv_detector.py      # OpenCV detection & determinism tests
│   ├── test_estimator.py        # Repair vs replace & pricing logic tests
│   └── test_pdf.py              # PDF compilation tests
├── cli.py                       # Command-line inspection tool
├── generate_samples.py          # Generates sample damage images for testing
├── Dockerfile                   # Multi-stage Docker container build
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
└── run.py                       # Local development launcher
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Devops-de/autodamage-cv.git
   cd autodamage-cv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Generate the built-in test sample images:
   ```bash
   python generate_samples.py
   ```

4. Start the application:
   ```bash
   python run.py
   ```

Open your browser to:
- **Web UI:** [http://localhost:8000](http://localhost:8000)
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Command Line Interface (CLI)

The repository includes a dedicated CLI tool for terminal users:

```bash
# Run inspection on an image and print the estimate breakdown
python cli.py inspect app/static/sample_images/sample_bumper_scratch.jpg --panel FRONT_BUMPER

# Run inspection and save the official PDF report
python cli.py inspect app/static/sample_images/sample_door_dent.jpg --panel DRIVER_FRONT_DOOR --pdf estimate.pdf

# List all supported vehicle panels, materials, and catalog rates
python cli.py list-panels
```

---

## Running Tests

Run the test suite using pytest:

```bash
pytest -v
```

All 8 tests run locally in ~1.2 seconds, verifying edge extraction, pricing rules, safety glass mandates, and PDF generation.

---

## Docker Deployment

Build and run the containerized application:

```bash
docker build -t autodamage-cv .
docker run -p 8000:8000 autodamage-cv
```

---

## Engineering Design & Trade-Offs

Detailed notes on architectural patterns, time and space complexity, and design choices can be found in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

This project is dual-licensed under either the [MIT License](LICENSE-MIT) or the [Apache License 2.0](LICENSE-APACHE). You may choose either license at your option.
