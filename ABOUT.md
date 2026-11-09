# Project Summary: AutoDamage CV

## Overview
**AutoDamage CV** is an open-source vehicle damage inspection and collision repair cost estimator. It analyzes digital photographs of car panels, identifies surface defects such as scratches, dents, and paint abrasions, measures the damaged area, and generates an itemized repair estimate using real-world collision industry standards.

Unlike typical automated claims demos that rely on opaque neural networks and expensive GPU hosting, AutoDamage CV uses **classical computer vision (OpenCV)** and **transparent actuarial rules (Mitchell / CCC ONE guidelines)** to produce explainable, verifiable appraisals in under 90 milliseconds on a standard CPU.

---

## The Problem
Getting a repair estimate after an accident or parking lot scrape is traditionally a slow and opaque process:
1. **Slow Turnarounds:** Vehicle owners often wait days or weeks for an insurance adjuster or body shop to examine photos and return a preliminary quote.
2. **Lack of Transparency:** Repair quotes frequently appear as arbitrary lumped totals, leaving owners and adjusters unsure of how specific labor hours or replacement decisions were calculated.
3. **The AI "Black Box" Problem:** Deep-learning models that output a confidence score without verifiable physical measurements are legally difficult to defend in insurance disputes and claim audits.

---

## The Solution
AutoDamage CV bridges the gap between image processing and insurance appraisal:
* **Pre-Flight Input Guardrails:** Before inspection begins, a classical CV verification filter validates that the photo depicts an authentic vehicle body panel with visible damage. It actively rejects non-car photos (selfies, documents, invoices, blur, domestic clutter, and undamaged cars), preventing invalid files from entering the pipeline.
* **Panel-Specific Material Intelligence:** Differentiates between flexible molded polyurethane plastic (front/rear bumper covers) and rigid sheet metal/aluminum (hoods, doors, fenders), validating the specific surface texture and damage pattern.
* **Measurable Damage Extraction:** Instead of guessing severity, the computer vision pipeline extracts concrete pixel gradients and boundary contours, mapping them to physical area in square centimeters (cm²).
* **Standardized Repair Logic:** Cost calculations follow standard US collision estimating guidelines:
  * **The 28% Surface Rule:** Panels with damage covering $\ge 28\%$ of the surface area (or severe tearing) are marked for full replacement; smaller damages are marked for sheet metal repair and refinishing.
  * **Safety Mandates:** Windshields and critical glass are always replaced upon fracture.
  * **Itemized Rates:** Calculates body labor ($65/hr), paint refinishing labor ($65/hr), paint supplies ($38/hr), EPA waste disposal fees, and state sales tax.
* **Instant Documentation:** Produces an official, itemized Motor Vehicle Damage Appraisal Report in PDF format matching standard insurance adjuster sheets.

---

## Architectural Highlights
* **Strategy Pattern (`app/cv_engine/base.py`):** The detection layer implements an abstract `BaseDamageDetector` interface, cleanly decoupling signal processing from pricing logic. A machine learning model (such as YOLOv8-seg) can be dropped in during future development without rewriting the financial rules or UI.
* **Pre-Flight Filter (`app/cv_engine/image_validator.py`):** Multi-stage validation checking resolution, lighting exposure, white document variance, human skin tone in YCrCb, focus via Laplacian variance & Sobel gradient magnitude, LAB chrominance uniformity, and damage contour presence.
* **Three Access Channels:**
  * **Web Dashboard:** An interactive browser interface with 1-click test scenarios, sensitivity sliders, pre-flight verification badges, and side-by-side mask inspection views.
  * **Command-Line Interface (`cli.py`):** A headless CLI tool for terminal workflows, automated scripting, and batch appraisal generation.
  * **REST API:** Clean FastAPI endpoints returning structured JSON schemas (validated via Pydantic).
* **CPU Efficiency:** Complete image processing and estimate calculation finish in under 90ms without GPUs or third-party cloud API costs.

---

## Project Status & Metrics
* **Test Suite:** 17 passing `pytest` tests covering input validation guardrails, boundary conditions, deductible bounds, safety glass rules, and edge detection reproducibility.
* **CI/CD:** Automated GitHub Actions pipeline verifying builds and test suites on every push.
* **Containerization:** Multi-stage `Dockerfile` with automated test verification during container build.
* **License:** Dual-licensed under the MIT License and Apache License 2.0.
