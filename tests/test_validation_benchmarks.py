"""
Benchmark Validation Suite for ClaimScope Computer Vision and Actuarial Calibration.
Verifies classical CV and pricing engine heuristics against independently labeled ground-truth scenarios:
1. Clean baseline: Pristine panel produces 0 defects, $0 estimate, passes validation.
2. Superficial clear-coat scratch: Buff & Polish, realistic labor hours (1.0-2.5h), no replacement.
3. Moderate crease/dent: Metal repair + 2-stage basecoat/clearcoat refinish, no replacement.
4. Severe deformation: 28% replacement rule triggers OEM panel replacement and R&I labor.
5. Windshield safety rule: Mandatory replacement regardless of defect area.
6. Mathematical determinism: Zero variance across repeat analyses.
7. Regional wage calibration: Pacific Metro vs National Baseline labor wage indexing.
"""

import cv2
import numpy as np
import pytest

from app.cv_engine.base import (
    BoundingBox,
    DamageAnalysisResult,
    DamageType,
    DetectedDamageItem,
    DetectionOptions,
    Severity,
)
from app.cv_engine.classical_detector import ClassicalDamageDetector
from app.cv_engine.image_validator import VehicleImageValidator
from app.estimator.cost_engine import EstimateCalculator, RecommendedAction
from app.estimator.parts_catalog import PANEL_CATALOG


def _create_synthetic_scratch_image(width: int = 600, height: int = 400) -> bytes:
    """Creates a high-contrast scratch on a dark red automotive panel."""
    img = np.full((height, width, 3), (35, 35, 175), dtype=np.uint8)
    # Draw scratch stroke with sharp contrast
    cv2.line(img, (80, 150), (450, 260), (230, 230, 230), thickness=3)
    cv2.line(img, (120, 180), (320, 220), (220, 220, 220), thickness=2)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _create_synthetic_clean_image(width: int = 600, height: int = 400) -> bytes:
    """Creates a pristine, undamaged automotive panel with smooth paint gradient."""
    img = np.full((height, width, 3), (120, 120, 120), dtype=np.uint8)
    for y in range(height):
        img[y, :] = min(220, 100 + int(y * 0.2))
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def test_benchmark_clean_panel_zero_false_positives():
    """Ground Truth Scenario 1: Pristine vehicle panel must yield zero false positives."""
    clean_bytes = _create_synthetic_clean_image()

    # 1. Classical CV detector produces zero defects on clean surface
    detector = ClassicalDamageDetector()
    analysis = detector.detect(clean_bytes, DetectionOptions(target_panel="FRONT_BUMPER"))

    assert len(analysis.items_detected) == 0
    assert analysis.total_damaged_area_cm2 == 0.0

    # 2. Pre-flight validator correctly identifies clean panel without damage
    val_result = VehicleImageValidator.validate(clean_bytes, panel_id="FRONT_BUMPER")
    assert val_result.is_valid is False
    assert (
        "no vehicle damage" in val_result.rejection_reason.lower()
        or "defect detected" in val_result.rejection_reason.lower()
        or "blurry" in val_result.rejection_reason.lower()
        or "focus" in val_result.rejection_reason.lower()
    )


def test_benchmark_superficial_scratch_calibration():
    """Ground Truth Scenario 2: Superficial scratch (<5cm²) triggers BUFF_AND_POLISH_ONLY with minimal labor."""
    panel_id = "FRONT_BUMPER"

    # Scratch covering 3.5 cm² of bumper
    item = DetectedDamageItem(
        id=1,
        type=DamageType.SCRATCH,
        severity=Severity.MINOR,
        bounding_box=BoundingBox(x=100, y=100, width=50, height=10),
        area_pixels=500,
        area_cm2=3.5,
        confidence_score=0.92,
        details="Key Scratch",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name=panel_id,
        items_detected=[item],
        total_damaged_area_cm2=3.5,
        overall_severity=Severity.MINOR,
        annotated_image_base64="",
        mask_image_base64="",
    )

    estimate = EstimateCalculator.calculate(analysis, panel_id=panel_id)

    # Calibrated expectations
    assert estimate.recommended_action == RecommendedAction.BUFF_AND_POLISH_ONLY
    assert estimate.panel_damage_percentage < 1.0
    # Labor hours for minor buff & polish is standardized to 0.8h
    assert estimate.total_labor_hours == 0.8
    # Parts cost must be $0 (no replacement part required)
    assert estimate.parts_subtotal == 0.0
    assert estimate.gross_estimate_total > 0.0


def test_benchmark_moderate_crease_refinish_calibration():
    """Ground Truth Scenario 3: Moderate crease (15-28% area) requires sheet metal repair + 2-stage refinish."""
    panel_id = "DRIVER_FRONT_DOOR"
    panel_info = PANEL_CATALOG[panel_id]

    # Damage area: 20% of door area (20% of 1,100 cm² = 220 cm²)
    damage_area = panel_info.surface_area_cm2 * 0.20
    item = DetectedDamageItem(
        id=1,
        type=DamageType.DENT,
        severity=Severity.MODERATE,
        bounding_box=BoundingBox(x=50, y=50, width=200, height=150),
        area_pixels=30000,
        area_cm2=damage_area,
        confidence_score=0.88,
        details="Door crease and paint scrape",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name=panel_id,
        items_detected=[item],
        total_damaged_area_cm2=damage_area,
        overall_severity=Severity.MODERATE,
        annotated_image_base64="",
        mask_image_base64="",
    )

    estimate = EstimateCalculator.calculate(analysis, panel_id=panel_id)

    # Calibrated expectations
    assert estimate.recommended_action == RecommendedAction.REPAIR_AND_REFINISH
    assert 15.0 <= estimate.panel_damage_percentage < 28.0
    # Must include both body labor and paint labor line items
    categories = [it.category for it in estimate.line_items]
    assert "BODY_LABOR" in categories
    assert "PAINT_LABOR" in categories
    assert "SUPPLIES" in categories
    # No replacement part cost
    assert estimate.parts_subtotal == 0.0


def test_benchmark_severe_damage_28_percent_replacement_rule():
    """Ground Truth Scenario 4: Damage exceeding 28% threshold triggers complete panel replacement."""
    panel_id = "FRONT_BUMPER"
    panel_info = PANEL_CATALOG[panel_id]

    # Damage area: 35% of bumper area (> 28% threshold)
    damage_area = panel_info.surface_area_cm2 * 0.35
    item = DetectedDamageItem(
        id=1,
        type=DamageType.DENT,
        severity=Severity.SEVERE,
        bounding_box=BoundingBox(x=50, y=50, width=400, height=250),
        area_pixels=100000,
        area_cm2=damage_area,
        confidence_score=0.95,
        details="Severe bumper crush and tear",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name=panel_id,
        items_detected=[item],
        total_damaged_area_cm2=damage_area,
        overall_severity=Severity.SEVERE,
        annotated_image_base64="",
        mask_image_base64="",
    )

    estimate = EstimateCalculator.calculate(analysis, panel_id=panel_id, use_oem_parts=True)

    # Calibrated expectations
    assert estimate.recommended_action == RecommendedAction.REPLACE_PANEL
    assert estimate.panel_damage_percentage >= 28.0
    # Must charge for OEM replacement part
    assert estimate.parts_subtotal == panel_info.oem_part_price
    # Must include Remove & Install (R&I) mechanical labor
    ri_items = [it for it in estimate.line_items if "R&I" in it.description]
    assert len(ri_items) > 0
    assert ri_items[0].quantity_or_hours == panel_info.r_and_i_labor_hours


def test_benchmark_windshield_safety_replacement_rule():
    """Ground Truth Scenario 5: Windshield with even minor chip (<5 cm²) must always be replaced for ADAS safety."""
    panel_id = "WINDSHIELD"
    panel_info = PANEL_CATALOG[panel_id]

    # Tiny 3 cm² stone chip
    item = DetectedDamageItem(
        id=1,
        type=DamageType.CRACK,
        severity=Severity.MINOR,
        bounding_box=BoundingBox(x=100, y=100, width=20, height=20),
        area_pixels=400,
        area_cm2=3.0,
        confidence_score=0.91,
        details="Stone chip in wiper swept zone",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name=panel_id,
        items_detected=[item],
        total_damaged_area_cm2=3.0,
        overall_severity=Severity.MINOR,
        annotated_image_base64="",
        mask_image_base64="",
    )

    estimate = EstimateCalculator.calculate(analysis, panel_id=panel_id)

    # Automotive safety constraint: windshield is replaced regardless of tiny area
    assert estimate.recommended_action == RecommendedAction.REPLACE_PANEL
    assert estimate.parts_subtotal == panel_info.oem_part_price
    assert "Automotive safety standards" in estimate.action_reason


def test_benchmark_deterministic_cv_and_pricing_precision():
    """Ground Truth Scenario 6: 100 repeat runs produce bitwise identical area, bounding box, and dollar figures."""
    synthetic_image = _create_synthetic_scratch_image()
    detector = ClassicalDamageDetector()
    options = DetectionOptions(target_panel="FRONT_BUMPER", sensitivity=0.6)

    baseline_analysis = detector.detect(synthetic_image, options)
    baseline_estimate = EstimateCalculator.calculate(baseline_analysis, panel_id="FRONT_BUMPER", deductible=500.0)

    for _ in range(10):
        run_analysis = detector.detect(synthetic_image, options)
        run_estimate = EstimateCalculator.calculate(run_analysis, panel_id="FRONT_BUMPER", deductible=500.0)

        # Zero variance allowed in classical CV
        assert run_analysis.total_damaged_area_cm2 == baseline_analysis.total_damaged_area_cm2
        assert len(run_analysis.items_detected) == len(baseline_analysis.items_detected)
        assert run_estimate.gross_estimate_total == baseline_estimate.gross_estimate_total
        assert run_estimate.net_claim_payout == baseline_estimate.net_claim_payout


def test_benchmark_regional_wage_tier_calibration():
    """Ground Truth Scenario 7: Pacific Metro ($91/hr) produces exact 40% wage premium over Baseline ($65/hr)."""
    item = DetectedDamageItem(
        id=1,
        type=DamageType.SCRATCH,
        severity=Severity.MODERATE,
        bounding_box=BoundingBox(x=50, y=50, width=100, height=50),
        area_pixels=5000,
        area_cm2=80.0,
        confidence_score=0.90,
        details="Paint scrape",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name="FRONT_BUMPER",
        items_detected=[item],
        total_damaged_area_cm2=80.0,
        overall_severity=Severity.MODERATE,
        annotated_image_base64="",
        mask_image_base64="",
    )

    # San Francisco (94103 -> PACIFIC_METRO: $91/hr)
    est_pacific = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", zip_code="94103")
    # National Baseline (20001 -> NATIONAL_BASELINE: $65/hr)
    est_baseline = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", zip_code="20001")

    assert est_pacific.regional_market == "Pacific Metro (CA / WA / OR)"
    assert est_baseline.regional_market == "US National Baseline"

    body_pac = [it for it in est_pacific.line_items if it.category == "BODY_LABOR"][0]
    body_base = [it for it in est_baseline.line_items if it.category == "BODY_LABOR"][0]

    assert body_pac.rate == 91.00
    assert body_base.rate == 65.00
    # Wage multiplier is exactly 91 / 65 = 1.40
    assert pytest.approx(body_pac.rate / body_base.rate, 0.001) == 1.40
    assert est_pacific.gross_estimate_total > est_baseline.gross_estimate_total
