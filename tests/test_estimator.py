"""
Unit and integration tests for the Actuarial Insurance Cost Estimator Engine.
Tests decision logic (Repair vs Replace vs Polish), line-item math, and deductible bounds.
"""

from app.cv_engine.base import (
    BoundingBox,
    DamageAnalysisResult,
    DamageType,
    DetectedDamageItem,
    Severity,
)
from app.estimator.cost_engine import (
    EstimateCalculator,
    RecommendedAction,
)
from app.estimator.parts_catalog import PANEL_CATALOG


def make_dummy_analysis(total_area: float, items=None, severity=Severity.MINOR):
    return DamageAnalysisResult(
        engine_used="TEST_CV_ENGINE",
        panel_name="FRONT_BUMPER",
        items_detected=items or [],
        total_damaged_area_cm2=total_area,
        overall_severity=severity,
        annotated_image_base64="",
        mask_image_base64="",
    )


def test_buff_and_polish_minor_damage():
    """Minor superficial damage (< 5 cm2) should recommend BUFF_AND_POLISH_ONLY."""
    item = DetectedDamageItem(
        id=1,
        type=DamageType.SCRATCH,
        severity=Severity.MINOR,
        bounding_box=BoundingBox(x=10, y=10, width=5, height=5),
        area_pixels=25,
        area_cm2=2.5,
        confidence_score=0.85,
    )
    analysis = make_dummy_analysis(total_area=2.5, items=[item], severity=Severity.MINOR)
    estimate = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", deductible=500.0)

    assert estimate.recommended_action == RecommendedAction.BUFF_AND_POLISH_ONLY
    assert estimate.parts_subtotal == 0.0
    assert estimate.gross_estimate_total < 500.0
    assert estimate.net_claim_payout == 0.0  # Under deductible


def test_repair_and_refinish_moderate_damage():
    """Moderate damage within threshold should trigger REPAIR_AND_REFINISH."""
    item = DetectedDamageItem(
        id=1,
        type=DamageType.DENT,
        severity=Severity.MODERATE,
        bounding_box=BoundingBox(x=100, y=100, width=40, height=40),
        area_pixels=1600,
        area_cm2=60.0,
        confidence_score=0.90,
    )
    analysis = make_dummy_analysis(total_area=60.0, items=[item], severity=Severity.MODERATE)
    estimate = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", deductible=500.0)

    assert estimate.recommended_action == RecommendedAction.REPAIR_AND_REFINISH
    assert estimate.parts_subtotal == 0.0  # No new part needed
    assert estimate.body_labor_subtotal > 0.0
    assert estimate.paint_labor_subtotal > 0.0
    assert estimate.paint_materials_subtotal > 0.0
    assert estimate.gross_estimate_total > 0.0


def test_replace_panel_for_extensive_damage():
    """Damage exceeding 28% of surface area should trigger complete REPLACE_PANEL."""
    bumper = PANEL_CATALOG["FRONT_BUMPER"]
    heavy_damage_area = bumper.surface_area_cm2 * 0.40  # 40% surface damage

    analysis = make_dummy_analysis(total_area=heavy_damage_area, severity=Severity.SEVERE)
    estimate = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", use_oem_parts=True, deductible=500.0)

    assert estimate.recommended_action == RecommendedAction.REPLACE_PANEL
    assert estimate.parts_subtotal == bumper.oem_part_price
    assert estimate.gross_estimate_total > bumper.oem_part_price
    # Net payout should subtract deductible
    expected_net = round(max(0.0, estimate.gross_estimate_total - 500.0), 2)
    assert estimate.net_claim_payout == expected_net


def test_windshield_always_replaced():
    """Automotive safety regulations require complete windshield replacement on crack."""
    analysis = make_dummy_analysis(total_area=15.0, severity=Severity.MODERATE)
    estimate = EstimateCalculator.calculate(analysis, panel_id="WINDSHIELD", deductible=250.0)

    assert estimate.recommended_action == RecommendedAction.REPLACE_PANEL
    assert "windshield" in estimate.action_reason.lower()


def test_negative_deductible_is_clamped():
    """Negative deductibles must be clamped to zero so net payout never exceeds gross total."""
    analysis = make_dummy_analysis(total_area=60.0, severity=Severity.MODERATE)
    estimate = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER", deductible=-200.0)

    assert estimate.deductible == 0.0
    assert estimate.net_claim_payout == estimate.gross_estimate_total
