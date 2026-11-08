"""
Tests for PDF report compilation.
"""
from app.cv_engine.base import DamageAnalysisResult, Severity
from app.estimator.cost_engine import EstimateCalculator
from app.reports.pdf_generator import generate_estimate_pdf


def test_pdf_generation_valid_bytes():
    analysis = DamageAnalysisResult(
        engine_used="CLASSICAL_CV_OPENCV",
        panel_name="FRONT_BUMPER",
        items_detected=[],
        total_damaged_area_cm2=45.0,
        overall_severity=Severity.MODERATE,
        annotated_image_base64="",
        mask_image_base64="",
    )
    estimate = EstimateCalculator.calculate(analysis, panel_id="FRONT_BUMPER")
    pdf_bytes = generate_estimate_pdf(estimate, analysis)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic byte header check
    assert pdf_bytes.startswith(b"%PDF-")
