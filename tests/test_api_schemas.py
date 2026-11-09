"""
Unit tests for API response schemas, Pydantic serialization, and catalog conversions.
Ensures strong typing and data integrity between backend algorithms and REST responses.
"""

import pytest

from app.cv_engine.base import (
    BoundingBox,
    DamageAnalysisResult,
    DamageType,
    DetectedDamageItem,
    Severity,
)
from app.estimator.cost_engine import (
    EstimateLineItem,
    InsuranceEstimate,
    RecommendedAction,
)
from app.estimator.parts_catalog import PANEL_CATALOG
from app.main import get_panels, get_regional_markets, get_settings, health_check
from app.schemas.api import (
    AnalysisResponse,
    ErrorResponse,
    ExportPDFRequest,
    HealthResponse,
    PanelResponseItem,
    SettingsResponse,
    TelemetryData,
    ValidationResponse,
)


def test_health_check_endpoint():
    """Verify health endpoint returns valid HealthResponse schema."""
    resp = health_check()
    assert isinstance(resp, HealthResponse)
    assert resp.status == "healthy"
    assert "OpenCV" in resp.engine
    assert resp.version == "1.0.0"
    data = resp.model_dump()
    assert data["status"] == "healthy"


def test_panels_catalog_endpoint():
    """Verify panels endpoint returns typed list matching parts catalog."""
    panels = get_panels()
    assert isinstance(panels, list)
    assert len(panels) == len(PANEL_CATALOG)
    for p in panels:
        assert isinstance(p, PanelResponseItem)
        assert p.id in PANEL_CATALOG
        assert p.surface_area_cm2 > 0
        assert p.oem_price > 0


def test_settings_endpoint():
    """Verify settings endpoint returns SettingsResponse with rates."""
    resp = get_settings()
    assert isinstance(resp, SettingsResponse)
    assert resp.rates.body_labor_rate_per_hour == 65.0
    assert resp.rates.paint_labor_rate_per_hour == 65.0
    assert resp.replace_threshold_area_cm2 == 250.0


def test_regional_markets_endpoint():
    """Verify regional markets returns valid list of RegionalMarket objects."""
    markets = get_regional_markets()
    assert isinstance(markets, list)
    assert len(markets) >= 4
    codes = [m.market_code for m in markets]
    assert "PACIFIC_METRO" in codes
    assert "NATIONAL_BASELINE" in codes


def test_validation_response_schema():
    """Verify ValidationResponse schema and diagnostic serialization."""
    val = ValidationResponse(
        is_valid=True,
        rejection_reason=None,
        blur_score=75.5,
        skin_tone_ratio=0.02,
        surface_smoothness_ratio=0.88,
        brightness_mean=140.2,
        defect_count=3,
        diagnostics={"focus_quality": "Sharp", "target_panel": "Front Bumper Cover"},
    )
    dumped = val.model_dump()
    assert dumped["is_valid"] is True
    assert dumped["defect_count"] == 3
    assert dumped["diagnostics"]["focus_quality"] == "Sharp"


def test_analysis_and_export_pdf_schemas():
    """Verify AnalysisResponse and ExportPDFRequest validation and serialization."""
    item = DetectedDamageItem(
        id=1,
        type=DamageType.SCRATCH,
        severity=Severity.MINOR,
        bounding_box=BoundingBox(x=10, y=10, width=50, height=20),
        area_pixels=1000,
        area_cm2=10.0,
        confidence_score=0.92,
        details="Key Scratch",
    )
    analysis = DamageAnalysisResult(
        engine_used="OpenCV Classical Pipeline",
        panel_name="FRONT_BUMPER",
        items_detected=[item],
        total_damaged_area_cm2=10.0,
        overall_severity=Severity.MINOR,
        annotated_image_base64="data:image/jpeg;base64,abc",
        mask_image_base64="data:image/png;base64,xyz",
    )
    line_item = EstimateLineItem(
        category="BODY_LABOR",
        description="Spot repair",
        quantity_or_hours=1.5,
        rate=65.0,
        subtotal=97.5,
    )
    estimate = InsuranceEstimate(
        claim_id="CLM-SCHEMA01",
        panel_name="Front Bumper Cover",
        panel_material="Plastic",
        recommended_action=RecommendedAction.REPAIR_AND_REFINISH,
        action_reason="Minor damage",
        use_oem_parts=True,
        deductible=500.0,
        regional_market="US National Baseline",
        line_items=[line_item],
        parts_subtotal=0.0,
        body_labor_subtotal=97.5,
        paint_labor_subtotal=0.0,
        paint_materials_subtotal=0.0,
        fees_subtotal=12.0,
        sales_tax=0.0,
        gross_estimate_total=109.5,
        net_claim_payout=0.0,
        total_labor_hours=1.5,
        damage_area_cm2=10.0,
        panel_damage_percentage=1.0,
    )
    telemetry = TelemetryData(
        cache_hit=False,
        cv_latency_ms=18.5,
        total_latency_ms=25.2,
        dhash="0123456789abcdef",
    )

    # 1. Test AnalysisResponse
    analysis_resp = AnalysisResponse(
        success=True,
        analysis=analysis,
        estimate=estimate,
        telemetry=telemetry,
    )
    assert analysis_resp.success is True
    assert analysis_resp.telemetry.dhash == "0123456789abcdef"

    # 2. Test ExportPDFRequest
    pdf_req = ExportPDFRequest(
        analysis=analysis,
        estimate=estimate,
    )
    assert pdf_req.estimate.claim_id == "CLM-SCHEMA01"
    assert len(pdf_req.analysis.items_detected) == 1


def test_error_response_schema():
    """Verify standardized error schema."""
    err = ErrorResponse(
        detail="File format not supported",
        error_type="VALIDATION_ERROR",
    )
    assert err.detail == "File format not supported"
    assert err.error_type == "VALIDATION_ERROR"


def test_upload_bounded_reader_size_limit():
    """Verify read_bounded_file rejects uploads exceeding max_upload_size_bytes with HTTP 413."""
    from fastapi import HTTPException

    from app.main import read_bounded_file

    class ReadyFuture:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            if False:
                yield
            return self.value

    class MockUpload:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def read(self, size=-1):
            val = self.chunks.pop(0) if self.chunks else b""
            return ReadyFuture(val)

    # 1. Normal file under limit (2 chunks of 250KB = 500KB)
    coro_small = read_bounded_file(MockUpload([b"x" * 250_000, b"y" * 250_000]))
    try:
        coro_small.send(None)
    except StopIteration as e:
        data = e.value
    assert len(data) == 500_000

    # 2. Oversized file (16 chunks of 1MB = 16MB > 15MB limit)
    coro_large = read_bounded_file(MockUpload([b"x" * (1024 * 1024)] * 16))
    with pytest.raises(HTTPException) as exc_info:
        coro_large.send(None)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum allowed file size" in exc_info.value.detail
