"""
Unit tests for Actuarial Precision (Regional Rates, Mitchell 50% Blend, Material Multipliers)
and Backend Performance (Perceptual Hashing & In-Memory Cache).
"""

from app.core.cache import PerceptualImageCache
from app.cv_engine.base import BoundingBox, DamageAnalysisResult, DamageType, DetectedDamageItem, Severity
from app.estimator.cost_engine import EstimateCalculator


def make_sample_damage_analysis(area_cm2: float = 45.0) -> DamageAnalysisResult:
    item = DetectedDamageItem(
        id=1,
        type=DamageType.SCRATCH,
        severity=Severity.MODERATE,
        bounding_box=BoundingBox(x=100, y=100, width=50, height=30),
        area_pixels=1200,
        area_cm2=area_cm2,
        confidence_score=0.92,
    )
    return DamageAnalysisResult(
        engine_used="TEST_CV",
        panel_name="FRONT_FENDER_LEFT",
        items_detected=[item],
        total_damaged_area_cm2=area_cm2,
        overall_severity=Severity.MODERATE,
        annotated_image_base64="",
        mask_image_base64="",
    )


def test_regional_wage_adjustments():
    """ZIP codes in high-cost metro regions (e.g. 90210 CA) should adjust labor rates and tax."""
    analysis = make_sample_damage_analysis(area_cm2=40.0)

    # 1. Baseline estimate (no ZIP)
    est_baseline = EstimateCalculator.calculate(analysis, panel_id="FRONT_FENDER_LEFT", zip_code=None)
    assert est_baseline.regional_market == "US National Baseline"

    # 2. California Pacific Metro ZIP (90210)
    est_ca = EstimateCalculator.calculate(analysis, panel_id="FRONT_FENDER_LEFT", zip_code="90210")
    assert "Pacific Metro" in est_ca.regional_market
    # Pacific Metro has 1.40x labor rate multiplier ($91/hr vs $65/hr)
    assert est_ca.body_labor_subtotal > est_baseline.body_labor_subtotal
    assert est_ca.gross_estimate_total > est_baseline.gross_estimate_total

    # 3. New York Northeast Metro ZIP (10001)
    est_ny = EstimateCalculator.calculate(analysis, panel_id="FRONT_FENDER_LEFT", zip_code="10001")
    assert "Northeast Metro" in est_ny.regional_market
    assert est_ny.body_labor_subtotal > est_baseline.body_labor_subtotal


def test_mitchell_adjacent_panel_blend_rule():
    """Enabling Mitchell 50% Rule should add adjacent panel clear-coat blend hours."""
    analysis = make_sample_damage_analysis(area_cm2=50.0)

    # Without blend
    est_no_blend = EstimateCalculator.calculate(analysis, panel_id="FRONT_FENDER_LEFT", blend_adjacent_panels=False)
    assert est_no_blend.adjacent_blend_applied is False
    blend_items_none = [it for it in est_no_blend.line_items if "Adjacent blend" in it.description]
    assert len(blend_items_none) == 0

    # With blend
    est_blend = EstimateCalculator.calculate(analysis, panel_id="FRONT_FENDER_LEFT", blend_adjacent_panels=True)
    assert est_blend.adjacent_blend_applied is True
    blend_items = [it for it in est_blend.line_items if "Adjacent blend" in it.description]
    assert len(blend_items) > 0
    # Refinish hours and gross estimate should be higher due to blend
    assert est_blend.paint_labor_subtotal > est_no_blend.paint_labor_subtotal
    assert est_blend.gross_estimate_total > est_no_blend.gross_estimate_total


def test_aluminum_material_multiplier():
    """Structural aluminum panels (Hood) should include specialized tooling surcharge."""
    analysis = make_sample_damage_analysis(area_cm2=40.0)
    est_hood = EstimateCalculator.calculate(analysis, panel_id="HOOD")

    # Hood has 1.25x material multiplier
    body_items = [it for it in est_hood.line_items if it.category == "BODY_LABOR"]
    assert any("specialized rate" in it.description for it in body_items)
    # Effective rate should be 65 * 1.25 = 81.25
    assert body_items[0].rate == 81.25


def test_perceptual_dhash_and_cache():
    """Perceptual difference hash (dHash) combined with SHA-256 caches and retrieves in sub-ms."""
    with open("app/static/sample_images/sample_bumper_scratch.jpg", "rb") as f:
        img_bytes = f.read()

    # 1. Compute SHA-256 and dHash
    sha256 = PerceptualImageCache.compute_sha256(img_bytes)
    dhash = PerceptualImageCache.compute_dhash(img_bytes)
    assert len(sha256) == 64
    assert len(dhash) == 16
    assert dhash != "0" * 16

    # 2. Cache miss initially
    assert (
        PerceptualImageCache.get_analysis(
            sha256=sha256,
            dhash=dhash,
            panel_id="FRONT_BUMPER",
            sensitivity=0.55,
            detect_scratches=True,
            detect_dents=True,
            min_defect_area_pixels=150,
            use_oem=True,
            deductible=500.0,
            blend=False,
            zip_code=None,
        )
        is None
    )

    # 3. Store mock payload
    mock_payload = {"success": True, "cached_key": "test_123"}
    PerceptualImageCache.set_analysis(
        sha256=sha256,
        dhash=dhash,
        panel_id="FRONT_BUMPER",
        sensitivity=0.55,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.0,
        blend=False,
        zip_code=None,
        data=mock_payload,
    )

    # 4. Cache hit
    cached = PerceptualImageCache.get_analysis(
        sha256=sha256,
        dhash=dhash,
        panel_id="FRONT_BUMPER",
        sensitivity=0.55,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.0,
        blend=False,
        zip_code=None,
    )
    assert cached is not None
    assert cached["cached_key"] == "test_123"


def test_cache_keys_preserve_parameter_precision_no_collision():
    """Verify distinct cache keys for precise deductible cents, sensitivity decimals, and options."""
    sha = "a" * 64
    dh = "b" * 16

    key_base = PerceptualImageCache._make_analysis_key(
        sha256=sha,
        dhash=dh,
        panel_id="FRONT_BUMPER",
        sensitivity=0.5000,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.00,
        blend=False,
        zip_code="94103",
    )

    # Different deductible in cents ($500.00 vs $500.50) must NOT collide
    key_diff_deductible = PerceptualImageCache._make_analysis_key(
        sha256=sha,
        dhash=dh,
        panel_id="FRONT_BUMPER",
        sensitivity=0.5000,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.50,
        blend=False,
        zip_code="94103",
    )
    assert key_base != key_diff_deductible

    # Different sensitivity (0.5000 vs 0.5050) must NOT collide
    key_diff_sensitivity = PerceptualImageCache._make_analysis_key(
        sha256=sha,
        dhash=dh,
        panel_id="FRONT_BUMPER",
        sensitivity=0.5050,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.00,
        blend=False,
        zip_code="94103",
    )
    assert key_base != key_diff_sensitivity

    # Different scratch flag must NOT collide
    key_diff_scratches = PerceptualImageCache._make_analysis_key(
        sha256=sha,
        dhash=dh,
        panel_id="FRONT_BUMPER",
        sensitivity=0.5000,
        detect_scratches=False,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.00,
        blend=False,
        zip_code="94103",
    )
    assert key_base != key_diff_scratches

    # Different SHA-256 must NOT collide even if dHash is identical
    key_diff_sha = PerceptualImageCache._make_analysis_key(
        sha256="c" * 64,
        dhash=dh,
        panel_id="FRONT_BUMPER",
        sensitivity=0.5000,
        detect_scratches=True,
        detect_dents=True,
        min_defect_area_pixels=150,
        use_oem=True,
        deductible=500.00,
        blend=False,
        zip_code="94103",
    )
    assert key_base != key_diff_sha


def test_cache_deepcopy_isolates_nested_state():
    """Mutating a retrieved cache entry must not corrupt nested dicts inside the LRU cache."""
    PerceptualImageCache.clear()
    sha = "d" * 64
    dh = "e" * 16
    payload = {"is_valid": True, "diagnostics": {"cache_hit": False}}

    PerceptualImageCache.set_validation(sha256=sha, dhash=dh, panel_id="FRONT_BUMPER", data=payload)
    first_read = PerceptualImageCache.get_validation(sha256=sha, dhash=dh, panel_id="FRONT_BUMPER")
    assert first_read is not None
    assert first_read["diagnostics"]["cache_hit"] is False

    # Mutate caller's dict after retrieval
    first_read["diagnostics"]["cache_hit"] = True

    second_read = PerceptualImageCache.get_validation(sha256=sha, dhash=dh, panel_id="FRONT_BUMPER")
    assert second_read is not None
    assert second_read["diagnostics"]["cache_hit"] is False
