"""
Main FastAPI Application for ClaimScope.
Provides RESTful endpoints for vehicle damage detection, actuarial calculation,
and insurance PDF appraisal generation.
"""
import io
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.cache import PerceptualImageCache
from app.cv_engine.base import DetectionOptions, DamageAnalysisResult
from app.cv_engine.classical_detector import ClassicalDamageDetector
from app.cv_engine.image_validator import VehicleImageValidator
from app.estimator.cost_engine import EstimateCalculator, InsuranceEstimate
from app.estimator.parts_catalog import PANEL_CATALOG
from app.estimator.regional_rates import REGIONAL_MARKETS, RegionalMarket
from app.reports.pdf_generator import generate_estimate_pdf
from app.schemas.api import (
    HealthResponse,
    PanelResponseItem,
    SettingsResponse,
    ValidationResponse,
    AnalysisResponse,
    ExportPDFRequest,
)

logger = logging.getLogger("claimscope")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Vehicle damage detection and collision repair cost appraisal engine using OpenCV.",
)

# Enable CORS for local and web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Classical CV Detector
detector = ClassicalDamageDetector()


@app.get("/api/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Returns application health, version, and active detection engine."""
    return HealthResponse(
        status="healthy",
        app=settings.app_name,
        engine=detector.engine_name,
        version=settings.version,
    )


@app.get("/api/panels", response_model=List[PanelResponseItem], tags=["Estimator Catalog"])
def get_panels():
    """Returns available vehicle panels, materials, and standard repair metrics."""
    return [
        PanelResponseItem(
            id=p.panel_id,
            name=p.name,
            material=p.material,
            surface_area_cm2=p.surface_area_cm2,
            oem_price=p.oem_part_price,
            aftermarket_price=p.aftermarket_part_price,
            r_and_i_hours=p.r_and_i_labor_hours,
            paint_hours=p.standard_paint_refinish_hours,
        )
        for p in PANEL_CATALOG.values()
    ]


@app.get("/api/settings", response_model=SettingsResponse, tags=["Diagnostics"])
def get_settings():
    """Returns system configuration and base actuarial labor rates."""
    return SettingsResponse(
        rates=settings.rates.model_dump(),
        replace_threshold_area_cm2=settings.replace_threshold_area_cm2,
    )


@app.get("/api/regional-markets", response_model=List[RegionalMarket], tags=["Estimator Catalog"])
def get_regional_markets():
    """Returns standard US regional collision repair markets and wage multipliers."""
    return list(REGIONAL_MARKETS.values())


@app.post("/api/validate-image", response_model=ValidationResponse, tags=["Computer Vision"])
async def validate_vehicle_image(
    file: UploadFile = File(...),
    panel_id: str = Form("FRONT_BUMPER"),
):
    """
    Pre-flight validation endpoint to verify that an uploaded photo
    is a focused, legitimate automotive body panel before loading it.
    Rejects selfies, paper documents, blur, screenshots, or non-vehicle textures.
    High-performance caching and thread-offloaded execution.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return ValidationResponse(
            is_valid=False,
            rejection_reason="Uploaded file must be a valid image (JPEG, PNG, WEBP).",
            blur_score=0.0,
            skin_tone_ratio=0.0,
            surface_smoothness_ratio=0.0,
            brightness_mean=0.0,
            defect_count=0,
            diagnostics={},
        )

    contents = await file.read()
    if len(contents) == 0:
        return ValidationResponse(
            is_valid=False,
            rejection_reason="Empty image file received.",
            blur_score=0.0,
            skin_tone_ratio=0.0,
            surface_smoothness_ratio=0.0,
            brightness_mean=0.0,
            defect_count=0,
            diagnostics={},
        )

    # Check perceptual hash cache for instantaneous sub-millisecond return
    dhash = await asyncio.to_thread(PerceptualImageCache.compute_dhash, contents)
    cached_val = PerceptualImageCache.get_validation(dhash, panel_id)
    if cached_val is not None:
        cached_val["diagnostics"]["cache_hit"] = True
        return ValidationResponse.model_validate(cached_val)

    # Run validation non-blockingly on Python's thread pool
    result = await asyncio.to_thread(VehicleImageValidator.validate, contents, panel_id=panel_id)
    result_dict = result.model_dump()
    result_dict["diagnostics"]["cache_hit"] = False

    # Store in perceptual cache
    PerceptualImageCache.set_validation(dhash, panel_id, result_dict)
    return ValidationResponse.model_validate(result_dict)


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Computer Vision"])
async def analyze_vehicle_damage(
    file: UploadFile = File(...),
    panel_id: str = Form("FRONT_BUMPER"),
    sensitivity: float = Form(0.55),
    detect_scratches: bool = Form(True),
    detect_dents: bool = Form(True),
    min_defect_area_pixels: int = Form(150),
    use_oem_parts: bool = Form(True),
    deductible: float = Form(500.00),
    blend_adjacent_panels: bool = Form(False),
    zip_code: Optional[str] = Form(None),
):
    """
    Accepts a vehicle damage photograph, applies Classical Computer Vision algorithms,
    and returns detected damage bounding boxes, visual mask, and actuarial cost estimate.
    """
    start_time = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, WEBP).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    # 1. Compute 64-bit Perceptual Hash
    dhash = await asyncio.to_thread(PerceptualImageCache.compute_dhash, contents)

    # 2. Check Cache
    cached_analysis = PerceptualImageCache.get_analysis(
        dhash=dhash,
        panel_id=panel_id,
        sensitivity=sensitivity,
        use_oem=use_oem_parts,
        deductible=deductible,
        blend=blend_adjacent_panels,
        zip_code=zip_code,
    )
    if cached_analysis is not None:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        cached_analysis["telemetry"] = {
            "cache_hit": True,
            "total_latency_ms": elapsed_ms,
            "dhash": dhash,
        }
        return AnalysisResponse.model_validate(cached_analysis)

    # 3. Pre-flight Validation (Non-blocking thread offload)
    validation = await asyncio.to_thread(VehicleImageValidator.validate, contents, panel_id=panel_id)
    if not validation.is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Image validation check failed: {validation.rejection_reason}"
        )

    val_time = time.perf_counter()

    options = DetectionOptions(
        target_panel=panel_id,
        sensitivity=max(0.1, min(1.0, sensitivity)),
        detect_scratches=detect_scratches,
        detect_dents=detect_dents,
        min_defect_area_pixels=min_defect_area_pixels,
    )

    try:
        # 4. Run Classical CV detection (Non-blocking thread offload)
        analysis_result = await asyncio.to_thread(detector.detect, contents, options)
        cv_time = time.perf_counter()

        # 5. Run Actuarial Cost Estimation with Regional Rates & 50% Blend Rule
        estimate = await asyncio.to_thread(
            EstimateCalculator.calculate,
            analysis=analysis_result,
            panel_id=panel_id,
            use_oem_parts=use_oem_parts,
            deductible=deductible,
            blend_adjacent_panels=blend_adjacent_panels,
            zip_code=zip_code,
        )
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        cv_latency_ms = round((cv_time - val_time) * 1000, 2)

        response_payload = {
            "success": True,
            "analysis": analysis_result.model_dump(),
            "estimate": estimate.model_dump(),
            "telemetry": {
                "cache_hit": False,
                "cv_latency_ms": cv_latency_ms,
                "total_latency_ms": total_time_ms,
                "dhash": dhash,
            },
        }

        # Save to perceptual cache
        PerceptualImageCache.set_analysis(
            dhash=dhash,
            panel_id=panel_id,
            sensitivity=sensitivity,
            use_oem=use_oem_parts,
            deductible=deductible,
            blend=blend_adjacent_panels,
            zip_code=zip_code,
            data=response_payload,
        )

        return AnalysisResponse.model_validate(response_payload)
    except Exception as e:
        logger.exception("Error processing vehicle damage analysis: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@app.post("/api/export-pdf", tags=["Reports"])
async def export_pdf(payload: ExportPDFRequest):
    """
    Generates an official insurance-standard PDF appraisal from analysis and estimate data.
    Directly validates structured Pydantic payload.
    """
    try:
        # Offload PDF compilation to worker thread
        pdf_bytes = await asyncio.to_thread(
            generate_estimate_pdf,
            payload.estimate,
            payload.analysis,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=ClaimScope_Appraisal_{payload.estimate.claim_id}.pdf"
            },
        )
    except Exception as e:
        logger.exception("PDF generation error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")


# Mount static directory for frontend
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
