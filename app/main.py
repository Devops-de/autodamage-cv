"""
Main FastAPI Application for AutoDamage CV.
Provides RESTful endpoints for vehicle damage detection, actuarial calculation,
and insurance PDF appraisal generation.
"""
import io
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from app.cv_engine.base import DetectionOptions
from app.cv_engine.classical_detector import ClassicalDamageDetector
from app.cv_engine.image_validator import VehicleImageValidator
from app.estimator.cost_engine import EstimateCalculator, InsuranceEstimate
from app.estimator.parts_catalog import PANEL_CATALOG
from app.reports.pdf_generator import generate_estimate_pdf


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


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "engine": detector.engine_name,
        "version": settings.version,
    }


@app.get("/api/panels")
def get_panels():
    """Returns available vehicle panels, materials, and standard repair metrics."""
    return [
        {
            "id": p.panel_id,
            "name": p.name,
            "material": p.material,
            "surface_area_cm2": p.surface_area_cm2,
            "oem_price": p.oem_part_price,
            "aftermarket_price": p.aftermarket_part_price,
            "r_and_i_hours": p.r_and_i_labor_hours,
            "paint_hours": p.standard_paint_refinish_hours,
        }
        for p in PANEL_CATALOG.values()
    ]


@app.get("/api/settings")
def get_settings():
    return {
        "rates": settings.rates.model_dump(),
        "replace_threshold_area_cm2": settings.replace_threshold_area_cm2,
    }


@app.post("/api/analyze")
async def analyze_vehicle_damage(
    file: UploadFile = File(...),
    panel_id: str = Form("FRONT_BUMPER"),
    sensitivity: float = Form(0.55),
    detect_scratches: bool = Form(True),
    detect_dents: bool = Form(True),
    min_defect_area_pixels: int = Form(150),
    use_oem_parts: bool = Form(True),
    deductible: float = Form(500.00),
):
    """
    Accepts a vehicle damage photograph, applies Classical Computer Vision algorithms,
    and returns detected damage bounding boxes, visual mask, and actuarial cost estimate.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, WEBP).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    # Validate image is a clear vehicle panel (rejects selfies, text documents, extreme blur)
    validation = VehicleImageValidator.validate(contents)
    if not validation.is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Image validation check failed: {validation.rejection_reason}"
        )

    options = DetectionOptions(
        target_panel=panel_id,
        sensitivity=max(0.1, min(1.0, sensitivity)),
        detect_scratches=detect_scratches,
        detect_dents=detect_dents,
        min_defect_area_pixels=min_defect_area_pixels,
    )

    try:
        # Run 100% Classical CV detection
        analysis_result = detector.detect(contents, options)

        # Run Actuarial Cost Estimation
        estimate = EstimateCalculator.calculate(
            analysis=analysis_result,
            panel_id=panel_id,
            use_oem_parts=use_oem_parts,
            deductible=deductible,
        )

        return {
            "success": True,
            "analysis": analysis_result.model_dump(),
            "estimate": estimate.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


class ExportPDFRequest(BaseModel):
    analysis: dict
    estimate: dict


@app.post("/api/export-pdf")
async def export_pdf(payload: ExportPDFRequest):
    """Generates an official insurance-standard PDF appraisal from analysis and estimate data."""
    try:
        from app.cv_engine.base import DamageAnalysisResult
        analysis_obj = DamageAnalysisResult.model_validate(payload.analysis)
        estimate_obj = InsuranceEstimate.model_validate(payload.estimate)

        pdf_bytes = generate_estimate_pdf(estimate_obj, analysis_obj)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=Damage_Appraisal_{estimate_obj.claim_id}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")


# Mount static directory for frontend
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
