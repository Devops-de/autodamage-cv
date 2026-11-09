"""
Pydantic API Schemas for AutoDamage CV REST Endpoints.
Provides strict validation, serialization, and OpenAPI documentation schemas.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

from app.cv_engine.base import DamageAnalysisResult
from app.estimator.cost_engine import InsuranceEstimate


class HealthResponse(BaseModel):
    """Schema for /api/health endpoint."""
    model_config = ConfigDict(extra="ignore")

    status: str = Field(..., description="Operational status, e.g. 'healthy'")
    app: str = Field(..., description="Application name")
    engine: str = Field(..., description="Active damage detection engine")
    version: str = Field(..., description="Semantic version string")


class PanelResponseItem(BaseModel):
    """Schema for individual panel in /api/panels response."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique panel identifier (e.g., 'FRONT_BUMPER')")
    name: str = Field(..., description="Human-readable panel title")
    material: str = Field(..., description="Primary substrate material (e.g., 'Plastic', 'Steel', 'Aluminum')")
    surface_area_cm2: float = Field(..., description="Standard total surface area in cm²")
    oem_price: float = Field(..., description="OEM replacement part catalog price in USD")
    aftermarket_price: float = Field(..., description="CAPA aftermarket part price in USD")
    r_and_i_hours: float = Field(..., description="Standard Remove & Install labor hours")
    paint_hours: float = Field(..., description="Standard refinish spray labor hours")


class SettingsRates(BaseModel):
    """Actuarial hourly repair rates."""
    model_config = ConfigDict(extra="ignore")

    body_labor_rate_per_hour: float
    paint_labor_rate_per_hour: float
    paint_materials_rate_per_hour: float
    mechanical_rate_per_hour: float
    sales_tax_rate: float


class SettingsResponse(BaseModel):
    """Schema for /api/settings endpoint."""
    model_config = ConfigDict(extra="ignore")

    rates: SettingsRates
    replace_threshold_area_cm2: float


class ValidationResponse(BaseModel):
    """Schema for /api/validate-image endpoint."""
    model_config = ConfigDict(extra="ignore")

    is_valid: bool = Field(..., description="True if image qualifies as a clear damaged automotive panel")
    rejection_reason: Optional[str] = Field(None, description="Detailed user-facing rejection reason if invalid")
    blur_score: float = Field(0.0, description="Laplacian focus variance metric")
    skin_tone_ratio: float = Field(0.0, description="Ratio of pixels matching human skin tones")
    surface_smoothness_ratio: float = Field(0.0, description="Automotive paint continuity score")
    brightness_mean: float = Field(0.0, description="Mean gray-level brightness (0-255)")
    defect_count: int = Field(0, description="Detected surface defect contours")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry attributes")


class TelemetryData(BaseModel):
    """Processing and cache telemetry metadata."""
    model_config = ConfigDict(extra="ignore")

    cache_hit: bool = Field(..., description="True if response was served from the in-memory dHash cache")
    cv_latency_ms: Optional[float] = Field(None, description="OpenCV processing duration in milliseconds")
    total_latency_ms: float = Field(..., description="Total request execution latency in milliseconds")
    dhash: str = Field(..., description="64-bit hexadecimal perceptual difference hash")


class AnalysisResponse(BaseModel):
    """Schema for /api/analyze endpoint."""
    model_config = ConfigDict(extra="ignore")

    success: bool = Field(True, description="Request execution status")
    analysis: DamageAnalysisResult = Field(..., description="Computer vision detection results and metrics")
    estimate: InsuranceEstimate = Field(..., description="Actuarial repair estimate and itemized line items")
    telemetry: TelemetryData = Field(..., description="Execution latency and cache hit metadata")


class ExportPDFRequest(BaseModel):
    """Schema for /api/export-pdf payload."""
    model_config = ConfigDict(extra="ignore")

    analysis: DamageAnalysisResult = Field(..., description="Structured computer vision analysis object")
    estimate: InsuranceEstimate = Field(..., description="Structured actuarial estimate object")


class ErrorResponse(BaseModel):
    """Standardized API error response payload."""
    model_config = ConfigDict(extra="ignore")

    detail: str = Field(..., description="Human-readable error description")
    error_type: Optional[str] = Field(None, description="Machine-readable error category")
