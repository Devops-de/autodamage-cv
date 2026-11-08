"""
Base interfaces and data schemas for Vehicle Damage Detection.
This design allows plugging in future AI/ML models seamlessly without modifying
the cost estimator, API, or frontend.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DamageType(str, Enum):
    SCRATCH = "SCRATCH"
    DENT = "DENT"
    PAINT_CHIP = "PAINT_CHIP"
    CRACK = "CRACK"
    TEAR = "TEAR"


class Severity(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedDamageItem(BaseModel):
    id: int
    type: DamageType
    severity: Severity
    bounding_box: BoundingBox
    area_pixels: int
    area_cm2: float
    confidence_score: float = Field(
        description="Classical CV metric or ML probability (0.0 to 1.0)"
    )
    details: str = ""


class DetectionOptions(BaseModel):
    target_panel: str = "FRONT_BUMPER"
    sensitivity: float = 0.5            # 0.1 to 1.0 (threshold adjustment)
    detect_scratches: bool = True
    detect_dents: bool = True
    min_defect_area_pixels: int = 150
    # Calibration factor: pixels per centimeter (default ~10 px/cm)
    pixels_per_cm: float = 10.0


class DamageAnalysisResult(BaseModel):
    engine_used: str                    # "CLASSICAL_CV_OPENCV" or "FUTURE_AI_MODEL"
    panel_name: str
    items_detected: List[DetectedDamageItem]
    total_damaged_area_cm2: float
    overall_severity: Severity
    annotated_image_base64: str         # Processed image with colored bounding boxes
    mask_image_base64: str              # Pure binary mask (edges & contours)
    diagnostics: Dict[str, Any] = {}


class BaseDamageDetector(ABC):
    """
    Abstract interface for vehicle damage detectors.
    Any future AI/ML model (e.g. YOLO, Mask R-CNN) simply implements this class.
    """
    @abstractmethod
    def detect(self, image_bytes: bytes, options: DetectionOptions) -> DamageAnalysisResult:
        """Process image bytes and return structured damage items + annotated visuals."""
        pass
