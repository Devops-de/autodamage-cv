"""
Backend adapter stub for machine learning detection models.

Implements BaseDamageDetector so an object-detection or segmentation model
(such as YOLOv8-seg, Mask R-CNN, or an ONNX runtime session) can be dropped
in without changing the cost estimator, API routes, or web UI.
"""
from app.cv_engine.base import (
    BaseDamageDetector,
    DamageAnalysisResult,
    DetectionOptions,
)


class FutureAIDamageDetector(BaseDamageDetector):
    """
    Placeholder adapter for ML-based damage detection models.
    """

    def __init__(self, model_weights_path: str = "models/yolov8_damage.pt"):
        self.model_weights_path = model_weights_path
        self.engine_name = "DEEP_LEARNING_DETECTOR (Planned)"

    def detect(self, image_bytes: bytes, options: DetectionOptions) -> DamageAnalysisResult:
        """
        Runs model inference on input image bytes and returns structured damage items.
        To implement: load model weights, run inference, format bounding boxes,
        and construct the standard DamageAnalysisResult object.
        """
        raise NotImplementedError(
            f"Model weights not configured at '{self.model_weights_path}'. "
            "Implement inference here when model weights are available."
        )
