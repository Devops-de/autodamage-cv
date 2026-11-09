"""
Vehicle and Panel Image Validation Filter.
Applies classical computer vision heuristics (NO AI / NO ML) to verify that
an uploaded photograph is a valid, focused vehicle panel suitable for appraisal,
and rejects non-automotive images (selfies, documents, extreme blur, indoor objects).
"""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from pydantic import BaseModel


class ValidationResult(BaseModel):
    is_valid: bool
    rejection_reason: Optional[str] = None
    blur_score: float
    skin_tone_ratio: float
    surface_smoothness_ratio: float
    brightness_mean: float
    diagnostics: Dict[str, Any] = {}


class VehicleImageValidator:
    """
    Multi-stage classical CV filter to ensure uploaded photos are clear,
    focused, and plausible automotive body panels.
    """

    # Minimum Laplacian variance for focus (below this is blurry)
    MIN_BLUR_THRESHOLD = 60.0

    # Max allowed ratio of human skin tone pixels (above this is likely a person/selfie)
    MAX_SKIN_RATIO = 0.22

    # Minimum dimension in pixels
    MIN_DIMENSION = 250

    @classmethod
    def validate(cls, image_bytes: bytes) -> ValidationResult:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return ValidationResult(
                is_valid=False,
                rejection_reason="File could not be decoded as a valid image. Please upload a JPEG or PNG.",
                blur_score=0.0,
                skin_tone_ratio=0.0,
                surface_smoothness_ratio=0.0,
                brightness_mean=0.0,
            )

        h, w = img.shape[:2]
        total_pixels = h * w

        # 1. Resolution Check
        if h < cls.MIN_DIMENSION or w < cls.MIN_DIMENSION:
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Image resolution is too low ({w}x{h}px). Minimum required is {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION}px for damage appraisal.",
                blur_score=0.0,
                skin_tone_ratio=0.0,
                surface_smoothness_ratio=0.0,
                brightness_mean=0.0,
            )

        # 2. Lighting & Exposure Check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))

        if brightness_mean < 18.0:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Photo is too dark to inspect vehicle damage. Please upload a photo taken with adequate lighting.",
                blur_score=0.0,
                skin_tone_ratio=0.0,
                surface_smoothness_ratio=0.0,
                brightness_mean=brightness_mean,
            )

        if brightness_mean > 248.0:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Photo is overexposed or nearly blank. Please upload a clear photo of the vehicle panel.",
                blur_score=0.0,
                skin_tone_ratio=0.0,
                surface_smoothness_ratio=0.0,
                brightness_mean=brightness_mean,
            )

        # 3. Document / White Paper / Screenshot Rejection Filter
        # Documents have high proportion of near-white pixels (> 230) and discrete line edges
        white_pixels = np.count_nonzero(gray > 230)
        white_ratio = white_pixels / total_pixels
        if white_ratio > 0.65:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Image appears to be a paper document, screenshot, or invoice. Please upload a photograph of the physical vehicle panel.",
                blur_score=0.0,
                skin_tone_ratio=0.0,
                surface_smoothness_ratio=0.0,
                brightness_mean=brightness_mean,
            )

        # 4. Human / Selfie Rejection Filter (YCrCb Skin Tone Segmentation)
        # Automotive panels do not consist of human skin. Rejects selfies, portraits, and people.
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Standard human skin chrominance bounding box in YCrCb
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_pixels = np.count_nonzero(skin_mask)
        skin_ratio = float(skin_pixels / total_pixels)

        if skin_ratio > cls.MAX_SKIN_RATIO:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Image appears to be a person or portrait. Please upload a photo focused specifically on the damaged vehicle panel.",
                blur_score=0.0,
                skin_tone_ratio=skin_ratio,
                surface_smoothness_ratio=0.0,
                brightness_mean=brightness_mean,
            )

        # 5. Blur & Focus Quality Check (Laplacian Variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < cls.MIN_BLUR_THRESHOLD:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Photo is too blurry or out of focus. A clear, sharp photo is required to detect surface damage accurately.",
                blur_score=blur_score,
                skin_tone_ratio=skin_ratio,
                surface_smoothness_ratio=0.0,
                brightness_mean=brightness_mean,
            )

        # 6. Automotive Surface & Specular Uniformity Check
        # Car panels are painted metal or molded plastic characterized by continuous base paint
        # with localized specular gloss, not high-frequency chaotic textures (like grass, carpet, or fabric).
        blurred_surface = cv2.GaussianBlur(gray, (21, 21), 0)
        surface_diff = cv2.absdiff(gray, blurred_surface)
        # In a real car panel, the majority of the surface area is smooth base paint
        smooth_pixels = np.count_nonzero(surface_diff < 15)
        smoothness_ratio = float(smooth_pixels / total_pixels)

        # Natural backgrounds like dense grass/carpet have smoothness < 0.20
        if smoothness_ratio < 0.18:
            return ValidationResult(
                is_valid=False,
                rejection_reason="Image texture does not match an automotive painted surface (appears to be foliage, fabric, or rough ground). Please ensure the camera is framed directly on the car panel.",
                blur_score=blur_score,
                skin_tone_ratio=skin_ratio,
                surface_smoothness_ratio=smoothness_ratio,
                brightness_mean=brightness_mean,
            )

        # Validation passed
        return ValidationResult(
            is_valid=True,
            rejection_reason=None,
            blur_score=blur_score,
            skin_tone_ratio=skin_ratio,
            surface_smoothness_ratio=smoothness_ratio,
            brightness_mean=brightness_mean,
            diagnostics={
                "resolution": f"{w}x{h}",
                "focus_quality": "Sharp" if blur_score > 150 else "Acceptable",
                "surface_type": "Automotive Painted Panel / Composite",
            },
        )
