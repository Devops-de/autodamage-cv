"""
Vehicle panel damage detector using standard OpenCV image processing.
Extracts scratches, dents, and paint abrasions using edge detection,
color-space thresholding, and contour geometry.
"""

import base64

import cv2
import numpy as np

from app.core.config import settings
from app.cv_engine.base import (
    BaseDamageDetector,
    BoundingBox,
    DamageAnalysisResult,
    DamageType,
    DetectedDamageItem,
    DetectionOptions,
    Severity,
)


class ClassicalDamageDetector(BaseDamageDetector):
    """
    Classical CV Detector using OpenCV.
    Deterministic, mathematical, reproducible, and requires zero neural networks or GPUs.
    """

    def __init__(self):
        self.engine_name = "OpenCV Classical Pipeline"

    def detect(self, image_bytes: bytes, options: DetectionOptions) -> DamageAnalysisResult:
        img = self._decode_and_normalize(image_bytes)
        h, w = img.shape[:2]
        annotated_img = img.copy()

        # Preprocessing: Bilateral filter smooths flat paint while preserving sharp scratch edges
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

        # CLAHE reveals subtle dents and clear-coat scrapes across uneven lighting
        clahe = cv2.createCLAHE(clipLimit=2.0 + (options.sensitivity * 2.0), tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        scratch_mask = (
            self._detect_scratches(enhanced_gray, options.sensitivity)
            if options.detect_scratches
            else np.zeros((h, w), dtype=np.uint8)
        )
        dent_mask = (
            self._detect_dents_and_paint_loss(smooth, options.sensitivity)
            if options.detect_dents
            else np.zeros((h, w), dtype=np.uint8)
        )

        combined_mask = cv2.bitwise_or(scratch_mask, dent_mask)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        damage_items = self._extract_damage_items(contours, scratch_mask, annotated_img, w, h, options)

        total_damaged_area = round(sum(i.area_cm2 for i in damage_items), 2)
        overall_severity = self._resolve_overall_severity(damage_items, total_damaged_area)

        return DamageAnalysisResult(
            engine_used=self.engine_name,
            panel_name=options.target_panel,
            items_detected=damage_items,
            total_damaged_area_cm2=total_damaged_area,
            overall_severity=overall_severity,
            annotated_image_base64=self._encode_jpeg_data_uri(annotated_img),
            mask_image_base64=self._encode_mask_data_uri(combined_mask),
            diagnostics={
                "image_width": w,
                "image_height": h,
                "total_contours_analyzed": len(contours),
                "sensitivity_used": options.sensitivity,
                "detection_method": "Canny + LAB Color Distance + Mathematical Contours",
            },
        )

    def _decode_and_normalize(self, image_bytes: bytes) -> np.ndarray:
        """Decodes image bytes, enforces resolution caps, and normalizes to <= 1280px."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image. Please upload a valid JPG/PNG/WEBP image.")

        h, w = img.shape[:2]
        max_dim_limit = settings.max_image_dimension
        max_px_limit = settings.max_image_pixels
        if h > max_dim_limit or w > max_dim_limit or (h * w) > max_px_limit:
            raise ValueError(
                f"Image dimensions ({w}x{h}px) exceed maximum allowed resolution limit of "
                f"{max_dim_limit}x{max_dim_limit}px ({max_px_limit // 1_000_000}MP)."
            )

        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _extract_damage_items(
        self,
        contours,
        scratch_mask: np.ndarray,
        annotated_img: np.ndarray,
        img_w: int,
        img_h: int,
        options: DetectionOptions,
    ) -> list[DetectedDamageItem]:
        """Filters valid defect contours, classifies damage types, and renders bounding annotations."""
        damage_items: list[DetectedDamageItem] = []
        px_per_cm = max(1.0, options.pixels_per_cm)
        px_per_cm2 = px_per_cm * px_per_cm

        for cnt in contours:
            area_pixels = cv2.contourArea(cnt)
            if area_pixels < options.min_defect_area_pixels:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Avoid full-screen frame boundary artifacts
            if bw > img_w * 0.9 and bh > img_h * 0.9:
                continue

            item_id = len(damage_items) + 1
            area_cm2 = round(area_pixels / px_per_cm2, 2)
            d_type, type_label = self._classify_defect(cnt, x, y, bw, bh, area_pixels, area_cm2, scratch_mask)
            severity, color = self._score_severity(area_cm2)
            confidence = self._compute_confidence(cnt, area_pixels, area_cm2)

            item = DetectedDamageItem(
                id=item_id,
                type=d_type,
                severity=severity,
                bounding_box=BoundingBox(x=x, y=y, width=bw, height=bh),
                area_pixels=int(area_pixels),
                area_cm2=area_cm2,
                confidence_score=confidence,
                details=f"{type_label} ({area_cm2} cm²)",
            )
            damage_items.append(item)
            self._draw_annotation(annotated_img, item_id, x, y, bw, bh, d_type, severity, color)

        return damage_items

    @staticmethod
    def _classify_defect(
        cnt,
        x: int,
        y: int,
        bw: int,
        bh: int,
        area_pixels: float,
        area_cm2: float,
        scratch_mask: np.ndarray,
    ) -> tuple[DamageType, str]:
        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        roi_scratch = scratch_mask[y : y + bh, x : x + bw]
        scratch_ratio = np.count_nonzero(roi_scratch) / max(1, area_pixels)

        if aspect_ratio >= 3.0 or scratch_ratio > 0.4:
            return DamageType.SCRATCH, "Scratch / Fracture"
        if area_cm2 > 80.0:
            d_type = DamageType.TEAR if aspect_ratio > 2.5 else DamageType.DENT
            return d_type, "Deep Dent / Deformity"
        d_type = DamageType.PAINT_CHIP if area_cm2 < 15.0 else DamageType.DENT
        return d_type, "Paint Scraping / Dent"

    @staticmethod
    def _score_severity(area_cm2: float) -> tuple[Severity, tuple[int, int, int]]:
        if area_cm2 < 20.0:
            return Severity.MINOR, (0, 220, 255)
        if area_cm2 < 100.0:
            return Severity.MODERATE, (0, 140, 255)
        return Severity.SEVERE, (0, 0, 255)

    @staticmethod
    def _compute_confidence(cnt, area_pixels: float, area_cm2: float) -> float:
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(area_pixels) / max(1.0, hull_area)
        return round(min(0.98, max(0.55, 0.5 + (solidity * 0.3) + (min(100, area_cm2) / 400.0))), 2)

    @staticmethod
    def _draw_annotation(
        annotated_img: np.ndarray,
        item_id: int,
        x: int,
        y: int,
        bw: int,
        bh: int,
        d_type: DamageType,
        severity: Severity,
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), color, 2)
        label_text = f"#{item_id} {d_type.value} ({severity.value})"
        (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_img, (x, max(0, y - 22)), (x + tw + 10, y), color, -1)
        cv2.putText(
            annotated_img,
            label_text,
            (x + 5, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _resolve_overall_severity(damage_items: list[DetectedDamageItem], total_damaged_area: float) -> Severity:
        if any(i.severity == Severity.SEVERE for i in damage_items) or total_damaged_area > 150.0:
            return Severity.SEVERE
        if any(i.severity == Severity.MODERATE for i in damage_items) or total_damaged_area > 30.0:
            return Severity.MODERATE
        return Severity.MINOR

    @staticmethod
    def _encode_jpeg_data_uri(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    @staticmethod
    def _encode_mask_data_uri(mask: np.ndarray) -> str:
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".png", colored_mask)
        return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

    def _detect_scratches(self, enhanced_gray: np.ndarray, sensitivity: float) -> np.ndarray:
        """Detects sharp linear scratches using Canny edge filtering and directional morphology."""
        med = np.median(enhanced_gray)
        lower = int(max(20, (1.0 - (sensitivity * 0.45)) * med))
        upper = int(min(240, (1.0 + (sensitivity * 0.45)) * med))

        edges = cv2.Canny(enhanced_gray, threshold1=lower, threshold2=upper)

        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        closed_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_horiz)
        closed_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_vert)

        scratch_mask = cv2.bitwise_or(closed_h, closed_v)
        return cv2.dilate(scratch_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    def _detect_dents_and_paint_loss(self, bgr_img: np.ndarray, sensitivity: float) -> np.ndarray:
        """Detects dents via localized shadow gradients in CIE LAB luminance space."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        blur_fine = cv2.GaussianBlur(l_channel, (9, 9), 2.0)
        blur_broad = cv2.GaussianBlur(l_channel, (31, 31), 8.0)
        dog = cv2.subtract(blur_broad, blur_fine)

        threshold_val = int(25 - (sensitivity * 12))
        _, dent_thresh = cv2.threshold(dog, threshold_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(dent_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
