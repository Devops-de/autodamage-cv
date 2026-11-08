"""
Unit tests for the Classical Computer Vision Engine.
Tests image decoding, edge/contour extraction, and mathematical reproducibility.
"""
import os
import cv2
import numpy as np
import pytest

from app.cv_engine.classical_detector import ClassicalDamageDetector
from app.cv_engine.base import DetectionOptions, Severity, DamageType


@pytest.fixture
def sample_scratch_image():
    """Generates a known synthetic image with a scratch line in memory."""
    img = np.full((400, 400, 3), (30, 30, 180), dtype=np.uint8) # Dark red panel
    # Draw high-contrast scratch
    cv2.line(img, (100, 100), (300, 300), (240, 240, 240), thickness=4)
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


@pytest.fixture
def clean_undamaged_image():
    """Generates a pristine, undamaged panel image in memory."""
    img = np.full((400, 400, 3), (120, 120, 120), dtype=np.uint8) # Flat grey
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def test_detector_detects_scratch(sample_scratch_image):
    detector = ClassicalDamageDetector()
    options = DetectionOptions(target_panel="FRONT_BUMPER", sensitivity=0.6)
    result = detector.detect(sample_scratch_image, options)

    assert "OpenCV" in result.engine_used
    assert len(result.items_detected) > 0
    assert result.total_damaged_area_cm2 > 0.0
    assert any(i.type == DamageType.SCRATCH for i in result.items_detected)


def test_detector_clean_image_produces_zero_defects(clean_undamaged_image):
    detector = ClassicalDamageDetector()
    options = DetectionOptions(target_panel="HOOD", sensitivity=0.5)
    result = detector.detect(clean_undamaged_image, options)

    assert len(result.items_detected) == 0
    assert result.total_damaged_area_cm2 == 0.0


def test_detector_reproducibility(sample_scratch_image):
    """Classical CV must produce the EXACT same results across repeated runs (100% deterministic)."""
    detector = ClassicalDamageDetector()
    options = DetectionOptions(target_panel="FRONT_BUMPER", sensitivity=0.5)

    res1 = detector.detect(sample_scratch_image, options)
    res2 = detector.detect(sample_scratch_image, options)

    assert res1.total_damaged_area_cm2 == res2.total_damaged_area_cm2
    assert len(res1.items_detected) == len(res2.items_detected)
    assert res1.overall_severity == res2.overall_severity
