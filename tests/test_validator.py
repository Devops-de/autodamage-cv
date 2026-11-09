"""
Unit tests for Vehicle and Panel Image Validation Filter.
"""
import cv2
import numpy as np
import pytest
from app.cv_engine.image_validator import VehicleImageValidator


def test_valid_car_panel():
    """A realistic painted vehicle panel with scratch should pass validation."""
    with open("app/static/sample_images/sample_bumper_scratch.jpg", "rb") as f:
        img_bytes = f.read()

    result = VehicleImageValidator.validate(img_bytes)
    assert result.is_valid is True
    assert result.rejection_reason is None


def test_reject_blurry_image():
    """Severely out-of-focus image should be rejected."""
    img = np.full((400, 400, 3), (120, 120, 120), dtype=np.uint8)
    # Add heavy Gaussian blur to eliminate all high-frequency gradients
    blurry = cv2.GaussianBlur(img, (51, 51), 20.0)
    _, buf = cv2.imencode(".jpg", blurry)

    result = VehicleImageValidator.validate(bytes(buf))
    assert result.is_valid is False
    assert "blurry" in result.rejection_reason.lower() or "focus" in result.rejection_reason.lower()


def test_reject_selfie_portrait():
    """Images with high skin tone ratios should be rejected as non-vehicle."""
    # Create an image dominated by skin tone in YCrCb (Cr ~150, Cb ~100)
    # Equivalent to approx BGR: (130, 160, 230)
    skin_img = np.full((400, 400, 3), (130, 160, 230), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", skin_img)

    result = VehicleImageValidator.validate(bytes(buf))
    assert result.is_valid is False
    assert "person" in result.rejection_reason.lower() or "portrait" in result.rejection_reason.lower()


def test_reject_blank_or_dark_image():
    """Extremely underexposed images should be rejected."""
    dark_img = np.full((300, 300, 3), 5, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dark_img)

    result = VehicleImageValidator.validate(bytes(buf))
    assert result.is_valid is False
    assert "dark" in result.rejection_reason.lower()
