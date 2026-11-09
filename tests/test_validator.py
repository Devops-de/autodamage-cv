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


def test_reject_document_or_invoice():
    """Paper documents, white sheets, and text invoices should be rejected."""
    doc_img = np.full((400, 400, 3), 245, dtype=np.uint8)
    # Add a few dark lines like text on white paper
    cv2.line(doc_img, (50, 100), (350, 100), (20, 20, 20), 2)
    cv2.line(doc_img, (50, 150), (350, 150), (20, 20, 20), 2)
    _, buf = cv2.imencode(".jpg", doc_img)

    result = VehicleImageValidator.validate(bytes(buf))
    assert result.is_valid is False
    assert "document" in result.rejection_reason.lower() or "invoice" in result.rejection_reason.lower()


def test_reject_undersized_image():
    """Low resolution images below minimum dimensions should be rejected."""
    tiny_img = np.full((120, 120, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", tiny_img)

    result = VehicleImageValidator.validate(bytes(buf))
    assert result.is_valid is False
    assert "resolution" in result.rejection_reason.lower()


def test_bumper_section_validation():
    """Bumper section validation should approve realistic damaged bumper and report bumper diagnostics."""
    with open("app/static/sample_images/sample_bumper_scratch.jpg", "rb") as f:
        img_bytes = f.read()

    result = VehicleImageValidator.validate(img_bytes, panel_id="FRONT_BUMPER")
    assert result.is_valid is True
    assert result.rejection_reason is None
    assert result.defect_count > 0
    assert "Front Bumper Cover" in result.diagnostics["target_panel"]
    assert "Polyurethane" in result.diagnostics["surface_type"]


def test_reject_clean_car_without_damage():
    """Images with zero detectable damage or surface anomalies must be rejected."""
    clean_car = np.full((400, 400, 3), (120, 120, 120), dtype=np.uint8)
    # Add gentle smooth lighting gradient without any scratch, dent, or crack
    for y in range(400):
        clean_car[y, :] = min(220, 100 + int(y * 0.25))
    _, buf = cv2.imencode(".jpg", clean_car)

    result = VehicleImageValidator.validate(bytes(buf), panel_id="FRONT_BUMPER")
    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert ("no vehicle damage" in result.rejection_reason.lower()
            or "defect detected" in result.rejection_reason.lower()
            or "blurry" in result.rejection_reason.lower()
            or "focus" in result.rejection_reason.lower())


def test_reject_multicolor_indoor_scene():
    """Multi-colored domestic scenes or furniture must be rejected."""
    room = np.full((400, 400, 3), (40, 40, 180), dtype=np.uint8) # red wall
    cv2.rectangle(room, (50, 100), (200, 300), (200, 100, 30), -1) # blue chair
    cv2.rectangle(room, (250, 150), (380, 350), (40, 180, 50), -1) # green table
    _, buf = cv2.imencode(".jpg", room)

    result = VehicleImageValidator.validate(bytes(buf), panel_id="FRONT_BUMPER")
    assert result.is_valid is False
    assert "indoor" in result.rejection_reason.lower() or "clutter" in result.rejection_reason.lower()
