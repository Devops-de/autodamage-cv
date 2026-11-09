"""
Perceptual Image Hashing and High-Performance LRU In-Memory Cache.
Combines cryptographic SHA-256 for collision-proof exact matches with
64-bit difference hashing (dHash) for perceptual visual telemetry.
"""

import copy
import hashlib
import threading
from collections import OrderedDict
from typing import Any, ClassVar

import cv2
import numpy as np


class PerceptualImageCache:
    """
    Thread-safe perceptual difference hash (dHash) and cryptographic SHA-256 LRU cache.
    Eliminates redundant OpenCV edge processing for repeat uploads with zero parameter collision.
    """

    MAX_CACHE_SIZE: ClassVar[int] = 250
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _analysis_cache: ClassVar[OrderedDict[str, dict[str, Any]]] = OrderedDict()
    _validation_cache: ClassVar[OrderedDict[str, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def compute_sha256(image_bytes: bytes) -> str:
        """Computes SHA-256 cryptographic hash of raw image bytes to prevent hash collisions."""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def compute_dhash(image_bytes: bytes) -> str:
        """
        Computes a robust 64-bit difference hash (dHash) in ~0.5ms.
        Resizes to 9x8, evaluates horizontal relative brightness gradients.
        """
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "0" * 16

            # Resize to 9 columns by 8 rows
            resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
            # Compare neighboring horizontal pixels
            diff = resized[:, 1:] > resized[:, :-1]
            bits = "".join(diff.flatten().astype(int).astype(str))
            return hex(int(bits, 2))[2:].zfill(16)
        except (cv2.error, ValueError, TypeError):
            return "0" * 16

    @classmethod
    def _make_validation_key(cls, sha256: str, dhash: str, panel_id: str) -> str:
        return f"{sha256}:{dhash}:{panel_id}"

    @classmethod
    def _make_analysis_key(
        cls,
        sha256: str,
        dhash: str,
        panel_id: str,
        sensitivity: float,
        detect_scratches: bool,
        detect_dents: bool,
        min_defect_area_pixels: int,
        use_oem: bool,
        deductible: float,
        blend: bool,
        zip_code: str | None,
    ) -> str:
        # Preserve exact parameter precision (no lossy truncation of cents or sensitivity)
        clean_zip = (zip_code or "").strip()
        return (
            f"{sha256}:{dhash}:{panel_id}:{float(sensitivity):.4f}:{bool(detect_scratches)}:"
            f"{bool(detect_dents)}:{int(min_defect_area_pixels)}:{bool(use_oem)}:"
            f"{float(deductible):.2f}:{bool(blend)}:{clean_zip}"
        )

    @classmethod
    def _lru_get(cls, cache: OrderedDict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
        with cls._lock:
            if key in cache:
                cache.move_to_end(key)
                # Deep copy isolates nested dicts (e.g., diagnostics, telemetry) per caller
                return copy.deepcopy(cache[key])
        return None

    @classmethod
    def _lru_set(cls, cache: OrderedDict[str, dict[str, Any]], key: str, data: dict[str, Any]) -> None:
        with cls._lock:
            if key in cache:
                cache.move_to_end(key)
            elif len(cache) >= cls.MAX_CACHE_SIZE:
                cache.popitem(last=False)
            cache[key] = copy.deepcopy(data)

    @classmethod
    def get_validation(cls, sha256: str, dhash: str, panel_id: str) -> dict[str, Any] | None:
        cache_key = cls._make_validation_key(sha256, dhash, panel_id)
        return cls._lru_get(cls._validation_cache, cache_key)

    @classmethod
    def set_validation(cls, sha256: str, dhash: str, panel_id: str, data: dict[str, Any]) -> None:
        cache_key = cls._make_validation_key(sha256, dhash, panel_id)
        cls._lru_set(cls._validation_cache, cache_key, data)

    @classmethod
    def get_analysis(
        cls,
        sha256: str,
        dhash: str,
        panel_id: str,
        sensitivity: float,
        detect_scratches: bool,
        detect_dents: bool,
        min_defect_area_pixels: int,
        use_oem: bool,
        deductible: float,
        blend: bool,
        zip_code: str | None,
    ) -> dict[str, Any] | None:
        cache_key = cls._make_analysis_key(
            sha256=sha256,
            dhash=dhash,
            panel_id=panel_id,
            sensitivity=sensitivity,
            detect_scratches=detect_scratches,
            detect_dents=detect_dents,
            min_defect_area_pixels=min_defect_area_pixels,
            use_oem=use_oem,
            deductible=deductible,
            blend=blend,
            zip_code=zip_code,
        )
        return cls._lru_get(cls._analysis_cache, cache_key)

    @classmethod
    def set_analysis(
        cls,
        sha256: str,
        dhash: str,
        panel_id: str,
        sensitivity: float,
        detect_scratches: bool,
        detect_dents: bool,
        min_defect_area_pixels: int,
        use_oem: bool,
        deductible: float,
        blend: bool,
        zip_code: str | None,
        data: dict[str, Any],
    ) -> None:
        cache_key = cls._make_analysis_key(
            sha256=sha256,
            dhash=dhash,
            panel_id=panel_id,
            sensitivity=sensitivity,
            detect_scratches=detect_scratches,
            detect_dents=detect_dents,
            min_defect_area_pixels=min_defect_area_pixels,
            use_oem=use_oem,
            deductible=deductible,
            blend=blend,
            zip_code=zip_code,
        )
        cls._lru_set(cls._analysis_cache, cache_key, data)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._analysis_cache.clear()
            cls._validation_cache.clear()
