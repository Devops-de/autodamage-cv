"""
Perceptual Image Hashing and High-Performance LRU In-Memory Cache.
Provides sub-millisecond lookups for duplicate and near-identical damage photos.
"""
import time
import threading
from typing import Optional, Dict, Any
from collections import OrderedDict
import cv2
import numpy as np


class PerceptualImageCache:
    """
    Thread-safe perceptual difference hash (dHash) LRU cache.
    Eliminates redundant OpenCV edge processing for repeat uploads.
    """

    MAX_CACHE_SIZE = 250
    _lock = threading.Lock()
    _analysis_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    _validation_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

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
        except Exception:
            return "0" * 16

    @classmethod
    def get_validation(cls, dhash: str, panel_id: str) -> Optional[Dict[str, Any]]:
        cache_key = f"{dhash}:{panel_id}"
        with cls._lock:
            if cache_key in cls._validation_cache:
                cls._validation_cache.move_to_end(cache_key)
                return cls._validation_cache[cache_key].copy()
        return None

    @classmethod
    def set_validation(cls, dhash: str, panel_id: str, data: Dict[str, Any]) -> None:
        cache_key = f"{dhash}:{panel_id}"
        with cls._lock:
            if cache_key in cls._validation_cache:
                cls._validation_cache.move_to_end(cache_key)
            else:
                if len(cls._validation_cache) >= cls.MAX_CACHE_SIZE:
                    cls._validation_cache.popitem(last=False)
            cls._validation_cache[cache_key] = data.copy()

    @classmethod
    def get_analysis(
        cls,
        dhash: str,
        panel_id: str,
        sensitivity: float,
        use_oem: bool,
        deductible: float,
        blend: bool,
        zip_code: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        cache_key = f"{dhash}:{panel_id}:{sensitivity:.2f}:{use_oem}:{deductible:.0f}:{blend}:{zip_code or ''}"
        with cls._lock:
            if cache_key in cls._analysis_cache:
                cls._analysis_cache.move_to_end(cache_key)
                return cls._analysis_cache[cache_key].copy()
        return None

    @classmethod
    def set_analysis(
        cls,
        dhash: str,
        panel_id: str,
        sensitivity: float,
        use_oem: bool,
        deductible: float,
        blend: bool,
        zip_code: Optional[str],
        data: Dict[str, Any],
    ) -> None:
        cache_key = f"{dhash}:{panel_id}:{sensitivity:.2f}:{use_oem}:{deductible:.0f}:{blend}:{zip_code or ''}"
        with cls._lock:
            if cache_key in cls._analysis_cache:
                cls._analysis_cache.move_to_end(cache_key)
            else:
                if len(cls._analysis_cache) >= cls.MAX_CACHE_SIZE:
                    cls._analysis_cache.popitem(last=False)
            cls._analysis_cache[cache_key] = data.copy()

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._analysis_cache.clear()
            cls._validation_cache.clear()
