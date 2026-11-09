"""
Configuration and Actuarial Collision Repair Standards (US Market).
Based on standard Mitchell / CCC ONE collision estimating guidelines.
"""

from pydantic import BaseModel


class RepairLaborRates(BaseModel):
    # Standard US market hourly rates for body shops
    body_labor_rate_per_hour: float = 65.00  # Sheet metal & structural repair
    paint_labor_rate_per_hour: float = 65.00  # Refinish & spray labor
    paint_materials_rate_per_hour: float = 38.00  # Clear coat, basecoat, primer, sandpaper
    mechanical_rate_per_hour: float = 110.00  # Suspension/mechanical if needed
    sales_tax_rate: float = 0.07  # 7% standard state/local parts tax


class Settings(BaseModel):
    app_name: str = "ClaimScope - Vehicle Damage & Collision Appraisal Engine"
    version: str = "1.0.0"
    rates: RepairLaborRates = RepairLaborRates()
    # Threshold: if damaged area exceeds 28% of panel area or 250 cm², recommend replacement
    replace_threshold_area_cm2: float = 250.0

    # Upload and resolution limits to prevent memory exhaustion & decompression bombs
    max_upload_size_bytes: int = 15 * 1024 * 1024  # 15 MB max file upload size
    max_image_dimension: int = 4096  # Max 4096px width or height
    max_image_pixels: int = 16_000_000  # Max 16 Megapixels total area


settings = Settings()
