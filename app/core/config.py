"""
Configuration and Actuarial Collision Repair Standards (US Market).
Based on standard Mitchell / CCC ONE collision estimating guidelines.
"""
from pydantic import BaseModel


class RepairLaborRates(BaseModel):
    # Standard US market hourly rates for body shops
    body_labor_rate_per_hour: float = 65.00      # Sheet metal & structural repair
    paint_labor_rate_per_hour: float = 65.00     # Refinish & spray labor
    paint_materials_rate_per_hour: float = 38.00 # Clear coat, basecoat, primer, sandpaper
    mechanical_rate_per_hour: float = 110.00     # Suspension/mechanical if needed
    sales_tax_rate: float = 0.07                 # 7% standard state/local parts tax


class Settings(BaseModel):
    app_name: str = "AutoDamage CV - Vehicle Damage & Estimate Engine"
    version: str = "1.0.0"
    rates: RepairLaborRates = RepairLaborRates()
    # Threshold: if damaged area exceeds 30% of panel area or 250 cm², recommend replacement
    replace_threshold_area_cm2: float = 250.0
    replace_threshold_severity_score: float = 75.0


settings = Settings()
