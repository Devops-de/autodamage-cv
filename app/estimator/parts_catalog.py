"""
Vehicle Panels and Parts Pricing Catalog.
Standard US Collision Repair baseline data (OEM and Certified Aftermarket parts).
"""
from typing import Dict
from pydantic import BaseModel


class PanelInfo(BaseModel):
    panel_id: str
    name: str
    material: str                     # "Steel", "Aluminum", "Plastic / Polyurethane"
    surface_area_cm2: float           # Approximate total surface area
    oem_part_price: float             # Original Equipment Manufacturer part price ($)
    aftermarket_part_price: float     # CAPA / Aftermarket alternative price ($)
    r_and_i_labor_hours: float        # Remove & Install labor hours
    standard_paint_refinish_hours: float # Base refinish spray hours


PANEL_CATALOG: Dict[str, PanelInfo] = {
    "FRONT_BUMPER": PanelInfo(
        panel_id="FRONT_BUMPER",
        name="Front Bumper Cover",
        material="Plastic / Polyurethane",
        surface_area_cm2=850.0,
        oem_part_price=380.00,
        aftermarket_part_price=220.00,
        r_and_i_labor_hours=2.2,
        standard_paint_refinish_hours=2.6,
    ),
    "REAR_BUMPER": PanelInfo(
        panel_id="REAR_BUMPER",
        name="Rear Bumper Cover",
        material="Plastic / Polyurethane",
        surface_area_cm2=820.0,
        oem_part_price=360.00,
        aftermarket_part_price=210.00,
        r_and_i_labor_hours=2.0,
        standard_paint_refinish_hours=2.5,
    ),
    "HOOD": PanelInfo(
        panel_id="HOOD",
        name="Engine Hood",
        material="Aluminum / Steel",
        surface_area_cm2=1400.0,
        oem_part_price=590.00,
        aftermarket_part_price=380.00,
        r_and_i_labor_hours=2.5,
        standard_paint_refinish_hours=3.5,
    ),
    "FRONT_FENDER_LEFT": PanelInfo(
        panel_id="FRONT_FENDER_LEFT",
        name="Front Left Fender",
        material="Steel",
        surface_area_cm2=650.0,
        oem_part_price=290.00,
        aftermarket_part_price=175.00,
        r_and_i_labor_hours=1.8,
        standard_paint_refinish_hours=2.0,
    ),
    "FRONT_FENDER_RIGHT": PanelInfo(
        panel_id="FRONT_FENDER_RIGHT",
        name="Front Right Fender",
        material="Steel",
        surface_area_cm2=650.0,
        oem_part_price=290.00,
        aftermarket_part_price=175.00,
        r_and_i_labor_hours=1.8,
        standard_paint_refinish_hours=2.0,
    ),
    "DRIVER_FRONT_DOOR": PanelInfo(
        panel_id="DRIVER_FRONT_DOOR",
        name="Driver Front Door Panel",
        material="High-Strength Steel",
        surface_area_cm2=1100.0,
        oem_part_price=680.00,
        aftermarket_part_price=440.00,
        r_and_i_labor_hours=3.2,
        standard_paint_refinish_hours=3.4,
    ),
    "PASSENGER_FRONT_DOOR": PanelInfo(
        panel_id="PASSENGER_FRONT_DOOR",
        name="Passenger Front Door Panel",
        material="High-Strength Steel",
        surface_area_cm2=1100.0,
        oem_part_price=680.00,
        aftermarket_part_price=440.00,
        r_and_i_labor_hours=3.2,
        standard_paint_refinish_hours=3.4,
    ),
    "TRUNK_TAILGATE": PanelInfo(
        panel_id="TRUNK_TAILGATE",
        name="Trunk Lid / Tailgate",
        material="Steel",
        surface_area_cm2=950.0,
        oem_part_price=490.00,
        aftermarket_part_price=310.00,
        r_and_i_labor_hours=2.4,
        standard_paint_refinish_hours=2.8,
    ),
    "WINDSHIELD": PanelInfo(
        panel_id="WINDSHIELD",
        name="Front Windshield Glass",
        material="Laminated Safety Glass",
        surface_area_cm2=1200.0,
        oem_part_price=420.00,
        aftermarket_part_price=280.00,
        r_and_i_labor_hours=2.8,
        standard_paint_refinish_hours=0.0,
    ),
}
