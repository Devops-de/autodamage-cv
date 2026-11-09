"""
Actuarial Insurance Claim Estimator Engine.
100% Deterministic business logic based on standard Mitchell / CCC ONE guidelines.
"""
import uuid
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.cv_engine.base import DamageAnalysisResult, DamageType, Severity
from app.estimator.parts_catalog import PANEL_CATALOG, PanelInfo
from app.estimator.regional_rates import lookup_regional_market, RegionalMarket


class RecommendedAction(str, Enum):
    REPAIR_AND_REFINISH = "REPAIR_AND_REFINISH"
    REPLACE_PANEL = "REPLACE_PANEL"
    BUFF_AND_POLISH_ONLY = "BUFF_AND_POLISH_ONLY"


class EstimateLineItem(BaseModel):
    category: str                     # "PARTS", "BODY_LABOR", "PAINT_LABOR", "SUPPLIES", "FEES"
    description: str
    quantity_or_hours: float
    rate: float
    subtotal: float


class InsuranceEstimate(BaseModel):
    claim_id: str
    panel_name: str
    panel_material: str
    recommended_action: RecommendedAction
    action_reason: str
    use_oem_parts: bool = True
    deductible: float = 500.00
    regional_market: str = "US National Baseline"
    zip_code: Optional[str] = None
    adjacent_blend_applied: bool = False
    
    # Financial breakdown
    line_items: List[EstimateLineItem]
    parts_subtotal: float
    body_labor_subtotal: float
    paint_labor_subtotal: float
    paint_materials_subtotal: float
    fees_subtotal: float
    sales_tax: float
    gross_estimate_total: float
    net_claim_payout: float           # Gross minus deductible

    total_labor_hours: float
    damage_area_cm2: float
    panel_damage_percentage: float


class EstimateCalculator:
    """
    Computes precise, transparent collision repair estimates using actuarial rules
    and standard Mitchell / CCC ONE collision repair guidelines.
    """

    @staticmethod
    def calculate(
        analysis: DamageAnalysisResult,
        panel_id: str = "FRONT_BUMPER",
        use_oem_parts: bool = True,
        deductible: float = 500.00,
        blend_adjacent_panels: bool = False,
        zip_code: Optional[str] = None,
    ) -> InsuranceEstimate:
        panel: PanelInfo = PANEL_CATALOG.get(panel_id, PANEL_CATALOG["FRONT_BUMPER"])
        market: RegionalMarket = lookup_regional_market(zip_code)

        # Regional & Material-adjusted labor rates
        reg_mult = market.labor_rate_multiplier
        base_body_rate = settings.rates.body_labor_rate_per_hour * reg_mult
        effective_body_rate = round(base_body_rate * panel.material_labor_multiplier, 2)
        effective_paint_rate = round(settings.rates.paint_labor_rate_per_hour * reg_mult, 2)
        paint_mat_rate = settings.rates.paint_materials_rate_per_hour

        damage_area = analysis.total_damaged_area_cm2
        panel_area = panel.surface_area_cm2
        pct_damaged = round((damage_area / max(1.0, panel_area)) * 100.0, 1)

        # 1. Decision Logic: Repair vs Replace (The 28% Rule)
        has_severe_tear = any(item.type == DamageType.TEAR or item.severity == Severity.SEVERE for item in analysis.items_detected)
        is_glass = "WINDSHIELD" in panel_id

        if is_glass:
            action = RecommendedAction.REPLACE_PANEL
            reason = "Automotive safety standards mandate complete windshield replacement for structural integrity."
        elif pct_damaged >= 28.0 or damage_area >= settings.replace_threshold_area_cm2 or has_severe_tear:
            action = RecommendedAction.REPLACE_PANEL
            reason = f"Damage area ({damage_area:.1f} cm² / {pct_damaged}%) exceeds the 28% repair threshold or exhibits severe structural deformity."
        elif damage_area < 5.0 and all(item.severity == Severity.MINOR for item in analysis.items_detected):
            action = RecommendedAction.BUFF_AND_POLISH_ONLY
            reason = f"Isolated superficial clear-coat marks ({damage_area:.1f} cm²). Paint buffing and spot correction recommended."
        else:
            action = RecommendedAction.REPAIR_AND_REFINISH
            reason = f"Damage is within repairable limits ({damage_area:.1f} cm² / {pct_damaged}% of panel). Panel structural integrity is intact."

        line_items: List[EstimateLineItem] = []
        parts_cost = 0.0
        body_labor_hours = 0.0
        paint_labor_hours = 0.0
        paint_materials_hours = 0.0

        # Description note for specialized material
        mat_note = ""
        if panel.material_labor_multiplier > 1.0:
            mat_note = f" [{panel.material} specialized rate: ${effective_body_rate:.2f}/hr]"

        # 2. Compute Line Items based on Action
        if action == RecommendedAction.REPLACE_PANEL:
            # New Part
            part_price = panel.oem_part_price if use_oem_parts else panel.aftermarket_part_price
            part_type_str = "OEM" if use_oem_parts else "CAPA Aftermarket"
            parts_cost += part_price
            line_items.append(
                EstimateLineItem(
                    category="PARTS",
                    description=f"New {panel.name} ({part_type_str})",
                    quantity_or_hours=1.0,
                    rate=part_price,
                    subtotal=part_price,
                )
            )

            # R&I Labor (Remove & Install)
            body_labor_hours += panel.r_and_i_labor_hours
            line_items.append(
                EstimateLineItem(
                    category="BODY_LABOR",
                    description=f"R&I (Remove & Install) {panel.name}{mat_note}",
                    quantity_or_hours=panel.r_and_i_labor_hours,
                    rate=effective_body_rate,
                    subtotal=round(panel.r_and_i_labor_hours * effective_body_rate, 2),
                )
            )

            # Refinish Paint Labor (if non-glass)
            if panel.standard_paint_refinish_hours > 0:
                paint_labor_hours += panel.standard_paint_refinish_hours
                line_items.append(
                    EstimateLineItem(
                        category="PAINT_LABOR",
                        description=f"Refinish spray paint {panel.name}",
                        quantity_or_hours=panel.standard_paint_refinish_hours,
                        rate=effective_paint_rate,
                        subtotal=round(panel.standard_paint_refinish_hours * effective_paint_rate, 2),
                    )
                )
                paint_materials_hours += panel.standard_paint_refinish_hours

        elif action == RecommendedAction.REPAIR_AND_REFINISH:
            # Sheet metal repair hours calculated from damaged cm²
            # Standard: ~1.0 hour of body repair labor per 40 cm² of damage, minimum 1.2 hrs
            body_labor_hours = round(max(1.2, damage_area / 40.0), 1)
            line_items.append(
                EstimateLineItem(
                    category="BODY_LABOR",
                    description=f"Dent extraction & metal repair ({damage_area:.1f} cm²){mat_note}",
                    quantity_or_hours=body_labor_hours,
                    rate=effective_body_rate,
                    subtotal=round(body_labor_hours * effective_body_rate, 2),
                )
            )

            # Paint refinish & feathering (standard panel paint hours * 0.85 for spot repair blend)
            paint_labor_hours = round(max(1.5, panel.standard_paint_refinish_hours * 0.85), 1)
            line_items.append(
                EstimateLineItem(
                    category="PAINT_LABOR",
                    description=f"Paint feather, prime, & clear coat blend",
                    quantity_or_hours=paint_labor_hours,
                    rate=effective_paint_rate,
                    subtotal=round(paint_labor_hours * effective_paint_rate, 2),
                )
            )
            paint_materials_hours = paint_labor_hours

        else: # BUFF_AND_POLISH_ONLY
            body_labor_hours = 0.8
            line_items.append(
                EstimateLineItem(
                    category="BODY_LABOR",
                    description=f"Spot clear-coat compound buff & micro-polish{mat_note}",
                    quantity_or_hours=0.8,
                    rate=effective_body_rate,
                    subtotal=round(0.8 * effective_body_rate, 2),
                )
            )

        # 3. Adjacent Panel Paint Blend (Mitchell 50% Rule)
        adjacent_blend_applied = False
        if blend_adjacent_panels and panel.standard_paint_refinish_hours > 0 and panel.adjacent_panels:
            for adj_id in panel.adjacent_panels[:2]:
                adj_panel = PANEL_CATALOG.get(adj_id)
                if adj_panel and adj_panel.standard_paint_refinish_hours > 0:
                    blend_hours = round(adj_panel.standard_paint_refinish_hours * 0.5, 1)
                    paint_labor_hours += blend_hours
                    paint_materials_hours += blend_hours
                    line_items.append(
                        EstimateLineItem(
                            category="PAINT_LABOR",
                            description=f"Adjacent blend: {adj_panel.name} (Mitchell 50% Rule)",
                            quantity_or_hours=blend_hours,
                            rate=effective_paint_rate,
                            subtotal=round(blend_hours * effective_paint_rate, 2),
                        )
                    )
                    adjacent_blend_applied = True

        # Paint Materials Surcharge
        paint_mat_subtotal = 0.0
        if paint_materials_hours > 0:
            paint_mat_subtotal = round(paint_materials_hours * paint_mat_rate, 2)
            line_items.append(
                EstimateLineItem(
                    category="SUPPLIES",
                    description=f"Paint supplies (Primer, Basecoat, Clear, Abrasives)",
                    quantity_or_hours=paint_materials_hours,
                    rate=paint_mat_rate,
                    subtotal=paint_mat_subtotal,
                )
            )

        # Environmental & EPA Hazardous Waste Disposal Fee
        epa_fee = 12.00
        line_items.append(
            EstimateLineItem(
                category="FEES",
                description="EPA Hazardous waste & paint disposal fee",
                quantity_or_hours=1.0,
                rate=epa_fee,
                subtotal=epa_fee,
            )
        )

        # Totals computation
        body_labor_subtotal = round(body_labor_hours * effective_body_rate, 2)
        paint_labor_subtotal = round(paint_labor_hours * effective_paint_rate, 2)
        fees_subtotal = epa_fee

        # Sales tax applied to Parts + Supplies
        taxable_amount = parts_cost + paint_mat_subtotal
        sales_tax = round(taxable_amount * market.sales_tax_rate, 2)

        gross_total = round(
            parts_cost + body_labor_subtotal + paint_labor_subtotal + paint_mat_subtotal + fees_subtotal + sales_tax,
            2,
        )

        net_claim = round(max(0.0, gross_total - deductible), 2)
        total_hours = round(body_labor_hours + paint_labor_hours, 1)

        claim_num = f"CLM-{uuid.uuid4().hex[:8].upper()}"

        return InsuranceEstimate(
            claim_id=claim_num,
            panel_name=panel.name,
            panel_material=panel.material,
            recommended_action=action,
            action_reason=reason,
            use_oem_parts=use_oem_parts,
            deductible=deductible,
            regional_market=market.region_name,
            zip_code=zip_code,
            adjacent_blend_applied=adjacent_blend_applied,
            line_items=line_items,
            parts_subtotal=parts_cost,
            body_labor_subtotal=body_labor_subtotal,
            paint_labor_subtotal=paint_labor_subtotal,
            paint_materials_subtotal=paint_mat_subtotal,
            fees_subtotal=fees_subtotal,
            sales_tax=sales_tax,
            gross_estimate_total=gross_total,
            net_claim_payout=net_claim,
            total_labor_hours=total_hours,
            damage_area_cm2=damage_area,
            panel_damage_percentage=pct_damaged,
        )
