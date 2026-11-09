"""
Actuarial Insurance Claim Estimator Engine.
100% Deterministic business logic based on standard Mitchell / CCC ONE guidelines.
"""

import uuid
from enum import Enum

from pydantic import BaseModel

from app.core.config import settings
from app.cv_engine.base import DamageAnalysisResult, DamageType, Severity
from app.estimator.parts_catalog import PANEL_CATALOG, PanelInfo
from app.estimator.regional_rates import RegionalMarket, lookup_regional_market


class RecommendedAction(str, Enum):
    REPAIR_AND_REFINISH = "REPAIR_AND_REFINISH"
    REPLACE_PANEL = "REPLACE_PANEL"
    BUFF_AND_POLISH_ONLY = "BUFF_AND_POLISH_ONLY"


class EstimateLineItem(BaseModel):
    category: str  # "PARTS", "BODY_LABOR", "PAINT_LABOR", "SUPPLIES", "FEES"
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
    zip_code: str | None = None
    adjacent_blend_applied: bool = False

    # Financial breakdown
    line_items: list[EstimateLineItem]
    parts_subtotal: float
    body_labor_subtotal: float
    paint_labor_subtotal: float
    paint_materials_subtotal: float
    fees_subtotal: float
    sales_tax: float
    gross_estimate_total: float
    net_claim_payout: float  # Gross minus deductible

    total_labor_hours: float
    damage_area_cm2: float
    panel_damage_percentage: float


class EstimateCalculator:
    """
    Computes precise, transparent collision repair estimates using actuarial rules
    and standard Mitchell / CCC ONE collision repair guidelines.
    """

    @classmethod
    def calculate(
        cls,
        analysis: DamageAnalysisResult,
        panel_id: str = "FRONT_BUMPER",
        use_oem_parts: bool = True,
        deductible: float = 500.00,
        blend_adjacent_panels: bool = False,
        zip_code: str | None = None,
    ) -> InsuranceEstimate:
        panel: PanelInfo = PANEL_CATALOG.get(panel_id, PANEL_CATALOG["FRONT_BUMPER"])
        market: RegionalMarket = lookup_regional_market(zip_code)
        safe_deductible = max(0.0, float(deductible))

        reg_mult = market.labor_rate_multiplier
        effective_body_rate = round(
            settings.rates.body_labor_rate_per_hour * reg_mult * panel.material_labor_multiplier, 2
        )
        effective_paint_rate = round(settings.rates.paint_labor_rate_per_hour * reg_mult, 2)
        paint_mat_rate = settings.rates.paint_materials_rate_per_hour

        damage_area = analysis.total_damaged_area_cm2
        pct_damaged = round((damage_area / max(1.0, panel.surface_area_cm2)) * 100.0, 1)

        action, reason = cls._determine_action(analysis, panel_id, damage_area, pct_damaged)
        mat_note = (
            f" [{panel.material} specialized rate: ${effective_body_rate:.2f}/hr]"
            if panel.material_labor_multiplier > 1.0
            else ""
        )

        line_items, parts_cost, body_labor_hours, paint_labor_hours, paint_materials_hours = cls._build_action_items(
            action=action,
            panel=panel,
            damage_area=damage_area,
            use_oem_parts=use_oem_parts,
            effective_body_rate=effective_body_rate,
            effective_paint_rate=effective_paint_rate,
            mat_note=mat_note,
        )

        blend_paint_hrs, adjacent_blend_applied = cls._append_adjacent_blend_items(
            line_items=line_items,
            panel=panel,
            blend_adjacent_panels=blend_adjacent_panels,
            effective_paint_rate=effective_paint_rate,
        )
        paint_labor_hours += blend_paint_hrs
        paint_materials_hours += blend_paint_hrs

        paint_mat_subtotal = 0.0
        if paint_materials_hours > 0:
            paint_mat_subtotal = round(paint_materials_hours * paint_mat_rate, 2)
            line_items.append(
                EstimateLineItem(
                    category="SUPPLIES",
                    description="Paint supplies (Primer, Basecoat, Clear, Abrasives)",
                    quantity_or_hours=paint_materials_hours,
                    rate=paint_mat_rate,
                    subtotal=paint_mat_subtotal,
                )
            )

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

        body_labor_subtotal = round(body_labor_hours * effective_body_rate, 2)
        paint_labor_subtotal = round(paint_labor_hours * effective_paint_rate, 2)
        sales_tax = round((parts_cost + paint_mat_subtotal) * market.sales_tax_rate, 2)
        gross_total = round(
            parts_cost + body_labor_subtotal + paint_labor_subtotal + paint_mat_subtotal + epa_fee + sales_tax,
            2,
        )

        return InsuranceEstimate(
            claim_id=f"CLM-{uuid.uuid4().hex[:8].upper()}",
            panel_name=panel.name,
            panel_material=panel.material,
            recommended_action=action,
            action_reason=reason,
            use_oem_parts=use_oem_parts,
            deductible=safe_deductible,
            regional_market=market.region_name,
            zip_code=zip_code,
            adjacent_blend_applied=adjacent_blend_applied,
            line_items=line_items,
            parts_subtotal=parts_cost,
            body_labor_subtotal=body_labor_subtotal,
            paint_labor_subtotal=paint_labor_subtotal,
            paint_materials_subtotal=paint_mat_subtotal,
            fees_subtotal=epa_fee,
            sales_tax=sales_tax,
            gross_estimate_total=gross_total,
            net_claim_payout=round(max(0.0, gross_total - safe_deductible), 2),
            total_labor_hours=round(body_labor_hours + paint_labor_hours, 1),
            damage_area_cm2=damage_area,
            panel_damage_percentage=pct_damaged,
        )

    @staticmethod
    def _determine_action(
        analysis: DamageAnalysisResult,
        panel_id: str,
        damage_area: float,
        pct_damaged: float,
    ) -> tuple[RecommendedAction, str]:
        if "WINDSHIELD" in panel_id:
            return (
                RecommendedAction.REPLACE_PANEL,
                "Automotive safety standards mandate complete windshield replacement for structural integrity.",
            )

        has_severe_tear = any(
            item.type == DamageType.TEAR or item.severity == Severity.SEVERE for item in analysis.items_detected
        )
        if pct_damaged >= 28.0 or damage_area >= settings.replace_threshold_area_cm2 or has_severe_tear:
            return (
                RecommendedAction.REPLACE_PANEL,
                f"Damage area ({damage_area:.1f} cm² / {pct_damaged}%) exceeds the 28% repair threshold or exhibits severe structural deformity.",
            )
        if damage_area < 5.0 and all(item.severity == Severity.MINOR for item in analysis.items_detected):
            return (
                RecommendedAction.BUFF_AND_POLISH_ONLY,
                f"Isolated superficial clear-coat marks ({damage_area:.1f} cm²). Paint buffing and spot correction recommended.",
            )
        return (
            RecommendedAction.REPAIR_AND_REFINISH,
            f"Damage is within repairable limits ({damage_area:.1f} cm² / {pct_damaged}% of panel). Panel structural integrity is intact.",
        )

    @staticmethod
    def _build_action_items(
        action: RecommendedAction,
        panel: PanelInfo,
        damage_area: float,
        use_oem_parts: bool,
        effective_body_rate: float,
        effective_paint_rate: float,
        mat_note: str,
    ) -> tuple[list[EstimateLineItem], float, float, float, float]:
        line_items: list[EstimateLineItem] = []
        parts_cost = 0.0
        body_labor_hours = 0.0
        paint_labor_hours = 0.0
        paint_materials_hours = 0.0

        if action == RecommendedAction.REPLACE_PANEL:
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
            paint_labor_hours = round(max(1.5, panel.standard_paint_refinish_hours * 0.85), 1)
            line_items.append(
                EstimateLineItem(
                    category="PAINT_LABOR",
                    description="Paint feather, prime, & clear coat blend",
                    quantity_or_hours=paint_labor_hours,
                    rate=effective_paint_rate,
                    subtotal=round(paint_labor_hours * effective_paint_rate, 2),
                )
            )
            paint_materials_hours = paint_labor_hours
        else:
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

        return line_items, parts_cost, body_labor_hours, paint_labor_hours, paint_materials_hours

    @staticmethod
    def _append_adjacent_blend_items(
        line_items: list[EstimateLineItem],
        panel: PanelInfo,
        blend_adjacent_panels: bool,
        effective_paint_rate: float,
    ) -> tuple[float, bool]:
        blend_hours_total = 0.0
        applied = False
        if not (blend_adjacent_panels and panel.standard_paint_refinish_hours > 0 and panel.adjacent_panels):
            return blend_hours_total, applied

        for adj_id in panel.adjacent_panels[:2]:
            adj_panel = PANEL_CATALOG.get(adj_id)
            if adj_panel and adj_panel.standard_paint_refinish_hours > 0:
                blend_hours = round(adj_panel.standard_paint_refinish_hours * 0.5, 1)
                blend_hours_total += blend_hours
                line_items.append(
                    EstimateLineItem(
                        category="PAINT_LABOR",
                        description=f"Adjacent blend: {adj_panel.name} (Mitchell 50% Rule)",
                        quantity_or_hours=blend_hours,
                        rate=effective_paint_rate,
                        subtotal=round(blend_hours * effective_paint_rate, 2),
                    )
                )
                applied = True
        return blend_hours_total, applied
