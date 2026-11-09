"""
Formal Motor Vehicle Damage Appraisal Report Generator.
Generates an official insurance-standard PDF document using fpdf2.
"""
import io
from datetime import datetime
from fpdf import FPDF

from app.estimator.cost_engine import InsuranceEstimate
from app.cv_engine.base import DamageAnalysisResult


class PDFReport(FPDF):
    def header(self):
        # Header banner
        self.set_fill_color(24, 43, 73) # Navy blue
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'MOTOR VEHICLE DAMAGE APPRAISAL REPORT', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 0, 'Standard US Collision Repair & Actuarial Audit - Generated via AutoDamage CV Engine', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | AutoDamage-CV Deterministic Appraisal | Strictly Confidential', align='C')


def clean_text(s: str) -> str:
    """Sanitizes unicode characters into clean ASCII/latin-1 equivalents for standard PDF fonts."""
    if not s:
        return ""
    replacements = {
        "cm²": "cm2",
        "²": "2",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s.encode("latin-1", "replace").decode("latin-1")


def generate_estimate_pdf(estimate: InsuranceEstimate, analysis: DamageAnalysisResult) -> bytes:
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Meta Section
    pdf.set_text_color(30, 41, 59)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 7, clean_text(f'Claim Number: {estimate.claim_id}'), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(100, 5, f'Appraisal Date: {datetime.now().strftime("%B %d, %Y %I:%M %p")}')
    pdf.cell(90, 5, clean_text(f'Inspected Panel: {estimate.panel_name}'), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 5, clean_text(f'Panel Material: {estimate.panel_material}'))
    pdf.cell(90, 5, clean_text(f'Detection Engine: {analysis.engine_used}'), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 5, f'Parts Type: {"OEM Original" if estimate.use_oem_parts else "CAPA Certified Aftermarket"}')
    pdf.cell(90, 5, clean_text(f'Panel Surface Damage: {estimate.panel_damage_percentage}% ({estimate.damage_area_cm2} cm2)'), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Recommendation Box
    pdf.set_fill_color(241, 245, 249) # Slate light
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 18, 'DF')
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, clean_text(f'APPRAISER ACTION: {estimate.recommended_action.value.replace("_", " ")}'), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(182, 4, clean_text(f'Determination Reason: {estimate.action_reason}'))
    pdf.ln(8)

    # Itemized Breakdown Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(226, 232, 240)
    pdf.set_text_color(15, 23, 42)
    
    # Table Header
    col_w = [25, 95, 25, 20, 25] # Total = 190
    headers = ["Category", "Line Item Description", "Qty / Hrs", "Rate", "Amount ($)"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align='C' if i in (0, 2, 3) else 'L' if i == 1 else 'R')
    pdf.ln()

    # Table Rows
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    for item in estimate.line_items:
        pdf.cell(col_w[0], 6, clean_text(item.category), border=1, align='C')
        pdf.cell(col_w[1], 6, clean_text(f'  {item.description}'), border=1, align='L')
        pdf.cell(col_w[2], 6, f'{item.quantity_or_hours:.1f}', border=1, align='C')
        pdf.cell(col_w[3], 6, f'${item.rate:.2f}', border=1, align='R')
        pdf.cell(col_w[4], 6, f'${item.subtotal:.2f}', border=1, align='R')
        pdf.ln()

    pdf.ln(3)

    # Totals Summary
    pdf.set_font('Helvetica', '', 9)
    summary_w1, summary_w2 = 145, 45
    totals = [
        ("Parts Replacement Total:", f"${estimate.parts_subtotal:.2f}"),
        ("Body Repair Labor:", f"${estimate.body_labor_subtotal:.2f}"),
        ("Paint Refinish Labor:", f"${estimate.paint_labor_subtotal:.2f}"),
        ("Paint Supplies & Materials:", f"${estimate.paint_materials_subtotal:.2f}"),
        ("EPA / Regulatory Fees:", f"${estimate.fees_subtotal:.2f}"),
        ("Sales Tax (Estimated):", f"${estimate.sales_tax:.2f}"),
    ]

    for label, val in totals:
        pdf.cell(summary_w1, 5, label, align='R')
        pdf.cell(summary_w2, 5, val, align='R', new_x="LMARGIN", new_y="NEXT")

    pdf.set_draw_color(15, 23, 42)
    pdf.line(100, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(1)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(summary_w1, 6, "GROSS REPAIR ESTIMATE:", align='R')
    pdf.cell(summary_w2, 6, f"${estimate.gross_estimate_total:.2f}", align='R', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('Helvetica', '', 9)
    pdf.cell(summary_w1, 5, f"Less Policy Deductible:", align='R')
    pdf.cell(summary_w2, 5, f"- ${estimate.deductible:.2f}", align='R', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(16, 185, 129) # Emerald Green
    pdf.cell(summary_w1, 7, "NET INSURANCE CLAIM SETTLEMENT:", align='R')
    pdf.cell(summary_w2, 7, f"${estimate.net_claim_payout:.2f}", align='R', new_x="LMARGIN", new_y="NEXT")

    # Output to raw bytes
    return bytes(pdf.output())
