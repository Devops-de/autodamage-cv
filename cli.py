#!/usr/bin/env python3
"""
AutoDamage CV - Terminal Command Line Interface (CLI)
Allows insurance adjusters and engineers to inspect vehicle photos and generate
estimates directly from the command line.
"""
import argparse
import os
import sys
from app.cv_engine.classical_detector import ClassicalDamageDetector
from app.cv_engine.base import DetectionOptions
from app.estimator.cost_engine import EstimateCalculator
from app.reports.pdf_generator import generate_estimate_pdf
from app.estimator.parts_catalog import PANEL_CATALOG


def main():
    parser = argparse.ArgumentParser(
        description="AutoDamage CV: Vehicle Damage Inspection & Repair Cost Estimator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py inspect app/static/sample_images/sample_bumper_scratch.jpg --panel FRONT_BUMPER
  python cli.py inspect car_photo.jpg --panel HOOD --sensitivity 0.65 --pdf claim_report.pdf
  python cli.py list-panels
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: list-panels
    subparsers.add_parser("list-panels", help="List supported vehicle panels and materials")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Run CV damage inspection and calculate repair estimate")
    inspect_parser.add_argument("image_path", help="Path to vehicle damage photograph")
    inspect_parser.add_argument(
        "--panel",
        default="FRONT_BUMPER",
        choices=list(PANEL_CATALOG.keys()),
        help="Target vehicle panel (default: FRONT_BUMPER)",
    )
    inspect_parser.add_argument("--sensitivity", type=float, default=0.55, help="Detection sensitivity (0.1 - 1.0)")
    inspect_parser.add_argument("--deductible", type=float, default=500.0, help="Policy deductible in USD")
    inspect_parser.add_argument("--aftermarket", action="store_true", help="Use CAPA aftermarket parts instead of OEM")
    inspect_parser.add_argument("--zip", type=str, default=None, help="5-digit US ZIP code for local prevailing wage adjustment")
    inspect_parser.add_argument("--blend", action="store_true", help="Apply Mitchell 50% Rule for adjacent panel paint blending")
    inspect_parser.add_argument("--pdf", type=str, default=None, help="Save appraisal report to PDF file path")

    args = parser.parse_args()

    if args.command == "list-panels":
        print("\nSupported Vehicle Panels:")
        print("-" * 65)
        print(f"{'Panel ID':<25} {'Material':<25} {'Surface Area (cm2)':<15}")
        print("-" * 65)
        for p in PANEL_CATALOG.values():
            print(f"{p.panel_id:<25} {p.material:<25} {p.surface_area_cm2:<15.1f}")
        print("-" * 65)
        return

    if args.command == "inspect":
        if not os.path.exists(args.image_path):
            print(f"Error: File not found at '{args.image_path}'", file=sys.stderr)
            sys.exit(1)

        print(f"\n[+] Loading image: {args.image_path}")
        with open(args.image_path, "rb") as f:
            img_bytes = f.read()

        print(f"[+] Inspecting panel: {args.panel} (Sensitivity: {args.sensitivity})...")
        detector = ClassicalDamageDetector()
        options = DetectionOptions(
            target_panel=args.panel,
            sensitivity=args.sensitivity,
        )
        analysis = detector.detect(img_bytes, options)

        print(f"\n{'='*60}")
        print(f"  CV INSPECTION RESULTS ({analysis.engine_used})")
        print(f"{'='*60}")
        print(f"  Defects Detected   : {len(analysis.items_detected)}")
        print(f"  Total Damage Area  : {analysis.total_damaged_area_cm2:.2f} cm2")
        print(f"  Overall Severity   : {analysis.overall_severity.value}")
        print(f"{'='*60}")

        for item in analysis.items_detected:
            clean_details = item.details.replace("cm²", "cm2")
            print(f"  #{item.id} {clean_details} [Conf: {item.confidence_score*100:.0f}% | Box: {item.bounding_box.width}x{item.bounding_box.height}px]")

        # Run Actuarial Estimator
        estimate = EstimateCalculator.calculate(
            analysis=analysis,
            panel_id=args.panel,
            use_oem_parts=not args.aftermarket,
            deductible=args.deductible,
            blend_adjacent_panels=args.blend,
            zip_code=args.zip,
        )

        clean_reason = estimate.action_reason.replace("cm²", "cm2")
        print(f"\n{'='*60}")
        print(f"  INSURANCE APPRAISAL: {estimate.claim_id}")
        print(f"{'='*60}")
        print(f"  Regional Market    : {estimate.regional_market}")
        print(f"  Recommended Action : {estimate.recommended_action.value}")
        print(f"  Justification      : {clean_reason}")
        print(f"{'-'*60}")
        print(f"{'Category':<15} {'Description':<32} {'Amount ($)':>10}")
        print(f"{'-'*60}")
        for item in estimate.line_items:
            clean_desc = item.description.replace("cm²", "cm2")[:32]
            print(f"{item.category:<15} {clean_desc:<32} {item.subtotal:>10.2f}")
        print(f"{'-'*60}")
        print(f"{'Gross Estimate Total:':<48} ${estimate.gross_estimate_total:>9.2f}")
        print(f"{'Less Deductible:':<48} -${estimate.deductible:>9.2f}")
        print(f"{'Net Insurance Payout:':<48} ${estimate.net_claim_payout:>9.2f}")
        print(f"{'='*60}\n")

        if args.pdf:
            pdf_bytes = generate_estimate_pdf(estimate, analysis)
            with open(args.pdf, "wb") as pf:
                pf.write(pdf_bytes)
            print(f"[OK] Official PDF Appraisal Report saved to: {args.pdf}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
