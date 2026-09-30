from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, KeepTogether


def _money(value: Any) -> str:
    return f"${float(value or 0):,.2f}"


def _pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def _footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 11 * mm, "Signal Chain - Recovery Analysis")
    canvas.drawRightString(192 * mm, 11 * mm, f"Page {document.page}")
    canvas.restoreState()


def _table(rows, widths, styles):
    table = Table([[Paragraph(str(cell), styles["Small"]) for cell in row] for row in rows], colWidths=widths, repeatRows=1 if len(rows) > 1 else 0)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def create_run_report(run_id: str, run: dict[str, Any]) -> Path:
    state = run["state"]
    results = run["experiment"]["results"]
    forecast = run.get("forecast") or state.get("forecast") or {}
    recommendation = forecast.get("recommendation") or {}
    verified = results.get("verified", {}).get("metrics", {})
    scenario = state.get("scenario") or {}
    imported = state.get("imported_network") or {}

    folder = Path(gettempdir()) / "signal-chain-reports"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"recovery-report-{run_id}.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=21, leading=25, textColor=colors.HexColor("#173b36")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#173b36"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name="Callout", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#166534"), backColor=colors.HexColor("#f0fdf4"), borderColor=colors.HexColor("#bbf7d0"), borderWidth=.6, borderPadding=8))

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm, title="Recovery Analysis Report")
    story = [
        Paragraph("Recovery Analysis Report", styles["ReportTitle"]),
        Paragraph(f"Run {run_id} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Small"]),
        Spacer(1, 5 * mm),
        Paragraph("Executive summary", styles["Section"]),
        Paragraph(
            f"The verified simulation completed at day {state.get('day', '-')}. Customer service averaged "
            f"{_pct(verified.get('average_service_level'))}, peak service loss was "
            f"{float(verified.get('peak_service_level_loss', 0) or 0):.3f}, and verified execution cost "
            f"{_money(verified.get('total_cost'))}. The verifier recorded {verified.get('verifier_intervention_rate', 0)} intervention(s).",
            styles["Callout"],
        ),
        Paragraph("Scenario and network", styles["Section"]),
    ]
    story.append(_table([
        ["Field", "Value"],
        ["Scenario", scenario.get("description") or "Not provided"],
        ["Affected nodes", ", ".join(scenario.get("affected_nodes") or []) or "Not specified"],
        ["Duration", f"{scenario.get('duration', '-')} days"],
        ["Severity", _pct(scenario.get("severity"))],
        ["Network status", state.get("network_status", "Not reported")],
        ["Physical imported nodes", imported.get("source_node_count", "Not reported")],
        ["Data quality", imported.get("data_quality", "Not reported")],
    ], [42 * mm, 132 * mm], styles))
    story += [Paragraph("Verifier and negotiation", styles["Section"])]
    verification = state.get("verification") or {}
    story.append(_table([
        ["Field", "Value"],
        ["Decision", verification.get("decision", "APPROVED")],
        ["Agreement valid", "Yes" if verification.get("valid", True) else "No"],
        ["Negotiation status", state.get("meta", {}).get("status", "Not reported")],
        ["Negotiation rounds", state.get("meta", {}).get("rounds", "Not reported")],
        ["Invalid executed rate", _pct(verified.get("invalid_agreement_rate"))],
    ], [60 * mm, 114 * mm], styles))
    ardn_table = _table([
        ["Field", "Value"],
        ["Recommended action", recommendation.get("label", "Not available")],
        ["Recovery prediction", f"{recommendation.get('recovery_days', '-')} +/- {recommendation.get('recovery_uncertainty_days', '-')} days"],
        ["Predicted service loss", recommendation.get("service_loss", "Not available")],
        ["Predicted cost", _money(recommendation.get("predicted_cost"))],
        ["Cost interval", " - ".join(_money(x) for x in (recommendation.get("cost_interval") or [0, 0]))],
        ["Overflow risk", recommendation.get("risk_label", "Not available")],
        ["OOD status", f"{recommendation.get('ood_score', '-')} / threshold {recommendation.get('ood_threshold', '-')}"],
        ["Execution status", recommendation.get("execution_status", "Not reported")],
    ], [52 * mm, 122 * mm], styles)
    story.append(KeepTogether([Paragraph("ARDN advisory forecast", styles["Section"]), ardn_table]))
    story += [PageBreak(), Paragraph("Strategy comparison", styles["Section"])]
    rows = [["Metric", "Rules only", "Unverified", "Verified"]]
    specs = [
        ("Recovery time", "recovery_time", lambda x: f"{x} days"),
        ("Invalid executed rate", "invalid_agreement_rate", _pct),
        ("Peak service loss", "peak_service_level_loss", lambda x: f"{float(x or 0):.3f}"),
        ("Average service", "average_service_level", _pct),
        ("Fairness variance", "fairness_variance", lambda x: f"{float(x or 0):.4f}"),
        ("Total cost", "total_cost", _money),
        ("Verifier interventions", "verifier_intervention_rate", lambda x: f"{x or 0} rounds"),
    ]
    for label, key, formatter in specs:
        rows.append([label] + [formatter(results.get(mode, {}).get("metrics", {}).get(key)) for mode in ("classical", "unverified", "verified")])
    comparison = _table(rows, [47 * mm, 42 * mm, 42 * mm, 42 * mm], styles)
    comparison.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#f0fdf4"))]))
    story.append(comparison)
    story += [Paragraph("Interpretation", styles["Section"]), Paragraph("ARDN is advisory. The verified simulator results are the ground-truth execution outcome, while ARDN values are forecasts with their own definitions and uncertainty.", styles["Small"])]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output
