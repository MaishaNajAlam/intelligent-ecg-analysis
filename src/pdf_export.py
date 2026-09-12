"""
Step 5 polish: one-click PDF export of a single ECG's analysis.

Takes the results app.py already computed for one record (the rendered
12-lead + saliency plot, the diagnosis/confidences, the generated report,
and optionally the cardiologist ground-truth report) and lays them out into
a formatted PDF, for a clinician to save/print/share.

Usage:
    python -m src.pdf_export --test
"""
import io
import os
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import config


def build_pdf_report(fig, title: str, diagnosis_label: str, confidences: dict,
                      report_text: str, ground_truth: str = None) -> str:
    """
    Args:
        fig: matplotlib Figure (the 12-lead + saliency plot, freshly rendered
             for this export -- not a Figure previously handed to Gradio).
        title: e.g. "Record 334" or "Uploaded ECG".
        diagnosis_label: e.g. "NORM — Normal ECG (67.6% confidence)".
        confidences: {class_name: probability} for every class.
        report_text: the model-generated clinical report.
        ground_truth: optional cardiologist ground-truth report text.

    Returns:
        Path to the written PDF file, under config.OUTPUT_DIR.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.OUTPUT_DIR, f"ecg_report_{safe_title}_{timestamp}.pdf")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    img_buf.seek(0)
    fig_w, fig_h = fig.get_size_inches()

    # diagnosis_label comes from app.py formatted for Gradio's Markdown component
    # (**bold**); ReportLab's Paragraph markup uses <b>bold</b> instead.
    diagnosis_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", diagnosis_label)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = [
        Paragraph("Intelligent ECG Analysis Report", styles["Title"]),
        Paragraph(title, styles["Heading2"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Diagnosis:</b> {diagnosis_html}", styles["Normal"]),
        Spacer(1, 0.1 * inch),
    ]

    table_data = [["Class", "Confidence"]] + [
        [cls, f"{prob * 100:.1f}%"]
        for cls, prob in sorted(confidences.items(), key=lambda kv: -kv[1])
    ]
    conf_table = Table(table_data, colWidths=[2.5 * inch, 1.5 * inch])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 0.2 * inch))

    img_width = 6.5 * inch
    story.append(Image(img_buf, width=img_width, height=img_width * (fig_h / fig_w)))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Generated Report:</b>", styles["Heading3"]))
    story.append(Paragraph(report_text.replace("\n", "<br/>"), styles["Normal"]))

    if ground_truth:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("<b>Cardiologist Ground Truth:</b>", styles["Heading3"]))
        story.append(Paragraph(ground_truth.replace("\n", "<br/>"), styles["Normal"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(
        "<i>This is a research/educational prototype trained on PTB-XL. "
        "It is not a certified diagnostic device and must not be used for "
        "real clinical decisions.</i>",
        styles["Normal"],
    ))

    doc.build(story)
    return out_path


def _test():
    """Quick sanity check without booting the full Gradio app."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
    ax.set_title("dummy signal")

    path = build_pdf_report(
        fig, title="Test Record",
        diagnosis_label="NORM — Normal ECG (99.0% confidence)",
        confidences={"NORM": 0.99, "MI": 0.02, "STTC": 0.01, "CD": 0.01, "HYP": 0.01},
        report_text="sinus rhythm. normal ecg.",
        ground_truth="sinus rhythm. no definite pathology.",
    )
    plt.close(fig)

    assert os.path.exists(path), f"FAIL: {path} was not written"
    size = os.path.getsize(path)
    assert size > 1000, f"FAIL: {path} looks too small ({size} bytes)"
    print(f"[OK] PDF written to {path} ({size} bytes)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the PDF export sanity check")
    args = parser.parse_args()
    if args.test:
        _test()
