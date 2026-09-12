"""
Step 5 polish: one-click PDF export of a single ECG's analysis, styled like
the auto-interpreted report a real 12-lead ECG cart prints (letterhead,
record/status header, boxed primary finding, signature line) rather than a
generic app export -- so a clinician can review, sign, and file it.

Takes the results app.py already computed for one record (the rendered
12-lead + saliency plot, the diagnosis/confidences, the generated report,
and optionally the cardiologist ground-truth report).

Usage:
    python -m src.pdf_export --test
"""
import io
import os
import re
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src import config

NAVY = colors.HexColor(config.BRAND_NAVY)
ACCENT = colors.HexColor(config.BRAND_ACCENT)
PAPER = colors.HexColor(config.BRAND_PAPER)
INK = colors.HexColor(config.BRAND_INK)
MUTED = colors.HexColor(config.BRAND_MUTED)
LINE = colors.HexColor(config.BRAND_LINE)

PAGE_W, PAGE_H = letter
MARGIN = 0.65 * inch
HEADER_H = 0.62 * inch
FOOTER_H = 0.45 * inch

_styles = getSampleStyleSheet()
_STYLE_LABEL = ParagraphStyle(
    "SectionLabel", parent=_styles["Normal"], fontName="Helvetica-Bold", fontSize=9,
    textColor=NAVY, spaceBefore=12, spaceAfter=4, leading=11,
)
_STYLE_BODY = ParagraphStyle(
    "Body", parent=_styles["Normal"], fontName="Helvetica", fontSize=10, textColor=INK, leading=14,
)
_STYLE_MUTED = ParagraphStyle(
    "Muted", parent=_styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=MUTED, leading=12,
)
_STYLE_FINDING = ParagraphStyle(
    "Finding", parent=_styles["Normal"], fontName="Times-Bold", fontSize=17, textColor=NAVY, leading=20,
)
_STYLE_FINDING_SUB = ParagraphStyle(
    "FindingSub", parent=_styles["Normal"], fontName="Helvetica", fontSize=10, textColor=INK, leading=13,
)
_STYLE_DISCLAIMER = ParagraphStyle(
    "Disclaimer", parent=_styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=INK, leading=11,
)
_STYLE_DISCLAIMER_BODY = ParagraphStyle(
    "DisclaimerBody", parent=_styles["Normal"], fontName="Helvetica", fontSize=8, textColor=INK, leading=11,
)


def _section_label(text: str):
    return [
        Paragraph(text.upper(), _STYLE_LABEL),
        HRFlowable(width="100%", thickness=0.75, color=NAVY, spaceAfter=6),
    ]


def _hr(color=LINE, thickness=0.5, space_after=8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=space_after)


def _na(value):
    """True for missing/NaN metadata -- PTB-XL leaves most optional clinical
    fields (heart_axis, pacemaker, etc.) blank for the large majority of records."""
    try:
        return value is None or pd.isna(value)
    except (TypeError, ValueError):
        return False


def _fmt_age(age):
    if _na(age):
        return "Not recorded"
    # PTB-XL caps recorded age at 300 to de-identify patients over 89
    # (documented dataset convention, not a data error).
    if age >= 200:
        return "Over 89 (de-identified per dataset policy)"
    return f"{int(age)} years"


def _fmt_sex(sex):
    if _na(sex):
        return "Not recorded"
    return "Female" if int(sex) == 1 else "Male"  # PTB-XL convention: 0=male, 1=female


def _fmt_measure(value, unit):
    if _na(value):
        return "Not recorded"
    return f"{value:g} {unit}"


def _fmt_id(value):
    if _na(value):
        return "Not recorded"
    # pandas upcasts an int column to float once any row has NaN, so a
    # patient_id often arrives as e.g. 2560.0 -- strip the trailing ".0".
    return str(int(value)) if float(value).is_integer() else str(value)


def _fmt_text(value):
    if _na(value):
        return "Not recorded"
    # PTB-XL's device field sometimes has irregular internal whitespace
    # (e.g. "AT-6    6") -- collapse it for a clean-looking report.
    return " ".join(str(value).split())


def _patient_recording_table(patient_meta: dict):
    """Builds the PATIENT / RECORDING INFORMATION block. Returns a list of
    flowables, or [] if no metadata was supplied (e.g. an uploaded CSV has
    no accompanying patient record)."""
    if not patient_meta:
        return []

    def cell(label, value):
        return Paragraph(f"<b>{label}:</b> {value}", _STYLE_BODY)

    rows = [
        [cell("Patient ID", _fmt_id(patient_meta.get("patient_id"))),
         cell("Age", _fmt_age(patient_meta.get("age"))),
         cell("Sex", _fmt_sex(patient_meta.get("sex")))],
        [cell("Height", _fmt_measure(patient_meta.get("height"), "cm")),
         cell("Weight", _fmt_measure(patient_meta.get("weight"), "kg")),
         cell("Device", _fmt_text(patient_meta.get("device")))],
    ]

    notes = []
    if not _na(patient_meta.get("recording_date")):
        notes.append(f"Recorded {patient_meta['recording_date']}")
    for label, key in [("Heart axis", "heart_axis"), ("Pacemaker", "pacemaker"),
                        ("Extra beats", "extra_beats"), ("Infarction stadium", "infarction_stadium1")]:
        if not _na(patient_meta.get(key)):
            notes.append(f"{label}: {patient_meta[key]}")
    if notes:
        rows.append([Paragraph(f"<b>Notes:</b> {'; '.join(notes)}", _STYLE_BODY), "", ""])

    table = Table(rows, colWidths=[2.23 * inch, 2.23 * inch, 2.24 * inch])
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 0)]
    if notes:
        style.append(("SPAN", (0, 2), (-1, 2)))
    table.setStyle(TableStyle(style))

    return _section_label("Patient & Recording Information") + [table]


def _draw_page_frame(canvas, doc):
    """Letterhead header bar + footer, drawn on every page."""
    canvas.saveState()

    # ── Header: navy letterhead bar ──
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - HEADER_H - 3, PAGE_W, 3, fill=1, stroke=0)

    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Bold", 13)
    canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, "CARDIOSAGE")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, PAGE_H - 0.47 * inch, "AI-Powered ECG Analysis & Clinical Reporting")

    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.32 * inch, "ECG DIAGNOSTIC REPORT")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.47 * inch, doc._report_id)

    # ── Footer: hairline + generation stamp + page number ──
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, FOOTER_H, PAGE_W - MARGIN, FOOTER_H)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, FOOTER_H - 12, f"Generated {doc._generated_at}  ·  Not a certified diagnostic device")
    canvas.drawRightString(PAGE_W - MARGIN, FOOTER_H - 12, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


def build_pdf_report(fig, title: str, diagnosis_label: str, confidences: dict,
                      report_text: str, ground_truth: str = None, patient_meta: dict = None) -> str:
    """
    Args:
        fig: matplotlib Figure (the 12-lead + saliency plot, freshly rendered
             for this export -- not a Figure previously handed to Gradio).
        title: e.g. "Record 334" or "Uploaded ECG".
        diagnosis_label: e.g. "**NORM** — Normal ECG (67.6% confidence)"
                         (Gradio-markdown bold syntax; converted to <b> here).
        confidences: {class_name: probability} for every class.
        report_text: the model-generated clinical report.
        ground_truth: optional cardiologist ground-truth report text.
        patient_meta: optional dict of PTB-XL fields for this record
                       (patient_id, age, sex, height, weight, device,
                       recording_date, heart_axis, pacemaker, extra_beats,
                       infarction_stadium1) -- omitted entirely (no section
                       rendered) when None, e.g. for an uploaded CSV that has
                       no accompanying patient record.

    Returns:
        Path to the written PDF file, under config.OUTPUT_DIR.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_id = f"ECG-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"
    out_path = os.path.join(config.OUTPUT_DIR, f"ecg_report_{safe_title}_{timestamp}.pdf")

    # diagnosis_label carries **bold** for Gradio's Markdown widget; split it
    # into the bare class name (headline) and the rest (sub-line) instead of
    # just swapping markup, since the PDF gives the finding its own type scale.
    # The confidence percentage is stripped back out of the sub-line since it
    # gets its own bolded line right below.
    top_class = max(confidences, key=confidences.get)
    top_conf = confidences[top_class]
    diagnosis_desc = diagnosis_label.split("—", 1)[1].strip() if "—" in diagnosis_label else diagnosis_label
    diagnosis_desc = re.sub(r"\s*\([\d.]+%\s*confidence\)\s*$", "", diagnosis_desc)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    img_buf.seek(0)
    fig_w, fig_h = fig.get_size_inches()

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        topMargin=HEADER_H + 0.25 * inch, bottomMargin=FOOTER_H + 0.2 * inch,
        leftMargin=MARGIN, rightMargin=MARGIN,
    )
    doc._report_id = report_id
    doc._generated_at = now.strftime("%Y-%m-%d %H:%M")

    story = []

    # ── Record metadata strip ──
    meta_table = Table(
        [[Paragraph(f"<b>Record:</b> {title}", _STYLE_BODY),
          Paragraph(f"<b>Report Date:</b> {now.strftime('%Y-%m-%d %H:%M')}", _STYLE_BODY),
          Paragraph(f"<b>Report ID:</b> {report_id}", _STYLE_BODY)]],
        colWidths=[2.4 * inch, 2.4 * inch, 2.1 * inch],
    )
    meta_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(meta_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Status:</b> Unconfirmed — Pending Physician Review", _STYLE_DISCLAIMER))
    story.append(_hr(space_after=8))

    # ── Patient & recording information (omitted if no metadata supplied) ──
    story += _patient_recording_table(patient_meta)

    # ── Primary finding, boxed ──
    finding_inner = Table(
        [[Paragraph(top_class, _STYLE_FINDING)],
         [Paragraph(diagnosis_desc, _STYLE_FINDING_SUB)],
         [Paragraph(f"Model confidence: <b>{top_conf * 100:.1f}%</b>", _STYLE_FINDING_SUB)]],
        colWidths=[6.7 * inch],
    )
    finding_inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 1.2, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (-1, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (0, -1), 2),
        ("BOTTOMPADDING", (0, 1), (0, -2), 2),
    ]))
    story += _section_label("Primary Finding")
    story.append(finding_inner)

    # ── Confidence breakdown table ──
    story += _section_label("Differential — Confidence by Class")
    header = ["Class", "Confidence"]
    rows = [header] + [
        [cls, f"{prob * 100:.1f}%"] for cls, prob in sorted(confidences.items(), key=lambda kv: -kv[1])
    ]
    conf_table = Table(rows, colWidths=[5.2 * inch, 1.5 * inch])
    row_styles = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            row_styles.append(("BACKGROUND", (0, i), (-1, i), PAPER))
    conf_table.setStyle(TableStyle(row_styles))
    story.append(conf_table)

    # ── Generated clinical impression ──
    story += _section_label("Clinical Impression (AI-Generated)")
    impression_box = Table([[Paragraph(report_text.replace("\n", "<br/>"), _STYLE_BODY)]],
                            colWidths=[6.7 * inch])
    impression_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(impression_box)

    # ── Disclaimer box ──
    story.append(Spacer(1, 10))
    disclaimer_cell = Table(
        [[Paragraph("IMPORTANT", _STYLE_DISCLAIMER)],
         [Paragraph(
             "This report was generated by a research/educational prototype trained on the "
             "PTB-XL dataset. It is NOT a certified diagnostic device and must not be used for "
             "real clinical decision-making. All findings require review and confirmation by a "
             "licensed physician.", _STYLE_DISCLAIMER_BODY)]],
        colWidths=[6.7 * inch],
    )
    disclaimer_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(disclaimer_cell)

    # ── Signature block ──
    story.append(Spacer(1, 14))
    sig_table = Table(
        [["Physician Signature: " + "_" * 32, "Date: " + "_" * 16]],
        colWidths=[4.4 * inch, 2.3 * inch],
    )
    sig_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)

    # ── Page 2: Dedicated Full 12-Lead ECG Tracing ──
    story.append(PageBreak())
    story += _section_label("12-Lead ECG Tracing & Saliency Waveform Analysis")
    story.append(Paragraph(
        "Standard 10-second 12-lead electrocardiogram (500 Hz). "
        "Red highlighted regions indicate saliency areas influencing the AI classification.",
        _STYLE_MUTED))
    story.append(Spacer(1, 6))
    img_width = 6.8 * inch
    img_height = min(8.0 * inch, img_width * (fig_h / fig_w))
    story.append(Image(img_buf, width=img_width, height=img_height))

    doc.build(story, onFirstPage=_draw_page_frame, onLaterPages=_draw_page_frame)
    return out_path


def render_preview_image(pdf_path: str, dpi: int = 140) -> str:
    """Rasterizes all pages of the built PDF into a continuous high-res PNG preview
    so the user sees the entire clinical document and waveform tracings."""
    import pymupdf
    from PIL import Image as PILImage

    doc = pymupdf.open(pdf_path)
    page_images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)
    doc.close()

    if not page_images:
        return ""

    if len(page_images) == 1:
        combined = page_images[0]
    else:
        pad = 20
        total_w = max(img.width for img in page_images)
        total_h = sum(img.height for img in page_images) + pad * (len(page_images) - 1)
        combined = PILImage.new("RGB", (total_w, total_h), color=(228, 233, 228))
        y = 0
        for img in page_images:
            combined.paste(img, ((total_w - img.width) // 2, y))
            y += img.height + pad

    png_path = os.path.splitext(pdf_path)[0] + "_preview.png"
    combined.save(png_path)
    return png_path


def _test():
    """Quick sanity check without booting the full Gradio app."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
    ax.set_title("dummy signal")

    path = build_pdf_report(
        fig, title="Test Record",
        diagnosis_label="**NORM** — Normal ECG (99.0% confidence)",
        confidences={"NORM": 0.99, "MI": 0.02, "STTC": 0.01, "CD": 0.01, "HYP": 0.01},
        report_text="sinus rhythm. normal ecg.",
        ground_truth="sinus rhythm. no definite pathology.",
        patient_meta={
            "patient_id": 2065, "age": 300, "sex": 1, "height": 160.0, "weight": float("nan"),
            "device": "AT-6 C 5.5", "recording_date": "1986-09-17 12:27:51",
            "heart_axis": float("nan"), "pacemaker": float("nan"),
            "extra_beats": float("nan"), "infarction_stadium1": float("nan"),
        },
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
