"""
Step 5: Gradio web interface.

Wraps the classifier and report generator in an interactive app:
  - Dropdown to pick a test-set ECG (or upload your own .dat/.hea/.csv)
  - 12-lead signal plot
  - Predicted diagnosis + confidence gauge
  - Generated clinical report
  - Side-by-side comparison against the ground-truth cardiologist report

Usage:
    python -m src.app
Then open the printed local URL (and/or the public gradio.live link).
"""
import os
import re
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from transformers import BartTokenizerFast

from src import config
from src.data import load_full_dataset, load_raw_signal, plot_12_lead
from src.encoder import ECGEncoder
from src.classifier import ClassifierHead, predict_single
from src.report_generator import ECGReportModel, generate_single, load_report_model
from src.datasets import ECGFeatureReportDataset
from src.saliency import compute_saliency
from src.pdf_export import build_pdf_report, render_preview_image

import torch

# Full names shown alongside the abbreviation in the diagnosis line.
LABEL_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction (Heart Attack)",
    "STTC": "ST/T Change (Ischemia or strain)",
    "CD": "Conduction Disturbance (e.g. bundle branch block)",
    "HYP": "Hypertrophy (thickened heart muscle)",
}

# Reuses the exact keyword list the training-time language filter uses
# (src/datasets.py) so "is this German?" means the same thing everywhere.
_GERMAN_PATTERN = re.compile(
    "|".join(ECGFeatureReportDataset._GERMAN_INDICATOR_WORDS), re.IGNORECASE
)


def _looks_german(text):
    return bool(_GERMAN_PATTERN.search(text))


# ---------------------------------------------------------------------
# Look & feel -- a sage/forest-green design system (separate from the PDF's
# navy brand, per explicit choice: PDF stays navy, web app goes green).
#
# Every custom CSS selector below was verified against Gradio's actual
# compiled frontend before use (grepped the installed package's shipped
# assets), not guessed:
#   - `.primary` is the real Button component class for variant="primary"
#     (found in Index-CcsdfK63.css: `.primary.svelte-xzq5jh{...}`).
#   - Tabs use `.selected` driven by the theme's --color-accent variable
#     (found in Index-D-h_yCjW.css) -- so setting color_accent in the theme
#     already colors the selected tab correctly; no tab CSS needed at all.
#   - Everything else (eyebrow labels, feature cards, badges) is rendered
#     inside gr.HTML blocks under our own `.ecg-*` classes we fully own,
#     with zero dependency on guessing Gradio's internal DOM structure.
# ---------------------------------------------------------------------
_SAGE_BG     = "#EAF3E6"   # page background
_FOREST      = "#16241C"   # headings / dark text
_SAGE_MUTED  = "#5F7A64"   # icons, eyebrow labels, secondary text
_ACCENT      = "#8EEB6E"   # pill-button / accent green
_ACCENT_HOVER = "#7BDD59"
_CREAM       = "#FCFEFA"   # card surfaces
_LINE        = "#D7E6D2"   # hairline borders
_INK         = "#20301F"   # body text
_INK_ON_ACCENT = "#16240E"  # text on the bright accent buttons

_THEME = gr.themes.Soft(
    primary_hue="green", secondary_hue="stone", neutral_hue="stone", radius_size="lg",
).set(
    body_background_fill=_SAGE_BG,
    background_fill_primary=_CREAM,
    background_fill_secondary=_SAGE_BG,
    block_background_fill=_CREAM,
    block_border_width="1px",
    block_border_color=_LINE,
    block_radius="18px",
    block_shadow="0 2px 10px rgba(22, 36, 28, 0.08)",
    block_label_text_color=_SAGE_MUTED,
    body_text_color=_INK,
    body_text_color_subdued=_SAGE_MUTED,
    color_accent=_ACCENT,
    color_accent_soft="#DFF6D6",
    button_primary_background_fill=_ACCENT,
    button_primary_background_fill_hover=_ACCENT_HOVER,
    button_primary_text_color=_INK_ON_ACCENT,
    button_primary_border_color=_ACCENT,
)

# `.primary` verified above; everything else here targets only our own
# `.ecg-*` classes. Lora is scoped to those classes too (via its own
# font-family declaration) so the rest of the UI keeps its normal sans font
# even if the Google Font fails to load offline -- Georgia/serif fallback.
_CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@500;600;700&display=swap');

body, .gradio-container {{ font-family: 'Inter', system-ui, sans-serif !important; }}

.primary {{ border-radius: 999px !important; font-weight: 600 !important; }}

/* ── Hero brand block ── */
.ecg-hero {{
    padding: 36px 0 28px 0;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0;
}}

.ecg-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: {_LINE};
    color: {_SAGE_MUTED};
    border-radius: 999px;
    padding: 5px 14px 5px 10px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 18px;
}}
.ecg-badge svg {{ width: 14px; height: 14px; flex-shrink: 0; }}

.ecg-wordmark {{
    font-family: 'Lora', Georgia, serif;
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1.15;
    padding-bottom: 0.1em;
    margin: 0 0 5px 0;
    background: linear-gradient(120deg, {_FOREST} 0%, #2E6645 60%, {_SAGE_MUTED} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}}

.ecg-tagline {{
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: {_SAGE_MUTED};
    margin: 0 0 6px 0;
    line-height: 1.55;
    max-width: 580px;
}}

.ecg-tagline strong {{
    color: {_FOREST};
    font-weight: 600;
}}

.ecg-tagline-sub {{
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.88rem;
    font-weight: 400;
    color: {_SAGE_MUTED};
    margin: 0 0 22px 0;
    line-height: 1.5;
    max-width: 500px;
    opacity: 0.8;
}}


.ecg-divider {{
    width: 100%;
    height: 1px;
    background: {_LINE};
    margin: 4px 0 26px 0;
}}

/* ── Section eyebrow labels ── */
.ecg-eyebrow {{
    display: flex; align-items: center; gap: 8px;
    color: {_SAGE_MUTED}; font-size: 0.73rem; letter-spacing: 0.13em;
    text-transform: uppercase; font-weight: 600; margin: 18px 0 6px 0;
}}
.ecg-eyebrow svg {{ width: 15px; height: 15px; flex-shrink: 0; }}

/* ── Feature capability cards ── */
.ecg-features {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px; margin: 0 0 8px 0;
}}
.ecg-feature-card {{
    background: {_CREAM}; border: 1px solid {_LINE}; border-radius: 16px;
    padding: 22px 20px 20px 20px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}}
.ecg-feature-card:hover {{
    box-shadow: 0 6px 24px rgba(22, 36, 28, 0.10);
    transform: translateY(-2px);
}}
.ecg-feature-card svg {{ width: 30px; height: 30px; color: #2E6645; margin-bottom: 14px; }}
.ecg-feature-card h3 {{
    font-family: 'Lora', Georgia, serif; color: {_FOREST};
    font-size: 0.98rem; font-weight: 600; margin: 0 0 7px 0;
}}
.ecg-feature-card p {{ color: {_SAGE_MUTED}; font-size: 0.83rem; line-height: 1.5; margin: 0; }}

.ecg-pdf-drawer {{
    max-height: 0px !important;
    opacity: 0 !important;
    transform: translateY(-14px) scale(0.98);
    overflow: hidden !important;
    transition: max-height 0.45s cubic-bezier(0.16, 1, 0.3, 1),
                opacity 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.4s cubic-bezier(0.16, 1, 0.3, 1),
                margin 0.35s ease !important;
    pointer-events: none;
    margin-top: 0px !important;
}}

.ecg-pdf-drawer.is-open {{
    max-height: 8000px !important;
    opacity: 1 !important;
    transform: translateY(0px) scale(1) !important;
    pointer-events: auto !important;
    margin-top: 14px !important;
}}

/* Ensure PDF preview image container displays cleanly aligned at top */
.ecg-pdf-drawer .image-frame {{
    align-items: flex-start !important;
    justify-content: center !important;
    height: auto !important;
    min-height: 0 !important;
    background: transparent !important;
}}

.ecg-pdf-drawer .image-container {{
    height: auto !important;
    min-height: 0 !important;
    background: transparent !important;
}}

.ecg-pdf-drawer .image-container > button {{
    display: block !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    cursor: default !important;
    width: 100% !important;
    height: auto !important;
}}

.ecg-pdf-drawer .block {{
    height: auto !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

.ecg-pdf-drawer img {{
    object-position: top center !important;
    vertical-align: top !important;
    width: 100% !important;
    max-width: 880px !important;
    margin: 0 auto !important;
    height: auto !important;
    max-height: none !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
    border-radius: 8px !important;
}}

/* Hide floating toolbar action buttons (download, share icons) on images/plots */
button.icon-button,
.image-container button.icon-button,
.image-frame button.icon-button,
.image-preview button.icon-button,
.floating-button,
button.download,
button.share,
button[aria-label="Download"],
button[aria-label="Download image"],
button[aria-label="Share"],
button[aria-label="Share image"],
button[title="Download"],
button[title="Download image"],
button[title="Share"],
button[title="Share image"] {{
    display: none !important;
}}

/* Ensure Download PDF button stays strictly in one line without wrapping */
.ecg-pdf-download-btn {{
    white-space: nowrap !important;
    word-break: keep-all !important;
    min-width: 140px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
}}

.ecg-pdf-bar {{
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: nowrap !important;
    gap: 16px !important;
    margin-bottom: 8px !important;
}}
"""

# Small stroke-style line icons (no photos/animation), matching the
# reference design's minimalist icon language. `currentColor` so each
# picks up the CSS `color` of its container.
_ICON_HEART = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 20s-7-4.35-9.5-8.5C.9 8.1 2.3 4.8 5.6 4.1c2-.4 3.9.6 4.9 2.2 '
    '1-1.6 2.9-2.6 4.9-2.2 3.3.7 4.7 4 3.1 7.4C19 15.65 12 20 12 20z"/></svg>'
)
_ICON_PULSE_CIRCLE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M6 12h2.5l1.5-4 2 8 1.5-5 1 1H18"/></svg>'
)
_ICON_SALIENCY = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 13h3l1.5-3 2 6 1.5-4 1 2h2"/><circle cx="15.5" cy="14.5" r="4.5"/>'
    '<path d="M19 18l2.5 2.5"/></svg>'
)
_ICON_REPORT = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>'
    '<path d="M14 3v4h4"/><path d="M8.5 13.5l1.8 1.8L15 11.5"/></svg>'
)


def _eyebrow(icon_svg, text):
    return f'<div class="ecg-eyebrow">{icon_svg}{text}</div>'


print("Loading models (this happens once at startup)...")
_data = load_full_dataset()
_test_df = _data["test"]
_encoder = ECGEncoder()

_classifier = ClassifierHead(in_dim=_encoder.feature_dim).to(config.DEVICE)
_clf_ckpt = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")
if os.path.exists(_clf_ckpt):
    _classifier.load_state_dict(torch.load(_clf_ckpt, map_location=config.DEVICE))
_classifier.eval()

_tokenizer = BartTokenizerFast.from_pretrained(config.BART_MODEL_ID)
_report_ckpt = os.path.join(config.CHECKPOINT_DIR, "report_generator.pt")
if os.path.exists(_report_ckpt):
    _report_model = load_report_model(_encoder.feature_dim, _report_ckpt)
else:
    _report_model = ECGReportModel(encoder_dim=_encoder.feature_dim).to(config.DEVICE)
_report_model.eval()

_models_trained = os.path.exists(_clf_ckpt) and os.path.exists(_report_ckpt)
if not _models_trained:
    print("\n" + "=" * 60)
    print("  WARNING: no trained checkpoint(s) found.")
    print(f"  classifier_head.pt found : {os.path.exists(_clf_ckpt)}")
    print(f"  report_generator.pt found: {os.path.exists(_report_ckpt)}")
    print("  Predictions and reports below will be RANDOM/untrained until")
    print("  you train the models (see README) and re-launch the app.")
    print("=" * 60 + "\n")

# Curated local demo set (strat_fold==10, English-language ground-truth reports,
# 3 per superclass except MI: NORM/STTC/CD/HYP) — the only test-set records whose
# .dat/.hea waveform files are actually present under data/ptbxl/records500/ locally.
# Avoids requiring the full ~1.7GB PTB-XL download just to run the app; only used
# for the "Test-set record" dropdown, not for training (training happened on the
# full dataset on Kaggle). "Upload your own" still works with any ECG regardless.
# NOTE: MI record 430 was dropped (its .dat download hung/failed and was skipped
# rather than retried), leaving MI at 2 records instead of 3.
_record_choices = [
    "334", "417", "440",   # NORM
    "765", "947",          # MI
    "427", "499", "1009",  # STTC
    "618", "950", "2146",  # CD
    "1219", "1522", "2119",  # HYP
]


def _plot_to_image(signal, title, saliency=None):
    """12-lead plot. If `saliency` ((T, n_leads) array) is given, overlay a
    per-lead red heatmap behind each trace showing which parts of the signal
    most influenced the predicted diagnosis (see src/saliency.py)."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(9, 11), sharex=True)

    sal_norm = None
    if saliency is not None:
        lead_max = saliency.max(axis=0, keepdims=True)
        lead_max = np.where(lead_max < 1e-12, 1.0, lead_max)
        sal_norm = saliency / lead_max  # per-lead normalize to [0, 1]

    for i, ax in enumerate(axes):
        ax.plot(signal[:, i], linewidth=0.7, color="#1a1a1a", zorder=2)
        if sal_norm is not None:
            ymin, ymax = ax.get_ylim()
            strip = sal_norm[:, i][np.newaxis, :]
            ax.imshow(strip, aspect="auto", cmap="Reds", alpha=0.5, vmin=0, vmax=1,
                      extent=[0, signal.shape[0], ymin, ymax], zorder=1)
            ax.set_ylim(ymin, ymax)  # imshow can shift ylim; pin it back
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=18, fontsize=8)
        ax.set_yticks([])
    axes[-1].set_xlabel("Samples")
    fig.suptitle(title)

    if sal_norm is not None:
        fig.text(0.5, 0.005,
                  "Red = regions of the signal that most influenced the predicted diagnosis "
                  "(gradient-based saliency; darker = higher influence).",
                  ha="center", fontsize=8, style="italic")
        fig.tight_layout(rect=[0, 0.02, 1, 1])
    else:
        fig.tight_layout()
    return fig


_PATIENT_META_FIELDS = [
    "patient_id", "age", "sex", "height", "weight", "device", "recording_date",
    "heart_axis", "pacemaker", "extra_beats", "infarction_stadium1",
]


def _extract_patient_meta(row):
    """Pulls the PTB-XL metadata fields relevant to a clinical-style PDF
    export out of a test-set record's row (see src/pdf_export.py for how
    they're rendered/formatted, including de-identified-age handling)."""
    return {field: row.get(field) for field in _PATIENT_META_FIELDS}


def toggle_pdf_preview(export_data, is_open):
    """Toggles the PDF preview and download drawer instantaneously with a smooth CSS transition."""
    if not is_open:
        if not export_data:
            return (
                gr.update(elem_classes=["ecg-pdf-drawer"]),
                gr.update(),
                gr.update(),
                gr.update(value="Preview PDF"),
                False,
            )
        pdf_path = export_data.get("pdf_path")
        preview_path = export_data.get("preview_path")
        if not pdf_path or not os.path.exists(pdf_path):
            fig = _plot_to_image(export_data["signal"], title=export_data["title"],
                                  saliency=export_data.get("saliency"))
            pdf_path = build_pdf_report(
                fig, title=export_data["title"], diagnosis_label=export_data["label_str"],
                confidences=export_data["confidences"], report_text=export_data["report_text"],
                ground_truth=None, patient_meta=export_data.get("patient_meta"),
            )
            plt.close(fig)
            preview_path = render_preview_image(pdf_path)
            export_data["pdf_path"] = pdf_path
            export_data["preview_path"] = preview_path

        return (
            gr.update(elem_classes=["ecg-pdf-drawer", "is-open"]),
            gr.update(value=preview_path),
            gr.update(value=pdf_path),
            gr.update(value="Hide PDF Preview"),
            True,
        )
    else:
        return (
            gr.update(elem_classes=["ecg-pdf-drawer"]),
            gr.update(),
            gr.update(),
            gr.update(value="Preview PDF"),
            False,
        )


def analyze_record(ecg_id_str):
    ecg_id = int(ecg_id_str)
    row = _test_df.loc[ecg_id]
    signal, _ = load_raw_signal(row)

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    top_desc = LABEL_DESCRIPTIONS.get(top_label, "")
    label_str = f"**{top_label}**: {top_desc} ({top_conf*100:.1f}% confidence)"
    diagnosis_md = f"### {top_label}: {top_desc}\n**Confidence:** {top_conf*100:.1f}%"

    saliency_result = compute_saliency(signal, encoder=_encoder, classifier=_classifier,
                                        target_class=top_label)
    fig = _plot_to_image(signal, title=f"Record {ecg_id}", saliency=saliency_result["saliency"])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)

    # Pre-generate PDF & preview during analysis so toggle button opens instantaneously
    patient_meta = _extract_patient_meta(row)
    pdf_path = build_pdf_report(
        fig, title=f"Record {ecg_id}", diagnosis_label=label_str,
        confidences=confidences, report_text=generated_report,
        ground_truth=None, patient_meta=patient_meta,
    )
    preview_path = render_preview_image(pdf_path)

    export_data = {
        "signal": signal,
        "saliency": saliency_result["saliency"],
        "title": f"Record {ecg_id}",
        "label_str": label_str,
        "confidences": confidences,
        "report_text": generated_report,
        "patient_meta": patient_meta,
        "pdf_path": pdf_path,
        "preview_path": preview_path,
    }

    return (
        fig, diagnosis_md, confidences, generated_report,
        export_data, gr.update(elem_classes=["ecg-pdf-drawer"]), gr.update(value="Preview PDF"), False
    )


def analyze_uploaded_csv(file_obj):
    """Optional path: accept a plain CSV of shape (T, 12) for a custom signal."""
    if file_obj is None:
        return (
            None, "No file uploaded.", None, "",
            None, gr.update(elem_classes=["ecg-pdf-drawer"]), gr.update(value="Preview PDF"), False
        )
    signal = np.loadtxt(file_obj.name, delimiter=",")
    if signal.shape[1] != config.N_LEADS:
        return (
            None, f"Expected {config.N_LEADS} columns (leads), got {signal.shape[1]}.", None, "",
            None, gr.update(elem_classes=["ecg-pdf-drawer"]), gr.update(value="Preview PDF"), False
        )

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    top_desc = LABEL_DESCRIPTIONS.get(top_label, "")
    label_str = f"**{top_label}**: {top_desc} ({top_conf*100:.1f}% confidence)"
    diagnosis_md = f"### {top_label}: {top_desc}\n**Confidence:** {top_conf*100:.1f}%"

    saliency_result = compute_saliency(signal, encoder=_encoder, classifier=_classifier,
                                        target_class=top_label)
    fig = _plot_to_image(signal, title="Uploaded ECG", saliency=saliency_result["saliency"])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)

    pdf_path = build_pdf_report(
        fig, title="Uploaded ECG", diagnosis_label=label_str,
        confidences=confidences, report_text=generated_report,
        ground_truth=None, patient_meta=None,
    )
    preview_path = render_preview_image(pdf_path)

    export_data = {
        "signal": signal,
        "saliency": saliency_result["saliency"],
        "title": "Uploaded ECG",
        "label_str": label_str,
        "confidences": confidences,
        "report_text": generated_report,
        "patient_meta": None,
        "pdf_path": pdf_path,
        "preview_path": preview_path,
    }

    return (
        fig, diagnosis_md, confidences, generated_report,
        export_data, gr.update(elem_classes=["ecg-pdf-drawer"]), gr.update(value="Preview PDF"), False
    )


def analyze_batch(record_ids, files):
    """Runs the existing single-record pipeline over several records/files at
    once and returns a summary table. No saliency map here (a heatmap image
    per row doesn't fit a table) -- see the other two tabs for the per-record
    visual explanation."""
    rows = []

    for rid in (record_ids or []):
        try:
            row = _test_df.loc[int(rid)]
            signal, _ = load_raw_signal(row)
            confidences = predict_single(signal, encoder=_encoder, model=_classifier)
            top_label = max(confidences, key=confidences.get)
            report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
            rows.append({"Source": f"Record {rid}", "Diagnosis": top_label,
                         "Confidence": f"{confidences[top_label]*100:.1f}%", "Report": report})
        except Exception as e:
            rows.append({"Source": f"Record {rid}", "Diagnosis": "ERROR", "Confidence": "", "Report": str(e)})

    for f in (files or []):
        source = os.path.basename(f.name)
        try:
            signal = np.loadtxt(f.name, delimiter=",")
            if signal.shape[1] != config.N_LEADS:
                raise ValueError(f"Expected {config.N_LEADS} columns (leads), got {signal.shape[1]}")
            confidences = predict_single(signal, encoder=_encoder, model=_classifier)
            top_label = max(confidences, key=confidences.get)
            report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
            rows.append({"Source": source, "Diagnosis": top_label,
                         "Confidence": f"{confidences[top_label]*100:.1f}%", "Report": report})
        except Exception as e:
            rows.append({"Source": source, "Diagnosis": "ERROR", "Confidence": "", "Report": str(e)})

    if not rows:
        return pd.DataFrame(columns=["Source", "Diagnosis", "Confidence", "Report"])
    return pd.DataFrame(rows)


with gr.Blocks(title="CardioSage - AI-Powered ECG Analysis & Clinical Reporting") as demo:
    gr.HTML(
        '<div class="ecg-hero">'
        '<div class="ecg-badge">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 20s-7-4.35-9.5-8.5C.9 8.1 2.3 4.8 5.6 4.1c2-.4 3.9.6 4.9 2.2 1-1.6 2.9-2.6 4.9-2.2 3.3.7 4.7 4 3.1 7.4C19 15.65 12 20 12 20z"/>'
        '</svg>'
        'AI-Powered Cardiac Intelligence'
        '</div>'
        '<div class="ecg-wordmark">CardioSage</div>'
        '<p class="ecg-tagline">AI-Powered ECG Analysis & Clinical Reporting</p>'
        '<div class="ecg-divider"></div>'
        '<div class="ecg-features">'
        f'<div class="ecg-feature-card">{_ICON_PULSE_CIRCLE}<h3>AI-Powered Diagnosis</h3>'
        '<p>HuBERT-ECG encodes all 12 leads into rich cardiac features, classified across five clinical superclasses with confidence scores.</p></div>'
        f'<div class="ecg-feature-card">{_ICON_SALIENCY}<h3>Explainable Saliency Maps</h3>'
        '<p>Gradient-based attribution highlights exactly which signal regions drove each prediction, designed for clinical trust and transparency.</p></div>'
        f'<div class="ecg-feature-card">{_ICON_REPORT}<h3>Clinical PDF Reports</h3>'
        '<p>Generate a structured, hospital-style diagnostic report with waveforms, findings, confidence data, and a physician sign-off block.</p></div>'
        '</div>'
        '</div>'
    )

    if not _models_trained:
        gr.Markdown("**Model not trained - predictions are random.**")

    with gr.Tab("Test-set Record"):
        with gr.Row():
            dropdown = gr.Dropdown(choices=_record_choices, label="Select a PTB-XL test-set record (Record ID)",
                                    value=_record_choices[0] if _record_choices else None, scale=4)
            run_btn = gr.Button("Run Analysis", variant="primary", scale=1)

        gr.HTML(_eyebrow(_ICON_SALIENCY, "12-Lead Signal & Saliency Map"))
        with gr.Group():
            with gr.Row():
                plot_out = gr.Plot(label="12-Lead ECG with saliency overlay", scale=3)
                with gr.Column(scale=2):
                    label_out = gr.Markdown(label="Primary Diagnosis")
                    conf_out = gr.Label(label="Confidence by class", num_top_classes=5)

        gr.HTML(_eyebrow(_ICON_REPORT, "AI-Generated Clinical Report"))
        with gr.Group():
            gen_report_out = gr.Textbox(label="Clinical Report", lines=6, interactive=False)

        gr.HTML(_eyebrow(_ICON_REPORT, "Clinical PDF Report"))
        with gr.Group():
            preview_btn = gr.Button("Preview PDF", variant="primary")
            with gr.Column(elem_classes=["ecg-pdf-drawer"]) as pdf_drawer:
                with gr.Row(equal_height=True, elem_classes=["ecg-pdf-bar"]):
                    gr.Markdown("**Diagnostic PDF Report** - Complete document with 12-lead waveforms & clinical summary.", scale=4)
                    download_btn = gr.DownloadButton("Download PDF", variant="primary", scale=1, min_width=140, elem_classes=["ecg-pdf-download-btn"])
                pdf_preview = gr.Image(label="PDF Report Preview", interactive=False, show_label=False)

        export_state = gr.State()
        pdf_open_state = gr.State(value=False)

        run_btn.click(
            analyze_record,
            inputs=dropdown,
            outputs=[plot_out, label_out, conf_out, gen_report_out, export_state, pdf_drawer, preview_btn, pdf_open_state]
        )
        preview_btn.click(
            toggle_pdf_preview,
            inputs=[export_state, pdf_open_state],
            outputs=[pdf_drawer, pdf_preview, download_btn, preview_btn, pdf_open_state]
        )

    with gr.Tab("Upload Your Own ECG"):
        gr.Markdown("Upload a CSV file with shape (n_samples, 12) - one column per lead, no header row.")
        with gr.Row():
            file_in = gr.File(label="Upload CSV (12-lead ECG)", file_types=[".csv"], scale=3)
            upload_btn = gr.Button("Run Analysis", variant="primary", scale=1)

        gr.HTML(_eyebrow(_ICON_SALIENCY, "12-Lead Signal & Saliency Map"))
        with gr.Group():
            with gr.Row():
                plot_out2 = gr.Plot(label="12-Lead ECG with saliency overlay", scale=3)
                with gr.Column(scale=2):
                    label_out2 = gr.Markdown(label="Primary Diagnosis")
                    conf_out2 = gr.Label(label="Confidence by class", num_top_classes=5)

        gr.HTML(_eyebrow(_ICON_REPORT, "AI-Generated Clinical Report"))
        with gr.Group():
            gen_report_out2 = gr.Textbox(label="Clinical Report", lines=6, interactive=False)

        gr.HTML(_eyebrow(_ICON_REPORT, "Clinical PDF Report"))
        with gr.Group():
            preview_btn2 = gr.Button("Preview PDF", variant="primary")
            with gr.Column(elem_classes=["ecg-pdf-drawer"]) as pdf_drawer2:
                with gr.Row(equal_height=True, elem_classes=["ecg-pdf-bar"]):
                    gr.Markdown("**Diagnostic PDF Report** - Complete document with 12-lead waveforms & clinical summary.", scale=4)
                    download_btn2 = gr.DownloadButton("Download PDF", variant="primary", scale=1, min_width=140, elem_classes=["ecg-pdf-download-btn"])
                pdf_preview2 = gr.Image(label="PDF Report Preview", interactive=False, show_label=False)

        export_state2 = gr.State()
        pdf_open_state2 = gr.State(value=False)

        upload_btn.click(
            analyze_uploaded_csv,
            inputs=file_in,
            outputs=[plot_out2, label_out2, conf_out2, gen_report_out2, export_state2, pdf_drawer2, preview_btn2, pdf_open_state2]
        )
        preview_btn2.click(
            toggle_pdf_preview,
            inputs=[export_state2, pdf_open_state2],
            outputs=[pdf_drawer2, pdf_preview2, download_btn2, preview_btn2, pdf_open_state2]
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown(
            "Analyze multiple ECGs in a single run and get a consolidated summary table. "
            "Select any combination of test-set records and uploaded CSV files, then click **Run Batch Analysis**."
        )
        with gr.Row():
            batch_dropdown = gr.Dropdown(choices=_record_choices, multiselect=True,
                                          label="Test-set records (Record IDs)")
            batch_files = gr.File(label="Upload CSV files", file_types=[".csv"],
                                   file_count="multiple")
        batch_btn = gr.Button("Run Batch Analysis", variant="primary")
        batch_out = gr.Dataframe(headers=["Source", "Diagnosis", "Confidence", "Report"],
                                  label="Batch Results", wrap=True)

        batch_btn.click(analyze_batch, inputs=[batch_dropdown, batch_files], outputs=batch_out)


if __name__ == "__main__":
    # share=True generates a public gradio.live URL — required on Kaggle/Colab
    # because those environments block direct localhost access from a browser.
    # The link is valid for 72 hours. Set share=False when running locally.
    share = os.path.exists("/kaggle/input") or os.path.exists("/content")
    # footer_links=[] hides Gradio's own default footer (API docs / "Built with
    # Gradio" / Settings / Runs links) -- the safety disclaimer itself still
    # lives prominently in the exported PDF's disclaimer box, not just here.
    demo.launch(share=share, theme=_THEME, css=_CUSTOM_CSS, footer_links=[])
