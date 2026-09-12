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
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&display=swap');

.primary {{ border-radius: 999px !important; font-weight: 600 !important; }}

.ecg-eyebrow {{
    display: flex; align-items: center; gap: 8px;
    color: {_SAGE_MUTED}; font-size: 0.78rem; letter-spacing: 0.12em;
    text-transform: uppercase; font-weight: 600; margin: 4px 0;
}}
.ecg-eyebrow svg {{ width: 16px; height: 16px; flex-shrink: 0; }}

.ecg-h1 {{
    font-family: 'Lora', Georgia, serif; color: {_FOREST};
    font-size: 2.1rem; font-weight: 600; margin: 0 0 6px 0; line-height: 1.15;
}}
.ecg-sub {{ color: {_SAGE_MUTED}; font-size: 0.95rem; margin: 0 0 18px 0; }}

.ecg-features {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px; margin: 4px 0 22px 0;
}}
.ecg-feature-card {{
    background: {_CREAM}; border: 1px solid {_LINE}; border-radius: 18px; padding: 20px;
}}
.ecg-feature-card svg {{ width: 34px; height: 34px; color: {_SAGE_MUTED}; margin-bottom: 12px; }}
.ecg-feature-card h3 {{
    font-family: 'Lora', Georgia, serif; color: {_FOREST};
    font-size: 1.02rem; font-weight: 600; margin: 0 0 6px 0;
}}
.ecg-feature-card p {{ color: {_SAGE_MUTED}; font-size: 0.85rem; line-height: 1.45; margin: 0; }}
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


def analyze_record(ecg_id_str):
    ecg_id = int(ecg_id_str)
    row = _test_df.loc[ecg_id]
    signal, _ = load_raw_signal(row)

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    top_desc = LABEL_DESCRIPTIONS.get(top_label, "")
    label_str = f"**{top_label}** — {top_desc} ({top_conf*100:.1f}% confidence)"
    diagnosis_md = f"### {top_label} — {top_desc}\n**Confidence:** {top_conf*100:.1f}%"

    saliency_result = compute_saliency(signal, encoder=_encoder, classifier=_classifier,
                                        target_class=top_label)
    fig = _plot_to_image(signal, title=f"Record {ecg_id}", saliency=saliency_result["saliency"])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
    ground_truth_raw = str(row.get("report", "N/A"))
    ground_truth = (
        "(Note: original cardiologist report is in German)"
        if _looks_german(ground_truth_raw)
        else ground_truth_raw
    )

    # Cached for the "Export to PDF" button -- regenerated there rather than
    # reusing `fig` directly, since Gradio may close/consume figures handed
    # to gr.Plot. Only lightweight, picklable data goes in gr.State.
    export_data = {
        "signal": signal,
        "saliency": saliency_result["saliency"],
        "title": f"Record {ecg_id}",
        "label_str": label_str,
        "confidences": confidences,
        "report_text": generated_report,
        "ground_truth": ground_truth,
        "patient_meta": _extract_patient_meta(row),
    }

    return fig, diagnosis_md, confidences, generated_report, ground_truth, export_data


def analyze_uploaded_csv(file_obj):
    """Optional path: accept a plain CSV of shape (T, 12) for a custom signal."""
    if file_obj is None:
        return None, "No file uploaded.", None, "", None
    signal = np.loadtxt(file_obj.name, delimiter=",")
    if signal.shape[1] != config.N_LEADS:
        return None, f"Expected {config.N_LEADS} columns (leads), got {signal.shape[1]}.", None, "", None

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    top_desc = LABEL_DESCRIPTIONS.get(top_label, "")
    label_str = f"**{top_label}** — {top_desc} ({top_conf*100:.1f}% confidence)"
    diagnosis_md = f"### {top_label} — {top_desc}\n**Confidence:** {top_conf*100:.1f}%"

    saliency_result = compute_saliency(signal, encoder=_encoder, classifier=_classifier,
                                        target_class=top_label)
    fig = _plot_to_image(signal, title="Uploaded ECG", saliency=saliency_result["saliency"])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)

    export_data = {
        "signal": signal,
        "saliency": saliency_result["saliency"],
        "title": "Uploaded ECG",
        "label_str": label_str,
        "confidences": confidences,
        "report_text": generated_report,
        "ground_truth": None,
        "patient_meta": None,  # no accompanying patient record for an uploaded signal
    }

    return fig, diagnosis_md, confidences, generated_report, export_data


def export_pdf(export_data):
    """Rebuilds the plot fresh (cheap -- matplotlib only, no model inference)
    and lays it out into a PDF via src/pdf_export.py, then rasterizes page 1
    so the user can preview it before downloading the actual file. The
    preview/download components start hidden (see the Blocks layout) and are
    only revealed once there's something real to show."""
    if not export_data:
        return gr.update(visible=False), gr.update(visible=False)
    fig = _plot_to_image(export_data["signal"], title=export_data["title"],
                          saliency=export_data["saliency"])
    path = build_pdf_report(
        fig, title=export_data["title"], diagnosis_label=export_data["label_str"],
        confidences=export_data["confidences"], report_text=export_data["report_text"],
        ground_truth=export_data.get("ground_truth"), patient_meta=export_data.get("patient_meta"),
    )
    plt.close(fig)
    preview_path = render_preview_image(path)
    return gr.update(value=preview_path, visible=True), gr.update(value=path, visible=True)


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


with gr.Blocks(title="Intelligent ECG Analysis Tool") as demo:
    gr.HTML(_eyebrow(_ICON_HEART, "AI-Powered Cardiac Analysis"))
    gr.HTML(
        '<div class="ecg-h1">Intelligent ECG Analysis</div>'
        '<p class="ecg-sub">Signal-to-report and signal-to-diagnosis with deep learning — '
        'built for clinicians who need to see the reasoning, not just the result.</p>'
    )

    gr.HTML(
        '<div class="ecg-features">'
        f'<div class="ecg-feature-card">{_ICON_PULSE_CIRCLE}<h3>AI-Powered Diagnosis</h3>'
        '<p>HuBERT-ECG + BART deep learning models analyze all 12 leads and classify five '
        'diagnostic superclasses in seconds.</p></div>'
        f'<div class="ecg-feature-card">{_ICON_SALIENCY}<h3>Explainable Saliency Mapping</h3>'
        '<p>See exactly which regions of the signal drove the diagnosis — built for '
        'clinical trust, not a black box.</p></div>'
        f'<div class="ecg-feature-card">{_ICON_REPORT}<h3>Clinical-Grade PDF Reports</h3>'
        '<p>Export a hospital-style diagnostic report with patient data, findings, and a '
        'physician sign-off line.</p></div>'
        '</div>'
    )

    if not _models_trained:
        gr.Markdown("⚠️ **Model not trained — predictions are random.**")

    with gr.Tab("Test-set record"):
        with gr.Row():
            dropdown = gr.Dropdown(choices=_record_choices, label="Select a test-set ECG (record ID)",
                                    value=_record_choices[0] if _record_choices else None, scale=4)
            run_btn = gr.Button("Analyze", variant="primary", scale=1)

        gr.HTML(_eyebrow(_ICON_SALIENCY, "ECG Signal & AI Saliency Map"))
        with gr.Group():
            with gr.Row():
                plot_out = gr.Plot(label="12-lead signal", scale=3)
                with gr.Column(scale=2):
                    label_out = gr.Markdown(label="Diagnosis")
                    conf_out = gr.Label(label="Confidence (all classes)", num_top_classes=5)

        gr.HTML(_eyebrow(_ICON_REPORT, "Clinical Report"))
        with gr.Group():
            with gr.Row():
                gen_report_out = gr.Textbox(label="🤖 AI-Generated Report", lines=6, interactive=False)
                truth_report_out = gr.Textbox(label="🩺 Cardiologist Ground Truth", lines=6, interactive=False)

        export_state = gr.State()
        export_btn = gr.Button("📄 Export to PDF", variant="primary")
        with gr.Row():
            pdf_preview = gr.Image(label="PDF Preview", scale=3, height=300, visible=False)
            pdf_out = gr.File(label="Download PDF report", scale=1, visible=False)

        run_btn.click(analyze_record, inputs=dropdown,
                       outputs=[plot_out, label_out, conf_out, gen_report_out, truth_report_out, export_state])
        export_btn.click(export_pdf, inputs=export_state, outputs=[pdf_preview, pdf_out])

    with gr.Tab("Upload your own"):
        gr.Markdown("CSV with shape (n_samples, 12), one column per lead, no header row.")
        with gr.Row():
            file_in = gr.File(label="Upload CSV", file_types=[".csv"], scale=3)
            upload_btn = gr.Button("Analyze uploaded ECG", variant="primary", scale=1)

        gr.HTML(_eyebrow(_ICON_SALIENCY, "ECG Signal & AI Saliency Map"))
        with gr.Group():
            with gr.Row():
                plot_out2 = gr.Plot(label="12-lead signal", scale=3)
                with gr.Column(scale=2):
                    label_out2 = gr.Markdown(label="Diagnosis")
                    conf_out2 = gr.Label(label="Confidence (all classes)", num_top_classes=5)

        gr.HTML(_eyebrow(_ICON_REPORT, "Clinical Report"))
        with gr.Group():
            gen_report_out2 = gr.Textbox(label="🤖 AI-Generated Report", lines=6, interactive=False)

        export_state2 = gr.State()
        export_btn2 = gr.Button("📄 Export to PDF", variant="primary")
        with gr.Row():
            pdf_preview2 = gr.Image(label="PDF Preview", scale=3, height=300, visible=False)
            pdf_out2 = gr.File(label="Download PDF report", scale=1, visible=False)

        upload_btn.click(analyze_uploaded_csv, inputs=file_in,
                          outputs=[plot_out2, label_out2, conf_out2, gen_report_out2, export_state2])
        export_btn2.click(export_pdf, inputs=export_state2, outputs=[pdf_preview2, pdf_out2])

    with gr.Tab("Batch analysis"):
        gr.Markdown(
            "Analyze several ECGs at once and get a summary table. "
            "Pick multiple test-set records and/or upload multiple CSVs, then run."
        )
        with gr.Row():
            batch_dropdown = gr.Dropdown(choices=_record_choices, multiselect=True,
                                          label="Test-set records (optional)")
            batch_files = gr.File(label="Upload CSVs (optional)", file_types=[".csv"],
                                   file_count="multiple")
        batch_btn = gr.Button("Run batch analysis", variant="primary")
        batch_out = gr.Dataframe(headers=["Source", "Diagnosis", "Confidence", "Report"],
                                  label="Batch results", wrap=True)

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
