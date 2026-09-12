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
from src.pdf_export import build_pdf_report

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


def analyze_record(ecg_id_str):
    ecg_id = int(ecg_id_str)
    row = _test_df.loc[ecg_id]
    signal, _ = load_raw_signal(row)

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    top_desc = LABEL_DESCRIPTIONS.get(top_label, "")
    label_str = f"**{top_label}** — {top_desc} ({top_conf*100:.1f}% confidence)"

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

    side_by_side = (
        f"**Generated report:**\n{generated_report}\n\n"
        f"**Cardiologist ground truth:**\n{ground_truth}"
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
    }

    return fig, label_str, confidences, side_by_side, export_data


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

    saliency_result = compute_saliency(signal, encoder=_encoder, classifier=_classifier,
                                        target_class=top_label)
    fig = _plot_to_image(signal, title="Uploaded ECG", saliency=saliency_result["saliency"])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
    report_str = f"**Generated report:**\n{generated_report}"

    export_data = {
        "signal": signal,
        "saliency": saliency_result["saliency"],
        "title": "Uploaded ECG",
        "label_str": label_str,
        "confidences": confidences,
        "report_text": generated_report,
        "ground_truth": None,
    }

    return fig, label_str, confidences, report_str, export_data


def export_pdf(export_data):
    """Rebuilds the plot fresh (cheap -- matplotlib only, no model inference)
    and lays it out into a downloadable PDF via src/pdf_export.py."""
    if not export_data:
        return None
    fig = _plot_to_image(export_data["signal"], title=export_data["title"],
                          saliency=export_data["saliency"])
    path = build_pdf_report(
        fig, title=export_data["title"], diagnosis_label=export_data["label_str"],
        confidences=export_data["confidences"], report_text=export_data["report_text"],
        ground_truth=export_data.get("ground_truth"),
    )
    plt.close(fig)
    return path


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
    gr.Markdown(
        "# Intelligent ECG Analysis Tool\n"
        "Signal-to-Report and Signal-to-Diagnosis with Deep Learning.\n\n"
        "Pick a record from the PTB-XL test set, or upload your own 12-lead CSV."
    )

    if not _models_trained:
        gr.Markdown("⚠️ **Model not trained — predictions are random.**")

    with gr.Tab("Test-set record"):
        with gr.Row():
            dropdown = gr.Dropdown(choices=_record_choices, label="Select a test-set ECG (record ID)",
                                    value=_record_choices[0] if _record_choices else None)
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            plot_out = gr.Plot(label="12-lead signal")
            with gr.Column():
                label_out = gr.Markdown(label="Diagnosis")
                conf_out = gr.Label(label="Confidence (all classes)", num_top_classes=5)
        report_out = gr.Markdown(label="Report comparison")
        export_state = gr.State()
        with gr.Row():
            export_btn = gr.Button("📄 Export to PDF")
            pdf_out = gr.File(label="Download PDF report")

        run_btn.click(analyze_record, inputs=dropdown,
                       outputs=[plot_out, label_out, conf_out, report_out, export_state])
        export_btn.click(export_pdf, inputs=export_state, outputs=pdf_out)

    with gr.Tab("Upload your own"):
        gr.Markdown("CSV with shape (n_samples, 12), one column per lead, no header row.")
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_btn = gr.Button("Analyze uploaded ECG", variant="primary")
        with gr.Row():
            plot_out2 = gr.Plot(label="12-lead signal")
            with gr.Column():
                label_out2 = gr.Markdown(label="Diagnosis")
                conf_out2 = gr.Label(label="Confidence (all classes)", num_top_classes=5)
        report_out2 = gr.Markdown(label="Generated report")
        export_state2 = gr.State()
        with gr.Row():
            export_btn2 = gr.Button("📄 Export to PDF")
            pdf_out2 = gr.File(label="Download PDF report")

        upload_btn.click(analyze_uploaded_csv, inputs=file_in,
                          outputs=[plot_out2, label_out2, conf_out2, report_out2, export_state2])
        export_btn2.click(export_pdf, inputs=export_state2, outputs=pdf_out2)

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

    gr.Markdown(
        "---\n"
        "*This is a research/educational prototype trained on PTB-XL. "
        "It is not a certified diagnostic device and must not be used for real clinical decisions.*"
    )


if __name__ == "__main__":
    # share=True generates a public gradio.live URL — required on Kaggle/Colab
    # because those environments block direct localhost access from a browser.
    # The link is valid for 72 hours. Set share=False when running locally.
    share = os.path.exists("/kaggle/input") or os.path.exists("/content")
    demo.launch(share=share)
