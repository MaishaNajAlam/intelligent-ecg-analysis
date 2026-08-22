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
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from transformers import BartTokenizerFast

from src import config
from src.data import load_full_dataset, load_raw_signal, plot_12_lead
from src.encoder import ECGEncoder
from src.classifier import ClassifierHead, predict_single
from src.report_generator import ECGReportModel, generate_single

import torch

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
_report_model = ECGReportModel(encoder_dim=_encoder.feature_dim).to(config.DEVICE)
_report_ckpt = os.path.join(config.CHECKPOINT_DIR, "report_generator.pt")
if os.path.exists(_report_ckpt):
    _report_model.load_state_dict(torch.load(_report_ckpt, map_location=config.DEVICE))
_report_model.eval()

_record_choices = [str(i) for i in _test_df.index[:200]]  # first 200 test records for the dropdown


def _plot_to_image(signal, title):
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(9, 11), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(signal[:, i], linewidth=0.7, color="#1a1a1a")
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=18, fontsize=8)
        ax.set_yticks([])
    axes[-1].set_xlabel("Samples")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def analyze_record(ecg_id_str):
    ecg_id = int(ecg_id_str)
    row = _test_df.loc[ecg_id]
    signal, _ = load_raw_signal(row)

    fig = _plot_to_image(signal, title=f"Record {ecg_id}")

    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    label_str = f"**{top_label}** ({top_conf*100:.1f}% confidence)"
    conf_table = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in sorted(confidences.items(), key=lambda kv: -kv[1])])

    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
    ground_truth = str(row.get("report", "N/A"))

    side_by_side = (
        f"**Generated report:**\n{generated_report}\n\n"
        f"**Cardiologist ground truth:**\n{ground_truth}"
    )

    return fig, label_str, conf_table, side_by_side


def analyze_uploaded_csv(file_obj):
    """Optional path: accept a plain CSV of shape (T, 12) for a custom signal."""
    if file_obj is None:
        return None, "No file uploaded.", "", ""
    signal = np.loadtxt(file_obj.name, delimiter=",")
    if signal.shape[1] != config.N_LEADS:
        return None, f"Expected {config.N_LEADS} columns (leads), got {signal.shape[1]}.", "", ""

    fig = _plot_to_image(signal, title="Uploaded ECG")
    confidences = predict_single(signal, encoder=_encoder, model=_classifier)
    top_label = max(confidences, key=confidences.get)
    top_conf = confidences[top_label]
    label_str = f"**{top_label}** ({top_conf*100:.1f}% confidence)"
    conf_table = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in sorted(confidences.items(), key=lambda kv: -kv[1])])
    generated_report = generate_single(signal, encoder=_encoder, model=_report_model, tokenizer=_tokenizer)
    return fig, label_str, conf_table, f"**Generated report:**\n{generated_report}"


with gr.Blocks(title="Intelligent ECG Analysis Tool") as demo:
    gr.Markdown(
        "# Intelligent ECG Analysis Tool\n"
        "Signal-to-Report and Signal-to-Diagnosis with Deep Learning.\n\n"
        "Pick a record from the PTB-XL test set, or upload your own 12-lead CSV."
    )

    with gr.Tab("Test-set record"):
        with gr.Row():
            dropdown = gr.Dropdown(choices=_record_choices, label="Select a test-set ECG (record ID)",
                                    value=_record_choices[0] if _record_choices else None)
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            plot_out = gr.Plot(label="12-lead signal")
            with gr.Column():
                label_out = gr.Markdown(label="Diagnosis")
                conf_out = gr.Markdown(label="Confidence (all classes)")
        report_out = gr.Markdown(label="Report comparison")

        run_btn.click(analyze_record, inputs=dropdown,
                       outputs=[plot_out, label_out, conf_out, report_out])

    with gr.Tab("Upload your own"):
        gr.Markdown("CSV with shape (n_samples, 12), one column per lead, no header row.")
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_btn = gr.Button("Analyze uploaded ECG", variant="primary")
        with gr.Row():
            plot_out2 = gr.Plot(label="12-lead signal")
            with gr.Column():
                label_out2 = gr.Markdown(label="Diagnosis")
                conf_out2 = gr.Markdown(label="Confidence (all classes)")
        report_out2 = gr.Markdown(label="Generated report")

        upload_btn.click(analyze_uploaded_csv, inputs=file_in,
                          outputs=[plot_out2, label_out2, conf_out2, report_out2])

    gr.Markdown(
        "---\n"
        "*This is a research/educational prototype trained on PTB-XL. "
        "It is not a certified diagnostic device and must not be used for real clinical decisions.*"
    )


if __name__ == "__main__":
    # share=True generates a public gradio.live URL — required on Kaggle/Colab
    # because those environments block direct localhost access from a browser.
    # The link is valid for 72 hours. Set share=False when running locally.
    share = not (os.path.exists("/kaggle/input") or os.path.exists("/content"))
    demo.launch(share=share)
