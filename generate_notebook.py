"""
generate_notebook.py
────────────────────
Reads the project source files and assembles a single self-contained
Kaggle-ready Jupyter notebook: ecg_project_kaggle.ipynb

Usage (run from the project root):
    python generate_notebook.py

The output file can be uploaded directly to Kaggle as a new notebook.
Requirements: nbformat  (pip install nbformat)
"""
import json
import os
import textwrap
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))


def read_src(filename):
    """Read a file from the src/ directory and return its text."""
    path = os.path.join(HERE, "src", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Helper: strip the module-level `if __name__ == "__main__":` block ──────
def strip_main_block(source: str) -> str:
    """Remove the `if __name__ == '__main__':` guard at the bottom of a file."""
    lines = source.splitlines()
    out = []
    skip = False
    for line in lines:
        if line.strip().startswith('if __name__ == "__main__"') or \
           line.strip().startswith("if __name__ == '__main__'"):
            skip = True
        if not skip:
            out.append(line)
    return "\n".join(out)


# ── Helper: remove relative `from src import ...` / `from src.x import ...` ─
def fix_imports(source: str) -> str:
    """
    In the notebook everything lives in one flat global namespace (every
    src/*.py file becomes its own cell, executed top-to-bottom), so any
    `from src import config`, `from src.data import load_raw_signal`, or
    `import src.encoder` is dead weight by the time later cells run — the
    names are already defined.

    Comment these lines out rather than rewriting them to `from data import
    ...` / `from encoder import ...` etc: there is no such top-level module
    in a notebook, so that used to raise ModuleNotFoundError (or, for a
    name like `datasets`, silently shadow the real HuggingFace `datasets`
    PyPI package and fail with a confusing ImportError instead).
    """
    import re

    def _comment(m):
        return f"# {m.group(0)}  # (removed by generate_notebook.py — already in notebook scope)"

    source = re.sub(r'^from src import \w+.*$', _comment, source, flags=re.MULTILINE)
    source = re.sub(r'^from src\.\w+ import .*$', _comment, source, flags=re.MULTILINE)
    source = re.sub(r'^import src\.\w+.*$', _comment, source, flags=re.MULTILINE)
    return source


def make_notebook():
    nb = new_notebook()
    cells = []

    # ────────────────────────────────────────────────────────────────────
    # Header
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(textwrap.dedent("""\
        # 🫀 Intelligent ECG Analysis Tool
        ### Signal-to-Report and Signal-to-Diagnosis with Deep Learning

        **Pipeline overview:**
        ```
        ECG Signal → [Frozen HuBERT-ECG Encoder] → features
                                                   ├─→ [Classifier Head]   → Label + Confidence
                                                   └─→ [Adapter + BART]   → Clinical Report
        ```

        **Steps covered in this notebook:**
        | Step | Description |
        |------|-------------|
        | 1 | Environment & data setup (PTB-XL) |
        | 2 | Load frozen HuBERT-ECG encoder, extract features |
        | 3 | Train classifier head (signal → label) |
        | 4 | Train report generator (signal → text) |
        | 5 | Launch Gradio web app |
        | 6 | Systematic evaluation & error analysis |

        > ⚠️ **Before running:** add the PTB-XL dataset to this notebook via  
        > *Add Data → search "PTB-XL: A Large Publicly Available ECG Dataset"*
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 1 — Install dependencies
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 📦 Step 0 — Install Dependencies\n*(runs once per Kaggle session, ~5 min)*"))
    cells.append(new_code_cell(textwrap.dedent("""\
        # Install all required packages
        # The HuBERT-ECG line installs the custom model class from GitHub
        import subprocess, sys

        packages = [
            "torch>=2.1.0",
            "torchaudio>=2.1.0",
            "transformers>=4.40.0",
            "wfdb>=4.1.2",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "scikit-learn>=1.3.0",
            "gradio>=4.30.0",
            "tqdm>=4.66.0",
            "evaluate>=0.4.1",
            "rouge-score>=0.1.2",
            "nltk>=3.8.1",
            "peft>=0.11.0",
            "huggingface_hub>=0.23.0",
            "scipy>=1.11.0",
        ]

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

        # HuBERT-ECG custom package MUST be installed BEFORE importing transformers
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/Edoar-do/HuBERT-ECG.git"
        ])

        print("✔  All packages installed.")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 2 — Configuration
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## ⚙️ Configuration\n*(edit `PTBXL_DIR` if your dataset path differs)*"))
    cells.append(new_code_cell(textwrap.dedent("""\
        import os
        import torch

        # ── Paths ──────────────────────────────────────────────────────────
        # Kaggle mounts PTB-XL datasets under /kaggle/input/
        # The folder name varies; we auto-detect it below.
        def _find_ptbxl():
            base = "/kaggle/input"
            if not os.path.exists(base):
                return None
            for entry in os.listdir(base):
                lower = entry.lower()
                if "ptb" in lower and ("xl" in lower or "ecg" in lower):
                    candidate = os.path.join(base, entry)
                    if os.path.exists(os.path.join(candidate, "ptbxl_database.csv")):
                        return candidate
            return None

        PTBXL_DIR = _find_ptbxl() or os.path.join(os.getcwd(), "data", "ptbxl")
        print(f"PTB-XL directory: {PTBXL_DIR}")
        assert os.path.exists(os.path.join(PTBXL_DIR, "ptbxl_database.csv")), \
            "❌ ptbxl_database.csv not found! Add the PTB-XL dataset via 'Add Data'."

        CHECKPOINT_DIR     = "/kaggle/working/checkpoints"
        OUTPUT_DIR         = "/kaggle/working/outputs"
        FEATURES_CACHE_DIR = "/kaggle/working/features_cache"

        for d in [CHECKPOINT_DIR, OUTPUT_DIR, FEATURES_CACHE_DIR]:
            os.makedirs(d, exist_ok=True)

        # ── Device ─────────────────────────────────────────────────────────
        DEVICE = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Device: {DEVICE}")

        # ── Signal params ──────────────────────────────────────────────────
        SAMPLING_RATE       = 500   # Hz  (PTB-XL records500/)
        SIGNAL_LENGTH_SEC   = 10
        N_LEADS             = 12
        PREPROCESSING_MODE  = "zscore"   # "zscore" | "minmax" | "bandpass_minmax"

        # ── Encoder ────────────────────────────────────────────────────────
        HUBERT_ECG_MODEL_ID           = "Edoardo-BS/hubert-ecg-base"
        ENCODER_FEATURE_DIM           = 768
        USE_FALLBACK_ENCODER_IF_UNAVAILABLE = True
        FALLBACK_GUARD                = False  # set True to block training on random features

        # ── Classifier ─────────────────────────────────────────────────────
        SUPERCLASSES          = ["NORM", "MI", "STTC", "CD", "HYP"]
        NUM_CLASSES           = len(SUPERCLASSES)
        CLASSIFIER_HIDDEN_DIM = 256
        CLASSIFIER_LR         = 1e-3
        CLASSIFIER_EPOCHS     = 30
        CLASSIFIER_BATCH_SIZE = 64

        # ── Report generator ───────────────────────────────────────────────
        BART_MODEL_ID         = "facebook/bart-base"
        ADAPTER_HIDDEN_DIM    = 512
        MAX_REPORT_TOKENS     = 128
        REPORT_GEN_LR         = 5e-5
        REPORT_GEN_EPOCHS     = 10
        REPORT_GEN_BATCH_SIZE = 8
        REPORT_GEN_NUM_BEAMS  = 4
        USE_LORA              = True

        # ── Misc ───────────────────────────────────────────────────────────
        RANDOM_SEED = 42

        # ── `config` namespace ────────────────────────────────────────────
        # src/*.py (the cells below) do `from src import config` and read
        # `config.PTBXL_DIR`, `config.SAMPLING_RATE`, etc. Rather than
        # duplicating every value under two names, wrap the settings above
        # in a real `config` object so `config.X` keeps working unchanged
        # (this mirrors src/config.py being an importable module there).
        # NOTE: built from a snapshot of globals() taken *here*, after every
        # setting above is assigned — it does not pick up unrelated
        # uppercase names defined in later cells.
        import types as _types
        config = _types.SimpleNamespace(**{
            k: v for k, v in dict(globals()).items()
            if k.isupper() and not k.startswith("_")
        })

        print("✔  Configuration loaded.")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 3 — Step 1: Data loading
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 📊 Step 1 — Load PTB-XL Dataset"))
    src_data = fix_imports(strip_main_block(read_src("data.py")))
    # Remove the module docstring to keep cell clean
    src_data_lines = src_data.split("\n")
    # Find end of docstring
    clean_lines = []
    in_doc = False
    doc_done = False
    for line in src_data_lines:
        if not doc_done and line.strip().startswith('"""'):
            if in_doc:
                doc_done = True
            else:
                in_doc = True
            continue
        if not doc_done and in_doc:
            continue
        clean_lines.append(line)
    src_data = "\n".join(clean_lines)
    cells.append(new_code_cell(src_data))

    # Step 1 checkpoint cell
    cells.append(new_markdown_cell("### ✅ Step 1 Checkpoint — Load & Plot One Record"))
    cells.append(new_code_cell(textwrap.dedent("""\
        # Load full dataset metadata
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for Kaggle

        data = load_full_dataset()
        df   = data["full"]
        row  = df.iloc[0]

        signal, meta = load_raw_signal(row)
        print(f"Record {row.name}: shape={signal.shape}, fs={meta['fs']} Hz")
        print(f"Diagnostic superclasses : {row['superclasses']}")
        print(f"Label vector {SUPERCLASSES}: {row['label_vector']}")
        print(f"Free-text report        : {row.get('report', 'N/A')}")

        train_df, val_df, test_df = get_official_splits(df)
        print(f"\\nSplits → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

        # Plot and save
        out_path = os.path.join(OUTPUT_DIR, "example_ecg_plot.png")
        plot_12_lead(signal, title=f"Record {row.name}", save_path=out_path)

        # Display inline
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(out_path)
        plt.figure(figsize=(10, 14))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Step 1 Checkpoint: 12-lead ECG")
        plt.show()
        print("\\n✔  Step 1 CHECKPOINT PASSED")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 4 — Step 2: Encoder
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 🧠 Step 2 — Load Frozen HuBERT-ECG Encoder"))
    src_encoder = fix_imports(strip_main_block(read_src("encoder.py")))
    cells.append(new_code_cell(src_encoder))

    # Step 2 checkpoint
    cells.append(new_markdown_cell("### ✅ Step 2 Checkpoint — Feature Extraction"))
    cells.append(new_code_cell(textwrap.dedent("""\
        import numpy as np

        enc = ECGEncoder()

        frozen = enc.confirm_frozen()
        print(f"Encoder frozen (zero gradients): {frozen}")
        print(f"Using fallback encoder         : {enc.is_fallback}")
        print(f"Feature dimension (d)          : {enc.feature_dim}")
        assert frozen, "FAIL: encoder not frozen!"

        # Single signal
        row    = data["test"].iloc[0]
        signal, meta = load_raw_signal(row)
        feats  = enc.encode(signal)
        print(f"\\nSingle signal → feature shape (L × d): {tuple(feats.shape)}")

        # Batch
        batch_rows    = [data["test"].iloc[i] for i in range(4)]
        batch_signals = np.stack([load_raw_signal(r)[0] for r in batch_rows])
        batch_feats   = enc.encode_batch(batch_signals)
        print(f"Batch (4) feature shape (B × L × d)  : {tuple(batch_feats.shape)}")

        print("\\n✔  Step 2 CHECKPOINT PASSED")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 5 — Datasets helper
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 🗄️ Dataset Utilities (Feature Caching)"))
    src_datasets = fix_imports(strip_main_block(read_src("datasets.py")))
    cells.append(new_code_cell(src_datasets))

    # ────────────────────────────────────────────────────────────────────
    # Cell 6 — Step 3: Classifier
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(textwrap.dedent("""\
        ## 🏷️ Step 3 — Train the Classifier Head (Signal → Label)

        Only the head's parameters are trained. The encoder stays frozen.  
        Features are pre-computed once and cached to disk to speed up all epochs.
    """)))
    src_clf = fix_imports(strip_main_block(read_src("classifier.py")))
    cells.append(new_code_cell(src_clf))

    cells.append(new_markdown_cell("### 🚀 Run Step 3 Training"))
    cells.append(new_code_cell(textwrap.dedent("""\
        # Pre-computes features (cached to FEATURES_CACHE_DIR) then trains for
        # CLASSIFIER_EPOCHS epochs, saving the best checkpoint by val AUROC.
        train()   # defined in the ClassifierHead cell above
    """)))

    cells.append(new_markdown_cell("### ✅ Step 3 Checkpoint — Test-set AUROC"))
    cells.append(new_code_cell(textwrap.dedent("""\
        macro_auroc, per_class = evaluate_test()
        print(f"\\nTest macro AUROC: {macro_auroc:.4f}")
        for cls, score in per_class.items():
            print(f"  {cls}: {score:.4f}")
        print("\\n✔  Step 3 CHECKPOINT PASSED")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 7 — Step 4: Report generator
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(textwrap.dedent("""\
        ## 📝 Step 4 — Train the Report Generator (Signal → Text)

        Architecture: `HuBERT-ECG features → FeatureAdapter MLP → BART cross-attention → text`  
        BART is fine-tuned with LoRA (lightweight, only ~2% of parameters are trainable).
    """)))
    src_rg = fix_imports(strip_main_block(read_src("report_generator.py")))
    cells.append(new_code_cell(src_rg))

    cells.append(new_markdown_cell("### 🚀 Run Step 4 Training"))
    cells.append(new_code_cell(textwrap.dedent("""\
        # Fine-tunes FeatureAdapter + LoRA-wrapped BART on (ECG features, report) pairs.
        # Reuses cached features from Step 3 if already computed.
        train()   # the train() defined in the Report Generator cell above
                  # Note: if ClassifierHead.train() is also in scope, be explicit:
                  # from the report_generator section, this is the correct one.
    """)))

    cells.append(new_markdown_cell("### ✅ Step 4 Checkpoint — BLEU & ROUGE + Sample Report"))
    cells.append(new_code_cell(textwrap.dedent("""\
        bleu_score, rouge_score = evaluate_test()
        print(f"\\nBLEU : {bleu_score['bleu']:.4f}")
        print(f"ROUGE: {rouge_score}")

        # Generate a sample report for one test record
        row    = data["test"].iloc[0]
        signal, _ = load_raw_signal(row)
        report = generate_single(signal)
        print(f"\\nGenerated report : {report}")
        print(f"Ground truth      : {row.get('report', 'N/A')}")
        print("\\n✔  Step 4 CHECKPOINT PASSED")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 8 — Step 6: Evaluation
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(textwrap.dedent("""\
        ## 📈 Step 6 — Systematic Evaluation & Error Analysis

        Produces three output files:
        - `per_diagnosis_performance.csv` — AUROC per class, sorted best → worst
        - `failure_cases.csv`            — confidently-wrong test cases
        - `evaluation_summary.txt`       — written paragraph summarising findings
    """)))
    src_eval = fix_imports(strip_main_block(read_src("evaluate_full.py")))
    cells.append(new_code_cell(src_eval))

    cells.append(new_markdown_cell("### ✅ Step 6 Checkpoint — Run Full Evaluation"))
    cells.append(new_code_cell(textwrap.dedent("""\
        run_classifier_error_analysis()

        # Display the performance table inline
        import pandas as pd
        perf_df = pd.read_csv(os.path.join(OUTPUT_DIR, "per_diagnosis_performance.csv"))
        print("\\nPer-Diagnosis Performance (sorted best → worst):")
        print(perf_df.to_string(index=False))

        fail_df = pd.read_csv(os.path.join(OUTPUT_DIR, "failure_cases.csv"))
        print(f"\\nTotal confidently-wrong failure cases: {len(fail_df)}")
        print(fail_df.head(10).to_string(index=False))

        with open(os.path.join(OUTPUT_DIR, "evaluation_summary.txt")) as f:
            print("\\n" + f.read())

        print("✔  Step 6 CHECKPOINT PASSED")
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Cell 9 — Step 5: Gradio web app
    # ────────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(textwrap.dedent("""\
        ## 🌐 Step 5 — Launch Gradio Web Application

        Runs the interactive ECG analysis app.  
        A **public `gradio.live` URL** is printed below — open it in any browser.  
        The link is valid for **72 hours**.

        > ⚠️ Keep this cell running. The server dies when the cell stops.
    """)))
    cells.append(new_code_cell(textwrap.dedent("""\
        import os
        import numpy as np
        import gradio as gr
        import matplotlib.pyplot as plt
        import torch
        from transformers import BartTokenizerFast

        print("Loading models (this happens once at startup)...")
        _data      = load_full_dataset()
        _test_df   = _data["test"]
        _encoder   = ECGEncoder()

        _classifier = ClassifierHead(in_dim=_encoder.feature_dim).to(DEVICE)
        _clf_ckpt   = os.path.join(CHECKPOINT_DIR, "classifier_head.pt")
        if os.path.exists(_clf_ckpt):
            _classifier.load_state_dict(torch.load(_clf_ckpt, map_location=DEVICE))
        _classifier.eval()

        _tokenizer    = BartTokenizerFast.from_pretrained(BART_MODEL_ID)
        _report_model = ECGReportModel(encoder_dim=_encoder.feature_dim).to(DEVICE)
        _report_ckpt  = os.path.join(CHECKPOINT_DIR, "report_generator.pt")
        if os.path.exists(_report_ckpt):
            _report_model.load_state_dict(torch.load(_report_ckpt, map_location=DEVICE))
        _report_model.eval()

        _record_choices = [str(i) for i in _test_df.index[:200]]


        def _plot_to_image(signal, title):
            lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
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
            row    = _test_df.loc[ecg_id]
            signal, _ = load_raw_signal(row)
            fig    = _plot_to_image(signal, title=f"Record {ecg_id}")
            confidences = predict_single(signal, encoder=_encoder, model=_classifier)
            top_label   = max(confidences, key=confidences.get)
            top_conf    = confidences[top_label]
            label_str   = f"**{top_label}** ({top_conf*100:.1f}% confidence)"
            conf_table  = "\\n".join(
                [f"- {k}: {v*100:.1f}%" for k, v in sorted(confidences.items(), key=lambda kv: -kv[1])]
            )
            generated_report = generate_single(signal, encoder=_encoder,
                                               model=_report_model, tokenizer=_tokenizer)
            ground_truth = str(row.get("report", "N/A"))
            side_by_side = (
                f"**Generated report:**\\n{generated_report}\\n\\n"
                f"**Cardiologist ground truth:**\\n{ground_truth}"
            )
            return fig, label_str, conf_table, side_by_side


        def analyze_uploaded_csv(file_obj):
            if file_obj is None:
                return None, "No file uploaded.", "", ""
            signal = np.loadtxt(file_obj.name, delimiter=",")
            if signal.shape[1] != N_LEADS:
                return None, f"Expected {N_LEADS} columns, got {signal.shape[1]}.", "", ""
            fig = _plot_to_image(signal, title="Uploaded ECG")
            confidences = predict_single(signal, encoder=_encoder, model=_classifier)
            top_label   = max(confidences, key=confidences.get)
            top_conf    = confidences[top_label]
            label_str   = f"**{top_label}** ({top_conf*100:.1f}% confidence)"
            conf_table  = "\\n".join(
                [f"- {k}: {v*100:.1f}%" for k, v in sorted(confidences.items(), key=lambda kv: -kv[1])]
            )
            generated_report = generate_single(signal, encoder=_encoder,
                                               model=_report_model, tokenizer=_tokenizer)
            return fig, label_str, conf_table, f"**Generated report:**\\n{generated_report}"


        with gr.Blocks(title="Intelligent ECG Analysis Tool") as demo:
            gr.Markdown(
                "# 🫀 Intelligent ECG Analysis Tool\\n"
                "Signal-to-Report and Signal-to-Diagnosis with Deep Learning.\\n\\n"
                "Pick a record from the PTB-XL test set, or upload your own 12-lead CSV."
            )
            with gr.Tab("Test-set record"):
                with gr.Row():
                    dropdown = gr.Dropdown(
                        choices=_record_choices, label="Select a test-set ECG (record ID)",
                        value=_record_choices[0] if _record_choices else None
                    )
                    run_btn = gr.Button("Analyze", variant="primary")
                with gr.Row():
                    plot_out  = gr.Plot(label="12-lead signal")
                    with gr.Column():
                        label_out = gr.Markdown(label="Diagnosis")
                        conf_out  = gr.Markdown(label="Confidence (all classes)")
                report_out = gr.Markdown(label="Report comparison")
                run_btn.click(analyze_record, inputs=dropdown,
                              outputs=[plot_out, label_out, conf_out, report_out])

            with gr.Tab("Upload your own"):
                gr.Markdown("CSV with shape (n_samples, 12), one column per lead, no header.")
                file_in    = gr.File(label="Upload CSV", file_types=[".csv"])
                upload_btn = gr.Button("Analyze uploaded ECG", variant="primary")
                with gr.Row():
                    plot_out2  = gr.Plot(label="12-lead signal")
                    with gr.Column():
                        label_out2 = gr.Markdown(label="Diagnosis")
                        conf_out2  = gr.Markdown(label="Confidence (all classes)")
                report_out2 = gr.Markdown(label="Generated report")
                upload_btn.click(analyze_uploaded_csv, inputs=file_in,
                                 outputs=[plot_out2, label_out2, conf_out2, report_out2])

            gr.Markdown(
                "---\\n"
                "*Research/educational prototype. Not a certified diagnostic device.*"
            )

        # Auto-detect Kaggle/Colab → share=True for public URL
        on_cloud = os.path.exists("/kaggle/input") or os.path.exists("/content")
        demo.launch(share=on_cloud)
    """)))

    # ────────────────────────────────────────────────────────────────────
    # Assemble and write
    # ────────────────────────────────────────────────────────────────────
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }

    out_path = os.path.join(HERE, "ecg_project_kaggle.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[OK] Notebook written to: {out_path}")
    print("     Upload it to Kaggle, add the PTB-XL dataset, enable GPU, and Run All.")


if __name__ == "__main__":
    make_notebook()