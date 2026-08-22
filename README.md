# Intelligent ECG Analysis Tool

Signal-to-Report and Signal-to-Diagnosis with Deep Learning — full implementation
of the project plan (Steps 1–6): data loading, frozen encoder feature extraction,
a classifier head, an adapter+BART report generator, a Gradio web app, and
systematic evaluation.

This code is a **complete, runnable skeleton**. It was written without internet
access to PhysioNet/Hugging Face, so nothing has been trained or tested against
real data yet — that's the part you'll do. Everything is wired so it runs
end-to-end once you plug in the dataset and (ideally) the real HuBERT-ECG
checkpoint.

```
ecg-project/
├── README.md                  <- you are here
├── requirements.txt
├── run_pipeline.py            <- one command to run everything in order
├── src/
│   ├── config.py               <- ALL paths & hyperparameters live here
│   ├── data.py                 <- Step 1: PTB-XL loading, splits, plotting
│   ├── encoder.py               <- Step 2: HuBERT-ECG wrapper (+ fallback encoder)
│   ├── datasets.py              <- feature caching + PyTorch Dataset classes
│   ├── classifier.py            <- Step 3: classification head + training
│   ├── report_generator.py      <- Step 4: adapter + BART + training
│   ├── evaluate_full.py         <- Step 6: per-diagnosis table + failure analysis
│   └── app.py                   <- Step 5: Gradio web interface
├── data/                        <- put PTB-XL here (see below)
├── checkpoints/                 <- trained models get saved here
└── outputs/                     <- plots, tables, predictions get saved here
```

---

## 0. Time-crunch quick path

If you have very little time, do these 5 things in order and you'll have a
working, demoable, evaluated tool:

1. `pip install -r requirements.txt`
2. Download PTB-XL, unzip into `data/ptbxl/` (Section 2 below — ~1.7GB, do this first, it's the long pole)
3. `python run_pipeline.py --all` (trains classifier + report generator, evaluates both)
4. `python -m src.app` → open the local URL → screenshot/record it for your demo
5. Check `outputs/` for your evaluation tables — paste `per_diagnosis_performance.csv`
   and `evaluation_summary.txt` straight into your write-up/slides.

Everything below explains each piece in case something breaks.

---

## 1. Environment setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU strongly recommended for Steps 3–4 (Google Colab Pro is the easiest option
if you don't have a local GPU — just `git clone` this folder or upload the zip
and `%cd` into it).

If you're on Colab:
```python
!pip install -r requirements.txt
```

## 2. Download PTB-XL (Step 1)

1. Go to https://physionet.org/content/ptb-xl/ and download the dataset
   (no application needed, just click "Download the ZIP file").
2. Unzip it so that `data/ptbxl/` directly contains `ptbxl_database.csv`,
   `scp_statements.csv`, `records100/`, `records500/`. If the zip creates a
   nested folder (e.g. `ptb-xl-a-large-.../`), move its *contents* up into
   `data/ptbxl/`.
3. Sanity check:
   ```bash
   python -m src.data --explore
   ```
   This is the Step 1 checkpoint: it loads one record, plots all 12 leads to
   `outputs/example_ecg_plot.png`, and prints the label + report text. If this
   works, your data is set up correctly.

## 3. Encoder checkpoint (Step 2) — read this before training

`src/config.py` has:
```python
HUBERT_ECG_MODEL_ID = "Edoardo-BS/hubert-ecg-base"
```
This is a **placeholder**. HuBERT-ECG's public checkpoint id on the Hugging
Face Hub may have moved by the time you read this. Before training:

1. Search "HuBERT-ECG" on https://huggingface.co/models
2. Copy the correct repo id (e.g. `something/hubert-ecg-base`) into
   `HUBERT_ECG_MODEL_ID` in `src/config.py`.
3. Also check the paper (Coppola et al., 2024) or its GitHub repo, linked
   from the project plan, for the official weights location if the Hub
   search doesn't turn it up directly.

**If you can't get the real checkpoint in time:** `src/encoder.py` automatically
falls back to a small, untrained CNN+Transformer encoder with the same
interface, so `run_pipeline.py --all` still works end-to-end without crashing.
This is enough to demo the *system* working, but the classifier/report
quality will be poor since the encoder isn't pre-trained on ECGs. Swap in the
real checkpoint as soon as you can and just re-run — no other code changes
needed, since everything downstream only depends on `encoder.encode()`'s
output shape.

Test the encoder on its own:
```bash
python -m src.encoder --test
```

## 4. Train the classifier (Step 3)

```bash
python -m src.classifier --train
python -m src.classifier --evaluate
```
- First run extracts and caches encoder features for every record to
  `data/features_cache/` (one `.npy` per record) — this is slow the first
  time, instant afterward.
- Trains a 2-layer MLP head on the 5 PTB-XL superclasses (NORM, MI, STTC, CD, HYP).
- Saves the best checkpoint (by validation macro-AUROC) to
  `checkpoints/classifier_head.pt`.
- `--evaluate` prints per-class and macro AUROC on the official test split.

To expand beyond the 5 superclasses to the full 71 SCP codes later, edit
`config.SUPERCLASSES` and `data.build_label_matrix` — the rest of the
pipeline doesn't need to change.

## 5. Train the report generator (Step 4)

```bash
python -m src.report_generator --train
python -m src.report_generator --evaluate
```
- Loads `facebook/bart-base` from Hugging Face, applies LoRA (lightweight
  fine-tuning — set `USE_LORA = False` in config.py to fully fine-tune instead).
- The `FeatureAdapter` bridges ECG-encoder-space to BART-hidden-space.
- Saves to `checkpoints/report_generator.pt`.
- `--evaluate` computes BLEU/ROUGE on the test split and saves
  `outputs/test_predictions.csv` (generated vs. ground-truth reports,
  side by side — feeds directly into the app's comparison feature).

This step is the slowest and most memory-hungry. If you're tight on time or
GPU memory, drop `REPORT_GEN_EPOCHS` in `config.py` to 2–3 for a quick,
demoable (if imperfect) model, then increase later if time allows.

## 6. Launch the web app (Step 5)

```bash
python -m src.app
```
Opens a Gradio app with two tabs:
- **Test-set record**: pick any PTB-XL test record from a dropdown, see the
  12-lead plot, predicted diagnosis + per-class confidence, and the generated
  report side-by-side with the cardiologist's ground truth.
- **Upload your own**: drop in a CSV (`n_samples x 12`, no header) for a
  custom signal.

To get a shareable public link for a live demo, change the last line of
`src/app.py` to `demo.launch(share=True)`.

Polish features already built in (from the plan's Step 5 table): per-lead
grid visualization, confidence display, predicted-vs-real report comparison,
CSV upload. Not yet built (add if you have time): attention/saliency map,
model-variant dropdown, PDF export, batch mode — see "Extending it" below.

## 7. Full evaluation & error analysis (Step 6)

```bash
python -m src.evaluate_full
```
Produces, in `outputs/`:
- `per_diagnosis_performance.csv` — AUROC per class, sorted best→worst, tagged
  rhythm/conduction vs. morphology (as suggested in the plan's narrative).
- `failure_cases.csv` — confidently-wrong test cases (record id, true vs.
  predicted diagnosis, confidence).
- `evaluation_summary.txt` — an auto-written paragraph summarizing the
  best/worst classes and the rhythm-vs-morphology pattern, ready to paste
  into your report or slides (edit/expand it in your own words afterward).

## 8. Run everything in one go

```bash
python run_pipeline.py --all
```
Runs Steps 1, 2 (checkpoints), 3 (train+eval), 4 (train+eval), 6, in order.
Step 5 (the app) is interactive so it's launched separately:
`python -m src.app`.

Or run individual steps: `python run_pipeline.py --step 3 4`

---

## Extending it (if you have extra time)

- **Attention/saliency map**: capture attention weights from the classifier
  head's input gradient (`x.requires_grad_(True)`, backward on the predicted
  class logit, plot `abs(x.grad)` over the signal) and overlay on the 12-lead
  plot in `app.py`.
- **PDF export**: use the `pdf` skill / `reportlab` to turn the report string
  + plot into a one-click downloadable PDF from the app.
- **Batch mode**: loop `predict_single` + `generate_single` over multiple
  uploaded records and render a summary `pandas` table in Gradio.
- **Full 71 SCP codes**: change `config.SUPERCLASSES` to the full code list
  from `scp_statements.csv` and re-run Step 3.
- **MIMIC-IV-ECG external validation**: once you have PhysioNet credentialed
  access, point a second `PTBXL_DIR`-style config at it and re-run
  `evaluate_full.py` against that split to test generalization.

## Troubleshooting

- **`FileNotFoundError: ptbxl_database.csv`** → PTB-XL isn't unzipped into
  `data/ptbxl/` correctly; see Section 2.
- **HuBERT-ECG fails to load** → expected if the placeholder model id in
  `config.py` is stale; find the correct id (Section 3) or proceed with the
  automatic fallback encoder to keep moving.
- **Out of memory during Step 4** → lower `REPORT_GEN_BATCH_SIZE` in
  `config.py`, or switch `SAMPLING_RATE` to 100 (already the default) rather
  than 500.
- **`bleu`/`rouge` metric download fails (offline)** → `pip install
  rouge-score nltk` and the `evaluate` library will compute them locally
  without needing external downloads for these two metrics specifically.

Good luck — the architecture and checkpoints in this plan already map
directly onto the checkpoints described in `Project_Plan.pdf` §9, so you can
report progress against that table as you go.
