"""
Central configuration for the ECG project.
Edit paths here once you've downloaded PTB-XL.

On Kaggle: paths are auto-detected via environment variables.
On Colab:  mount Drive first, then set PTBXL_DIR below to your Drive path.
Locally:   just set PTBXL_DIR to wherever you unzipped PTB-XL.
"""
import os
import torch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Where you put the PTB-XL data ─────────────────────────────────────
# Kaggle: add the PTB-XL dataset to your notebook; it will appear at
#         /kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset/
# Colab:  download to /content/ptbxl/ or mount Drive
# Local:  wherever you unzipped it
#
# Expected structure inside this folder:
#   ptbxl_database.csv
#   scp_statements.csv
#   records100/   ← 100 Hz waveforms
#   records500/   ← 500 Hz waveforms
PTBXL_DIR = os.environ.get(
    "PTBXL_DIR",
    os.path.join(DATA_DIR, "ptbxl"),   # default: ecg-project/data/ptbxl/
)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
FEATURES_CACHE_DIR = os.path.join(DATA_DIR, "features_cache")

for d in [DATA_DIR, PTBXL_DIR, CHECKPOINT_DIR, OUTPUT_DIR, FEATURES_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------
# Device  (auto-detected: CUDA → MPS → CPU)
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ---------------------------------------------------------------------
# Signal params & Preprocessing
# ---------------------------------------------------------------------
SAMPLING_RATE      = 500   # HuBERT-ECG standard sampling rate (500 Hz = 5000 samples per 10s lead)
                           # Note: If fine-tuning on custom 100Hz models, set to 100 and use records100/
                           # PTB-XL provides both records100/ and records500/
SIGNAL_LENGTH_SEC  = 10
N_LEADS            = 12

# Preprocessing strategy: "zscore" (default per-lead normalization: zero-mean unit-variance),
# "minmax" (scale to [-1, 1] per lead), or "bandpass_minmax" (0.5-50Hz Butterworth filter + [-1, 1] scaling).
# Note: "zscore" matches standard practice, while some community notebooks use "bandpass_minmax".
PREPROCESSING_MODE = "zscore"

# ---------------------------------------------------------------------
# Encoder — HuBERT-ECG  (Coppola et al., 2024)
# ---------------------------------------------------------------------
# GitHub Repo: https://github.com/Edoar-do/HuBERT-ECG.git (owner: Edoar-do)
# Hugging Face Hub organization name: Edoardo-BS
# Verified Hugging Face Hub IDs:
#   small  → "Edoardo-BS/hubert-ecg-small"   feature_dim = 512
#   base   → "Edoardo-BS/hubert-ecg-base"    feature_dim = 768  ← default
#   large  → "Edoardo-BS/hubert-ecg-large"   feature_dim = 1024
# All require trust_remote_code=True (already set in encoder.py).
HUBERT_ECG_MODEL_ID  = "Edoardo-BS/hubert-ecg-base"
ENCODER_FEATURE_DIM  = 768   # must match the variant above

# If HuBERT-ECG cannot be downloaded (offline / network error), encoder.py
# can fall back to a small untrained CNN+Transformer with the same interface.
USE_FALLBACK_ENCODER_IF_UNAVAILABLE = True

# SAFETY FLAG ── set True during real training to RAISE an error rather than
# silently train on random features. Flip to False only for a quick pipeline
# demo when you know the real checkpoint isn't available yet.
FALLBACK_GUARD = True

# ---------------------------------------------------------------------
# Classifier  (Step 3) — 5 PTB-XL superclasses
# ---------------------------------------------------------------------
SUPERCLASSES          = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES           = len(SUPERCLASSES)
CLASSIFIER_HIDDEN_DIM = 256
CLASSIFIER_LR         = 1e-3
CLASSIFIER_EPOCHS     = 30
CLASSIFIER_BATCH_SIZE = 64

# ---------------------------------------------------------------------
# Report generator  (Step 4) — BART + adapter
# ---------------------------------------------------------------------
BART_MODEL_ID       = "facebook/bart-base"
ADAPTER_HIDDEN_DIM  = 512
MAX_REPORT_TOKENS   = 128
REPORT_GEN_LR       = 5e-5
REPORT_GEN_EPOCHS   = 10
REPORT_GEN_BATCH_SIZE = 8
REPORT_GEN_NUM_BEAMS  = 4    # beam search width; higher = better quality but slower
USE_LORA            = True   # LoRA fine-tuning (lightweight); False = full fine-tune

# ---------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------
RANDOM_SEED = 42
