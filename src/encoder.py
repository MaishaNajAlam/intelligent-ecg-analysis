"""
Step 2: Load the (frozen) HuBERT-ECG encoder and extract features.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW HuBERT-ECG LOADING WORKS (read this before training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The real HuBERT-ECG uses a custom model class registered with
Hugging Face via the `hubert_ecg` pip package from the authors'
GitHub repo. You need to do TWO things before it loads:

  1. pip install git+https://github.com/Edoar-do/HuBERT-ECG.git
     (this is already in requirements.txt)

  2. import hubert_ecg  ← registers the custom class with AutoModel
     (this file does it automatically)

Expected input format for HuBERT-ECG:
  - Shape: (batch_size, n_leads=12, n_samples)
  - Sampling rate: 500 Hz  (10 s signal → 5000 samples)
  - Normalization: per-lead z-score  (mean=0, std=1)
  - The model config says it was pre-trained at 500 Hz.

  PTB-XL has both 100Hz and 500Hz versions. config.py sets
  SAMPLING_RATE=500 when using the real encoder so the signal
  length matches what the model was trained on.

Fallback path:
  If the real checkpoint cannot be loaded (network error, wrong ID,
  first run before install), a small untrained CNN+Transformer encoder
  is used. It produces valid tensor shapes so the rest of the pipeline
  can be tested, but quality will be poor — it is NOT pre-trained.

Step 2 Checkpoint (from Project_Plan.md §6.2):
  "A raw ECG goes in, a feature tensor of known shape comes out,
  and the encoder's parameters are confirmed frozen (zero gradients)."

Usage:
    python -m src.encoder --test
"""
import os
import numpy as np
import torch
import torch.nn as nn

from src import config


# ──────────────────────────────────────────────────────────────────────
# Fallback encoder (untrained — only for pipeline demos)
# ──────────────────────────────────────────────────────────────────────

class FallbackECGEncoder(nn.Module):
    """
    Small CNN + Transformer used ONLY if HuBERT-ECG cannot be loaded.
    NOT pre-trained — produces valid tensor shapes but random features.
    Swap in the real checkpoint (see README) before reporting any results.
    """

    def __init__(self, in_channels=config.N_LEADS,
                 feature_dim=config.ENCODER_FEATURE_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128,          kernel_size=9,  stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, feature_dim, kernel_size=5,  stride=2, padding=2),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8,
            dim_feedforward=feature_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        # x: (B, n_leads, T)
        h = self.conv(x)          # (B, feature_dim, T')
        h = h.transpose(1, 2)    # (B, T', feature_dim)
        h = self.transformer(h)  # (B, T', feature_dim)
        return h


# ──────────────────────────────────────────────────────────────────────
# Pre-processing helpers (Step 2, To-do item 3)
# ──────────────────────────────────────────────────────────────────────

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply per-lead z-score normalization as required by HuBERT-ECG.

    Args:
        signal: (T, n_leads) numpy array — raw output from wfdb.rdsamp()
    Returns:
        (T, n_leads) numpy array — normalized, same dtype
    """
    # Per-lead normalization: subtract mean, divide by std
    mean = signal.mean(axis=0, keepdims=True)   # (1, n_leads)
    std  = signal.std(axis=0,  keepdims=True)   # (1, n_leads)
    std  = np.where(std < 1e-8, 1.0, std)       # avoid div-by-zero for flat leads
    return ((signal - mean) / std).astype(np.float32)


def signal_to_tensor(signal: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert (T, n_leads) numpy array → (1, n_leads, T) tensor on device.
    This is the input shape expected by HuBERT-ECG and the fallback CNN.
    """
    x = torch.tensor(signal, dtype=torch.float32)  # (T, n_leads)
    x = x.T.unsqueeze(0)                            # (1, n_leads, T)
    return x.to(device)


def batch_signals_to_tensor(signals: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert (B, T, n_leads) numpy array → (B, n_leads, T) tensor on device.
    """
    x = torch.tensor(signals, dtype=torch.float32)  # (B, T, n_leads)
    x = x.permute(0, 2, 1)                          # (B, n_leads, T)
    return x.to(device)


# ──────────────────────────────────────────────────────────────────────
# Main encoder wrapper
# ──────────────────────────────────────────────────────────────────────

class ECGEncoder:
    """
    Unified wrapper around HuBERT-ECG (primary) or FallbackECGEncoder.

    Both expose the same interface:
        encode(signal)        → (L, d) tensor    [single signal]
        encode_batch(signals) → (B, L, d) tensor [batch]

    where L = number of time frames output by the encoder,
          d = feature dimension (768 for hubert-ecg-base).
    """

    def __init__(self, device=config.DEVICE):
        self.device      = device
        self.is_fallback = False
        self.model       = None
        self.feature_dim = config.ENCODER_FEATURE_DIM
        self._load()

    def _load(self):
        try:
            # Step 1 of the loading dance: register HuBERTECG with AutoModel
            # This import does nothing else — it just registers the class.
            try:
                import hubert_ecg  # noqa: F401  ← registers custom class
                print("hubert_ecg package imported (custom class registered).")
            except ImportError:
                print(
                    "[WARNING] hubert_ecg package not found. "
                    "Run:  pip install git+https://github.com/Edoar-do/HuBERT-ECG.git\n"
                    "Falling back to untrained CNN encoder."
                )
                raise  # triggers fallback

            from transformers import AutoModel
            print(f"Loading HuBERT-ECG from '{config.HUBERT_ECG_MODEL_ID}' …")
            self.model = AutoModel.from_pretrained(
                config.HUBERT_ECG_MODEL_ID,
                trust_remote_code=True,
            )
            self.feature_dim = getattr(
                self.model.config, "hidden_size", config.ENCODER_FEATURE_DIM
            )
            print(f"✔  HuBERT-ECG loaded. Feature dim = {self.feature_dim}")

        except Exception as e:
            if not config.USE_FALLBACK_ENCODER_IF_UNAVAILABLE:
                raise

            if config.FALLBACK_GUARD:
                raise RuntimeError(
                    "\n"
                    "══════════════════════════════════════════════════════════\n"
                    "  FALLBACK ENCODER GUARD TRIGGERED\n"
                    "══════════════════════════════════════════════════════════\n"
                    f"  Could not load HuBERT-ECG: {e}\n\n"
                    "  FALLBACK_GUARD = True in config.py, so we refuse to\n"
                    "  continue with the untrained CNN encoder during training.\n"
                    "  Training on random features would produce meaningless results.\n\n"
                    "  Fix options:\n"
                    "  1. Install the hubert_ecg package:\n"
                    "       pip install git+https://github.com/Edoar-do/HuBERT-ECG.git\n"
                    "  2. Make sure you have internet access on Kaggle/Colab.\n"
                    "  3. To test the pipeline shape without the real encoder, set\n"
                    "       FALLBACK_GUARD = False  in src/config.py\n"
                    "══════════════════════════════════════════════════════════\n"
                ) from e

            # FALLBACK_GUARD = False → warn loudly but continue
            print("\n" + "=" * 60)
            print("  ⚠  WARNING: Using UNTRAINED fallback encoder!")
            print(f"  Reason: {e}")
            print("  Results will NOT be meaningful until the real")
            print("  HuBERT-ECG checkpoint is loaded.")
            print("=" * 60 + "\n")
            self.model       = FallbackECGEncoder()
            self.is_fallback = True

        self.model.to(self.device)
        self.model.eval()
        # Freeze all parameters — the encoder is used as a fixed feature extractor
        for p in self.model.parameters():
            p.requires_grad = False

    # ── public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, signal: np.ndarray) -> torch.Tensor:
        """
        Step 2, To-do item 4: pass one ECG signal through the encoder.

        Args:
            signal: (T, n_leads) numpy array from wfdb.rdsamp() or data.load_raw_signal()
                    T = SIGNAL_LENGTH_SEC * SAMPLING_RATE  (e.g. 10 * 500 = 5000)
        Returns:
            (L, d) torch tensor — frozen feature representation
        """
        signal = preprocess_signal(signal)       # per-lead z-score normalization
        x      = signal_to_tensor(signal, self.device)  # (1, n_leads, T)

        if self.is_fallback:
            out = self.model(x)                       # (1, L, d)
        else:
            out = self.model(x).last_hidden_state     # HF models expose this

        return out.squeeze(0).cpu()  # (L, d)

    @torch.no_grad()
    def encode_batch(self, signals: np.ndarray) -> torch.Tensor:
        """
        Step 2, To-do item 5: encode a batch of signals.

        Args:
            signals: (B, T, n_leads) numpy array
        Returns:
            (B, L, d) torch tensor
        """
        signals = np.stack([preprocess_signal(s) for s in signals])  # normalize each
        x       = batch_signals_to_tensor(signals, self.device)       # (B, n_leads, T)

        if self.is_fallback:
            out = self.model(x)
        else:
            out = self.model(x).last_hidden_state
        return out.cpu()

    def confirm_frozen(self) -> bool:
        """Returns True if ALL encoder parameters have requires_grad=False."""
        return all(not p.requires_grad for p in self.model.parameters())


# ──────────────────────────────────────────────────────────────────────
# Step 2 checkpoint test  (run with:  python -m src.encoder --test)
# ──────────────────────────────────────────────────────────────────────

def _test():
    """
    Full Step 2 checkpoint from Project_Plan.md §6.2:

    1. Load the encoder + confirm frozen (To-do 1 & 2)
    2. Preprocess a real PTB-XL ECG signal (To-do 3)
    3. Pass it through → inspect output shape (To-do 4)
    4. Encode a small BATCH and save features to disk (To-do 5)
    """
    import os
    import numpy as np
    from src.data import load_full_dataset, load_raw_signal

    print("\n" + "=" * 60)
    print("  Step 2 Checkpoint — Encoder Feature Extraction")
    print("=" * 60 + "\n")

    # ── 1 & 2: Load encoder, confirm frozen ───────────────────────────
    enc = ECGEncoder()
    frozen = enc.confirm_frozen()
    print(f"  Encoder frozen (zero gradients): {frozen}")
    print(f"  Using fallback encoder         : {enc.is_fallback}")
    print(f"  Feature dimension (d)          : {enc.feature_dim}")
    assert frozen, "FAIL: encoder parameters are not frozen!"

    # ── 3 & 4: Preprocess + single forward pass ────────────────────────
    print("\n  Loading one real ECG from PTB-XL …")
    data   = load_full_dataset()
    row    = data["test"].iloc[0]
    signal, meta = load_raw_signal(row)
    print(f"  Raw signal shape  : {signal.shape}   (T={signal.shape[0]}, leads={signal.shape[1]})")
    print(f"  Sampling rate     : {meta['fs']} Hz")

    # Apply preprocessing (per-lead z-score)
    signal_norm = preprocess_signal(signal)
    print(f"  After normalization: mean={signal_norm.mean():.4f}, std={signal_norm.std():.4f}")

    feats = enc.encode(signal)
    print(f"\n  ✔ Feature tensor shape (L × d): {tuple(feats.shape)}")
    print(f"     L = {feats.shape[0]} time frames")
    print(f"     d = {feats.shape[1]} feature dimensions")

    # ── 5: Batch encoding + save to disk ──────────────────────────────
    print("\n  Encoding a small batch of 4 ECGs …")
    batch_rows    = [data["test"].iloc[i] for i in range(4)]
    batch_signals = np.stack([load_raw_signal(r)[0] for r in batch_rows])  # (4, T, 12)
    batch_feats   = enc.encode_batch(batch_signals)
    print(f"  ✔ Batch feature shape (B × L × d): {tuple(batch_feats.shape)}")

    # Save the 4 feature arrays to disk for inspection
    save_dir = os.path.join(config.OUTPUT_DIR, "step2_feature_samples")
    os.makedirs(save_dir, exist_ok=True)
    for i, row in enumerate(batch_rows):
        np.save(os.path.join(save_dir, f"features_ecg_{row.name}.npy"),
                batch_feats[i].numpy())
    print(f"  ✔ Saved {len(batch_rows)} feature files to {save_dir}/")

    print("\n" + "=" * 60)
    print("  CHECKPOINT PASSED ✔")
    print("  A raw ECG went in, a feature tensor of known shape came out.")
    print(f"  Encoder frozen: {frozen}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run the Step 2 checkpoint")
    args = parser.parse_args()
    if args.test:
        _test()
