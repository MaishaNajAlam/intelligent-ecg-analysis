"""
PyTorch Dataset wrappers used by both the classifier (Step 3) and the
report generator (Step 4). Features are extracted once through the frozen
encoder and cached to disk (data/features_cache/) so repeated epochs don't
re-run the encoder forward pass every time.
"""
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src import config
from src.data import load_raw_signal


def _cache_path(ecg_id):
    return os.path.join(config.FEATURES_CACHE_DIR, f"{ecg_id}.npy")


def precompute_features(df, encoder, desc="Extracting features"):
    """Run every record in df through the frozen encoder once, caching to disk."""
    for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        cpath = _cache_path(ecg_id)
        if os.path.exists(cpath):
            continue
        signal, _ = load_raw_signal(row)
        feats = encoder.encode(signal)  # (L, d)
        np.save(cpath, feats.numpy())


class ECGFeatureClassificationDataset(Dataset):
    """(cached feature, multi-hot label) pairs for Step 3."""

    def __init__(self, df):
        self.df = df.reset_index() if "ecg_id" not in df.columns else df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = row["ecg_id"] if "ecg_id" in row else row.name
        feats = np.load(_cache_path(ecg_id))          # (L, d)
        pooled = feats.mean(axis=0)                    # (d,)  simple mean-pool over time
        label = np.array(row["label_vector"], dtype=np.float32)
        return torch.tensor(pooled, dtype=torch.float32), torch.tensor(label)


class ECGFeatureReportDataset(Dataset):
    """(cached feature sequence, tokenized report) pairs for Step 4."""

    def __init__(self, df, tokenizer, max_len=config.MAX_REPORT_TOKENS, report_col="report"):
        self.df = df.reset_index() if "ecg_id" not in df.columns else df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.report_col = report_col
        # Drop rows with empty/NaN reports — Step 4 needs text targets
        self.df = self.df[self.df[report_col].notna() & (self.df[report_col].str.strip() != "")]
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = row["ecg_id"] if "ecg_id" in row else row.name
        feats = np.load(_cache_path(ecg_id))  # (L, d)

        target_text = str(row[self.report_col])
        tok = self.tokenizer(
            target_text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "encoder_features": torch.tensor(feats, dtype=torch.float32),
            "labels": tok["input_ids"].squeeze(0),
            "report_text": target_text,
        }


def collate_report_batch(batch):
    """Pad variable-length encoder feature sequences within a batch."""
    max_len = max(item["encoder_features"].shape[0] for item in batch)
    d = batch[0]["encoder_features"].shape[1]

    feats = torch.zeros(len(batch), max_len, d)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["report_text"] for item in batch]

    for i, item in enumerate(batch):
        L = item["encoder_features"].shape[0]
        feats[i, :L] = item["encoder_features"]
        attn_mask[i, :L] = 1

    return {
        "encoder_features": feats,
        "attention_mask": attn_mask,
        "labels": labels,
        "report_text": texts,
    }
