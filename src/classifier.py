"""
Step 3: Train a small classification head on top of frozen encoder features.

Usage:
    python -m src.classifier --train
    python -m src.classifier --evaluate
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src import config
from src.data import load_full_dataset
from src.encoder import ECGEncoder
from src.datasets import ECGFeatureClassificationDataset, precompute_features


class ClassifierHead(nn.Module):
    """One or two linear layers on top of the (pooled) encoder feature vector."""

    def __init__(self, in_dim=config.ENCODER_FEATURE_DIM,
                 hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits; apply sigmoid outside for multi-label


def train():
    torch.manual_seed(config.RANDOM_SEED)
    data = load_full_dataset()
    encoder = ECGEncoder()

    print("Pre-computing (and caching) encoder features for all splits...")
    precompute_features(data["train"], encoder, desc="train features")
    precompute_features(data["val"], encoder, desc="val features")
    precompute_features(data["test"], encoder, desc="test features")

    train_ds = ECGFeatureClassificationDataset(data["train"])
    val_ds = ECGFeatureClassificationDataset(data["val"])
    train_loader = DataLoader(train_ds, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=False)

    model = ClassifierHead(in_dim=encoder.feature_dim).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CLASSIFIER_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")

    for epoch in range(config.CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLASSIFIER_EPOCHS}"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_ds)
        val_auroc = evaluate_loader(model, val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_macro_AUROC={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    print(f"Training done. Best val macro AUROC: {best_auroc:.4f}")


@torch.no_grad()
def evaluate_loader(model, loader, per_class=False):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(config.DEVICE)
        logits = model(x).cpu()
        all_logits.append(logits)
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    aurocs = []
    per_class_scores = {}
    for i, cls in enumerate(config.SUPERCLASSES):
        if len(np.unique(labels[:, i])) < 2:
            continue  # AUROC undefined if only one class present in this split
        score = roc_auc_score(labels[:, i], probs[:, i])
        aurocs.append(score)
        per_class_scores[cls] = score

    macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    if per_class:
        return macro_auroc, per_class_scores
    return macro_auroc


def evaluate_test():
    data = load_full_dataset()
    encoder = ECGEncoder()
    precompute_features(data["test"], encoder, desc="test features")

    test_ds = ECGFeatureClassificationDataset(data["test"])
    test_loader = DataLoader(test_ds, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=False)

    model = ClassifierHead(in_dim=encoder.feature_dim).to(config.DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

    macro_auroc, per_class = evaluate_loader(model, test_loader, per_class=True)
    print(f"Test macro AUROC: {macro_auroc:.4f}")
    for cls, score in per_class.items():
        print(f"  {cls}: {score:.4f}")
    return macro_auroc, per_class


@torch.no_grad()
def predict_single(signal, encoder=None, model=None):
    """Used by the Gradio app: raw signal -> (label, confidence dict)."""
    if encoder is None:
        encoder = ECGEncoder()
    if model is None:
        model = ClassifierHead(in_dim=encoder.feature_dim).to(config.DEVICE)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        model.eval()

    feats = encoder.encode(signal)          # (L, d)
    pooled = feats.mean(dim=0, keepdim=True).to(config.DEVICE)  # (1, d)
    logits = model(pooled)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {cls: float(p) for cls, p in zip(config.SUPERCLASSES, probs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    if args.evaluate:
        evaluate_test()
