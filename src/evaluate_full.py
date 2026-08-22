"""
Step 6: Systematic evaluation and error analysis.

Produces:
  outputs/per_diagnosis_performance.csv  - AUROC per class, sorted best-to-worst
  outputs/failure_cases.csv              - confidently-wrong test cases
  outputs/evaluation_summary.txt         - written paragraph-style summary

Usage:
    python -m src.evaluate_full
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src import config
from src.data import load_full_dataset
from src.encoder import ECGEncoder
from src.datasets import ECGFeatureClassificationDataset, precompute_features
from src.classifier import ClassifierHead

# Diagnoses grouped by type, used for the "rhythm vs morphology" breakdown
# mentioned in the project plan. Extend this mapping if you expand beyond
# the 5 superclasses to the full 71 SCP codes.
DIAGNOSIS_TYPE = {
    "NORM": "baseline",
    "MI": "morphology",
    "STTC": "morphology",
    "CD": "rhythm/conduction",
    "HYP": "morphology",
}


def run_classifier_error_analysis():
    data = load_full_dataset()
    encoder = ECGEncoder()
    precompute_features(data["test"], encoder, desc="test features")

    test_ds = ECGFeatureClassificationDataset(data["test"])
    test_loader = DataLoader(test_ds, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=False)

    model = ClassifierHead(in_dim=encoder.feature_dim).to(config.DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(config.DEVICE)).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    # --- Per-diagnosis performance table ---
    rows = []
    for i, cls in enumerate(config.SUPERCLASSES):
        if len(np.unique(labels[:, i])) < 2:
            continue
        auroc = roc_auc_score(labels[:, i], probs[:, i])
        rows.append({
            "diagnosis": cls,
            "type": DIAGNOSIS_TYPE.get(cls, "unknown"),
            "AUROC": round(auroc, 4),
            "n_positive": int(labels[:, i].sum()),
        })
    perf_df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    perf_path = os.path.join(config.OUTPUT_DIR, "per_diagnosis_performance.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"Saved per-diagnosis performance table to {perf_path}")
    print(perf_df.to_string(index=False))

    # --- Failure case list: confidently wrong predictions ---
    test_ids = test_ds.df["ecg_id"].values if "ecg_id" in test_ds.df.columns else test_ds.df.index.values
    failure_rows = []
    for row_idx in range(len(labels)):
        wrong_mask = preds[row_idx] != labels[row_idx]
        if not wrong_mask.any():
            continue
        confidence = np.max(np.abs(probs[row_idx] - 0.5)) * 2  # 0..1, how far from the decision boundary
        if confidence < 0.5:
            continue  # only keep "confidently wrong" cases
        true_classes = [config.SUPERCLASSES[i] for i in range(config.NUM_CLASSES) if labels[row_idx, i] == 1]
        pred_classes = [config.SUPERCLASSES[i] for i in range(config.NUM_CLASSES) if preds[row_idx, i] == 1]
        failure_rows.append({
            "ecg_id": test_ids[row_idx],
            "true_diagnosis": ", ".join(true_classes) or "none",
            "predicted_diagnosis": ", ".join(pred_classes) or "none",
            "confidence": round(float(confidence), 3),
        })
    failure_df = pd.DataFrame(failure_rows).sort_values("confidence", ascending=False)
    failure_path = os.path.join(config.OUTPUT_DIR, "failure_cases.csv")
    failure_df.to_csv(failure_path, index=False)
    print(f"\nSaved {len(failure_df)} confidently-wrong failure cases to {failure_path}")

    # --- Written summary paragraph ---
    if len(perf_df) > 0:
        best = perf_df.iloc[0]
        worst = perf_df.iloc[-1]
        rhythm_scores = perf_df[perf_df.type == "rhythm/conduction"]["AUROC"]
        morph_scores = perf_df[perf_df.type == "morphology"]["AUROC"]
        summary = (
            f"Evaluation summary\n"
            f"===================\n\n"
            f"The classifier performs best on '{best.diagnosis}' (AUROC={best.AUROC}) "
            f"and worst on '{worst.diagnosis}' (AUROC={worst.AUROC}).\n\n"
        )
        if len(rhythm_scores) and len(morph_scores):
            summary += (
                f"Rhythm/conduction diagnoses averaged AUROC={rhythm_scores.mean():.4f}, "
                f"while morphology-defined diagnoses averaged AUROC={morph_scores.mean():.4f}. "
            )
            if rhythm_scores.mean() > morph_scores.mean():
                summary += (
                    "This matches the pattern described in the project plan: rhythm abnormalities "
                    "tend to be easier to detect from waveform shape alone, while morphology-based "
                    "diagnoses (e.g. infarction, hypertrophy) require finer-grained amplitude/interval "
                    "cues that a frozen, general-purpose encoder may under-represent.\n\n"
                )
            else:
                summary += "\n\n"
        summary += (
            f"{len(failure_df)} confidently-wrong test cases were identified "
            f"(confidence > 0.5 away from the decision boundary, prediction wrong). "
            f"See failure_cases.csv for the full list and per_diagnosis_performance.csv "
            f"for the sorted performance table.\n"
        )
        summary_path = os.path.join(config.OUTPUT_DIR, "evaluation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"\nSaved written summary to {summary_path}")
        print("\n" + summary)


if __name__ == "__main__":
    run_classifier_error_analysis()
