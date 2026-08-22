"""
Step 1: PTB-XL data loading, exploration, and splitting.

Usage:
    python -m src.data --explore
"""
import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

from src import config


def load_ptbxl_metadata():
    """Load the main PTB-XL metadata CSV and parse SCP-code annotations."""
    csv_path = os.path.join(config.PTBXL_DIR, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}.\n"
            "Download PTB-XL from https://physionet.org/content/ptb-xl/ and "
            f"unzip it into {config.PTBXL_DIR} (see README)."
        )
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def load_scp_statements():
    path = os.path.join(config.PTBXL_DIR, "scp_statements.csv")
    scp_df = pd.read_csv(path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]
    return scp_df


def aggregate_diagnostic_superclass(scp_codes, scp_df):
    """Map raw SCP codes -> one or more of the 5 superclasses (NORM, MI, STTC, CD, HYP)."""
    classes = set()
    for code in scp_codes.keys():
        if code in scp_df.index:
            classes.add(scp_df.loc[code].diagnostic_class)
    return list(classes)


def build_label_matrix(df, scp_df):
    """Attach a multi-hot label vector (over config.SUPERCLASSES) to every record."""
    df = df.copy()
    df["superclasses"] = df.scp_codes.apply(lambda codes: aggregate_diagnostic_superclass(codes, scp_df))

    label_matrix = np.zeros((len(df), config.NUM_CLASSES), dtype=np.float32)
    for i, classes in enumerate(df["superclasses"]):
        for c in classes:
            if c in config.SUPERCLASSES:
                label_matrix[i, config.SUPERCLASSES.index(c)] = 1.0
    df["label_vector"] = list(label_matrix)
    return df


def get_record_path(row, sampling_rate=config.SAMPLING_RATE):
    """Return the on-disk path (without extension) for a given metadata row."""
    if sampling_rate == 100:
        rel = row.filename_lr
    else:
        rel = row.filename_hr
    return os.path.join(config.PTBXL_DIR, rel)


def load_raw_signal(row, sampling_rate=config.SAMPLING_RATE):
    """Load one 12-lead ECG as a (n_samples, 12) numpy array."""
    path = get_record_path(row, sampling_rate)
    signal, meta = wfdb.rdsamp(path)
    return signal, meta


def get_official_splits(df):
    """
    PTB-XL ships an official 10-fold split in `strat_fold` (1-10).
    Standard convention: fold 9 = validation, fold 10 = test, 1-8 = train.
    """
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]
    return train_df, val_df, test_df


def load_full_dataset():
    """Convenience: metadata + scp statements + labels + official splits, all in one call."""
    df = load_ptbxl_metadata()
    scp_df = load_scp_statements()
    df = build_label_matrix(df, scp_df)
    train_df, val_df, test_df = get_official_splits(df)
    return {
        "full": df,
        "scp_df": scp_df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def plot_12_lead(signal, title="ECG", save_path=None):
    """Plot all 12 leads stacked vertically, PTB-XL lead order."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(signal[:, i], linewidth=0.8, color="black")
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=20, fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("Samples")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _explore():
    """CHECKPOINT for Step 1: load one record, plot it, print label + report text."""
    data = load_full_dataset()
    df = data["full"]
    row = df.iloc[0]

    signal, meta = load_raw_signal(row)
    print(f"Loaded record {row.name}: shape={signal.shape}, fs={meta['fs']}")
    print(f"Diagnostic superclasses: {row['superclasses']}")
    print(f"Label vector ({config.SUPERCLASSES}): {row['label_vector']}")
    print(f"Free-text cardiologist report: {row.get('report', 'N/A')}")

    train_df, val_df, test_df = get_official_splits(df)
    print(f"Splits -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    out_path = os.path.join(config.OUTPUT_DIR, "example_ecg_plot.png")
    plot_12_lead(signal, title=f"Record {row.name}", save_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true", help="Run the Step 1 checkpoint")
    args = parser.parse_args()
    if args.explore:
        _explore()
