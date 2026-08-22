"""
Step 4: Adapter + BART decoder that turns encoder feature sequences into
free-text clinical reports.

The adapter is a small MLP that projects encoder feature vectors (dim d)
into BART's hidden size, then feeds them to BART as "encoder_outputs" so
BART's own cross-attention and decoder do the language generation.

Usage:
    python -m src.report_generator --train
    python -m src.report_generator --evaluate
    python -m src.report_generator --generate --ecg_id 1
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

from src import config
from src.data import load_full_dataset, load_raw_signal
from src.encoder import ECGEncoder
from src.datasets import ECGFeatureReportDataset, precompute_features, collate_report_batch


class FeatureAdapter(nn.Module):
    """Projects (L, encoder_dim) ECG features into (L, bart_hidden_dim)."""

    def __init__(self, in_dim, out_dim, hidden_dim=config.ADAPTER_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)  # (B, L, out_dim)


class ECGReportModel(nn.Module):
    """Wraps the adapter + BART so encoder features go in, token logits come out."""

    def __init__(self, encoder_dim, bart_model_id=config.BART_MODEL_ID, use_lora=config.USE_LORA):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_id)
        bart_hidden = self.bart.config.d_model
        self.adapter = FeatureAdapter(encoder_dim, bart_hidden)

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                lora_cfg = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"],
                )
                self.bart = get_peft_model(self.bart, lora_cfg)
                print("LoRA applied to BART (lightweight fine-tuning).")
            except Exception as e:
                print(f"[WARNING] Could not apply LoRA ({e}); fine-tuning BART fully instead.")

    def forward(self, encoder_features, attention_mask, labels=None):
        adapted = self.adapter(encoder_features)  # (B, L, bart_hidden)
        encoder_outputs = BaseModelOutput(last_hidden_state=adapted)
        out = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    @torch.no_grad()
    def generate(self, encoder_features, attention_mask, tokenizer, num_beams=4, max_length=config.MAX_REPORT_TOKENS):
        adapted = self.adapter(encoder_features)
        encoder_outputs = BaseModelOutput(last_hidden_state=adapted)
        gen_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def train():
    torch.manual_seed(config.RANDOM_SEED)
    data = load_full_dataset()
    encoder = ECGEncoder()

    report_col = "report" if "report" in data["train"].columns else None
    if report_col is None:
        raise ValueError(
            "No 'report' column found in PTB-XL metadata. "
            "Check ptbxl_database.csv for the free-text report field name."
        )

    print("Pre-computing encoder features (reused from classifier cache if already run)...")
    precompute_features(data["train"], encoder, desc="train features")
    precompute_features(data["val"], encoder, desc="val features")

    tokenizer = BartTokenizerFast.from_pretrained(config.BART_MODEL_ID)

    train_ds = ECGFeatureReportDataset(data["train"], tokenizer, report_col=report_col)
    val_ds = ECGFeatureReportDataset(data["val"], tokenizer, report_col=report_col)
    train_loader = DataLoader(train_ds, batch_size=config.REPORT_GEN_BATCH_SIZE, shuffle=True,
                               collate_fn=collate_report_batch)
    val_loader = DataLoader(val_ds, batch_size=config.REPORT_GEN_BATCH_SIZE, shuffle=False,
                             collate_fn=collate_report_batch)

    model = ECGReportModel(encoder_dim=encoder.feature_dim).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.REPORT_GEN_LR)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "report_generator.pt")

    for epoch in range(config.REPORT_GEN_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.REPORT_GEN_EPOCHS}"):
            feats = batch["encoder_features"].to(config.DEVICE)
            attn = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

            optimizer.zero_grad()
            out = model(feats, attn, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item() * feats.size(0)

        avg_train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["encoder_features"].to(config.DEVICE)
                attn = batch["attention_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
                out = model(feats, attn, labels=labels)
                val_loss += out.loss.item() * feats.size(0)
        avg_val_loss = val_loss / len(val_ds)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")

    print(f"Training done. Best val loss: {best_val_loss:.4f}")


def evaluate_test():
    """Compute BLEU / ROUGE on the held-out test split."""
    import evaluate as hf_evaluate

    data = load_full_dataset()
    encoder = ECGEncoder()
    report_col = "report"
    precompute_features(data["test"], encoder, desc="test features")

    tokenizer = BartTokenizerFast.from_pretrained(config.BART_MODEL_ID)
    test_ds = ECGFeatureReportDataset(data["test"], tokenizer, report_col=report_col)
    test_loader = DataLoader(test_ds, batch_size=config.REPORT_GEN_BATCH_SIZE, shuffle=False,
                              collate_fn=collate_report_batch)

    model = ECGReportModel(encoder_dim=encoder.feature_dim).to(config.DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "report_generator.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    bleu = hf_evaluate.load("bleu")
    rouge = hf_evaluate.load("rouge")

    predictions, references = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reports"):
            feats = batch["encoder_features"].to(config.DEVICE)
            attn = batch["attention_mask"].to(config.DEVICE)
            preds = model.generate(feats, attn, tokenizer)
            predictions.extend(preds)
            references.extend(batch["report_text"])

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_score = rouge.compute(predictions=predictions, references=references)

    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"ROUGE: {rouge_score}")

    # Save a side-by-side comparison for the "predicted vs real report" app feature
    out_path = os.path.join(config.OUTPUT_DIR, "test_predictions.csv")
    import pandas as pd
    pd.DataFrame({"predicted": predictions, "ground_truth": references}).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    return bleu_score, rouge_score


@torch.no_grad()
def generate_single(signal, encoder=None, model=None, tokenizer=None):
    """Used by the Gradio app: raw signal -> generated report string."""
    if encoder is None:
        encoder = ECGEncoder()
    if tokenizer is None:
        tokenizer = BartTokenizerFast.from_pretrained(config.BART_MODEL_ID)
    if model is None:
        model = ECGReportModel(encoder_dim=encoder.feature_dim).to(config.DEVICE)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "report_generator.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        model.eval()

    feats = encoder.encode(signal).unsqueeze(0).to(config.DEVICE)      # (1, L, d)
    attn = torch.ones(feats.shape[:2], dtype=torch.long).to(config.DEVICE)
    report = model.generate(feats, attn, tokenizer)[0]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--ecg_id", type=int, default=None)
    args = parser.parse_args()

    if args.train:
        train()
    if args.evaluate:
        evaluate_test()
    if args.generate:
        data = load_full_dataset()
        row = data["full"].loc[args.ecg_id] if args.ecg_id else data["full"].iloc[0]
        signal, _ = load_raw_signal(row)
        report = generate_single(signal)
        print(f"Generated report: {report}")
        print(f"Ground truth: {row.get('report', 'N/A')}")
