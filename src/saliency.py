"""
Step 6 (explainability): gradient-based saliency maps for the classifier.

Answers "why did the model predict this diagnosis?" by backpropagating the
predicted class's logit back to the raw input signal (Simonyan et al. vanilla
gradients). Large |gradient| at a (timestep, lead) means that point had a
large effect on the model's decision -- rendered as a heatmap overlay on the
12-lead plot in app.py so a clinician can sanity-check the model against
where they'd expect it to be looking (e.g. ST segment, QRS complex).

ECGEncoder.encode() and classifier.predict_single() are both @torch.no_grad()
by design (fast, memory-light normal inference) so they cannot be reused here.
This module runs its own gradient-enabled forward pass instead, mirroring
ECGEncoder.encode()'s internals exactly.

Usage:
    python -m src.saliency --test
"""
import numpy as np
import torch

from src import config
from src.encoder import ECGEncoder, preprocess_signal
from src.classifier import ClassifierHead


def compute_saliency(signal: np.ndarray, encoder: ECGEncoder, classifier: ClassifierHead,
                      target_class: str = None) -> dict:
    """
    Args:
        signal: (T, n_leads) raw ECG waveform, same input predict_single() takes.
        encoder, classifier: already-loaded, eval()-mode objects (e.g. app.py's
                              module-level _encoder / _classifier).
        target_class: one of config.SUPERCLASSES to explain. If None, uses the
                      model's own top predicted class.

    Returns:
        {
          "saliency": (T, n_leads) float32 np.ndarray, abs(d(logit)/d(input)),
          "target_class": str,
          "target_index": int,
          "confidence": float,  # sigmoid probability of target_class
        }
    """
    device = encoder.device
    signal_norm = preprocess_signal(signal)  # plain numpy fn -- runs before any tensor/graph exists

    encoder.model.eval()
    classifier.eval()

    if encoder.is_fallback:
        # Mirrors encoder.py's signal_to_tensor(): (T, n_leads) -> (1, n_leads, T)
        x = torch.tensor(signal_norm, dtype=torch.float32).T.contiguous().unsqueeze(0).to(device)
        x.requires_grad_(True)
        out = encoder.model(x)          # (1, L, d)
        feats = out.squeeze(0)          # (L, d)
    else:
        # Mirrors ECGEncoder.encode()'s real-model path exactly:
        # x_2d = torch.tensor(signal.T, ...) -> (n_leads, T)
        x = torch.tensor(signal_norm.T, dtype=torch.float32).contiguous().to(device)
        x.requires_grad_(True)
        out = encoder.model(x).last_hidden_state   # (n_leads, L, d)
        feats = out.mean(dim=0)                     # (L, d)

    pooled = feats.mean(dim=0, keepdim=True)   # (1, d)
    logits = classifier(pooled)                 # (1, num_classes)
    probs = torch.sigmoid(logits)[0]             # (num_classes,)

    if target_class is None:
        target_idx = int(torch.argmax(probs).item())
    else:
        target_idx = config.SUPERCLASSES.index(target_class)

    # torch.autograd.grad (not logits.backward()): classifier is a shared global
    # module with requires_grad=True params (unlike the frozen encoder). A plain
    # .backward() would populate .grad on every classifier parameter on that
    # shared object -- mutable global state with a race window under concurrent
    # Gradio requests. autograd.grad computes the gradient only for the named
    # input tensor `x` and never touches any nn.Module parameter's .grad.
    grad, = torch.autograd.grad(outputs=logits[0, target_idx], inputs=x)

    grad_np = grad.detach().cpu().numpy()
    if encoder.is_fallback:
        grad_np = grad_np.squeeze(0)             # (n_leads, T)
    saliency = np.abs(grad_np).T.astype(np.float32)  # -> (T, n_leads), matches raw signal's own layout

    return {
        "saliency": saliency,
        "target_class": config.SUPERCLASSES[target_idx],
        "target_index": target_idx,
        "confidence": float(probs[target_idx].item()),
    }


def _test():
    """Quick sanity check without booting the full Gradio app."""
    import os
    from src.data import load_full_dataset, load_raw_signal

    print("\n" + "=" * 60)
    print("  Saliency map sanity check")
    print("=" * 60 + "\n")

    encoder = ECGEncoder()
    classifier = ClassifierHead(in_dim=encoder.feature_dim).to(config.DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_head.pt")
    if os.path.exists(ckpt_path):
        classifier.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    classifier.eval()

    data = load_full_dataset()
    # Use a record ID known to have its .dat/.hea waveform files present locally
    # (see the curated _record_choices list in app.py) rather than .iloc[0],
    # which may point at a record whose waveform was never downloaded.
    row = data["test"].loc[334] if 334 in data["test"].index else data["test"].iloc[0]
    signal, _ = load_raw_signal(row)
    print(f"  Raw signal shape: {signal.shape}")

    result = compute_saliency(signal, encoder=encoder, classifier=classifier)
    saliency = result["saliency"]

    assert saliency.shape == signal.shape, (
        f"FAIL: saliency shape {saliency.shape} != signal shape {signal.shape}"
    )
    print(f"  [OK] saliency shape matches signal shape: {saliency.shape}")
    print(f"  target_class = {result['target_class']}  (confidence={result['confidence']:.4f})")
    print(f"  saliency min={saliency.min():.6g}  max={saliency.max():.6g}  mean={saliency.mean():.6g}")
    print("\n" + "=" * 60)
    print("  CHECKPOINT PASSED [OK]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the saliency sanity check")
    args = parser.parse_args()
    if args.test:
        _test()
