# ECG Project: Notebook Review + Train-Once/Run-Anywhere Deployment Plan

## Context

The user built `notebook/ECG_project.ipynb` (mirrored by the `src/` package) implementing
`Project_Plan.md`'s 6-step build: PTB-XL loading → frozen HuBERT-ECG encoder → classifier
head → adapter+BART report generator → Gradio app → evaluation. They run it on Kaggle.

Two problems to solve, purely as a plan (no code changes yet):
1. **Verify completeness** — does the notebook actually satisfy each step's checkpoint from
   the project plan, and what's missing or risky?
2. **Deployment pain** — today, opening the Gradio app requires re-running the *entire*
   multi-hour pipeline (data load → train classifier → train BART+LoRA → evaluate) in the
   same Kaggle session, and if the Kaggle draft resets, the trained checkpoints and the app
   both vanish. The user wants: train once, then open the web app anytime without retraining.
   They confirmed they only need this to work **locally on their own machine** (no public
   hosting needed), and they already have a Hugging Face account.

This document is the analysis + the plan. Nothing has been implemented yet.

---

## Part 1 — Notebook vs. `Project_Plan.md`: what's actually done

Verified by reading every cell of `notebook/ECG_project.ipynb` end-to-end and comparing
against each step's to-do list and checkpoint in `Project_Plan.md` §6.

| Step | Plan requirement | Status | Notes |
|---|---|---|---|
| **1. Env & data** | Load PTB-XL, plot 12 leads, print label+report, load official splits | ✅ Done | `data.py` cells (4, 6, 8) match the checkpoint exactly: auto-detects the Kaggle input path, uses the official `strat_fold` splits (1–8 train / 9 val / 10 test). |
| **2. Encoder** | Load frozen HuBERT-ECG, confirm zero-grad, inspect feature shape, batch-encode & cache | ✅ Done, with one now-resolved risk | `ECGEncoder` wraps the real model correctly (see risk note below) and has an untrained CNN+Transformer **fallback** if the download fails, so the pipeline never hard-crashes. |
| **3. Classifier** | 1–2 linear layers on frozen features, BCE loss, per-class + macro AUROC on test | ✅ Done | 5 superclasses (NORM/MI/STTC/CD/HYP) as the plan recommends starting with. Confidence via sigmoid. Best checkpoint saved by val AUROC. |
| **4. Report generator** | Adapter + BART, fine-tune on (features, report) pairs, beam search, BLEU/ROUGE | ✅ Done | `FeatureAdapter` → BART cross-attention via `encoder_outputs`, LoRA fine-tuning (only q/v proj, ~2% of params), beam search generation, BLEU+ROUGE computed and predictions saved to CSV for the app's comparison view. |
| **5. Web interface** | Plot, label+confidence, report, upload, ≥2 polish features | 🟡 Mostly done | Has: 12-lead plot, dropdown over test records, CSV upload tab, label+confidence, **predicted-vs-ground-truth side-by-side** (this alone satisfies "≥2 extra features" together with per-lead viz + upload). **Missing** vs. the plan's polish table: attention/saliency map, model-variant selector, PDF export, batch mode, and a real confidence *gauge* widget (currently a markdown text list). |
| **6. Evaluation** | Per-class AUROC table, BLEU/ROUGE, failure case list, rhythm-vs-morphology pattern, written summary | ✅ Done | Produces `per_diagnosis_performance.csv`, `failure_cases.csv` (confidently-wrong cases), `evaluation_summary.txt` (auto-written paragraph). **Caveat**: with only the 5 superclasses, `CD` is used as a lone stand-in "proxy" for the rhythm/conduction category — this is honestly commented in the code, but it means the rhythm-vs-morphology comparison is thin (n=1 class) until you expand to full SCP codes. |
| PTB-XL+ (supporting dataset) | Optional: interval/amplitude measurements for feature analysis | ❌ Not used | Not required by the checkpoints; would strengthen error analysis (e.g., "did the model fail on borderline QRS/QT cases?"). |
| MIMIC-IV-ECG (external validation) | Optional, credentialed | ❌ Not used | Requires PhysioNet credentialing; explicitly optional in the plan. |

### Key findings worth your attention

1. **HuBERT-ECG checkpoint — verified real, not a stale placeholder.** I checked
   `Edoardo-BS/hubert-ecg-base` on Hugging Face directly: it exists, is not gated, is under
   CC BY-NC 4.0, and its own model card shows the exact same loading pattern the code
   uses (`import hubert_ecg; AutoModel.from_pretrained(..., trust_remote_code=True)`).
   README's "this is a placeholder, may be stale" warning can be downgraded — it's correct.
   Only remaining requirement: **internet must be turned ON in the Kaggle notebook's
   settings panel**, or the download silently fails and triggers the fallback path.

2. **Biggest real risk: the notebook can silently train on garbage features.**
   `src/config.py` sets `FALLBACK_GUARD = True` (fails loudly if the real encoder can't
   load), but the **notebook's config cell (cell 4) sets `FALLBACK_GUARD = False`**. If the
   real HuBERT-ECG download fails for any reason (internet toggle off, transient network
   error, HF rate limit) during a real training run, Steps 3–4 will train for hours on an
   *untrained, random* CNN+Transformer encoder and produce a plausible-looking but
   meaningless classifier/report — with only an easy-to-miss printed warning, no error.
   **Fix before your next real training run:** flip `FALLBACK_GUARD = True` in the
   notebook's config cell (cell 4) so this fails fast instead of wasting a multi-hour run.

3. **Step 5 satisfies its checkpoint but skips the plan's most narratively valuable
   feature.** `Project_Plan.md` §8 ("Framing: Suggested Narrative") explicitly calls out
   *"Show the attention map"* as part of the strongest presentation arc. It's not built yet
   (README already flags it as future work). This is the highest-value optional addition.

4. **CD-as-rhythm-proxy is a known simplification, not a bug** — but if your final
   writeup leans on the plan's "rhythm vs. morphology" narrative (§8, step 5), it will be a
   one-class-vs-three-class comparison, which is a weak version of that story. Expanding to
   the full 71 SCP codes (mentioned as a later option in both the plan and the README) would
   make that comparison real.

**Bottom line:** Steps 1, 3, 4, 6 are faithful, checkpoint-passing implementations. Step 2 is
solid and now de-risked. Step 5 passes its checkpoint but has real headroom. Nothing is
half-finished — what's missing is additive polish, not core plumbing.

---

## Part 2 — Deployment: train once on Kaggle, run the app anytime locally

### Why this is happening (the actual mechanism, not just symptoms)

Two separate Kaggle behaviors are combining to cause the pain:

- **Kaggle "draft" sessions are ephemeral.** Interactively running cells in the editor is a
  *draft* session. `/kaggle/working/` (where `CHECKPOINT_DIR` points) is only durably saved
  when you explicitly **"Save Version" → "Save & Run All (Commit)"**, which creates a
  permanent, versioned snapshot whose output files stay downloadable forever — without ever
  re-running anything. If you've only ever hit "Run" on cells in a live draft, that explains
  why everything disappears when the draft resets: it was never committed.
- **The Gradio cell (`demo.launch(share=True)`) is not a hosting mechanism.** It opens a
  temporary tunnel through Gradio's own servers, valid ~72h, but it is only alive while that
  *exact Kaggle kernel process* is running. The moment the kernel stops (timeout, quota,
  closing the tab), the URL dies — by design, not a bug. Kaggle was never meant to be a
  always-on app host; it's a compute environment for the training job.

The real fix is architectural, not a Kaggle setting: **decouple training (heavy, needs GPU,
done occasionally) from serving (light, CPU is fine, needs to be "always available").**
`src/app.py` is already structured the right way for this — the Step 5 cell *loads*
checkpoint files from disk rather than calling `train()` itself. The only reason it currently
still requires a full pipeline run first is that the checkpoint files it loads
(`classifier_head.pt`, `report_generator.pt`) only exist inside that same ephemeral
`/kaggle/working/checkpoints/`.

### The plan: train once → export two small files → run the app locally, anytime

```
┌─────────────── Kaggle (occasional, GPU) ───────────────┐      ┌──────── Your machine (anytime, CPU) ────────┐
│  Steps 1-4 run once → produces:                         │      │  python -m src.app                          │
│    checkpoints/classifier_head.pt   (~1 MB)             │─────▶│    loads frozen HuBERT-ECG (auto-downloaded,│
│    checkpoints/report_generator.pt  (LoRA adapter, small)│      │    cached after first run)                  │
│  "Save & Run All" (Commit) so the run is permanent       │      │    loads the 2 small checkpoint files       │
└──────────────────────────────────────────────────────────┘      │    Gradio serves at http://localhost:7860   │
                                                                    └───────────────────────────────────────────┘
```

Only the classifier head and the BART adapter/LoRA weights are things *you* trained — they're
small (a few MB to a few tens of MB). The HuBERT-ECG encoder itself is frozen and pretrained,
so it's never something you need to "export" — it's downloaded straight from Hugging Face
Hub the same way on Kaggle or on your laptop, and gets cached locally (`~/.cache/huggingface`)
after the first run, so it's a one-time download either way, not a retrain.

### Concrete steps (for when you're ready to execute this)

1. **Do one real, committed training run on Kaggle.**
   - First flip `FALLBACK_GUARD = True` in the notebook's config cell (see Part 1, finding 2)
     so a network hiccup fails loudly instead of quietly wasting the run.
   - Confirm internet is ON in the notebook's settings panel.
   - Run Steps 1–4 (and 6, for your evaluation numbers) through to completion.
   - Use **"Save Version" → "Save & Run All" (Commit)**, not just interactive "Run" — this
     is what makes the run's `/kaggle/working/checkpoints/*.pt` files permanent and
     downloadable from the Kaggle version page indefinitely, independent of future drafts.

2. **Pull the two checkpoint files out of Kaggle, onto your machine (or HF Hub as a backup).**
   Two options, not mutually exclusive:
   - **Simplest:** download `classifier_head.pt` and `report_generator.pt` directly from the
     committed version's Output tab on Kaggle, straight to a local `checkpoints/` folder.
   - **More durable (recommended, since you already have an HF account):** push those same
     two files to a small Hugging Face **model repo** (e.g. `your-username/ecg-report-tool`)
     via `huggingface_hub.upload_file(...)` — one-time, ~5 lines, from inside the Kaggle
     notebook right after training. This protects you from ever losing a good checkpoint to
     Kaggle storage limits/quota resets again, and gives you versioning if you retrain later
     with better hyperparameters. Not required for local-only use, but cheap insurance.

3. **Run the app locally — no training, no Kaggle, anytime.**
   - `src/config.py` already defaults `CHECKPOINT_DIR` to a local `checkpoints/` folder and
     `PTBXL_DIR` to `data/ptbxl/` when `PTBXL_DIR` isn't set as an env var — it's already
     environment-agnostic, unlike the notebook's hardcoded Kaggle-path config cell.
   - `pip install -r requirements.txt` once, locally.
   - Put the two downloaded checkpoint files into `checkpoints/`.
   - `python -m src.app` → opens Gradio on `http://localhost:7860`, every time, instantly,
     with no training step. CPU is sufficient here — inference is one forward pass through a
     ~93M-param frozen encoder + a tiny head, or one beam-search decode through BART-base;
     both are a few seconds on CPU for a single record, which is fine for a demo tool.

4. **One nuance about the "Test-set record" tab specifically:** it depends on the PTB-XL
   metadata + waveform files being present locally (to populate the dropdown and show the
   cardiologist's ground-truth report side-by-side). The "Upload your own CSV" tab does
   **not** need the dataset at all. You have two options, pick based on how much you want the
   full demo experience locally:
   - Download the full PTB-XL dataset locally too (~1.7 GB, one-time, free, same PhysioNet
     link already in the README) — gives full parity with the Kaggle app.
   - Or bundle a small fixed sample (e.g. 20–50 records + their ground-truth reports) into
     the repo itself as tiny CSVs, so the dropdown tab works standalone without the full
     dataset. Lighter-weight, good enough for demos, doesn't need any code you don't already
     have (the CSV-upload code path can be reused to *read* the bundled samples too).

### Why not Hugging Face Spaces (noting the tradeoff, since you already have the account)

You said local-only is enough, so this isn't the plan — but worth knowing the option exists
for later: if you ever want a public, always-on link you don't have to keep your own machine
on for, HF Spaces (free CPU tier, Gradio SDK) is the natural next step *on top of* this same
setup — same `app.py`, same two checkpoint files (now already living in your HF model repo
from step 2 above), just deployed to a Space instead of run locally. Nothing above is wasted
if you change your mind later.

---

## Part 3 — Optional polish phase (roadmap, do after the above works)

Per your preference, these are staged as a later phase, not blocking the deploy work:

- **Attention/saliency map** — highest narrative value per the plan's own suggested framing
  (§8). README already sketches the approach: gradient w.r.t. input, `abs(x.grad)` overlaid
  on the 12-lead plot.
- **Real confidence gauge** — swap the current `gr.Markdown` confidence list for
  `gr.Label(num_top_classes=5)`, which Gradio renders as a bar/gauge natively — this is
  close to a drop-in change, not a new feature to build from scratch.
- **PDF export** — one-click formatted report via `reportlab`, as already sketched in the
  README's "Extending it" section.
- **Batch mode** — loop `predict_single` + `generate_single` over multiple uploaded records,
  render a summary table.
- **Full 71 SCP codes** — makes Step 6's rhythm-vs-morphology comparison a real multi-class
  comparison instead of the current CD-as-single-proxy stand-in.
- **PTB-XL+ / MIMIC-IV-ECG** — only worth it if you want the stronger "generalizes to a
  different hospital" story for the final writeup; both are optional in the plan itself.

---

## Verification (once this is actually executed in a future session)

1. **Notebook fix check:** re-open the notebook, confirm cell 4 has `FALLBACK_GUARD = True`,
   re-run Step 2's checkpoint cell, confirm it prints `is_fallback: False` (i.e. the real
   encoder loaded, not the CNN fallback).
2. **Committed-run check:** on the Kaggle version page (not the draft), confirm
   `checkpoints/classifier_head.pt` and `checkpoints/report_generator.pt` are listed under
   Output and downloadable.
3. **Local serving check:** after placing the two files in a local `checkpoints/` folder and
   running `python -m src.app`, confirm the app opens at `localhost:7860` with **no training
   log output at all** — only "Loading models..." followed by the Gradio URL — and that
   picking a test record or uploading a CSV returns a diagnosis + report within a few
   seconds, entirely offline aside from the one-time HuBERT-ECG/BART download.
4. **Persistence check:** close the terminal, reopen it days later, run `python -m src.app`
   again with no internet at all (after the first-run cache is warm) — it should still work,
   proving training is genuinely decoupled from serving.
