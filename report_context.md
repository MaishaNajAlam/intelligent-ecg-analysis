# INTELLIGENT ECG ANALYSIS TOOL — COMPLETE PROJECT CONTEXT & TECHNICAL SPECIFICATION

> **Document Purpose**: This comprehensive context document contains every detail of the *Intelligent ECG Analysis Tool* project—including clinical motivation, theoretical foundations, end-to-end architecture, mathematical definitions, dataset schemas, model parameters, implementation details, user interface mechanics, PDF report generation, error analysis, and deployment strategy. It is designed to provide full context to LLMs, collaborators, researchers, and engineers.

---

## TABLE OF CONTENTS
1. [Project Identity & Clinical Motivation](#1-project-identity--clinical-motivation)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Repository Directory Map & File Inventory](#3-repository-directory-map--file-inventory)
4. [Dataset & Biomedical Signal Pipeline (`PTB-XL`)](#4-dataset--biomedical-signal-pipeline-ptb-xl)
5. [Self-Supervised Feature Extractor (`HuBERT-ECG`)](#5-self-supervised-feature-extractor-hubert-ecg)
6. [Multi-Label Diagnostic Classifier Head](#6-multi-label-diagnostic-classifier-head)
7. [Clinical Report Generator (`Adapter + BART`)](#7-clinical-report-generator-adapter--bart)
8. [Explainability & Gradient-Based Saliency Mapping](#8-explainability--gradient-based-saliency-mapping)
9. [Clinical-Grade PDF Report Engine (`ReportLab + PyMuPDF`)](#9-clinical-grade-pdf-report-engine-reportlab--pymupdf)
10. [Web Application & User Interface (`Gradio`)](#10-web-application--user-interface-gradio)
11. [Systematic Evaluation & Failure Analysis](#11-systematic-evaluation--failure-analysis)
12. [Execution, Training & Deployment Workflows](#12-execution-training--deployment-workflows)
13. [Dependencies & Environment Configuration](#13-dependencies--environment-configuration)
14. [Medical & Diagnostic Reference Glossary](#14-medical--diagnostic-reference-glossary)

---

## 1. PROJECT IDENTITY & CLINICAL MOTIVATION

### 1.1 The Global Problem
Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for an estimated **17.9 million deaths annually** (WHO, 2021). The standard 10-second 12-lead electrocardiogram (ECG) is the most pervasive, non-invasive, cost-effective diagnostic tool in cardiology. 

While recording an ECG takes seconds and requires minimal equipment, **interpreting an ECG requires years of specialized clinical training**. In low-resource settings, rural clinics, and overwhelmed emergency departments, expert cardiologists are often unavailable. Unread or misread ECGs cause diagnostic delays for acute conditions like ST-Elevation Myocardial Infarction (STEMI), life-threatening arrhythmias, and severe conduction blocks.

### 1.2 The Project Mission
This project builds a complete, production-grade, end-to-end deep learning system that takes a raw 12-lead ECG signal and generates:
1. **Signal-to-Diagnosis (Classification)**: Multi-label disease classification with confidence percentages across the 5 primary clinical superclasses.
2. **Signal-to-Report (Natural Language Generation)**: Free-text clinical narrative mimicking how a board-certified cardiologist documents findings (rhythm, axis, morphology, ischemic changes).
3. **Visual Explainability (Saliency Mapping)**: Gradient-based temporal attribution overlaid on all 12 leads, highlighting the exact waveform regions (e.g. ST segment elevation, pathological Q-waves, QRS widening) driving the AI prediction.
4. **Hospital-Grade PDF Export**: One-click generation of an authentic clinical ECG document with patient demographics, boxed primary findings, differential confidence tables, AI narrative, waveform tracings, and physician signature lines.
5. **Interactive Web Application**: A polished Gradio interface with responsive tabbed workflows for test-set review, custom CSV uploads, and batch multi-record processing.

---

## 2. END-TO-END SYSTEM ARCHITECTURE

The system implements a **dual-head multi-modal transformer architecture** anchored by a frozen, pre-trained self-supervised biomedical foundation model.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │      Raw 12-Lead ECG Signal (10 sec @ 500 Hz)            │
                    │      Shape: (12 leads, 5000 samples)                     │
                    └─────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │   Signal Preprocessing (Per-lead Z-score / Bandpass)     │
                    │   Normalized Waveform (12, 5000)                         │
                    └─────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │            HuBERT-ECG Foundation Encoder                 │
                    │     (Frozen SSL 1D-CNN + 12-Layer Transformer)           │
                    │            Pre-trained on 9.1M ECGs                      │
                    └─────────────────────────────┬────────────────────────────┘
                                                  │
                                 Feature Tensor (L=249, d=768)
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │    Mean Temporal Pooling         │             │    Feature Adapter (MLP)         │
        │    (L, 768) ──▶ (768,)           │             │    Linear(768,512) ──▶ GELU      │
        └────────────────┬─────────────────┘             │    ──▶ LayerNorm ──▶ Linear(768) │
                         │                               └────────────────┬─────────────────┘
                         ▼                                                │
        ┌──────────────────────────────────┐                              ▼
        │   Classifier Head (2-Layer MLP)  │             ┌──────────────────────────────────┐
        │   Linear(768, 256) ──▶ ReLU      │             │    BART-Base Decoder             │
        │   ──▶ Dropout(0.3) ──▶ Linear(5) │             │    (Pretrained Seq2Seq LM + LoRA)│
        └────────────────┬─────────────────┘             │    Cross-Attention over Adapter  │
                         │                               └────────────────┬─────────────────┘
                         ▼                                                │
        ┌──────────────────────────────────┐                              ▼
        │   Multi-Label Probabilities      │             ┌──────────────────────────────────┐
        │   Sigmoid: NORM, MI, STTC,       │             │   Autoregressive Text Generation │
        │            CD, HYP               │             │   Beam Search (width=4)          │
        │   Top class + Confidence %       │             │   "sinus rhythm. normal ecg..."  │
        └────────────────┬─────────────────┘             └────────────────┬─────────────────┘
                         │                                                │
                         ├────────────────────────┬───────────────────────┤
                         ▼                        ▼                       ▼
        ┌───────────────────────────────────────────────────────────────────────────────────┐
        │                       Gradient-Based Saliency Engine                              │
        │       abs(d(logit_target) / d(input_signal)) ──▶ (12, 5000) Heatmap               │
        └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                                  │
                         ┌────────────────────────┴───────────────────────┐
                         ▼                                                ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │      Gradio Web Interface        │             │   Clinical-Grade PDF Engine      │
        │  Interactive plots, confidence   │             │   Letterhead, Patient Metadata,  │
        │  gauges, side-by-side comparison,│             │   Finding Box, 12-Lead Saliency, │
        │  instantaneous PDF preview drawer│             │   Physician Signature Block      │
        └──────────────────────────────────┘             └──────────────────────────────────┘
```

---

## 3. REPOSITORY DIRECTORY MAP & FILE INVENTORY

```
intelligent-ecg-analysis/
├── README.md                  <- Comprehensive user & developer guide
├── Project_Plan.md            <- Original academic thesis / semester project blueprint
├── DEPLOYMENT_PLAN.md         <- Architectural audit & local decoupling strategy
├── requirements.txt           <- Complete Python dependencies
├── run_pipeline.py            <- Unified CLI entry point for Steps 1–6
├── setup_kaggle.py            <- Automated Kaggle / Google Colab runner
├── report_context.md          <- (This file) Complete technical documentation for LLMs
│
├── src/
│   ├── __init__.py            <- Package initializer
│   ├── config.py              <- Central hyperparameters, paths, brand colors, device detection
│   ├── data.py                <- Step 1: PTB-XL loading, SCP label parsing, official splits, plotting
│   ├── datasets.py            <- Feature caching engine, PyTorch Datasets, German language filter
│   ├── encoder.py             <- Step 2: HuBERT-ECG wrapper + fallback CNN-Transformer encoder
│   ├── classifier.py          <- Step 3: Multi-label classification head, training & evaluation
│   ├── report_generator.py    <- Step 4: FeatureAdapter + BART Seq2Seq generator (with LoRA)
│   ├── saliency.py            <- Step 6 explainability: gradient backpropagation saliency maps
│   ├── pdf_export.py          <- Step 5 polish: ReportLab PDF builder + PyMuPDF rasterizer
│   ├── app.py                 <- Step 5: Sage-themed Gradio web application
│   └── evaluate_full.py       <- Step 6: Per-class AUROC, failure case analysis, summary generator
│
├── checkpoints/               <- Trained PyTorch weights
│   ├── classifier_head.pt     <- Trained classification head weights (~795 KB)
│   └── report_generator.pt    <- Trained adapter + BART weights (~561 MB)
│
├── data/
│   ├── ptbxl/                 <- PTB-XL dataset directory (WFDB waveforms & CSV metadata)
│   └── features_cache/        <- Cached HuBERT-ECG feature arrays (*.npy per ecg_id)
│
├── notebook/
│   └── ECG_project.ipynb      <- Kaggle/Colab training notebook mirroring the src/ package
│
└── outputs/                   <- Evaluation tables, generated PDFs, preview PNGs, example plots
    ├── per_diagnosis_performance.csv
    ├── failure_cases.csv
    ├── evaluation_summary.txt
    └── ecg_report_*.pdf
```

---

## 4. DATASET & BIOMEDICAL SIGNAL PIPELINE (`PTB-XL`)

### 4.1 PTB-XL Dataset Overview
- **Reference**: Wagner et al., *Scientific Data* (2020), PhysioNet open access.
- **Volume**: 21,799 clinical 12-lead ECG records from 18,869 distinct patients.
- **Duration**: 10.0 seconds per recording.
- **Sampling Rates**: Available in 100 Hz (`records100/`, 1000 samples) and 500 Hz (`records500/`, 5000 samples). This project defaults to **500 Hz** to match HuBERT-ECG's native pre-training sampling rate.
- **Format**: WFDB binary format (`.dat` 16-bit waveform signals, `.hea` text header files).

### 4.2 Metadata & Diagnostic Label Mapping (`src/data.py`)
PTB-XL annotates records using the **SCP-ECG (Standard Communications Protocol for Computer-Assisted Electrocardiography)** standard. The annotations are structured as dictionaries of SCP codes mapped to diagnostic likelihoods (e.g. `{'NORM': 100.0, 'SR': 100.0}`).

The pipeline aggregates the granular diagnostic codes into **5 Clinical Superclasses**:
1. `NORM` — Normal Electrocardiogram
2. `MI` — Myocardial Infarction (Anterior, Inferior, Lateral, Posterior)
3. `STTC` — ST/T Change (Non-specific ST depression/elevation, T-wave inversion, ischemia)
4. `CD` — Conduction Disturbance (Left/Right Bundle Branch Block, Fascicular Block, AV Block)
5. `HYP` — Ventricular / Atrial Hypertrophy (LVH, RVH, Atrial Enlargement)

#### Label Vector Construction
- `load_scp_statements()` filters `scp_statements.csv` for `diagnostic == 1`.
- `aggregate_diagnostic_superclass()` maps SCP codes to their parent superclass.
- `build_label_matrix()` creates a binary multi-hot vector $y \in \{0, 1\}^5$ for each record.

### 4.3 Official Split Strategy
PTB-XL provides an official 10-fold patient-stratified split column (`strat_fold`):
- **Training Set**: Folds 1–8 ($\approx 80\%$, 17,441 records)
- **Validation Set**: Fold 9 ($\approx 10\%$, 2,183 records)
- **Held-Out Test Set**: Fold 10 ($\approx 10\%$, 2,163 records)
*Ensures zero patient leakage across splits (recordings from the same patient never appear in both train and test).*

### 4.4 Signal Preprocessing Modes (`src/encoder.py`)
Raw signals $S \in \mathbb{R}^{5000 \times 12}$ are processed via `preprocess_signal(signal, mode)`:
- `"zscore"` (Default): Per-lead zero-mean, unit-variance normalization:
  $$\hat{S}_{:, c} = \frac{S_{:, c} - \mu_c}{\sigma_c + \epsilon}$$
- `"minmax"`: Per-lead linear scaling to $[-1, 1]$:
  $$\hat{S}_{:, c} = 2 \cdot \frac{S_{:, c} - \min(S_{:, c})}{\max(S_{:, c}) - \min(S_{:, c}) + \epsilon} - 1$$
- `"bandpass_minmax"`: 3rd-order Butterworth bandpass filter ($0.5\text{ Hz} - 50.0\text{ Hz}$) followed by $[-1, 1]$ min-max scaling to remove baseline wander and powerline interference.

### 4.5 Language Filtering Heuristic (`src/datasets.py`)
PTB-XL contains both English and German reports. To prevent the English-pretrained BART tokenizer from producing degraded multilingual outputs, `ECGFeatureReportDataset` enforces a regex keyword filter that drops German reports:
- Filter keywords: `["sinusrhythmus", "lagetyp", "anhalt", "keine", "normaler", "fuer", "rhythmus", "vorhof", "schenkel", "achse"]`
- Retains verified English clinical reports without requiring heavy NLP language-ID libraries.

### 4.6 Feature Caching Architecture (`src/datasets.py`)
To prevent re-running the 93M parameter encoder during every training epoch:
- `precompute_features(df, encoder)` passes every signal through the frozen encoder once.
- The resulting $(L, d) = (249, 768)$ float32 tensor is saved to disk as `data/features_cache/<ecg_id>.npy`.
- Repeated classifier and report generator training epochs load directly from disk, speeding up epoch training by $>50\times$.

---

## 5. SELF-SUPERVISED FEATURE EXTRACTOR (`HuBERT-ECG`)

### 5.1 Model Specifications (`src/encoder.py`)
- **Reference**: Coppola et al., "HuBERT-ECG: Self-Supervised Learning for 12-Lead Electrocardiograms", 2024.
- **Hugging Face Hub ID**: `Edoardo-BS/hubert-ecg-base`
- **Architecture**:
  - 1D Convolutional feature extractor (temporal downsampling)
  - 12 Transformer encoder layers ($d_{\text{model}} = 768$, 12 attention heads, feedforward dim 3072)
  - Trained on 9.1 million 12-lead ECG records via masked prediction of acoustic/signal units.
- **Input Shape**: $(B, 12, 5000)$ at 500 Hz.
- **Output Shape**: $(B, L=249, d=768)$ feature sequence.
- **Frozen Protocol**: All encoder parameters $\theta_{\text{enc}}$ are locked (`requires_grad = False`).

### 5.2 Loading Mechanism & Registration
HuBERT-ECG uses a custom Hugging Face model class. To load:
```python
import hubert_ecg  # Registers custom class with transformers AutoModel
from transformers import AutoModel
model = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True)
```

### 5.3 Fallback Architecture & `FALLBACK_GUARD`
If internet access is unavailable or the package is uninstalled:
- `FallbackECGEncoder` initializes a lightweight untrained 3-layer Conv1D + 4-layer TransformerEncoder ($d_{\text{model}} = 768$) to preserve identical tensor interfaces.
- `FALLBACK_GUARD = True` in `src/config.py`: Safety barrier that **raises a hard RuntimeError** during training rather than silently training downstream heads on random uninitialized features.

---

## 6. MULTI-LABEL DIAGNOSTIC CLASSIFIER HEAD

### 6.1 Architecture (`src/classifier.py`)
The classifier maps the pooled temporal feature vector to 5 diagnostic superclasses:

```
Encoder Output (L=249, d=768)
        │
        ▼ Mean Temporal Pooling: x_pooled = (1/L) * sum(h_t) -> (768,)
┌──────────────────────────────────────┐
│ Linear(in_features=768, out=256)     │
├──────────────────────────────────────┤
│ ReLU Activation                      │
├──────────────────────────────────────┤
│ Dropout(p = 0.30)                    │
├──────────────────────────────────────┤
│ Linear(in_features=256, out=5)       │
└──────────────────────────────────────┘
        │
        ▼ Raw Logits: z in R^5
        │
        ▼ Sigmoid: p_i = 1 / (1 + exp(-z_i))
```

### 6.2 Loss Formulation & Optimization
- **Loss Function**: Binary Cross-Entropy with Logits across all classes:
  $$\mathcal{L}_{\text{BCE}}(z, y) = -\frac{1}{C}\sum_{c=1}^C \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$)
- **Batch Size**: 64
- **Epochs**: 30
- **Model Selection**: Best checkpoint saved to `checkpoints/classifier_head.pt` based on Validation Macro-AUROC.

### 6.3 Single Inference API
```python
def predict_single(signal, encoder=None, model=None) -> Dict[str, float]:
    # Returns {"NORM": 0.02, "MI": 0.94, "STTC": 0.31, "CD": 0.05, "HYP": 0.01}
```

---

## 7. CLINICAL REPORT GENERATOR (`ADAPTER + BART`)

### 7.1 Cross-Modal Bridging Concept
Generating natural language text from a biomedical signal requires bridging continuous temporal representation space $\mathbb{R}^{L \times 768}$ into discrete semantic token embedding space.

### 7.2 Feature Adapter Architecture (`src/report_generator.py`)
A dedicated projection module aligns the feature dimensions and non-linearities:
```python
class FeatureAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # 768 -> 512
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),  # 512 -> 768 (BART d_model)
        )
```

### 7.3 Seq2Seq Generation Engine (`ECGReportModel`)
- **Base LLM**: `facebook/bart-base` (6 encoder layers, 6 decoder layers, $d_{\text{model}}=768$, vocabulary size 50,265).
- **Encoder Substitution**: Instead of token embeddings, BART receives the adapter output $\tilde{H} = \text{Adapter}(H_{\text{ECG}})$ directly injected into `encoder_outputs = BaseModelOutput(last_hidden_state=adapted)`.
- **Parameter-Efficient Fine-Tuning (LoRA)**:
  - Low-Rank Adaptation applied via `peft`:
    - Rank $r = 16$
    - Scaling $\alpha = 32$
    - Target modules: Query (`q_proj`) and Value (`v_proj`) projection matrices in cross-attention and self-attention layers.
    - Trainable parameter reduction: fine-tunes only $\approx 2\%$ of BART parameters.

### 7.4 Autoregressive Decoding & Hyperparameters
- **Optimizer**: AdamW ($\text{lr} = 5 \times 10^{-5}$)
- **Batch Size**: 8
- **Epochs**: 10
- **Beam Search Width**: `num_beams = 4`
- **Max Target Length**: `max_length = 128` tokens
- **Loss**: Cross-Entropy over target report token IDs (ignoring pad token $-100$).
- **Metrics**: BLEU (1–4 gram precision), ROUGE-1, ROUGE-2, ROUGE-L.

---

## 8. EXPLAINABILITY & GRADIENT-BASED SALIENCY MAPPING

### 8.1 Clinical Explainability Rationale
Clinicians will not trust a black-box diagnosis. When the classifier predicts *Myocardial Infarction*, the physician must inspect whether the model focused on ischemic markers (ST-elevation, pathological Q-waves, T-wave inversion) or on artifact noise.

### 8.2 Mathematical Formulation (`src/saliency.py`)
Let $x \in \mathbb{R}^{5000 \times 12}$ be the preprocessed input ECG, $f(x)$ be the end-to-end classifier, and $z_c$ be the unnormalized logit for the predicted class $c = \arg\max_k p_k$.

The saliency map $S \in \mathbb{R}^{5000 \times 12}$ is defined via vanilla gradient backpropagation:
$$S_{t, j} = \left| \frac{\partial z_c}{\partial x_{t, j}} \right|$$

```python
def compute_saliency(signal, encoder, classifier, target_class=None):
    # Runs gradient-enabled pass on x (requires_grad=True)
    # Uses torch.autograd.grad(outputs=logits[0, target_idx], inputs=x)
    # Returns (5000, 12) absolute gradient heatmap
```

### 8.3 Concurrency & Thread-Safety Design
Normal `.backward()` populates `.grad` attributes on all shared `nn.Module` parameters, creating race conditions under concurrent web requests. `saliency.py` explicitly uses `torch.autograd.grad(outputs=..., inputs=x)`, computing gradients **strictly for the input tensor $x$** without mutating module parameters.

### 8.4 Heatmap Visualization
The saliency tensor is normalized per-lead:
$$S^{\text{norm}}_{:, j} = \frac{S_{:, j}}{\max(S_{:, j}) + \epsilon}$$
Rendered as an alpha-blended `Reds` colormap overlay directly behind each of the 12 black waveform traces in Matplotlib.

---

## 9. CLINICAL-GRADE PDF REPORT ENGINE (`REPORTLAB + PYMUPDF`)

### 9.1 Professional Design Standard (`src/pdf_export.py`)
Rather than generic web exports, the PDF generator produces an authentic, publication-quality diagnostic report modeled after GE Marquette / Philips clinical ECG cart printouts.

### 9.2 Brand Design System
| Color Token | Hex Code | Visual Application |
|---|---|---|
| `BRAND_NAVY` | `#12283F` | Letterhead header bar, section headers, primary table headers |
| `BRAND_ACCENT` | `#8C2F2F` | Primary finding callout box border, warning rules |
| `BRAND_PAPER` | `#F7F5F0` | Finding box surface, alternating table row shading |
| `BRAND_INK` | `#22282E` | High-contrast body typography |
| `BRAND_MUTED` | `#6B7280` | Subtitles, metadata labels, footer stamps |
| `BRAND_LINE` | `#D8D3C9` | Hairline dividing rules |

### 9.3 Document Layout Structure
1. **Page 1: Clinical Diagnostic Summary**
   - **Header Letterhead**: Dark Navy bar with "INTELLIGENT ECG ANALYSIS SYSTEM" in Times-Bold and unique Report ID (`ECG-YYYYMMDD-HHMMSS`).
   - **Record & Status Strip**: Timestamp, unconfirmed physician review status flag.
   - **Patient & Recording Table**: 3-column table with Patient ID, Age (with de-identification handling for $\ge 200$), Sex, Height, Weight, Recording Device, Heart Axis, Pacemaker, Extra Beats.
   - **Primary Finding Box**: Large Times-Bold diagnosis name, descriptive subtitle, model confidence percentage in Accent Red box.
   - **Differential Table**: Complete sorted class breakdown with percentage bars.
   - **AI-Generated Clinical Impression**: Framed box containing generated medical narrative.
   - **Regulatory Disclaimer**: Prominent safety notice stating educational/research prototype status.
   - **Physician Sign-Off Block**: Signature and Date lines.
2. **Page 2: 12-Lead Tracings & Saliency Waveform**
   - Full-page high-resolution Matplotlib render of all 12 leads ($I, II, III, aVR, aVL, aVF, V_1, V_2, V_3, V_4, V_5, V_6$) with red saliency overlays.

### 9.4 Dual-Page High-Res Rasterization Preview
`render_preview_image(pdf_path, dpi=140)` leverages `pymupdf` (`fitz`) and `PIL` to render the multi-page PDF into a single continuous vertical PNG image preview displayed inside the Gradio UI.

---

## 10. WEB APPLICATION & USER INTERFACE (`GRADIO`)

### 10.1 UI Design Philosophy (`src/app.py`)
The web application uses a custom **Sage / Forest Green** palette designed for clinical calming aesthetics, distinct from the Navy PDF branding:
- **Background**: Soft Sage `#EAF3E6`
- **Headings**: Deep Forest `#16241C` (Lora Serif typography)
- **Primary CTA Buttons**: Vibrant Green `#8EEB6E` pill buttons with smooth hover states
- **Surfaces**: Crisp Cream `#FCFEFA` cards with subtle elevation shadows

### 10.2 Three Interactive Workflows (Tabs)
1. **Tab 1: Test-Set Record Explorer**
   - Curated dropdown of 14 representative PTB-XL test records across all superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
   - Displays 12-lead signal plot with saliency heatmaps.
   - Top diagnosis markdown headline + `gr.Label` confidence gauge.
   - Generated clinical narrative textbox.
   - Instantaneous animated PDF preview drawer with direct Download button.
2. **Tab 2: Custom Waveform Upload**
   - Accepts user CSV files of shape $(T, 12)$ (samples $\times$ channels, no header).
   - Runs full inference, saliency computation, narrative generation, and PDF construction.
3. **Tab 3: Batch Multi-Record Analysis**
   - Accepts multiple selected test IDs and/or multi-file CSV uploads.
   - Produces a consolidated `pandas.DataFrame` table: `[Source, Diagnosis, Confidence, Report]`.

### 10.3 Instant Animated PDF Drawer
Uses custom CSS keyframes and transitions (`max-height`, `opacity`, `transform`) so that clicking "📄 Preview PDF" smoothly expands the document viewer without lag or UI jumping.

---

## 11. SYSTEMATIC EVALUATION & FAILURE ANALYSIS

### 11.1 Systematic Evaluation Framework (`src/evaluate_full.py`)
Step 6 moves beyond basic accuracy to clinical subgroup evaluation:
- **Quantitative Metrics**:
  - Per-class and Macro-AUROC on the official PTB-XL test split (Fold 10).
  - NLG Metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L.
- **Output Artifacts**:
  - `outputs/per_diagnosis_performance.csv`: AUROC per diagnosis sorted best-to-worst.
  - `outputs/failure_cases.csv`: Confidently-wrong predictions ($|\text{confidence} - 0.5| \ge 0.5$ where prediction was erroneous).
  - `outputs/evaluation_summary.txt`: Automated clinical narrative summary.

### 11.2 Core Thesis Finding: Rhythm vs. Morphology Discrepancy
The evaluation documents a critical clinical pattern:
- **Rhythm / Conduction Abnormalities (e.g. CD)**: The model achieves high AUROC because rhythm disorders exhibit macroscopic, periodic temporal signatures easily captured by convolutional downsampling and self-attention.
- **Morphology / Interval Abnormalities (e.g. MI, HYP, STTC)**: The model exhibits lower AUROC because ischemic and hypertrophic changes rely on localized micro-volt amplitudes (e.g. $1\text{ mm}$ ST elevation) and millisecond interval variations that can be smoothed out by frozen general-purpose encoders.

---

## 12. EXECUTION, TRAINING & DEPLOYMENT WORKFLOWS

### 12.1 Centralized Pipeline CLI (`run_pipeline.py`)
```bash
# Execute the complete 6-step pipeline in order
python run_pipeline.py --all

# Execute specific steps (e.g. train and evaluate classifier)
python run_pipeline.py --step 3 3.5
```

### 12.2 Step-by-Step Execution Commands
```bash
# Step 1: Verify data loading and plot sample ECG
python -m src.data --explore

# Step 2: Verify frozen HuBERT-ECG feature extraction
python -m src.encoder --test

# Step 3: Train and evaluate classifier head
python -m src.classifier --train
python -m src.classifier --evaluate

# Step 4: Train and evaluate report generator
python -m src.report_generator --train
python -m src.report_generator --evaluate

# Step 5: Launch interactive web application
python -m src.app

# Step 6: Generate full evaluation and failure analysis
python -m src.evaluate_full
```

### 12.3 Cloud GPU Training vs. Local Serving Architecture
To avoid repeating multi-hour training runs:
1. **Train on Cloud (Kaggle / Colab GPU)**:
   - Run `setup_kaggle.py` -> train classifier -> train BART+LoRA.
   - Commit version -> download `classifier_head.pt` (~800 KB) and `report_generator.pt` (~560 MB).
2. **Serve Locally (Any Laptop / CPU)**:
   - Place weights into `checkpoints/`.
   - Run `python -m src.app`.
   - Inference takes $< 2$ seconds on standard CPU.

---

## 13. DEPENDENCIES & ENVIRONMENT CONFIGURATION

### 13.1 `requirements.txt` Specifications
```txt
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.40.0
wfdb>=4.1.2
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
gradio>=4.30.0
tqdm>=4.66.0
evaluate>=0.4.1
rouge-score>=0.1.2
nltk>=3.8.1
peft>=0.11.0
huggingface_hub>=0.23.0
scipy>=1.11.0
reportlab>=4.0.0
pymupdf>=1.24.0
git+https://github.com/Edoar-do/HuBERT-ECG.git
```

### 13.2 Hyperparameter Summary (`src/config.py`)
| Parameter | Value | Description |
|---|---|---|
| `SAMPLING_RATE` | `500` Hz | 5000 samples per 10-sec lead |
| `N_LEADS` | `12` | Standard 12-lead arrangement |
| `ENCODER_FEATURE_DIM` | `768` | HuBERT-ECG Base hidden dimension |
| `CLASSIFIER_HIDDEN_DIM` | `256` | Intermediate MLP layer dimension |
| `CLASSIFIER_LR` | `1e-3` | Adam learning rate for classifier |
| `CLASSIFIER_EPOCHS` | `30` | Classifier training epochs |
| `CLASSIFIER_BATCH_SIZE` | `64` | Classifier batch size |
| `ADAPTER_HIDDEN_DIM` | `512` | FeatureAdapter hidden bottleneck |
| `MAX_REPORT_TOKENS` | `128` | Maximum report generation length |
| `REPORT_GEN_LR` | `5e-5` | AdamW learning rate for BART |
| `REPORT_GEN_EPOCHS` | `10` | Report generator training epochs |
| `REPORT_GEN_BATCH_SIZE` | `8` | Seq2Seq batch size |
| `REPORT_GEN_NUM_BEAMS` | `4` | Beam search generation width |
| `USE_LORA` | `True` | LoRA parameter-efficient fine-tuning |
| `FALLBACK_GUARD` | `True` | Safety check preventing random feature training |

---

## 14. MEDICAL & DIAGNOSTIC REFERENCE GLOSSARY

### 14.1 The 12 ECG Leads
- **Limb Leads (Frontal Plane)**:
  - Lead I: Right Arm (-) to Left Arm (+)
  - Lead II: Right Arm (-) to Left Leg (+)
  - Lead III: Left Arm (-) to Left Leg (+)
- **Augmented Vector Leads (Frontal Plane)**:
  - aVR: Augmented Vector Right
  - aVL: Augmented Vector Left
  - aVF: Augmented Vector Foot
- **Precordial / Chest Leads (Horizontal Plane)**:
  - $V_1, V_2$: Septal surface of the heart
  - $V_3, V_4$: Anterior wall of the left ventricle
  - $V_5, V_6$: Lateral wall of the left ventricle

### 14.2 The Cardiac Electrical Cycle
- **P-wave**: Atrial depolarization.
- **PR-interval**: Conduction time through AV node ($0.12 - 0.20\text{ s}$).
- **QRS complex**: Ventricular depolarization ($< 0.12\text{ s}$).
- **ST-segment**: Plateau phase of ventricular repolarization (elevation/depression indicates acute ischemia/infarction).
- **T-wave**: Ventricular repolarization.
- **QT-interval**: Total duration of ventricular electrical activity.

### 14.3 Diagnostic Superclasses
1. **NORM (Normal)**: Normal sinus rhythm ($60-100\text{ bpm}$), normal P-QRS-T morphology, normal PR and QT intervals.
2. **MI (Myocardial Infarction)**: Ischemic heart attack. Marked by ST-elevation (STEMI), pathological Q-waves ($>0.04\text{ s}$ or $>25\%$ of R-wave), and T-wave inversion.
3. **STTC (ST/T Changes)**: Non-infarction repolarization abnormalities, subendocardial ischemia, strain patterns, or electrolyte imbalances.
4. **CD (Conduction Disturbance)**: Delayed or blocked intraventricular electrical transmission (e.g. Left/Right Bundle Branch Block, AV blocks, hemiblocks).
5. **HYP (Hypertrophy)**: Thickening of atrial or ventricular myocardium (e.g., Left Ventricular Hypertrophy due to chronic hypertension).

---
*End of Intelligent ECG Analysis Project Context Specification.*
