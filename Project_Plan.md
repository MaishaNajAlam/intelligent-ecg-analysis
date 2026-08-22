Project Plan Intelligent ECG Analysis Tool: Signal-to-Report and Signal-to-Diagnosis with Deep Learning 

_Every year, 17.9 million people die from cardiovascular diseases, making it the leading cause of death globally. A cardiologist reads an ECG in under 30 seconds. Most of the world has no cardiologist. The question is whether a machine can read it too._ 

# **Contents** 

|**1**|**Pro**|**ject Overview**|**3**|
|---|---|---|---|
||1.1|What this project builds . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>3|
||1.2|What this project is **not** . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>3|
|**2**|**Wh**|**y This Tool Matters: The Real-World Need**|**3**|
||2.1|The scale of the problem . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>3|
||2.2|What an automated tool would change . . . . . . . . . . . . . . . . .|. . . . . . .<br>4|
||2.3|Why this is a credible semester project . . . . . . . . . . . . . . . . .|. . . . . . .<br>4|
|**3**|**Bac**|**kground Knowledge: What to Learn and Where**|**4**|
||3.1|Tier A: Visual explainers (start here) . . . . . . . . . . . . . . . . . .|. . . . . . .<br>4|
||3.2|Tier B: Primary research papers . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>5|
||3.3|Tier C: Reference textbook (free) . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>6|
||3.4|Tier D: Clinical ECG knowledge<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>6|
||3.5|Tier E: Tools and frameworks . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>7|
||3.6|How to study (not just read)<br>. . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>7|
|**4**|**Dat**|**asets: What Data Is Used and Why**|**8**|
||4.1|Primary dataset: PTB-XL . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>8|
||4.2|Supporting dataset: PTB-XL+ . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>8|
||4.3|Optional external validation: MIMIC-IV-ECG . . . . . . . . . . . . .|. . . . . . .<br>8|
|**5**|**Mo**|**dels: What Is Used and How**|**8**|
||5.1|The encoder: HuBERT-ECG<br>. . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
||5.2|The decoder: BART . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
||5.3|The classifer head<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
||5.4|How the pieces connect . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
||5.5|Reference systems to study and compare against<br>. . . . . . . . . . .|. . . . . . .<br>10|
|**6**|**Ste**|**p-by-Step Build Plan**|**10**|
||6.1|Step 1: Environment and data setup . . . . . . . . . . . . . . . . . .|. . . . . . .<br>10|
||6.2|Step 2: Load the encoder and extract features . . . . . . . . . . . . .|. . . . . . .<br>11|
||6.3|Step 3: Build the classifer (signal to label)<br>. . . . . . . . . . . . . .|. . . . . . .<br>11|
||6.4|Step 4: Build the report generator (signal to text)<br>. . . . . . . . . .|. . . . . . .<br>12|
||6.5|Step 5: Build the web interface . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>13|



1 

||6.6<br>Step 6: Systematic evaluation and error analysis<br>. . . . . . . . .|. . . . . . . . .<br>14|
|---|---|---|
|**7**|**Expected Outputs and Deliverables**|**15**|
|**8**|**Framing: Suggested Narrative**|**15**|
|**9**|**Summary of Checkpoints**|**15**|
|**10 **|**Final Note**|**16**|



2 

# **1 Project Overview** 

## **1.1 What this project builds** 

This project builds an end-to-end intelligent ECG analysis tool. It is a working application that takes a raw 12-lead electrocardiogram signal as input and produces two outputs: 

1. **A diagnostic classification:** a predicted disease label with a confidence score (e.g. “Atrial Fibrillation, 94% confidence”). 

2. **A clinical text report:** a natural-language description of the findings, generated word by word, mimicking how a cardiologist writes an interpretation (e.g. “Sinus rhythm, normal axis, no ST-segment abnormality”). 

The tool is wrapped in a clean, interactive web interface suitable for demonstration. 

```
+-->classificationhead-->label+confidence
ECGsignal-->ENCODER|("AtrialFibrillation,94%")
+-->textdecoder-------->clinicalreport
("Sinusrhythm,normalaxis,...")
```

## **1.2 What this project is not** 

This is not a toy demo. It uses real clinical data, real pre-trained models, and produces outputs that can be meaningfully compared against real cardiologist reports. At the same time, it is not a research paper. It is an engineering project that applies existing models to build a functional, polished tool. Understanding comes from building. 

# **2 Why This Tool Matters: The Real-World Need** 

## **2.1 The scale of the problem** 

Cardiovascular disease (CVD) is the number-one cause of death worldwide, responsible for an estimated 17.9 million deaths per year (WHO, 2021). The 12-lead electrocardiogram (ECG) is the single most common diagnostic test in cardiology. It is cheap, fast, non-invasive, and available in nearly every clinic on earth. 

The problem is interpretation. Reading an ECG requires years of specialist training. Rural hospitals, primary-care clinics, and low-resource settings frequently lack trained cardiologists. A recorded ECG may sit unread for hours or be misread by a non-specialist. People die not because the test was unavailable, but because no one was there to read it. 

3 

## **2.2 What an automated tool would change** 

|**Scenario**|**Impact of an intelligent ECG tool**|
|---|---|
|Rural clinic, no<br>cardiologist|A nurse records the ECG; the tool provides an immediate prelim-<br>inary interpretation and fags urgent fndings. The patient is re-<br>ferred or treated faster|
|Emergency department|Hundreds of ECGs arrive per shift. The tool pre-screens and high-|
|triage|lights the critical ones, because a heart attack cannot wait in a<br>queue|
|Telemedicine|A remote ECG is uploaded; the tool generates a structured re-<br>port that a distant specialist can review, reducing turnaround from<br>hours to seconds|
|Medical education|Students compare the tool’s generated report against the ground<br>truth to learn ECG interpretation interactively|



## **2.3 Why this is a credible semester project** 

The project sits at the intersection of three high-demand fields: **biomedical signal processing** , **natural language processing (NLP)** , and **clinical AI deployment** . It demonstrates the ability to handle real clinical data responsibly, understanding of self-supervised learning and transformer architectures, end-to-end system design from raw signal to polished user interface, and direct, measurable societal impact. 

# **3 Background Knowledge: What to Learn and Where** 

Before writing any code, a solid understanding of the core ideas is essential. This section provides a structured learning path, ordered from the most accessible materials to the primary research papers. 

## **3.1 Tier A: Visual explainers (start here)** 

These are free, visual, beginner-friendly blog posts. Each one can be read in a single sitting and makes the corresponding research paper far easier to approach. 

4 

|**#**|**Resource**|**What it teaches**|
|---|---|---|
|A1|The Illustrated Transformer (Jay Alammar).<br>`https://`<br>`jalammar.github.io/illustrated-transformer/`|How a transformer reads a<br>sequence of numbers using<br>attention. This is the single<br>most important architecture<br>in the project. Worth<br>reading twice|
|A2|The Illustrated BERT (Jay Alammar). `https://jalammar.`<br>`github.io/illustrated-bert/`|How pre-training works: a<br>model learns from<br>unlabelled data by guessing<br>missing pieces|
|A3|The Illustrated Word2Vec (Jay Alammar).<br>`https://`<br>`jalammar.github.io/illustrated-word2vec/`|Where “embeddings” come<br>from, i.e. turning words (or<br>signals) into vectors of<br>numbers|
|A4|3Blue1Brown, Neural Networks series (YouTube). `https://`<br>`www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_`<br>`67000Dx_ZCJB-3pi`|The clearest visual<br>explanation of what a neural<br>network actually does, built<br>from scratch|
|A5|StatQuest, Attention and Transformers (YouTube). `https:`<br>`//www.youtube.com/c/joshstarmer`|Step-by-step, no-jargon<br>walkthroughs of attention,<br>encoder-decoder models,<br>and training|



## **3.2 Tier B: Primary research papers** 

These are the actual papers behind the models used in this project. Reading the Tier A explainers first makes these substantially more approachable. 

5 

|**#**|**Paper**|**The one idea to take away**|
|---|---|---|
|B1|Vaswani et al., “Attention<br>Is All You Need,” 2017.<br>`https://arxiv.org/abs/`<br>`1706.03762`|One architecture, the transformer, can read any sequence of<br>numbers. This is the foundation of everything here|
|B2|Devlin et al., “BERT,”<br>2019. `https://arxiv.`<br>`org/abs/1810.04805`|A model can learn powerful representations from unlabelled<br>data by predicting masked tokens. This is the ancestor of how<br>the ECG encoder learns|
|B3|Sutskever et al., “Sequence<br>to Sequence Learning,”<br>2014. `https://arxiv.`<br>`org/abs/1409.3215`|The “translation” framing: a sequence goes in, a diferent se-<br>quence comes out. The ECG-to-report pipeline is exactly this|
|B4|Lewis et al., “BART,”<br>2020. `https://arxiv.`<br>`org/abs/1910.13461`|The text decoder used in this project to generate clinical re-<br>ports|
|B5|Bahdanau et al., “Neural<br>Machine Translation by<br>Jointly Learning to Align<br>and Translate,” 2015.<br>`https://arxiv.org/abs/`<br>`1409.0473`|Attention as an alignment mechanism:<br>how the decoder<br>knows which part of the input to focus on when generating<br>each word|
|B6|Sennrich et al., “BPE<br>Subword Units,” 2016.<br>`https://arxiv.org/abs/`<br>`1508.07909`|Even human text is not naturally discrete. It is chopped into<br>invented subword pieces.<br>A key concept for understanding<br>tokenization|



## **3.3 Tier C: Reference textbook (free)** 

|**#**|**Resource**|**How to use it**|
|---|---|---|
|C1|Jurafsky & Martin, _Speech_|Free standard NLP textbook. The chapters on tokenization,|
||_and Language Processing_,|embeddings, seq2seq, transformers, and BERT resolve most|
||3rd ed.|confusions left by a paper. Dip in as needed; not meant to be|
||`https://web.stanford.`|read cover to cover|
||`edu/~jurafsky/slp3/`||



## **3.4 Tier D: Clinical ECG knowledge** 

Understanding the medical side is not optional. The tool analyses ECGs, so one must know what an ECG actually says. 

6 

|**#**|**Resource**|**What it teaches**|
|---|---|---|
|D1|Dubin, _Rapid_<br>_Interpretation of EKG’s_<br>(textbook, widely<br>available)|The fastest route to understanding how heart attacks, rhythm<br>disorders, and conduction blocks appear on an ECG. Written<br>for beginners, not cardiologists|
|D2|ECG Library, Life in the<br>Fast Lane. `https://`<br>`litfl.com/ecg-library/`|Free online reference with annotated ECG examples for every<br>major diagnosis|
|D3|Khan Academy,|Video-based introduction to cardiac physiology and what each|
||Circulatory System and<br>ECG. `https:`|wave on an ECG represents|
||`//www.khanacademy.org/`<br>`science/`||
||`health-and-medicine`||



## **3.5 Tier E: Tools and frameworks** 

|**#**|**Resource**|**What it covers**|
|---|---|---|
|E1|PyTorch tutorials.|Ofcial tutorials for the deep learning framework used|
||`https://pytorch.org/`<br>`tutorials/`|throughout the project|
|E2|Hugging Face<br>Transformers<br>documentation.|How to load, use, and fne-tune pre-trained transformer mod-<br>els (BART, encoders) in Python|
||`https://huggingface.`<br>`co/docs/transformers/`||
|E3|Gradio documentation.<br>`https:`<br>`//www.gradio.app/docs/`|How to build interactive web interfaces around ML models<br>with minimal code|
|E4|Streamlit documentation.<br>`https:`<br>`//docs.streamlit.io/`|Alternative to Gradio for building data-centric web dash-<br>boards|



## **3.6 How to study (not just read)** 

Passive reading does not produce understanding. A productive habit after each Tier B paper is to close it and write three sentences from memory: what problem it solves, what idea it proposes, what result it shows. If that is not possible, the matching Tier A post or Tier C chapter is the place to return to. 

**CHECKPOINT:** the learning stage is finished when the following statement can be explained to another person without notes. “A transformer model turns a sequence of numbers into a representation using attention. A pre-trained encoder learns from unlabelled data. A decoder generates text one token at a time. An ECG signal can be treated as a sequence and translated into a clinical report, just like translating one language into another.” 

7 

# **4 Datasets: What Data Is Used and Why** 

## **4.1 Primary dataset: PTB-XL** 

|**Property**|**Detail**|
|---|---|
|Full name|PTB-XL: A Large Publicly Available Electrocardiography Dataset|
|Reference|Wagner et al., 2020. `https://physionet.org/content/ptb-xl/`|
|Size|21,799 clinical 12-lead ECG recordings from 18,869 patients|
|Duration per<br>recording|10 seconds|
|Sampling rates|500 Hz and 100 Hz versions available|
|Labels|Multi-label SCP-ECG diagnostic statements with cardiologist likelihood<br>scores (how confdent the cardiologist was, from 15 to 100)|
|Reports|9,839 unique free-text clinical reports written by cardiologists|
|Splits|Ofcial 10-fold stratifed train/validation/test splits provided|
|Access|**Fully open**, free download, no application needed|



**Why PTB-XL:** It is the largest freely available 12-lead ECG dataset that comes with both diagnostic labels _and_ free-text reports. The labels provide the training targets for classification; the reports provide the training targets for text generation. The official splits ensure reproducibility. 

## **4.2 Supporting dataset: PTB-XL+** 

|**Property**|**Detail**|
|---|---|
|Full name|PTB-XL+|
|Reference|Strodthof et al., 2023. Aligned to PTB-XL|
|Content|Clinical-grade interval and amplitude measurements (PR interval, QRS<br>duration, QT/QTc, ST levels, R amplitudes) computed by validated clin-<br>ical software|
|Role|Provides precise ground-truth measurements for each ECG. Useful for<br>feature analysis and understanding what the signal contains beyond the<br>raw waveform|
|Access|**Open**|



## **4.3 Optional external validation: MIMIC-IV-ECG** 

|**Property**|**Detail**|
|---|---|
|Full name|MIMIC-IV-ECG|
|Scale|Approximately 800,000 ECG recordings|
|Reports|Machine-generated clinical reports|
|Role|If the tool works on PTB-XL, testing it on a completely diferent hospi-<br>tal’s data shows whether the tool generalises or merely memorises|
|Access|Credentialed (requires a short ethics training course on PhysioNet)|



# **5 Models: What Is Used and How** 

Two pre-trained models form the backbone. Neither is trained from scratch; both are downloaded and used. 

8 

## **5.1 The encoder: HuBERT-ECG** 

|**Property**|**Detail**|
|---|---|
|What it is|A self-supervised transformer pre-trained on 9.1 million 12-lead ECG<br>recordings across 164 cardiovascular conditions|
|Reference|Coppola et al., 2024. Public checkpoint available|
|What it does|Takes a raw ECG signal and produces a _feature representation_, which<br>is a compact numerical summary that captures the clinically relevant<br>patterns in the signal|
|How it is used here|**Frozen** (not trained further). It serves as the front-end: raw signal goes<br>in, feature vectors come out. These feature vectors feed both the classifer<br>and the text decoder|
|Why this model|Pre-trained at massive scale; public checkpoint eliminates the need for<br>expensive pre-training; produces state-of-the-art representations|
|Variants|SMALL, BASE, and LARGE, difering in capacity. Start with SMALL<br>or BASE|



## **5.2 The decoder: BART** 

|**Property**|**Detail**|
|---|---|
|What it is|A pre-trained sequence-to-sequence transformer designed for text gener-<br>ation|
|Reference|Lewis et al., 2020. Available via Hugging Face|
|What it does|Takes a sequence of feature vectors and generates natural-language text<br>one token at a time|
|How it is used here|Connected to the encoder’s output through a small adapter layer. Fine-<br>tuned on the (signal, report) pairs from PTB-XL to learn to write clinical<br>reports|
|Why this model|Simple to wire up, well-documented, strong baseline for conditional text<br>generation|



## **5.3 The classifier head** 

|**Property**|**Detail**|
|---|---|
|What it is|A small, simple neural network (one or two linear layers) added on top<br>of the encoder’s output|
|What it does|Maps the encoder’s feature vectors to a probability distribution over di-<br>agnostic classes|
|How it is used here|Trained from scratch on PTB-XL labels (the encoder stays frozen)|
|Output|A predicted label and a confdence percentage|



## **5.4 How the pieces connect** 

```
Raw12-leadECGsignal(PTB-XL,10seconds,12channels)
```

```
|
```

```
v
```

```
+-------------------+
```

```
|HuBERT-ECG|<--frozen,downloaded,nottrained
```

9 

```
|(Encoder)|
+-------------------+
|
|featurevectors
|
+-----+-----+
||
vv
+--------++-------------------+
|Linear||Adapter+BART|<--fine-tunedonPTB-XLreports
|Head||(TextDecoder)|
+--------++-------------------+
||
vv
Label+Clinicaltextreport
Confidence(generatedwordbyword)
```

## **5.5 Reference systems to study and compare against** 

These are existing ECG-to-report systems from the literature. They are not rebuilt from scratch but studied for design decisions and used as comparison baselines. 

|**System**|**What to learn from it**|
|---|---|
|MEIT (Wan et<br>al., 2024)|A complete ECG-to-report pipeline. Study the architecture and how signal<br>features are converted into text|
|ECG-Chat|Adds retrieval and prompting to reduce hallucination. Study how it handles|
|(Zhao et al.,<br>2024)|cases the model is unsure about|



# **6 Step-by-Step Build Plan** 

Each step ends with a **CHECKPOINT** : a concrete, testable condition. The step is finished when the checkpoint holds. If it does not hold, the step is not done. That is all “done” means here. 

## **6.1 Step 1: Environment and data setup** 

**Goal:** a working development environment with the dataset loaded and explorable. 

### **To-do list:** 

1. Set up a GPU-enabled Python environment. Options include Google Colab Pro (easiest, cloud-based, GPU included), a university lab machine with a GPU, or a local machine with an NVIDIA GPU and CUDA installed. 

2. Install the core libraries: 

   - `torch` (PyTorch, the deep learning framework). 

   - `transformers` (Hugging Face, for loading BART and pre-trained models). 

   - `wfdb` (for reading PhysioNet ECG files). 

   - `matplotlib` , `numpy` , `pandas` (data handling and plotting). 

10 

3. Download PTB-XL from PhysioNet ( `https://physionet.org/content/ptb-xl/` ). 

4. Load one ECG record. Plot all 12 leads. Inspect the waveform visually. 

5. Open the accompanying metadata: find the diagnostic label and the free-text report for that same record. Print both. 

6. Understand the data splits: PTB-XL provides official folds. Load the split assignments. 

### **Knowledge required:** 

- Basic Python and NumPy. 

- What a 12-lead ECG looks like (Tier D1, D2). 

- How to use Google Colab or a terminal. 

**CHECKPOINT:** any ECG record from PTB-XL can be loaded, its 12 leads plotted, and its label and report text printed in the console. 

## **6.2 Step 2: Load the encoder and extract features** 

**Goal:** pass a raw ECG signal through the frozen HuBERT-ECG encoder and obtain its feature representation. 

### **To-do list:** 

1. Download the HuBERT-ECG checkpoint from its public repository. 

2. Load the model in PyTorch. Set it to evaluation mode ( `model.eval()` ) and freeze all parameters (no training). 

3. Preprocess one ECG signal to match the encoder’s expected input format (check the model’s documentation for expected sampling rate, normalisation, and input shape). 

4. Pass the signal through the encoder. Inspect the output tensor: note its shape ( _L × d_ , where _L_ is the number of time frames and _d_ is the feature dimension). 

5. Repeat for a small batch of signals. Save the features to disk for reuse. 

### **Knowledge required:** 

- What “freezing” a model means: the encoder’s weights are fixed and it is used as a feature extractor, not trained further (Tier A2, B2). 

- What a feature vector is: a compressed numerical summary of the input (Tier A3). 

- Basic PyTorch: loading a model, running a forward pass, inspecting tensor shapes. 

**CHECKPOINT:** a raw ECG goes in, a feature tensor of known shape comes out, and the encoder’s parameters are confirmed frozen (zero gradients). 

## **6.3 Step 3: Build the classifier (signal to label)** 

**Goal:** train a small classification head on top of the frozen encoder features to predict diagnostic labels. 

### **To-do list:** 

1. Design the classification head: one or two linear layers, followed by a sigmoid or softmax activation. Input dimension = the encoder’s feature dimension _d_ . Output dimension = number of diagnostic classes. 

11 

2. Prepare the labels: PTB-XL uses multi-label SCP-ECG statements. Decide on a label set. Starting with the 5 superclasses (NORM, MI, STTC, CD, HYP) is recommended; expanding to the full 71 SCP codes can come later. 

3. Train the classification head using the official PTB-XL train/validation splits. Only the head’s parameters are updated; the encoder stays frozen. 

4. Evaluate on the test split. Report per-class AUROC and overall AUROC. 

5. Extract a confidence score from the output probabilities. 

### **Knowledge required:** 

- What multi-label classification is: one signal can have multiple diagnoses simultaneously. 

- Binary cross-entropy loss (the standard loss for multi-label problems). 

- AUROC as an evaluation metric (Tier C1 or any ML textbook). 

- PyTorch training loop: forward pass, loss, backward pass, optimizer step. 

**CHECKPOINT:** a signal goes in; the tool prints a diagnosis label with a confidence percentage. Test-set AUROC is computed and recorded. 

## **6.4 Step 4: Build the report generator (signal to text)** 

**Goal:** connect the encoder features to a BART decoder and fine-tune the decoder to generate clinical reports. 

This is the most conceptually challenging step. It is where signal processing meets NLP. Take it slow. 

### **To-do list:** 

1. Load a pre-trained BART model from Hugging Face ( `facebook/bart-base` is a good starting point). 

2. Build an adapter layer: a small neural network that maps the encoder’s feature vectors into the dimensionality expected by BART’s cross-attention. This is the bridge between the ECG world and the text world. 

3. Prepare training pairs: each pair is (encoder features, target report text). The target text is tokenized using BART’s built-in tokenizer. 

4. Fine-tune the adapter + BART decoder on the training set. The encoder remains frozen. BART’s parameters can be fine-tuned directly or kept frozen with a LoRA adapter, which is a lightweight fine-tuning method (see Hu et al., 2022). 

5. Generate a report for a test signal using beam search or greedy decoding. Observe the text emerging token by token. 

6. Evaluate using standard NLG metrics: BLEU and ROUGE scores against the ground-truth reports. 

### **Knowledge required:** 

- The seq2seq (sequence-to-sequence) paradigm: an encoder reads input, a decoder generates output (Tier B3, B4). 

- Cross-attention: how the decoder “looks at” the encoder’s output when deciding what word to generate next (Tier A1, B5). 

12 

- Tokenization: how text is broken into subword units (Tier B6). 

- Beam search vs. greedy decoding: strategies for generating text (Tier C1). 

- What BLEU and ROUGE measure: overlap between generated and reference text. 

**CHECKPOINT:** a signal goes in; the tool generates a complete clinical report sentence. BLEU and ROUGE scores are computed on the test set. 

## **6.5 Step 5: Build the web interface** 

**Goal:** wrap the classifier and report generator in a clean, interactive web application. 

**Recommended framework: Gradio** (Python, free). Gradio turns a model into a shareable web page in approximately 30 lines of code with no web-development experience required. Streamlit is a viable alternative for a more dashboard-style layout. 

### **To-do list:** 

1. Create the basic interface: 

   - An upload button (or a dropdown to select a test ECG). 

   - A signal plot showing all 12 leads. 

   - The predicted label with confidence. 

   - The generated report. 

2. Add polishing features. Select several from the table below; basic implementations are sufficient. 

|**Feature**|**What it adds**|
|---|---|
|Per-lead visualization|All 12 leads rendered in a clean clinical grid layout. Looks profes-<br>sional and medically authentic|
|Attention / saliency map|Highlight the region of the signal the model focused on. Provides<br>visual explainability, which is very strong for presenting the project|
|Confdence gauge|A visual dial or bar showing model certainty. Immediately commu-<br>nicates reliability to a non-technical audience|
|Model selector dropdown|Allow<br>switching<br>between<br>encoder<br>or<br>decoder<br>variants<br>(e.g.<br>HuBERT-ECG SMALL vs. BASE; BART vs. T5). Demonstrates<br>understanding that components are interchangeable|
|Predicted vs. real report|Side-by-side display comparing the generated report with the car-<br>diologist’s ground-truth report. Immediately exposes quality difer-<br>ences|
|Export to PDF|One-click generation of a formatted clinical report document. Pol-<br>ished and practical|
|Batch mode|Analyse multiple ECGs at once and display a summary table with<br>diagnoses and confdence scores|



### **Knowledge required:** 

- Gradio or Streamlit basics (Tier E3 or E4). 

- Matplotlib for signal plotting. 

- Basic UI/UX sense: clean layout, readable fonts, no clutter. 

13 

**CHECKPOINT:** a person other than the developer can open the app in a browser, select or upload an ECG, and see the plotted signal, a diagnosis with confidence, a generated report, and at least two additional features working. It should be polished enough for a demo or a GitHub screenshot. 

## **6.6 Step 6: Systematic evaluation and error analysis** 

**Goal:** go beyond a working demo. Systematically evaluate the tool’s performance and identify where it succeeds and where it fails. 

This step transforms the project from “a demo that works” into “a tool whose strengths and limitations are understood.” 

### **To-do list:** 

1. **Quantitative evaluation:** 

   - Compute per-class AUROC for the classifier across all diagnostic categories, not just the overall average. 

   - Compute BLEU and ROUGE for the report generator. 

   - Break results down by diagnosis type: rhythm diagnoses (e.g. atrial fibrillation, sinus rhythm) versus morphology/interval diagnoses (e.g. myocardial infarction, bundle branch block, left ventricular hypertrophy). 

2. **Qualitative error analysis:** 

   - Select test cases where the classifier is confidently wrong. 

   - Use the side-by-side feature (Step 5) to compare generated reports against ground truth for difficult cases, especially heart attacks and conduction blocks. 

   - Build a list of failure cases. For each, note the true diagnosis and what the model predicted instead. 

3. **Pattern identification:** 

   - Examine the failure list. Ask: what do the failed diagnoses share? What do the succeeded diagnoses share? 

   - Categorise diagnoses into those the tool handles well and those it does not. Document the pattern. 

4. **Summary table:** produce a clean table showing per-diagnosis performance, sorted from best to worst. This single table tells the story of where the tool is strong and where it is weak. 

### **Knowledge required:** 

- Per-class vs. macro-average metrics. 

- Basic understanding of ECG diagnoses: what defines a rhythm diagnosis vs. a morphology diagnosis (Tier D1, D2). 

- Scientific honesty: reporting failures is a strength, not a weakness. A tool with known, characterised limitations is more credible than one that claims to work perfectly. 

**CHECKPOINT:** a summary table exists showing per-diagnosis performance. A documented list of failure cases is available, with a written paragraph describing the pattern observed. The tool’s strengths and limitations can be explained clearly. 

14 

# **7 Expected Outputs and Deliverables** 

At the end of all steps, the project produces: 

|**Deliverable**|**Description**|
|---|---|
|Working web<br>application|An interactive tool: ECG in, diagnosis + report out, with visualization,<br>explainability features, and a clean interface|
|Trained classifer|A diagnostic classifer achieving measurable AUROC on the PTB-XL test<br>set, with per-class breakdown|
|Trained report<br>generator|A text generation model producing clinical reports, evaluated with BLEU<br>and ROUGE|
|Evaluation report|Quantitative metrics (per-class AUROC, BLEU, ROUGE) plus qualita-<br>tive error analysis with identifed patterns|
|Code repository<br>presentation|Clean, documented, reproducible code on GitHub<br>A slide deck or poster summarising the motivation, architecture, results,<br>and identifed limitations|



# **8 Framing: Suggested Narrative** 

A strong entry tells a story, not a feature list. Here is a suggested narrative arc: 

1. **The problem.** 17.9 million cardiovascular deaths per year. ECGs are cheap but require expert interpretation. Most of the world has no cardiologist. 

2. **The solution.** An intelligent tool that reads an ECG and writes a clinical report, the same task a cardiologist performs, automated. 

3. **The architecture.** A self-supervised encoder (HuBERT-ECG) captures the signal’s meaning; a transformer decoder (BART) translates it into language. Show the diagram. 

4. **The demo.** Live. Upload an ECG. Show the 12-lead plot, the diagnosis, the generated report. Show the attention map. 

5. **The honest analysis.** The tool works well on rhythm diagnoses. It struggles on morphologydefined diagnoses. Show the per-class table. Explain the pattern. _This is the most impressive framing_ , because it shows understanding, not just engineering. 

6. **The impact.** If deployed in a rural clinic, this tool provides immediate preliminary interpretation. Limitations are known and stated. The path to improvement is clear. 

# **9 Summary of Checkpoints** 

|**Step**|**Checkpoint condition**|
|---|---|
|Learning|The core concepts can be explained to another person without notes|
|Step 1|Any PTB-XL record can be loaded, plotted, and its label/report printed|
|Step 2|A raw ECG produces a feature tensor of known shape; encoder is confrmed frozen|
|Step 3|A signal produces a diagnosis label with confdence; test AUROC is recorded|
|Step 4|A signal produces a complete clinical report; BLEU/ROUGE are computed|
|Step 5|A non-developer can open the app, upload an ECG, and see plot, diagnosis, report,<br>and at least two extra features|
|Step 6|A per-diagnosis performance table and a documented failure pattern exist and can<br>be explained clearly|



15 

# **10 Final Note** 

Every step builds on the previous one. No step is skipped. The learning comes first because without it the building is mechanical and fragile. The building comes before the evaluation because failures cannot be understood on a system that does not exist yet. And the honest evaluation, showing where the tool fails and why, is what separates a student project from a professional one. 

The goal is not to build something perfect. The goal is to build something real, understand it deeply, and present it with clarity and honesty. 

16 

