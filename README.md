# AI Chatbot for Mental Health

**MSc Information Technology (Artificial Intelligence) Final Year Project**

An end-to-end research and production-grade machine learning system designed to detect mental health distress / stress states from conversational text, benchmark classical machine learning models, and provide empathetic, safety-aware chatbot support.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Overview](#dataset-overview)
3. [System Architecture](#system-architecture)
4. [NLP Preprocessing Pipeline](#nlp-preprocessing-pipeline)
5. [Models Evaluated](#models-evaluated)
6. [Evaluation Framework & Metrics](#evaluation-framework--metrics)
7. [Mental Health Chatbot & Safety Layer](#mental-health-chatbot--safety-layer)
8. [Project Structure](#project-structure)
9. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
10. [REST API Endpoints](#rest-api-endpoints)
11. [Ethical Considerations & Limitations](#ethical-considerations--limitations)

---

## 1. Project Overview
This project develops an empathetic, AI-driven mental health support chatbot powered by supervised machine learning models trained on social media mental health data. The system:
- Ingests and cleans social media texts related to stress, anxiety, trauma, and mental distress.
- Performs stratified splitting (70% train, 15% validation, 15% held-out test).
- Extracts vocabulary and n-gram representations using TF-IDF (fitted strictly on training data to prevent data leakage).
- Trains and benchmarks **5 baseline ML models**:
  - **Random Forest** (`RandomForestClassifier`)
  - **Support Vector Machine (SVM)** (`SVC(probability=True)`)
  - **Logistic Regression** (`LogisticRegression`)
  - **K-Nearest Neighbors (KNN)** (`KNeighborsClassifier`)
  - **Naive Bayes** (`MultinomialNB`)
- Evaluates models using rigorous research metrics:
  - **Accuracy**
  - **Prediction Confidence** (mean, median, high-confidence ratio)
  - **Confusion Matrix** (heatmaps & normalized counts)
  - **F1-Score** (Macro & Weighted)
  - **Precision & Recall** (Macro & Weighted)
  - **Specificity** (True Negative Rate)
  - **ROC-AUC** & ROC Curves
  - **Inference Latency** (milliseconds per sample)
- Deploys the best model behind a safety-first conversational chatbot with multi-tier crisis detection, escalation hotlines, and grounding coping exercises.

---

## 2. Dataset Overview
- **Dataset**: Dreaddit Mental Health & Stress Dataset (`Dataset/Dreaddit_combine_data.csv`).
- **Samples**: 3,553 labeled posts across mental health communities (`ptsd`, `anxiety`, `stress`, `assistance`, `relationships`, `survivorsofabuse`, `domesticviolence`, `homeless`, etc.).
- **Primary Task**: Binary Stress Classification (`0: Non-Stress`, `1: Stress`).
- **Annotator Confidence**: Ground truth confidence ratings provided per post.

---

## 3. System Architecture
```text
Raw Dataset (Dreaddit)
         │
         ▼
[ Data Cleaning & Deduplication ]
         │
         ▼
[ NLP Text Normalization & Preprocessing ]
 (HTML/URL removal, contractions expansion, tokenization, lemmatization)
         │
         ▼
[ Stratified Split (70/15/15) ] ── Train / Val / Test
         │
         ▼
[ Feature Extraction ] ── TF-IDF Vectorizer (fit strictly on Train)
         │
         ▼
[ Model Training & Checkpointing ]
 ├── Random Forest
 ├── Support Vector Machine (SVM)
 ├── Logistic Regression
 ├── K-Nearest Neighbors (KNN)
 └── Naive Bayes
         │
         ▼
[ Comprehensive Evaluation ]
 ├── Accuracy & Confidence Analysis
 ├── Confusion Matrices & Heatmaps
 ├── ROC-AUC & Precision-Recall
 └── Model Comparison (results/model_comparison.csv)
         │
         ▼
[ Best Model Selection ] ──> models/best_model/
         │
         ▼
[ Safety & Chatbot Inference Layer ]
 ├── Multi-tier Crisis Detection (Self-harm / Suicide keywords)
 ├── Immediate Escalation Hotlines (988, Crisis Text Line)
 ├── Empathetic Response Generator (Box breathing, 5-4-3-2-1 Grounding)
 └── Non-Clinical Medical Disclaimer Enforcement
         │
         ▼
[ FastAPI Backend / Interactive CLI Chat ]
```

---

## 4. NLP Preprocessing Pipeline
Implemented in `src/preprocessing/`:
1. **`TextCleaner`**: Strips URLs, HTML entities, Reddit username/sub mentions (`r/...`), and non-printable characters.
2. **`TextNormalizer`**: Expands English contractions (`"don't"` -> `"do not"`, `"i'm"` -> `"i am"`) and normalizes case.
3. **`WordTokenizer`**: Tokenizes text into discrete word tokens.
4. **`StopwordsHandler`**: Emotion & negation aware filtering. Preserves crucial mental health negations (`"not"`, `"never"`, `"no"`, `"cannot"`) and first-person self-referential markers (`"i"`, `"me"`, `"myself"`).
5. **`TextLemmatizer`**: Reduces words to their morphological base form.
6. **`TextPreprocessingPipeline`**: Integrates all stages into an end-to-end callable pipeline.

---

## 5. Models Evaluated
Implemented in `src/models/baseline/`:
1. **Random Forest Classifier**: Non-linear ensemble model with 100 estimators, balanced class weights, and tree-based decision aggregation.
2. **Support Vector Machine (SVM)**: Maximum-margin hyperplane with linear kernel and Platt-scaled calibrated probability estimates (`probability=True`).
3. **Logistic Regression**: Linear classifier with L2 regularization, L-BFGS solver, and class weighting.
4. **K-Nearest Neighbors (KNN)**: Non-parametric instance-based classifier utilizing cosine distance and distance-weighted neighbor votes (`n_neighbors=5`).
5. **Naive Bayes**: Probabilistic classifier with Laplace smoothing (`alpha=1.0`).

---

## 6. Evaluation Framework & Metrics
Implemented in `src/evaluation/`:
- **Accuracy**: Overall classification correctness.
- **Prediction Confidence**: Mean probability allocated to the winning class, capturing model certainty.
- **Confusion Matrix**: Visualized as publication-ready heatmaps in `results/figures/` and exported to CSV in `results/confusion_matrices/`.
- **F1-Score, Precision, Recall**: Macro-averaged and weighted-averaged scores.
- **Specificity**: Measures True Negative Rate (correct rejection of non-stress).
- **ROC-AUC**: Evaluates discrimination ability across classification thresholds.
- **Latency**: Measures wall-clock training duration and per-sample inference latency (ms).

All models are automatically ranked in `results/model_comparison.csv`.

---

## 7. Mental Health Chatbot & Safety Layer
Implemented in `chatbot/safety/` and `src/inference/`:
- **Crisis Detection (`CrisisDetector`)**: Regex-driven detection of acute suicidal ideation, self-harm, severe trauma, and abuse. Categorizes risk as `NONE`, `LOW`, `MODERATE`, `SEVERE`, `CRITICAL`.
- **Immediate Escalation (`CrisisEscalator`)**: Directly provides verified national and global hotlines:
  - **988** Suicide & Crisis Lifeline (US/Canada)
  - **741741** Crisis Text Line (Text HOME)
  - **Tele-MANAS** 14416 (India)
  - **Samaritans** 116 123 (UK)
  - International directory: https://findahelpline.com
- **Empathetic Response Generation (`MentalHealthResponseGenerator`)**: Generates evidence-based coping interventions (box breathing, 5-4-3-2-1 sensory grounding, cognitive reframing).
- **Ethical Non-Diagnosis Disclaimer**: Strictly enforces that the system is an AI supportive tool and does not provide clinical diagnoses or replace mental health professionals.

---

## 8. Project Structure
```text
ai-chatbot-mental-health/
├── configs/
│   ├── dataset.yaml            # Dataset path, schema, split ratios
│   ├── models.yaml             # Hyperparameters for all 5 models
│   ├── preprocessing.yaml      # NLP pipeline settings
│   ├── training.yaml           # Training configurations
│   └── evaluation.yaml         # Metric selection and plot settings
│
├── data/
│   ├── raw/                    # Raw data location
│   └── processed/              # Cleaned train.csv, val.csv, test.csv
│
├── Dataset/
│   └── Dreaddit_combine_data.csv # Raw Dreaddit mental health dataset
│
├── src/
│   ├── data/                   # Data loader, validator, splitter
│   ├── preprocessing/          # Cleaner, normalizer, stopwords, lemmatizer, pipeline
│   ├── features/               # TF-IDF, BoW, Semantic features
│   ├── models/baseline/        # Random Forest, SVM, Logistic Regression, KNN, Naive Bayes
│   ├── training/               # Baseline trainer, hyperparameter tuning
│   ├── evaluation/             # Metrics, confusion matrix, ROC curve, model comparison
│   ├── inference/              # Predictor, response generator, chatbot engine
│   ├── api/                    # FastAPI backend (main, routes, schemas)
│   ├── utils/                  # Seed, logger, file utils, device
│   └── main.py                 # Master end-to-end pipeline orchestrator
│
├── chatbot/
│   ├── conversation_manager.py # Multi-turn dialogue state
│   └── safety/                 # Crisis detection, rules, hotlines escalation
│
├── scripts/
│   ├── prepare_dataset.py      # Clean, preprocess, and partition dataset
│   ├── train.py                # Train all 5 models and save checkpoints
│   ├── evaluate.py             # Evaluate all models on held-out test set
│   ├── compare_models.py       # Aggregate results into model_comparison.csv
│   └── predict.py              # CLI prediction & interactive chatbot
│
├── results/
│   ├── model_comparison.csv    # Final benchmark comparison table
│   ├── confusion_matrices/     # Matrix CSV files
│   ├── figures/                # Heatmaps, comparison bar charts, ROC curves
│   └── metrics/                # Per-model JSON metrics reports
│
└── tests/                      # Pytest unit tests for all components
```

---

## 9. Step-by-Step Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
Cleans, preprocesses, and creates stratified train/val/test splits:
```bash
python scripts/prepare_dataset.py
```

### Step 3: Train All 5 Models
Trains Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes, validating each and saving checkpoints:
```bash
python scripts/train.py
```

### Step 4: Evaluate on Held-out Test Set
Computes accuracy, confidence, confusion matrices, ROC curves, F1-scores, and generates comparative plots:
```bash
python scripts/evaluate.py
```

### Step 5: Compare Models & Identify Best Model
Aggregates test performance into `results/model_comparison.csv`:
```bash
python scripts/compare_models.py
```

### Step 6: Interactive Chatbot Session
Launch interactive command-line mental health support chat:
```bash
python scripts/predict.py --chat
```
Or run a single inference check:
```bash
python scripts/predict.py --text "I have been having panic attacks and cannot sleep."
```

### Step 7: Launch FastAPI Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is accessible at `http://localhost:8000/docs`.

### Run All Pipeline Steps Automatically
```bash
python src/main.py
```

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## 10. REST API Endpoints
- `GET /health`: System status and model health.
- `POST /predict`: Classify text into stress/non-stress with confidence score and class probabilities.
- `POST /chat`: Multi-turn conversational endpoint with crisis analysis and empathetic guidance.
- `GET /hotlines`: Emergency hotline directory.
- `GET /model-info`: Metadata of currently deployed best model.

---

## 11. Ethical Considerations & Limitations
1. **Non-Clinical System**: This chatbot is intended solely for psychoeducational support, coping exercises, and distress screening. It must not be marketed or used as a medical diagnosis instrument.
2. **Crisis Protocol**: Severe distress triggers immediate emergency intervention and hotlines rather than conversational chit-chat.
3. **Data Bias**: The Dreaddit dataset originates from Reddit communities; phrasing patterns may reflect specific internet demographics.
