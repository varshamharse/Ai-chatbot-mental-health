# Project Structure Prompt — AI Chatbot Mental Health

You are an expert AI/ML engineer and research project architect.

I am working on my **MSc Information Technology (Artificial Intelligence) Final Year Project** titled:

**“AI Chatbot for Mental Health”**

I want to build a complete, research-oriented machine learning system that can **train and evaluate ML/DL models on a mental-health-related dataset and measure model accuracy and other relevant evaluation metrics**.

Create a production-ready and academically suitable project structure for this project.

## 1. Project Objectives

The system should:

- Load and validate the mental-health dataset.
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Handle missing values, duplicates, inconsistent text, and class imbalance.
- Perform NLP preprocessing where required.
- Extract meaningful features from the dataset.
- Train multiple suitable ML/DL models.
- Compare model performance.
- Calculate:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Specificity
  - ROC-AUC where applicable
  - Confusion Matrix
  - Training time
  - Inference time
- Identify the best-performing model.
- Save trained models and preprocessing components.
- Provide a chatbot inference layer using the best model.
- Provide a clear separation between training, evaluation, inference, and UI/API components.
- Generate research-ready results, graphs, tables, and reports.

## 2. Recommended Technology Stack

Use:

- Python 3.11+
- PyTorch and/or TensorFlow
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK / spaCy
- Hugging Face Transformers where appropriate
- FastAPI for backend API
- Streamlit or React/Next.js for frontend
- Jupyter Notebook for experimentation
- YAML/JSON configuration files
- Git/GitHub
- Docker support

Prefer **PyTorch + Hugging Face Transformers** for deep-learning/NLP experiments if the dataset contains conversational or textual data.

## 3. Dataset Requirements

Design the project so that the dataset can be configured rather than hard-coded.

The dataset may contain fields such as:

- user input / question
- chatbot response
- intent
- mental health category
- emotion
- sentiment
- label/class
- conversation ID
- metadata

Do not assume that all these columns exist.

Create a configurable dataset loader that allows the actual dataset schema to be specified in:

`configs/dataset.yaml`

The system must automatically inspect the dataset and provide:

- Number of records
- Number of features
- Data types
- Missing values
- Duplicate records
- Class distribution
- Text length statistics
- Unique labels
- Potential data leakage
- Outliers where applicable

## 4. Project Structure

Create the following recommended structure:

```text
ai-chatbot-mental-health/
│
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── dataset.yaml
│   ├── preprocessing.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── models.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── external/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_deep_learning_models.ipynb
│   ├── 07_transformer_models.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_final_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── validator.py
│   │   ├── splitter.py
│   │   └── dataset.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── text_normalizer.py
│   │   ├── tokenizer.py
│   │   ├── stopwords.py
│   │   ├── lemmatizer.py
│   │   ├── feature_extraction.py
│   │   └── pipeline.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tfidf.py
│   │   ├── bow.py
│   │   ├── embeddings.py
│   │   └── semantic_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline/
│   │   │   ├── logistic_regression.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── svm.py
│   │   │   └── random_forest.py
│   │   │
│   │   ├── deep_learning/
│   │   │   ├── lstm.py
│   │   │   ├── bilstm.py
│   │   │   └── cnn_text.py
│   │   │
│   │   └── transformers/
│   │       ├── bert.py
│   │       ├── roberta.py
│   │       └── distilbert.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── train_baseline.py
│   │   ├── train_dl.py
│   │   ├── train_transformer.py
│   │   └── hyperparameter_tuning.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── roc_curve.py
│   │   ├── evaluator.py
│   │   └── model_comparison.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── chatbot_engine.py
│   │   └── response_generator.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   ├── device.py
│   │   └── file_utils.py
│   │
│   └── main.py
│
├── chatbot/
│   ├── prompts/
│   ├── safety/
│   │   ├── crisis_detection.py
│   │   ├── safety_rules.py
│   │   └── escalation.py
│   └── conversation_manager.py
│
├── models/
│   ├── checkpoints/
│   ├── tokenizer/
│   ├── vectorizers/
│   └── best_model/
│
├── experiments/
│   ├── experiment_001/
│   ├── experiment_002/
│   └── experiment_003/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── predictions/
│   ├── model_comparison.csv
│   └── final_report.csv
│
├── logs/
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_api.py
│
└── scripts/
    ├── prepare_dataset.py
    ├── train.py
    ├── evaluate.py
    ├── compare_models.py
    └── predict.py
```

## 5. Data Processing Pipeline

Implement the following pipeline:

```text
Raw Dataset
     ↓
Dataset Validation
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Missing Value Handling
     ↓
Text Normalization
     ↓
Tokenization
     ↓
Stopword Handling
     ↓
Lemmatization/Stemming
     ↓
Feature Extraction
     ↓
Train/Validation/Test Split
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Chatbot Inference
```

Ensure that preprocessing fitted on training data is not independently fitted on validation/test data.

Prevent data leakage throughout the pipeline.

## 6. Dataset Splitting

Use a reproducible split such as:

- Training: 70%
- Validation: 15%
- Testing: 15%

If the dataset is small, allow configurable alternatives such as:

- 80/10/10
- 70/15/15
- Stratified K-Fold Cross Validation

Use stratification for classification tasks wherever appropriate.

Set a fixed random seed.

## 7. Models

Implement baseline models first:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine
4. Random Forest

Then implement deep-learning models where justified:

1. CNN Text Classifier
2. LSTM
3. BiLSTM

Finally evaluate transformer-based models where computational resources and dataset size justify them:

1. DistilBERT
2. BERT
3. RoBERTa

Do not train every model blindly.

The system should allow models to be enabled/disabled through:

`configs/models.yaml`

## 8. Accuracy Evaluation

Create a standardized evaluation framework.

For every model calculate:

```text
Accuracy
Precision
Recall
F1-Score
Specificity
ROC-AUC
Confusion Matrix
Training Time
Inference Time
```

For multiclass classification, report:

- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1

Do not rely only on accuracy, especially if the dataset is imbalanced.

Generate a final comparison table:

```text
Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | Training Time | Inference Time
```

Save it to:

`results/model_comparison.csv`

## 9. Visualization

Generate research-quality plots:

- Class distribution
- Missing-value analysis
- Text-length distribution
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Model accuracy comparison
- F1-score comparison
- Precision comparison
- Recall comparison
- Training time comparison
- Inference time comparison

Save all figures under:

`results/figures/`

Use high-resolution output suitable for an MSc dissertation.

## 10. Experiment Tracking

Each experiment should record:

- Experiment ID
- Date/time
- Dataset version
- Dataset size
- Preprocessing configuration
- Feature extraction method
- Model
- Hyperparameters
- Random seed
- Training duration
- Validation metrics
- Test metrics
- Model checkpoint path

Store experiment metadata in a structured format such as JSON or CSV.

## 11. Reproducibility

The entire project must be reproducible.

Implement:

- Fixed random seeds
- Configuration-driven experiments
- Versioned preprocessing
- Versioned model configurations
- Saved tokenizer/vectorizer
- Saved label encoder
- Saved model checkpoints
- Logged experiment parameters
- Logged evaluation metrics

A single command should be able to execute the complete training/evaluation pipeline.

Example:

```bash
python scripts/train.py
```

And evaluation:

```bash
python scripts/evaluate.py
```

Model comparison:

```bash
python scripts/compare_models.py
```

Prediction:

```bash
python scripts/predict.py
```

## 12. Chatbot Architecture

Separate the ML classification model from the conversational layer.

Use:

```text
User Message
      ↓
Input Validation
      ↓
Text Preprocessing
      ↓
Mental Health Intent/Category Detection
      ↓
Risk/Crisis Detection
      ↓
Response Selection/Generation
      ↓
Safety Filtering
      ↓
Final Response
```

The chatbot must not claim to diagnose a medical or psychiatric condition.

Include appropriate safety mechanisms for:

- Self-harm
- Suicidal ideation
- Crisis situations
- Severe distress
- Emergency situations

For high-risk messages, the system should provide an appropriate crisis/escalation response rather than attempting diagnosis or treatment.

## 13. API

Create a FastAPI backend with endpoints such as:

```text
GET  /health
POST /predict
POST /chat
POST /evaluate
GET  /model-info
```

Example `/predict` response:

```json
{
  "input": "I have been feeling very anxious lately",
  "predicted_class": "anxiety",
  "confidence": 0.91
}
```

Do not expose sensitive training data through API responses.

## 14. Model Selection

Create an automated model-selection process.

The best model should not be selected based only on accuracy.

Use a configurable primary metric such as:

```text
F1-score
```

and consider:

- Recall
- Specificity
- ROC-AUC
- Class imbalance
- Inference latency
- Model size

The final selected model must be justified using quantitative results.

## 15. Research Output

Generate files that can be directly used in the MSc project documentation:

```text
results/
├── dataset_statistics.csv
├── preprocessing_summary.csv
├── model_comparison.csv
├── final_metrics.csv
├── predictions.csv
├── confusion_matrices/
├── roc_curves/
├── figures/
└── experiment_summary.json
```

Also generate a final research summary containing:

- Dataset description
- Preprocessing methodology
- Models evaluated
- Experimental setup
- Hyperparameters
- Results
- Best model
- Limitations
- Future improvements

## 16. Important Research Requirements

Follow these rules:

1. Do not fabricate dataset results.
2. Do not claim a model has high accuracy until it has actually been trained and evaluated.
3. Clearly distinguish validation performance from test performance.
4. Never use the test set during model tuning.
5. Prevent data leakage.
6. Report class imbalance.
7. Use reproducible experiments.
8. Keep raw data unchanged.
9. Save processed datasets separately.
10. Log every experiment.
11. Use confidence intervals or cross-validation where appropriate.
12. Report limitations honestly.
13. Do not make medical diagnosis claims.
14. Treat the chatbot as an AI support/assistance system rather than a replacement for a qualified mental-health professional.

## 17. Deliverables

Create the complete project with:

- Clean project structure
- Dataset loader
- Dataset validator
- EDA
- Preprocessing pipeline
- Feature extraction
- Baseline ML models
- Deep-learning models
- Transformer models where appropriate
- Training pipeline
- Evaluation pipeline
- Model comparison
- Visualization
- Experiment tracking
- Model checkpointing
- Inference system
- Mental-health chatbot layer
- Safety layer
- FastAPI backend
- Unit tests
- Docker configuration
- Complete README
- Research-ready result files

## 18. Implementation Priority

Implement in this order:

### Phase 1 — Dataset

```text
Dataset Loading
→ Validation
→ EDA
→ Cleaning
→ Dataset Statistics
```

### Phase 2 — Preprocessing

```text
Text Cleaning
→ Tokenization
→ Feature Extraction
→ Train/Validation/Test Split
```

### Phase 3 — Baseline Models

```text
Logistic Regression
→ Naive Bayes
→ SVM
→ Random Forest
```

### Phase 4 — Deep Learning

```text
CNN
→ LSTM
→ BiLSTM
```

### Phase 5 — Transformers

```text
DistilBERT
→ BERT
→ RoBERTa
```

Only implement transformer models if the dataset and available GPU/compute resources make them practical.

### Phase 6 — Evaluation

```text
Accuracy
→ Precision
→ Recall
→ F1
→ Specificity
→ ROC-AUC
→ Confusion Matrix
→ Model Comparison
```

### Phase 7 — Chatbot

```text
Best Model
→ Prediction API
→ Intent/Category Detection
→ Risk Detection
→ Response Generation
→ Safety Filtering
```

### Phase 8 — Final Research Results

Generate all tables, figures, metrics, experiment logs, and conclusions required for the MSc final-year project.

## Final Requirement

Before implementing the project, inspect the actual dataset structure and determine:

1. Dataset format
2. Number of records
3. Number of columns
4. Text columns
5. Target/label column
6. Number of classes
7. Class distribution
8. Missing values
9. Duplicate records
10. Whether the task is binary or multiclass classification
11. Whether the dataset is suitable for the proposed models
12. Whether there is any risk of data leakage

Do not make assumptions about the dataset.

Build the implementation around the **actual dataset structure** after inspection.

The final system should be suitable for an **MSc Information Technology (Artificial Intelligence) final-year research project**, with reproducible experiments, scientifically valid evaluation, and clearly documented model accuracy.