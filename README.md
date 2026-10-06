# 🛒 Sentiment Analysis on E-Commerce Product Reviews

An end-to-end NLP pipeline that classifies customer product reviews as **Positive** or **Negative**, covering the full lifecycle from raw data to a deployed Streamlit app.


## Project Overview

| | |
|---|---|
| **Task** | Binary sentiment classification (Positive / Negative) |
| **Domain** | E-commerce product reviews |
| **Approach** | Classical ML (TF-IDF + Naive Bayes / Logistic Regression / SVM / Random Forest / Gradient Boosting) |
| **Deployment** | Streamlit web app |

## Business Problem

Companies receive far more customer review text than any team can manually read. This project builds a system to automatically triage that volume by sentiment — see `reports/phase10_business_insights.md` for the full business framing and actionable insights.

## Dataset

- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle, originally Stanford SNAP)
- **License:** CC0: Public Domain
- **Size:** ~568,000 reviews (full dataset); the included generator can create a 4,000-row synthetic stand-in
- **Target construction:** `Score` 4-5 → positive, `Score` 1-2 → negative, `Score` 3 (neutral) dropped

To use the real data: download `Reviews.csv` from the Kaggle link above and place it at `data/raw/Reviews.csv`, then run the pipeline (below).

## Project Structure

```
Sentiment-Analysis/
├── data/
│├── raw/              # Place a dataset here, or generate the synthetic demo
│└── processed/        # Generated locally after Phase 4
├── notebooks/            # End-to-end Jupyter notebook
├── src/                  # All pipeline source code (see below)
├── models/                # Generated model + vectorizer (.pkl)
├── reports/               # Metrics, error analysis, business insights, screenshots
├── figures/               # EDA visualizations (.png)
├── deployment/            # Deployment-related notes
├── tests/                 # Unit tests
├── app.py                 # Streamlit deployment app
├── requirements.txt
└── .gitignore
```

## Pipeline / How to Run

```bash
pip install -r requirements.txt

cd src
python generate_synthetic_dataset.py   # OPTIONAL: only if you don't have the real Reviews.csv yet
python data_understanding.py           # Phase 3: data inventory report
python text_cleaning.py                # Phase 4: cleaning -> data/processed/cleaned_reviews.csv
python eda.py                          # Phase 5: visualizations -> figures/
python train_models.py                 # Phase 6-8: feature extraction, training, evaluation
python error_analysis.py               # Phase 9: misclassification analysis
python business_insights.py            # Phase 10: business insights report

cd ..
streamlit run app.py                   # Phase 11: launch the web app
```

## Source Code Modules (`src/`)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 2 | Load raw CSV, construct binary sentiment label |
| `data_understanding.py` | 3 | Shape, missing values, duplicates, class balance, length stats |
| `text_cleaning.py` | 4 | Full text cleaning + negation-aware tokenization |
| `eda.py` | 5 | Sentiment distribution, word clouds, n-grams, TF-IDF top terms |
| `train_models.py` | 6-8 | Feature extraction comparison, 5-model training, evaluation |
| `error_analysis.py` | 9 | Misclassification categorization |
| `business_insights.py` | 10 | Business-facing insight report |
| `generate_synthetic_dataset.py` | — | Demo data generator (see data note above) |

## Model Performance

See `reports/model_comparison.csv` and `reports/model_metrics_full.json` for the full comparison across Naive Bayes, Logistic Regression, SVM, Random Forest, and Gradient Boosting (accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation).

## Deployment

The Streamlit app (`app.py`) provides:
- Free-text review input with validation
- One-click example reviews
- Predicted sentiment label + confidence score
- Probability bar chart
- Downloadable CSV prediction report

Screenshots: `reports/screenshot_01_home.png`, `screenshot_02_example_selected.png`, `screenshot_03_prediction_result.png`

## Future Enhancements

- Aspect-Based Sentiment Analysis (ABSA) for feature-level sentiment
- Fine-grained (5-class) sentiment instead of binary
- Transformer-based models (e.g. fine-tuned DistilBERT) for higher accuracy at the cost of interpretability
- Real-time streaming pipeline (Kafka + model microservice) for production-scale review ingestion
- A/B testing sentiment-driven UI changes against business KPIs (return rate, NPS)


> **Portfolio note:** The 4,000-row synthetic stand-in is generated locally and is not committed. Any metrics produced from it are demonstrations only and should not be presented as performance on the full Amazon Fine Food Reviews dataset. The full review dataset is not included.

## Archive note

This repository includes the curated source-only bundle `sentiment-analysis.zip` alongside the original uploaded archive `Sentiment-Analysis-Project_.zip`. The original archive is separate from the curated bundle; review its contents and source terms before reusing or redistributing it.
