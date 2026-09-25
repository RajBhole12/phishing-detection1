# PhishGuard AI — AI-Based Phishing Website Detection System

A full-stack cybersecurity web application that analyzes website URLs with
trained Machine Learning models and returns a real, explainable prediction
(Legitimate / Suspicious / Phishing) with a risk score, confidence, model
attribution, and feature-level reasoning.

This is a URL-detection-only build: no email/message scanning, no deep
learning — the algorithm set is deliberately restricted to the standard
classical ML algorithms covered in a diploma-level (MSBTE) Machine Learning
curriculum.

## Algorithms Used

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Support Vector Machine (SVM)

All five are trained and evaluated on the same train/test split; the best
performer (by ROC-AUC) is automatically selected as the active prediction
model. No ensemble methods (Random Forest, Gradient Boosting) and no deep
learning (TensorFlow/Keras) are used.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Provide a dataset (see "Dataset" below), then:
python src/generate_dataset.py   # only runs if data/phishing.csv doesn't already exist
python src/train_models.py       # trains all 5 models, saves them to /models
python src/evaluate_models.py    # generates evaluation plots into static/images

# 2. Run the app
python app.py
```

Then open **http://localhost:5000**.

## Troubleshooting installation

**"No module named flask" even after installing packages manually:** make
sure `pip install` and `python app.py` use the exact same Python
interpreter. Use `python -m pip install -r requirements.txt` and
`python app.py` with the same `python` command each time, or use a virtual
environment:
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Running via Streamlit (`streamlit run app.py`) will fail** — this is a
Flask app, not a Streamlit app. Run it with `python app.py` and open it in
your browser at http://localhost:5000.

## Dataset

Expects `data/phishing.csv` with two columns: `url,label` (`label`: `1` =
phishing, `0` = legitimate). Real-world datasets you can drop in as-is
include Kaggle's "Phishing Website Detector" dataset, PhishTank exports, or
the UCI Phishing Websites dataset (reshaped to `url,label`).

If no dataset is present, `src/generate_dataset.py` builds a
structurally-realistic starter dataset (~2,400 URLs) by composing real
legitimate domains with known phishing URL construction patterns (IP-hosted
links, brand names stuffed into subdomains, URL shorteners, typosquatting,
suspicious keywords). Labels are assigned by construction, and every model
metric shown in the app (accuracy, precision, recall, F1, ROC-AUC,
confusion matrices) is computed for real from an actual train/test split —
nothing is hard-coded. For production use, swap in a real dataset.

## How Predictions Work

1. The submitted URL string is parsed and 31 lexical/structural features
   are extracted (`src/features/url_features.py`) — length, entropy,
   subdomain count, IP-address hosting, suspicious keywords, etc.
2. **The target site is never opened, fetched, or executed** — this is
   string-level analysis only.
3. The best model (selected automatically by ROC-AUC during training)
   scores the feature vector, producing a phishing probability.
4. Because all five models above are **binary classifiers**, their raw
   probability tends to cluster near 0% or 100%. To get a genuinely
   three-level result, the app blends a temperature-calibrated version of
   that probability with a transparent, feature-based heuristic risk score
   (see `compute_feature_risk_score` in `url_features.py`) into a final
   0–100 risk score:
   - 0–35 → Legitimate
   - 36–70 → Suspicious
   - 71–100 → Phishing

   "Suspicious" is a risk tier applied on top of a binary model's output —
   none of the underlying models are trained as 3-class classifiers.

## Project Structure

```
phishing-detection/
├── app.py                     # Flask app + API routes
├── requirements.txt
├── data/phishing.csv          # dataset (user-supplied or generated)
├── models/                    # trained models, scaler, metrics (generated)
├── database/
│   ├── database.py            # SQLite persistence layer
│   └── scans.db                # generated on first run
├── src/
│   ├── features/url_features.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── generate_dataset.py
├── templates/                 # Jinja2 pages
└── static/{css,js,images}/
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | `{ "url": "..." }` → prediction, risk score, confidence, features, reasons |
| `/api/history` | GET / DELETE | list / clear scan history (supports `search`, `prediction`, `sort_by`, `sort_dir`) |
| `/api/statistics` | GET | dashboard totals, recent scans, activity over time |
| `/api/model-performance` | GET | metrics for every trained model + best-model info |
| `/api/train` | POST | kicks off a background training run |
| `/api/train-status` | GET | polling endpoint for training progress |

## Notes

- If models haven't been trained, the scanner clearly tells the user to
  train them rather than faking a prediction.
- If the dataset is missing, training clearly reports that instead of
  generating fake results.
- All database queries are parameterized (no SQL injection surface).
