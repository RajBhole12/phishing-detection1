"""
train_models.py
----------------
Full training pipeline:
  Dataset -> Clean -> Extract features -> Split -> Train ML models
  -> Evaluate -> Pick best model -> Save everything to /models

Algorithms used (matches the standard MSBTE diploma Machine Learning
curriculum -- classic supervised classification algorithms, no ensemble
methods and no deep learning):
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (Gaussian)
  - Support Vector Machine (SVM)

Run directly:
    python src/train_models.py

Or call run_training_pipeline() from the Flask app (used by the
"Train Models" button / POST /api/train).
"""

import os
import sys
import json
import time
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

from src.preprocessing import get_train_test_data, DatasetNotFoundError

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model.json")

# Models that are distance/gradient-based benefit from scaled features.
# Decision Tree is scale-invariant, so it trains on the raw features.
SCALED_MODELS = {"Logistic Regression", "KNN", "Naive Bayes", "SVM"}


def get_model_definitions():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist()[::max(1, len(fpr)//50)], "tpr": tpr.tolist()[::max(1, len(tpr)//50)]},
    }


def run_training_pipeline(verbose=True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()

    data = get_train_test_data()
    joblib.dump(data["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(data["feature_names"], os.path.join(MODEL_DIR, "feature_names.pkl"))

    results = {}
    trained = {}

    for name, model in get_model_definitions().items():
        if verbose:
            print(f"Training {name}...")
        if name in SCALED_MODELS:
            model.fit(data["X_train_scaled"], data["y_train"])
            metrics = evaluate_model(model, data["X_test_scaled"], data["y_test"])
        else:
            model.fit(data["X_train"], data["y_train"])
            metrics = evaluate_model(model, data["X_test"], data["y_test"])

        results[name] = metrics
        trained[name] = model

        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))

    # Feature importance -- Decision Tree is the only model in this set
    # that natively exposes feature_importances_.
    if "Decision Tree" in trained:
        importances = trained["Decision Tree"].feature_importances_
        feat_importance = sorted(
            zip(data["feature_names"], importances.tolist()),
            key=lambda x: x[1], reverse=True
        )
        with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
            json.dump(feat_importance, f, indent=2)

    # Best model = highest ROC-AUC (a good all-round metric for binary classification).
    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])

    best_info = {
        "best_model": best_name,
        "metric_used": "roc_auc",
        "trained_at": time.time(),
        "training_duration_sec": round(time.time() - t0, 2),
        "dataset_size": data["dataset_size"],
        "n_features": data["n_features"],
        "n_train": data["n_train"],
        "n_test": data["n_test"],
        "class_distribution": data["class_distribution"],
        "algorithms_used": list(get_model_definitions().keys()),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(BEST_MODEL_INFO_PATH, "w") as f:
        json.dump(best_info, f, indent=2)

    if verbose:
        print(f"\nBest model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")
        print(f"Training complete in {best_info['training_duration_sec']}s")

    return results, best_info


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except DatasetNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
