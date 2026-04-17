"""
train_model.py
--------------
Model Training, Evaluation & Saving
Trains multiple classifiers and selects the best one.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script mode
import matplotlib.pyplot as plt

from src.preprocess import (
    load_data, clean_data, engineer_features,
    encode_features, get_features_target, scale_features, FEATURE_COLS
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODEL_PATH   = "models/best_model.pkl"
RESULTS_PATH = "outputs/model_results.json"
CM_IMG_PATH  = "images/confusion_matrix.png"
FI_IMG_PATH  = "images/feature_importance.png"

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("images",  exist_ok=True)

# ─────────────────────────────────────────────
# CANDIDATE MODELS
# ─────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                       max_depth=5, random_state=42),
}

LABEL_NAMES = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────
# TRAIN & COMPARE ALL MODELS
# ─────────────────────────────────────────────
def train_and_compare(X_train_s, X_test_s, y_train, y_test):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in MODELS.items():
        print(f"\n🔄 Training: {name}")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc    = accuracy_score(y_test, y_pred)
        cv_acc = cross_val_score(model, X_train_s, y_train,
                                  cv=cv, scoring="accuracy").mean()

        results[name] = {
            "accuracy"   : round(acc, 4),
            "cv_accuracy": round(cv_acc, 4),
            "model"      : model,
            "y_pred"     : y_pred
        }
        print(f"   Test Accuracy : {acc:.4f}")
        print(f"   CV   Accuracy : {cv_acc:.4f}")

    return results


# ─────────────────────────────────────────────
# PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name, save_path=CM_IMG_PATH):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# PLOT FEATURE IMPORTANCE (Random Forest / GB)
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, save_path=FI_IMG_PATH):
    if not hasattr(model, "feature_importances_"):
        print("ℹ️  Model has no feature_importances_ attribute; skipping plot.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    ax.bar(range(len(feature_names)), importances[idx], color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importance Score")
    ax.set_title("Feature Importance — Top Predictors of Employee Performance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Feature importance saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_training(data_path="data/employee_data.csv"):
    # ── 1. Load & Preprocess ──────────────────
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_features(df)

    X, y = get_features_target(df)

    # ── 2. Train / Test Split ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Scale ──────────────────────────────
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)

    # ── 4. Train all models ───────────────────
    results = train_and_compare(X_train_s, X_test_s, y_train, y_test)

    # ── 5. Pick best model by test accuracy ───
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_info = results[best_name]
    best_model = best_info["model"]
    y_pred_best = best_info["y_pred"]

    print(f"\n🏆 Best Model : {best_name}")
    print(f"   Accuracy   : {best_info['accuracy']:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=LABEL_NAMES))

    # ── 6. Save best model ────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")

    # ── 7. Save results JSON ──────────────────
    summary = {k: {"accuracy": v["accuracy"], "cv_accuracy": v["cv_accuracy"]}
               for k, v in results.items()}
    summary["best_model"] = best_name
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Results saved → {RESULTS_PATH}")

    # ── 8. Plots ──────────────────────────────
    plot_confusion_matrix(y_test, y_pred_best, best_name)
    plot_feature_importance(best_model, FEATURE_COLS)

    return best_model, results


if __name__ == "__main__":
    run_training()
