"""
Wearable Fatigue Detection — ML Training Pipeline (local CSV version)
========================================================================

Trains a classifier to predict fatigue from your wearable's sensor
readings, using your ground-truth button presses ("marked") as labels.

This version works directly with the all_sensors_*.csv files saved by
log_all_sensors.py — no cloud/ThingSpeak needed. It automatically finds
and combines ALL matching session files in the same folder, so you can
just keep dropping new recording sessions in and re-run this.

SETUP:
    pip install pandas numpy scikit-learn matplotlib joblib

USAGE:
    Put this script in the same folder as your all_sensors_*.csv files,
    then run:  python train_fatigue_model.py
"""

import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib


# ============================================================
# CONFIG
# ============================================================

# Matches all_sensors_20260913_233848.csv, all_sensors_20260908_032437.csv, etc.
FILE_PATTERN = "all_sensors_*.csv"

FEATURE_COLS = ["ppg_bpm", "motion_magnitude", "ecg_bpm", "ecg_hrv", "skin_temp"]


# ============================================================
# STEP 1: LOAD AND COMBINE ALL SESSION FILES
# ============================================================

def load_all_sessions(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in this folder. "
            "Make sure this script is in the same directory as your CSVs."
        )

    print(f"Found {len(files)} session file(s):")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f
        print(f"  {f}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined total: {len(combined)} rows across all sessions")
    return combined


# ============================================================
# STEP 2: CLEAN
# ============================================================

def clean_data(df):
    numeric_cols = FEATURE_COLS + ["ecg_leads_off", "fatigue_alert", "marked"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    # Drop rows with no usable heart signal at all
    df = df[~((df["ppg_bpm"] == 0) & (df["ecg_leads_off"] == 1))]

    # Drop rows missing anything we need
    df = df.dropna(subset=FEATURE_COLS + ["marked"])

    after = len(df)
    print(f"\nCleaned: kept {after} of {before} rows "
          f"({before - after} dropped for missing/unusable readings)")

    return df


# ============================================================
# STEP 3: FEATURES + LABELS
# ============================================================

def prepare_features(df):
    X = df[FEATURE_COLS]
    y = df["marked"].astype(int)

    print("\nLabel distribution:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError(
            "Only one class present — you need both marked and unmarked "
            "moments in your data to train a classifier."
        )

    return X, y


# ============================================================
# STEP 4: TRAIN
# ============================================================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Trivial baseline: what accuracy would "always guess the majority
    # class" get? Your model needs to clearly beat this to mean anything.
    majority_baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nTraining accuracy:      {train_acc:.2f}")
    print(f"Test accuracy:          {test_acc:.2f}")
    print(f"Trivial baseline would: {majority_baseline:.2f} "
          f"(always guessing the majority class)")

    if test_acc <= majority_baseline + 0.03:
        print("⚠ Test accuracy is barely above (or below) the trivial "
              "baseline — the model isn't learning much real signal yet. "
              "More labeled data would likely help.")
    else:
        print("✓ Test accuracy meaningfully beats the trivial baseline — "
              "the model is learning a real pattern, not just guessing "
              "the majority class.")

    if train_acc - test_acc > 0.2:
        print("⚠ Training accuracy is much higher than test accuracy — "
              "sign of overfitting, common with small datasets.")

    print("\nClassification report (test set):")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=["normal", "fatigued"],
                                 zero_division=0))

    print("Confusion matrix (test set):")
    print(confusion_matrix(y_test, model.predict(X_test)))

    return model


# ============================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(model):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(FEATURE_COLS)), importances[order])
    plt.xticks(range(len(FEATURE_COLS)), [FEATURE_COLS[i] for i in order], rotation=30)
    plt.ylabel("Importance")
    plt.title("Which signals matter most for predicting fatigue")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nSaved feature_importance.png")

    print("\nFeature importances:")
    for i in order:
        print(f"  {FEATURE_COLS[i]}: {importances[i]:.3f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_all_sessions(FILE_PATTERN)
    df = clean_data(df)
    X, y = prepare_features(df)
    model = train_model(X, y)
    plot_feature_importance(model)

    joblib.dump(model, "fatigue_model.pkl")
    print("\nSaved trained model to fatigue_model.pkl")
    print("Reload it later with: joblib.load('fatigue_model.pkl')")
