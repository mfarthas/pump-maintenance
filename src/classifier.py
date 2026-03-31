"""
Classifier with and without synthetic augmentation.

Trains two Random Forest classifiers — Model A on real data only, Model B
on real + synthetic faults — evaluates both on a time-based held-out test set,
saves metrics to results/metrics/classifier_comparison.json, and plots ROC curves.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
IN_FEATURES = ROOT / "data" / "features.parquet"
IN_FAULT_BANK = ROOT / "data" / "fault_bank.parquet"
OUT_METRICS = ROOT / "results" / "metrics" / "classifier_comparison.json"
OUT_FIG = ROOT / "results" / "figures" / "roc_comparison.png"
OUT_MODEL_B = ROOT / "data" / "model_b.pkl"

TEST_FRACTION = 0.20
RNG_SEED = 42


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in ("machine_status", "is_synthetic")]


def build_labels(df):
    """1 = fault (BROKEN or RECOVERING or SYNTHETIC_FAULT), 0 = NORMAL."""
    return (df["machine_status"] != "NORMAL").astype(int)


def evaluate(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else 0.0
    print(f"\n{name}:")
    print(f"  Precision : {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}  AUC-ROC: {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4), "auc_roc": round(auc, 4)}, fpr, tpr


def main():

    df = pd.read_parquet(IN_FEATURES)
    fault_bank = pd.read_parquet(IN_FAULT_BANK)

    feat_cols = get_feature_cols(df)

    # Time-based split: put 80% of fault events in train, 20% in test.
    # A plain last-20% split leaves the test set with zero fault rows because
    # all BROKEN/RECOVERING events are clustered in the first ~80% of the timeline.
    fault_positions = np.where(df["machine_status"].isin(["BROKEN", "RECOVERING"]).values)[0]
    if len(fault_positions) == 0:
        n_train = int(len(df) * (1 - TEST_FRACTION))
    else:
        n_train = int(fault_positions[int(len(fault_positions) * 0.80)])

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    X_test = test_df[feat_cols].ffill().bfill().values
    y_test = build_labels(test_df).values

    print(f"Test fault rate: {y_test.mean():.3f}")

    X_train_A = train_df[feat_cols].ffill().bfill().values
    y_train_A = build_labels(train_df).values
    print(f"Model A — NORMAL: {(y_train_A == 0).sum()}  FAULT: {(y_train_A == 1).sum()}")

    clf_A = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG_SEED,
                                   class_weight="balanced")
    print("Training Model A ...")
    clf_A.fit(X_train_A, y_train_A)

    # --- Model B: real + synthetic ---
    synth_only = fault_bank[fault_bank["is_synthetic"] == True].copy()
    synth_feat_cols = [c for c in feat_cols if c in synth_only.columns]
    synth_X = synth_only[synth_feat_cols].fillna(0.0).values
    synth_y = np.ones(len(synth_X), dtype=int)

    X_train_B = np.vstack([X_train_A, synth_X])
    y_train_B = np.concatenate([y_train_A, synth_y])
    print(f"Model B — NORMAL: {(y_train_B == 0).sum()}  FAULT: {(y_train_B == 1).sum()}")

    clf_B = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RNG_SEED,
                                   class_weight="balanced")
    print("Training Model B ...")
    clf_B.fit(X_train_B, y_train_B)

    joblib.dump(clf_B, OUT_MODEL_B)
    print(f"Model B saved to {OUT_MODEL_B}")

    # Evaluate
    metrics_A, fpr_A, tpr_A = evaluate("Model A (real only)", clf_A, X_test, y_test)
    metrics_B, fpr_B, tpr_B = evaluate("Model B (real + synthetic)", clf_B, X_test, y_test)

    results = {"model_A_real_only": metrics_A, "model_B_real_plus_synthetic": metrics_B}
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS, "w") as fh:
        json.dump(results, fh, indent=2)

    # ROC curves
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_A, tpr_A, lw=2, color="steelblue",
            label=f"Model A — real only  (AUC={metrics_A['auc_roc']:.3f})")
    ax.plot(fpr_B, tpr_B, lw=2, color="darkorange",
            label=f"Model B — real+synth (AUC={metrics_B['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model A vs Model B", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
