"""
Anomaly detector baseline.

Trains an Isolation Forest on NORMAL-only windows from features.parquet,
scores all windows, evaluates against true machine_status labels, saves
metrics to results/metrics/baseline_anomaly.json, and plots anomaly score
over time with BROKEN windows highlighted.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
IN_FEATURES = ROOT / "data" / "features.parquet"
OUT_METRICS = ROOT / "results" / "metrics" / "baseline_anomaly.json"
OUT_FIG = ROOT / "results" / "figures" / "anomaly_baseline.png"


def get_feature_cols(df):
    return [c for c in df.columns if c != "machine_status"]


def main():

    df = pd.read_parquet(IN_FEATURES)
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].ffill().bfill().values

    normal_mask = df["machine_status"] == "NORMAL"
    X_train = X[normal_mask.values]
    print(f"Train (NORMAL): {X_train.shape[0]}  Eval (all): {X.shape[0]}")
    clf = IsolationForest(contamination=0.01, n_estimators=100,
                          random_state=42, n_jobs=-1)
    clf.fit(X_train)

    # score_samples returns higher = more normal
    raw_scores = clf.score_samples(X)
    # Invert so higher = more anomalous
    anomaly_score = -raw_scores
    predictions = clf.predict(X)  # 1=normal, -1=anomaly
    pred_labels = (predictions == -1).astype(int)  # 1=anomaly

    # Use BROKEN + RECOVERING as fault class — only 7 pure BROKEN rows exist,
    # RECOVERING (14K rows) is the meaningful fault signal in this dataset.
    true_labels = (df["machine_status"].isin(["BROKEN", "RECOVERING"])).astype(int).values

    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    metrics = {"precision": round(precision, 4),
               "recall": round(recall, 4),
               "f1": round(f1, 4)}

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Plot
    fault_mask = df["machine_status"].isin(["BROKEN", "RECOVERING"])

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df.index, anomaly_score, linewidth=0.4, color="steelblue",
            label="Anomaly score")
    ax.fill_between(df.index, anomaly_score.min(), anomaly_score.max(),
                    where=fault_mask.values, color="red", alpha=0.3,
                    label="BROKEN/RECOVERING (true)")
    ax.set_title("Isolation Forest anomaly score — red = true fault events", fontsize=12)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Anomaly score (higher = more anomalous)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
