"""
Uncertainty quantification for the fault classifier using split conformal prediction.

Split conformal prediction wraps any trained classifier without retraining: a held-out
calibration set is used to compute nonconformity scores, and a quantile threshold (q_hat)
determines which classes are included in each prediction set. This guarantees that the
true label is contained in the prediction set at least (1 - alpha) of the time on new data,
with no distributional assumptions.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
IN_FEATURES = ROOT / "data" / "features.parquet"
IN_MODEL_B = ROOT / "data" / "model_b.pkl"
OUT_METRICS = ROOT / "results" / "metrics" / "uncertainty_coverage.json"
OUT_FIG = ROOT / "results" / "figures" / "uncertainty_timeline.png"

ALPHA = 0.10
TEST_FRACTION = 0.20


def get_feature_cols(df):
    return [c for c in df.columns if c not in ("machine_status", "is_synthetic")]


def build_labels(df):
    """1 = fault (BROKEN or RECOVERING or SYNTHETIC_FAULT), 0 = NORMAL."""
    return (df["machine_status"] != "NORMAL").astype(int)


def main():
    # --- Load data and model ---
    print("Loading features and Model B ...")
    df = pd.read_parquet(IN_FEATURES)
    model_b = joblib.load(IN_MODEL_B)
    print(f"  Loaded {len(df)} rows. Model B: {model_b}")

    feat_cols = get_feature_cols(df)

    # --- Reconstruct time-based train/test split (mirrors classifier.py) ---
    print("Reconstructing train/test split ...")
    fault_positions = np.where(df["machine_status"].isin(["BROKEN", "RECOVERING"]).values)[0]
    if len(fault_positions) == 0:
        n_train = int(len(df) * (1 - TEST_FRACTION))
    else:
        n_train = int(fault_positions[int(len(fault_positions) * 0.80)])

    test_df = df.iloc[n_train:].copy().reset_index(drop=False)
    print(f"  Test set size: {len(test_df)}")

    # --- Split test set into calibration (first half) and evaluation (second half) ---
    mid = len(test_df) // 2
    cal_df = test_df.iloc[:mid]
    eval_df = test_df.iloc[mid:]
    print(f"  Calibration: {len(cal_df)}  Evaluation: {len(eval_df)}")

    X_cal = cal_df[feat_cols].ffill().bfill().values
    y_cal = build_labels(cal_df).values
    X_eval = eval_df[feat_cols].ffill().bfill().values
    y_eval = build_labels(eval_df).values

    # --- Compute nonconformity scores on calibration set ---
    print("Computing nonconformity scores ...")
    proba_cal = model_b.predict_proba(X_cal)
    scores = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]

    q_hat = np.quantile(scores, 1.0 - ALPHA)
    print(f"  q_hat (alpha={ALPHA}): {q_hat:.4f}")

    # Warn if score distribution looks degenerate
    if np.std(scores) < 1e-4:
        print("  WARNING: nonconformity scores have near-zero variance — model may be miscalibrated.")

    # --- Build prediction sets for evaluation set ---
    print("Building prediction sets on evaluation set ...")
    proba_eval = model_b.predict_proba(X_eval)
    n_classes = proba_eval.shape[1]

    # prediction_sets[i] = list of classes included for example i
    in_set = (1.0 - proba_eval) <= q_hat  # shape (n_eval, n_classes)

    # Coverage: true label is in prediction set
    covered = in_set[np.arange(len(y_eval)), y_eval]
    empirical_coverage = covered.mean()

    # Uncertain: prediction set contains both classes
    is_uncertain = in_set.all(axis=1) if n_classes == 2 else (in_set.sum(axis=1) > 1)

    uncertain_fraction = is_uncertain.mean()

    fault_mask = y_eval == 1
    uncertain_fault_fraction = is_uncertain[fault_mask].mean() if fault_mask.sum() > 0 else 0.0

    print(f"\nResults:")
    print(f"  Empirical coverage : {empirical_coverage:.4f}  (target >= {1 - ALPHA:.2f})")
    print(f"  Uncertain fraction : {uncertain_fraction:.4f}")
    print(f"  Uncertain fault fraction: {uncertain_fault_fraction:.4f}")

    if not (0.85 <= empirical_coverage <= 0.99):
        print("  WARNING: empirical coverage is outside [0.85, 0.99] — check calibration logic.")

    # --- Save metrics ---
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "alpha": ALPHA,
        "target_coverage": 1.0 - ALPHA,
        "empirical_coverage": round(float(empirical_coverage), 4),
        "uncertain_fraction": round(float(uncertain_fraction), 4),
        "uncertain_fault_fraction": round(float(uncertain_fault_fraction), 4),
        "q_hat": round(float(q_hat), 6),
    }
    with open(OUT_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nMetrics saved to {OUT_METRICS}")

    # --- Plot uncertainty timeline ---
    print("Plotting uncertainty timeline ...")
    fault_prob = proba_eval[:, 1]

    # Recover timestamps if available
    time_index = eval_df.index if "timestamp" not in eval_df.columns else eval_df["timestamp"]
    x = np.arange(len(eval_df))

    sns.set_style("darkgrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    # Shade true BROKEN windows on both panels
    is_broken = eval_df["machine_status"].isin(["BROKEN", "RECOVERING"]).values
    in_fault = False
    fault_start = None
    for i, broken in enumerate(np.append(is_broken, False)):
        if broken and not in_fault:
            fault_start = i
            in_fault = True
        elif not broken and in_fault:
            for ax in (ax_top, ax_bot):
                ax.axvspan(fault_start, i, color="salmon", alpha=0.25, label="_fault_bg")
            in_fault = False

    # Top panel: fault probability
    ax_top.plot(x, fault_prob, color="steelblue", lw=1.2, label="Fault probability")
    ax_top.axhline(0.5, color="gray", lw=0.8, linestyle="--")
    ax_top.set_ylabel("P(fault)")
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.legend(loc="upper right", fontsize=9)

    # Add a single legend entry for fault windows
    from matplotlib.patches import Patch
    fault_patch = Patch(color="salmon", alpha=0.4, label="True fault window")
    ax_top.legend(handles=[ax_top.lines[0], fault_patch], loc="upper right", fontsize=9)

    # Bottom panel: uncertain flag as binary bars
    ax_bot.bar(x, is_uncertain.astype(int), width=1.0, color="crimson",
               align="edge", label="Uncertain")
    ax_bot.set_ylabel("Uncertain")
    ax_bot.set_yticks([0, 1])
    ax_bot.set_xlabel("Evaluation timestep")

    fig.suptitle("Fault Prediction with Conformal Uncertainty — 90% Coverage Target", fontsize=13)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"Figure saved to {OUT_FIG}")


if __name__ == "__main__":
    main()
