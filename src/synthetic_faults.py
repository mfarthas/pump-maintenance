"""
Step 5 - Synthetic fault generation with KMeansSMOTE + ENN filtering.

Uses KMeansSMOTE to generate synthetic fault samples. KMeansSMOTE first
clusters the full dataset so that the NORMAL class acts as a spatial boundary
reference — synthetic faults are only interpolated within minority-dominated
clusters, preventing cross-cluster generation into NORMAL space.

After generation, EditedNearestNeighbours (ENN) filters out any synthetic
sample whose k nearest neighbours in the original data are majority-NORMAL —
i.e. samples that crossed the decision boundary during interpolation.

This two-step approach (KMeansSMOTE → ENN) is equivalent to SMOTEENN but
with explicit tracking so only the clean synthetic rows are saved to the
fault bank, keeping the fault_bank.parquet interface compatible with step 6.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import EditedNearestNeighbours

ROOT = Path(__file__).resolve().parents[1]
IN_FEATURES = ROOT / "data" / "features.parquet"
OUT_FAULT_BANK = ROOT / "data" / "fault_bank.parquet"
OUT_FIG = ROOT / "results" / "figures" / "synthetic_vs_real.png"

SYNTHETIC_MULTIPLIER = 5
RNG_SEED = 42


def get_feature_cols(df):
    return [c for c in df.columns if c != "machine_status"]


def main():

    df = pd.read_parquet(IN_FEATURES)
    feat_cols = get_feature_cols(df)

    fault_mask = df["machine_status"].isin(["BROKEN", "RECOVERING"])
    n_real = fault_mask.sum()
    n_normal = (~fault_mask).sum()
    print(f"Real faults: {n_real}  Normal: {n_normal}")

    X_full = df[feat_cols].ffill().bfill().values
    y_full = fault_mask.astype(int).values

    target_n_fault = n_real * SYNTHETIC_MULTIPLIER
    sampling_ratio = target_n_fault / n_normal
    cluster_thresh = (n_real / len(df)) * 0.5

    print(f"KMeansSMOTE target: {target_n_fault} faults (ratio={sampling_ratio:.3f}, "
          f"cluster_thresh={cluster_thresh:.4f})")
    kmsmote = KMeansSMOTE(
        sampling_strategy=sampling_ratio,
        k_neighbors=5,
        cluster_balance_threshold=cluster_thresh,
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    X_over, y_over = kmsmote.fit_resample(X_full, y_full)

    # Synthetic rows are appended after the original n rows by imblearn
    n_orig = len(X_full)
    n_generated = len(X_over) - n_orig
    X_synthetic = X_over[n_orig:]
    print(f"Generated: {n_generated} synthetic samples")
    X_combined = np.vstack([X_full, X_synthetic])
    y_combined = np.concatenate([y_full, np.ones(n_generated, dtype=int)])

    enn = EditedNearestNeighbours(n_neighbors=5)
    enn.fit_resample(X_combined, y_combined)

    # sample_indices_ gives the rows from X_combined that SURVIVED ENN
    kept_indices = enn.sample_indices_
    synthetic_start = n_orig
    # Indices >= synthetic_start correspond to synthetic rows
    synthetic_survived_local = kept_indices[kept_indices >= synthetic_start] - synthetic_start
    X_synthetic_clean = X_synthetic[synthetic_survived_local]

    n_removed = n_generated - len(X_synthetic_clean)
    print(f"ENN removed: {n_removed} ({n_removed / n_generated * 100:.1f}%)  "
          f"Retained: {len(X_synthetic_clean)}")

    # --- Build fault bank ---
    real_faults = df[fault_mask].reset_index(drop=True).copy()
    real_faults["is_synthetic"] = False

    synth_df = pd.DataFrame(X_synthetic_clean, columns=feat_cols)
    synth_df["machine_status"] = "SYNTHETIC_FAULT"
    synth_df["is_synthetic"] = True

    fault_bank = pd.concat([real_faults, synth_df], ignore_index=True)
    real_count = (~fault_bank['is_synthetic']).sum()
    synth_count = fault_bank['is_synthetic'].sum()
    print(f"Fault bank: {real_count} real + {synth_count} synthetic = {len(fault_bank)} rows")

    OUT_FAULT_BANK.parent.mkdir(parents=True, exist_ok=True)
    fault_bank.to_parquet(OUT_FAULT_BANK)

    fault_variances = real_faults[feat_cols].var()
    rep_sensor = fault_variances.idxmax()

    col_idx = feat_cols.index(rep_sensor)
    real_vals = X_full[y_full == 1, col_idx]
    synth_vals = X_synthetic_clean[:, col_idx]

    n_examples = 3
    segment_len = 60

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 8), sharey=True)
    fig.suptitle(
        f"Real vs KMeansSMOTE+ENN synthetic fault examples — {rep_sensor}", fontsize=13
    )

    for i in range(n_examples):
        start_r = min(i * segment_len, max(0, len(real_vals) - segment_len))
        seg_r = real_vals[start_r: start_r + segment_len]
        axes[i, 0].plot(seg_r, color="steelblue", linewidth=1.2)
        axes[i, 0].set_title(f"Real #{i + 1}", fontsize=9)
        axes[i, 0].set_ylabel("Value")

        start_s = i * segment_len
        seg_s = synth_vals[start_s: start_s + segment_len]
        axes[i, 1].plot(seg_s, color="darkorange", linewidth=1.2)
        axes[i, 1].set_title(f"Synthetic #{i + 1}", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("Time step within segment")

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
