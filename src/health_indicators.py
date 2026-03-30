"""
Step 3 - Health indicator engineering.

Loads sensor_clean.parquet, drops near-zero variance sensors, computes rolling
statistics (mean, std, rate-of-change) per sensor over a 60-cycle window,
saves engineered features to data/features.parquet, and saves a correlation
heatmap of NORMAL windows to results/figures/correlation_heatmap.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "sensor_clean.parquet"
OUT_FEATURES = ROOT / "data" / "features.parquet"
OUT_HEATMAP = ROOT / "results" / "figures" / "correlation_heatmap.png"

WINDOW = 60

# Identified in step 2: sensor_15 is 100% NaN
LOW_VARIANCE_SENSORS = [
    "sensor_15",
]


def main():

    df = pd.read_parquet(IN_PARQUET)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    drop_cols = [s for s in LOW_VARIANCE_SENSORS if s in sensor_cols]
    keep_cols = [s for s in sensor_cols if s not in drop_cols]

    print(f"Dropping sensors: {drop_cols}")
    print(f"Rolling features: {len(keep_cols)} sensors × 3 stats, window={WINDOW}")

    # Fill scattered NaNs in raw sensor columns before rolling so they don't propagate
    df[keep_cols] = df[keep_cols].ffill().bfill()

    feature_frames = [df[["machine_status"]]]

    for col in keep_cols:
        s = df[col]
        feature_frames.append(s.rename(col))
        feature_frames.append(
            s.rolling(WINDOW, min_periods=1).mean().rename(f"{col}_roll_mean")
        )
        feature_frames.append(
            s.rolling(WINDOW, min_periods=1).std().rename(f"{col}_roll_std")
        )
        feature_frames.append(
            s.diff().rolling(WINDOW, min_periods=1).mean().rename(f"{col}_roll_roc")
        )

    features = pd.concat(feature_frames, axis=1)
    print(f"Feature matrix: {features.shape}")

    post_window = features.iloc[WINDOW:]
    nan_count = post_window.drop(columns=["machine_status"]).isna().sum().sum()
    print(f"NaNs beyond window: {nan_count}")

    # Correlation heatmap on NORMAL windows
    normal_df = features[features["machine_status"] == "NORMAL"]
    numeric_cols = [c for c in normal_df.columns if c != "machine_status"]
    # Use only raw sensor columns for readability
    raw_only = [c for c in numeric_cols if not any(
        c.endswith(s) for s in ["_roll_mean", "_roll_std", "_roll_roc"]
    )]
    corr = normal_df[raw_only].corr()

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                xticklabels=True, yticklabels=True,
                linewidths=0.1, linecolor="grey")
    ax.set_title("Cross-sensor Pearson correlation (NORMAL windows only)", fontsize=13)
    plt.tight_layout()
    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_HEATMAP, dpi=150)
    plt.close()

    features.to_parquet(OUT_FEATURES)


if __name__ == "__main__":
    main()
