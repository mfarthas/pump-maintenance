"""

Exploratory data analysis.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
IN_PARQUET = ROOT / "data" / "sensor_clean.parquet"
OUT_FIG = ROOT / "results" / "figures" / "sensor_overview.png"

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-6


def pick_representative_sensors(df, n=6):
    """Return n sensor column names spread across the variance distribution."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var().sort_values(ascending=False)
    # Drop near-zero variance first
    useful = variances[variances > NEAR_ZERO_VARIANCE_THRESHOLD]
    # Pick evenly spaced across the sorted list
    idx = np.linspace(0, len(useful) - 1, n, dtype=int)
    return list(useful.iloc[idx].index)


def main():

    df = pd.read_parquet(IN_PARQUET)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # Near-zero variance detection — also catches all-NaN columns (var()=NaN)
    variances = df[sensor_cols].var()
    all_nan = df[sensor_cols].isna().all()
    low_var = variances[
        (variances <= NEAR_ZERO_VARIANCE_THRESHOLD) | all_nan
    ].index.tolist()
    print(f"Near-zero variance sensors ({len(low_var)} found):")
    for s in low_var:
        nan_pct = df[s].isna().mean() * 100
        print(f"  {s}  (var={variances[s]:.2e}, NaN={nan_pct:.1f}%)")

    chosen = pick_representative_sensors(df, n=6)

    # BROKEN mask
    broken_mask = df["machine_status"] == "BROKEN"

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    fig.suptitle("Pump Sensor Overview — BROKEN windows in red", fontsize=14)

    for ax, col in zip(axes, chosen):
        ax.plot(df.index, df[col], linewidth=0.5, color="steelblue", label=col)
        # Shade BROKEN regions
        broken_times = df.index[broken_mask]
        if len(broken_times):
            ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=broken_mask.values, color="red", alpha=0.25,
                            label="BROKEN")
        ax.set_ylabel(col, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
