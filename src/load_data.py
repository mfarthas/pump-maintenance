"""
Step 1 - Load and validate data.

Loads data/sensor.csv, parses timestamps, reports shape/dtypes/NaNs,
and saves a cleaned dataframe to data/sensor_clean.parquet.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "sensor.csv"
OUT_PARQUET = ROOT / "data" / "sensor_clean.parquet"


def main():

    df = pd.read_csv(RAW_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    print(f"Shape: {df.shape}")
    print(f"\nmachine_status distribution:\n{df['machine_status'].value_counts()}")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    nan_counts = df[sensor_cols].isna().sum()
    nan_report = nan_counts[nan_counts > 0]
    if nan_report.empty:
        print("\nNo NaNs found in sensor columns.")
    else:
        print(f"\nNaN counts per sensor (non-zero only):\n{nan_report}")

    df.to_parquet(OUT_PARQUET)


if __name__ == "__main__":
    main()
