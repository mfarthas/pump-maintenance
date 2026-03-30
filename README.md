# pump-health-ml

Predictive maintenance pipeline for an industrial pump. Raw sensor data in, trained fault classifier out.

The problem this is designed around: 220,320 rows of sensor readings across 153 days, with exactly 7 labelled BROKEN. 

---

## Dataset

[Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) on Kaggle.

- 220,320 rows, one per minute, ~153 days
- 52 sensors (sensor_00 to sensor_51)
- Labels: `NORMAL` (205,836), `RECOVERING` (14,477), `BROKEN` (7)

Download `sensor.csv` and place it at `data/sensor.csv` first.

---

## Results

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Isolation Forest (no labels) | 38.9% | 9.0% | 14.6% | — |
| Random Forest, real data only | 95.7% | 96.2% | 96.0% | ~1.000 |
| Random Forest, real + synthetic | 95.6% | 96.1% | 95.9% | ~1.000 |

The Isolation Forest is trained on normal operation only, so it had no fault examples at all. Both Random Forests use the same time-ordered split and the same test set. Synthetic augmentation uses KMeansSMOTE with ENN filtering: normal rows act as boundary context during generation, and any synthetic fault sample whose nearest neighbours vote normal gets removed.

---

## Quickstart

```bash
pip install -r requirements.txt
# place sensor.csv at data/sensor.csv
python run.py
```

Each step can also run independently, scripts are self-contained:

```bash
python src/01_load_data.py  # through
python src/07_evaluate.py
```

---

## What each step does

`01_load_data.py` parses the CSV, sets timestamp as index, prints label distribution and per-sensor NaN counts, saves `data/sensor_clean.parquet`.

`02_eda.py` plots 6 representative sensors over the full timeline with fault windows marked, detects all-NaN columns, saves `results/figures/sensor_overview.png`.

`03_health_indicators.py` drops `sensor_15` (100% NaN) and computes rolling mean, std, and rate of change over a 60-sample window for each remaining sensor. NaNs are filled on the raw columns before the rolling step — doing it after propagates them through every window they touch. Output: `data/features.parquet` (205 columns), `results/figures/correlation_heatmap.png`.

`04_anomaly_detector.py` trains an Isolation Forest on NORMAL rows only (contamination=0.01), scores the full dataset, evaluates against BROKEN+RECOVERING as the positive class. Output: `results/metrics/baseline_anomaly.json`, `results/figures/anomaly_baseline.png`.

`05_synthetic_faults.py` generates 5x the real fault count with KMeansSMOTE. The full dataset goes to the sampler, so NORMAL data defines the cluster boundaries. ENN removes synthetic samples that land in NORMAL-dominated neighbourhoods. Output: `data/fault_bank.parquet`, `results/figures/synthetic_vs_real.png`.

`06_classifier.py` trains Model A (real data only) and Model B (real + synthetic), evaluates both on the same test set. Output: `results/metrics/classifier_comparison.json`, `results/figures/roc_comparison.png`.

`07_evaluate.py` loads all metrics, prints the comparison table, writes `results/metrics/final_summary.json`.

---

## The train/test split

A last-20%-of-rows split doesn't work. All fault events are in the first part of the timeline, so that split produces a test set with zero positive labels. First run, both models returned AUC 0.0.

The split point is the row where 80% of fault events have already occurred. Faults land in both splits, time order is preserved.

Random splits are also out — rolling features over a 60-row window leak future data into training when a window straddles the boundary.

---

## Structure

```
pump-health-ml/
├── run.py               # runs all steps, prints per-step timing
├── requirements.txt
├── CLAUDE.md
├── data/
│   └── sensor.csv       # download from Kaggle, not committed
└── src/
    ├── 01_load_data.py
    ├── 02_eda.py
    ├── 03_health_indicators.py
    ├── 04_anomaly_detector.py
    ├── 05_synthetic_faults.py
    ├── 06_classifier.py
    └── 07_evaluate.py
```

`data/` and `results/` are gitignored. All outputs reproduce by running `run.py`.

---

## Dependencies

Python 3.11+. `pip install -r requirements.txt` covers everything: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, pyarrow.
