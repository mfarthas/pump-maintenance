import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
ANOMALY_FILE = METRICS_DIR / "baseline_anomaly.json"
CLASSIFIER_FILE = METRICS_DIR / "classifier_comparison.json"
OUT_SUMMARY = METRICS_DIR / "final_summary.json"


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def fmt(val, pct=True):
    if val is None:
        return "  N/A "
    return f"{val * 100:5.1f}%" if pct else f"{val:.4f}"


def print_table(anomaly, model_a, model_b):
    header = f"{'Metric':<12} | {'Baseline (IF)':>14} | {'Model A (real)':>14} | {'Model B (+synth)':>16}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for metric in ["precision", "recall", "f1", "auc_roc"]:
        a_val = anomaly.get(metric)
        b_val = model_a.get(metric)
        c_val = model_b.get(metric)
        row = (f"{metric:<12} | {fmt(a_val):>14} | {fmt(b_val):>14} | {fmt(c_val):>16}")
        print(row)
    print(sep)


def main():

    anomaly = load_json(ANOMALY_FILE)
    classifiers = load_json(CLASSIFIER_FILE)
    model_a = classifiers["model_A_real_only"]
    model_b = classifiers["model_B_real_plus_synthetic"]

    print("\n=== RESULTS COMPARISON TABLE ===\n")
    print_table(anomaly, model_a, model_b)

    summary = {
        "baseline_isolation_forest": anomaly,
        "model_A_real_only": model_a,
        "model_B_real_plus_synthetic": model_b,
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
