"""
run.py — single entry point for the full pump health pipeline.

Executes all seven steps in order. Stops immediately if any step fails.
"""

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("01_load_data.py",        "Load and validate data"),
    ("02_eda.py",              "Exploratory data analysis"),
    ("03_health_indicators.py","Health indicator engineering"),
    ("04_anomaly_detector.py", "Anomaly detector baseline"),
    ("05_synthetic_faults.py", "Synthetic fault generation"),
    ("06_classifier.py",       "Classifier comparison"),
    ("07_evaluate.py",         "Final evaluation"),
]


def load_and_run(filename):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location("_step", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():

    pipeline_start = time.perf_counter()
    timings = []

    for filename, label in STEPS:
        print(f"\n[{label}]")
        t0 = time.perf_counter()
        try:
            load_and_run(filename)
        except Exception as exc:
            print(f"\nFAILED: {exc}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        timings.append((label, elapsed))

    total = time.perf_counter() - pipeline_start

    print("\n  Timing summary")
    print("-" * 60)
    for label, elapsed in timings:
        print(f"  {elapsed:5.1f}s  {label}")
    print(f"  ------")
    print(f"  {total:5.1f}s  total")


if __name__ == "__main__":
    main()
