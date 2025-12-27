import json
from pathlib import Path

import pandas as pd


def write_metrics(metrics: dict, out_dir: Path):
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))


def write_report(report_lines: list, out_dir: Path):
    path = out_dir / "report.txt"
    path.write_text("\n".join(report_lines))


def write_predictions(df: pd.DataFrame, probs, out_dir: Path, split_name: str):
    report = df.copy().reset_index(drop=True)
    report["pred_prob"] = probs
    report["split"] = split_name
    report["label"] = report["label"].astype(int)
    report.to_csv(out_dir / "predictions_test.csv", index=False)
