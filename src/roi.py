from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

def evaluate_thresholds(
    y_true: np.ndarray,
    probability: np.ndarray,
    monthly_margin: float,
    intervention_cost: float,
    save_probability: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    total_churners = int(y_true.sum())
    for threshold in thresholds:
        targeted = probability >= threshold
        targeted_n = int(targeted.sum())
        captured = int(y_true[targeted].sum())
        gross = captured * save_probability * monthly_margin
        cost = targeted_n * intervention_cost
        rows.append({
            "threshold": float(threshold),
            "targeted_customers": targeted_n,
            "targeted_share": targeted_n / len(y_true) if len(y_true) else 0,
            "captured_churners": captured,
            "recall": captured / total_churners if total_churners else 0,
            "precision": captured / targeted_n if targeted_n else 0,
            "gross_expected_value_inr": gross,
            "intervention_cost_inr": cost,
            "net_expected_value_inr": gross - cost,
        })
    return pd.DataFrame(rows)

def optimize_threshold(y_true, probability, config_path="config.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    thresholds = np.round(np.arange(
        cfg["roi"]["threshold_start"],
        cfg["roi"]["threshold_end"] + 1e-9,
        cfg["roi"]["threshold_step"],
    ), 2)
    table = evaluate_thresholds(
        np.asarray(y_true), np.asarray(probability),
        cfg["business"]["monthly_margin_inr"],
        cfg["business"]["retention_cost_inr"],
        cfg["business"]["save_probability_if_targeted"],
        thresholds,
    )
    best = table.loc[table["net_expected_value_inr"].idxmax()].to_dict()
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/roi_thresholds.csv", index=False)
    Path("reports/roi_optimal.json").write_text(json.dumps(best, indent=2))
    return table, best
