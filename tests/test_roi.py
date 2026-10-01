import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from roi import evaluate_thresholds

def test_roi_has_expected_columns():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.3, 0.9])
    df = evaluate_thresholds(y, p, 300, 80, 0.35, np.array([0.5, 0.8]))
    assert "net_expected_value_inr" in df.columns
    assert len(df) == 2
    assert (df["targeted_customers"] >= 0).all()
