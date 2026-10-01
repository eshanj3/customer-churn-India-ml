import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train import make_preprocessor

def test_preprocessor_handles_mixed_types():
    data = pd.DataFrame({
        "tenure_months": [3, 20],
        "city": ["Indore", "Mumbai"],
        "avg_data_gb": [2.0, 6.0],
    })
    transformed = make_preprocessor(data).fit_transform(data)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 3
