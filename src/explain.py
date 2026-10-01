from __future__ import annotations
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

def explain_model() -> None:
    artifact = joblib.load("models/churn_model.joblib")
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    model = pipeline.named_steps["model"]
    data = pd.read_csv("data/model_features.csv")
    X = data[features]
    transformed = pipeline.named_steps["preprocess"].transform(X)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.Explainer(model, transformed, feature_names=names)
    sample = transformed[: min(1000, transformed.shape[0])]
    values = explainer(sample)
    Path("reports").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": abs(values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.head(30).to_csv("reports/shap_top_features.csv", index=False)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, features=sample, feature_names=names, plot_type="bar", show=False, max_display=12)
    plt.tight_layout()
    plt.savefig("assets/shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    explain_model()
