from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/churn_model.joblib"
METRICS_PATH = ROOT / "reports/test_metrics.json"
ROI_PATH = ROOT / "reports/roi_thresholds.csv"
ROI_OPTIMAL_PATH = ROOT / "reports/roi_optimal.json"
SHAP_PATH = ROOT / "reports/shap_top_features.csv"
CONFIG_PATH = ROOT / "config.yaml"

if not MODEL_PATH.exists():
    raise RuntimeError("Trained model artifact not found. Run python src/train.py first.")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
features = artifact["features"]
model_name = artifact["model_name"]
decision_threshold = float(artifact["decision_threshold"])
model_version = artifact.get("model_version", "unknown")

app = FastAPI(
    title="Indian Subscription Churn API",
    version="1.2.0",
    description="Churn prediction, model metrics, ROI optimization, and SHAP explainability.",
)

class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "error": "validation_error",
        "message": "Request validation failed.",
        "request_id": request.headers.get("x-request-id"),
        "details": exc.errors(),
    })

class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    model_version: str
    decision_threshold: float

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model": model_name,
        "model_version": model_version,
        "decision_threshold": decision_threshold,
    }

class DatasetMetrics(BaseModel):
    test_size: int
    test_churn_rate: float

class ClassificationMetrics(BaseModel):
    precision: float
    recall: float
    f1: float

class RankingMetrics(BaseModel):
    roc_auc: float
    pr_auc: float

class ROIMetrics(BaseModel):
    targeted_share: float
    captured_churners: int
    gross_expected_value_inr: float
    intervention_cost_inr: float
    net_expected_value_inr: float

class MetricsResponse(BaseModel):
    model_name: str
    model_version: str
    selection_metric: str
    dataset: DatasetMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    decision_threshold: float
    roi: ROIMetrics

@app.get("/metrics", response_model=MetricsResponse)
def metrics_endpoint():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Metrics artifact not available.")
    data = json.loads(METRICS_PATH.read_text())
    return {
        "model_name": data["model_name"],
        "model_version": data.get("model_version", model_version),
        "selection_metric": data.get("selection_metric", "cv_pr_auc_mean"),
        "dataset": {"test_size": data["test_size"], "test_churn_rate": data["test_churn_rate"]},
        "classification": {
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
        },
        "ranking": {"roc_auc": data["roc_auc"], "pr_auc": data["pr_auc"]},
        "decision_threshold": data["validation_optimal_threshold"],
        "roi": {
            "targeted_share": data["validation_targeted_share"],
            "captured_churners": data["validation_captured_churners"],
            "gross_expected_value_inr": data["validation_gross_expected_value_inr"],
            "intervention_cost_inr": data["validation_intervention_cost_inr"],
            "net_expected_value_inr": data["validation_net_expected_value_inr"],
        },
    }

class ROIAssumptions(BaseModel):
    monthly_margin_inr: float
    intervention_cost_inr: float
    save_probability_if_targeted: float

class ROIScenario(BaseModel):
    threshold: float
    targeted_customers: int
    targeted_share: float
    captured_churners: int
    recall: float
    precision: float
    gross_expected_value_inr: float
    intervention_cost_inr: float
    net_expected_value_inr: float

class ROIResponse(BaseModel):
    model_name: str
    model_version: str
    assumptions: ROIAssumptions
    optimal: ROIScenario
    curve: list[ROIScenario]

@app.get("/roi", response_model=ROIResponse)
def roi_endpoint():
    if not ROI_PATH.exists() or not ROI_OPTIMAL_PATH.exists():
        raise HTTPException(status_code=503, detail="ROI artifacts not available.")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    curve_df = pd.read_csv(ROI_PATH)
    optimal = json.loads(ROI_OPTIMAL_PATH.read_text())
    curve = [ROIScenario(**row) for row in curve_df.to_dict(orient="records")]
    return {
        "model_name": model_name,
        "model_version": model_version,
        "assumptions": {
            "monthly_margin_inr": cfg["business"]["monthly_margin_inr"],
            "intervention_cost_inr": cfg["business"]["retention_cost_inr"],
            "save_probability_if_targeted": cfg["business"]["save_probability_if_targeted"],
        },
        "optimal": ROIScenario(**optimal),
        "curve": curve,
    }

class ExplainabilityDriver(BaseModel):
    feature: str
    display_name: str
    mean_abs_shap: float
    interpretation: str

class ExplainabilityResponse(BaseModel):
    model_name: str
    model_version: str
    method: Literal["SHAP"]
    top_n: int
    drivers: list[ExplainabilityDriver]

def friendly_feature(feature: str) -> tuple[str, str]:
    raw = feature.replace("num__", "").replace("cat__", "")
    mapping = {
        "support_tickets": ("Support tickets", "Support-ticket activity is an important model signal."),
        "avg_days_since_recharge": ("Recharge recency", "Longer recharge gaps are an important model signal."),
        "max_days_since_recharge": ("Maximum recharge gap", "Large recharge gaps are an important model signal."),
        "avg_monthly_data_delta": ("Monthly usage change", "Month-to-month usage change is an important model signal."),
        "avg_recharge_amount_inr": ("Average recharge amount", "Recharge value is an important model signal."),
        "tenure_months": ("Tenure", "Customer tenure is an important model signal."),
        "avg_data_gb": ("Average data usage", "Average data consumption is an important model signal."),
        "avg_rolling_3m_data_gb": ("Rolling 3-month data usage", "Recent average data usage is an important model signal."),
        "recharge_frequency": ("Recharge frequency", "Recharge activity frequency is an important model signal."),
        "usage_increase_months": ("Months with increasing usage", "Positive usage movement is an important model signal."),
        "data_cap_gb": ("Data cap", "Plan data capacity is an important model signal."),
        "avg_calls": ("Average calls", "Calling activity is an important model signal."),
        "max_rolling_recharge_recency": ("Rolling recharge recency", "Recent recharge recency is an important model signal."),
    }
    if raw in mapping:
        return mapping[raw]
    if raw.startswith("contract_type_"):
        return f"Contract: {raw.replace('contract_type_', '').replace('_', ' ')}", "Contract type is an important model signal."
    if raw.startswith("autopay_type_"):
        return f"Payment: {raw.replace('autopay_type_', '').replace('_', ' ')}", "Payment method is an important model signal."
    if raw.startswith("plan_name_"):
        return f"Plan: {raw.replace('plan_name_', '').replace('_', ' ')}", "Plan choice is an important model signal."
    return raw.replace("_", " ").title(), "This feature is an important model signal."

@app.get("/explainability", response_model=ExplainabilityResponse)
def explainability_endpoint(top_n: int = Query(10, ge=1, le=30)):
    if not SHAP_PATH.exists():
        raise HTTPException(status_code=503, detail="SHAP artifact not available.")
    df = pd.read_csv(SHAP_PATH).head(top_n)
    drivers = []
    for row in df.to_dict(orient="records"):
        display_name, interpretation = friendly_feature(row["feature"])
        drivers.append({
            "feature": row["feature"],
            "display_name": display_name,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "interpretation": interpretation,
        })
    return {
        "model_name": model_name,
        "model_version": model_version,
        "method": "SHAP",
        "top_n": len(drivers),
        "drivers": drivers,
    }

class CustomerInput(BaseModel):
    city: str = "Indore"
    city_tier: int = Field(2, ge=1, le=3)
    plan_name: str = "Jio 299"
    autopay_type: str = "UPI Autopay"
    tenure_months: int = Field(10, ge=1)
    tenure_bucket: str = "7-12"
    support_tickets: int = Field(1, ge=0)
    contract_type: str = "Monthly"
    data_cap_gb: float = Field(6, ge=0)
    avg_data_gb: float = Field(4.5, ge=0)
    avg_calls: float = Field(200, ge=0)
    avg_days_since_recharge: float = Field(8, ge=0)
    max_days_since_recharge: float = Field(14, ge=0)
    avg_recharge_amount_inr: float = Field(300, ge=0)
    recharge_frequency: float = Field(6, ge=0)
    avg_rolling_3m_data_gb: float = Field(4.5, ge=0)
    usage_increase_months: float = Field(2, ge=0)
    avg_monthly_data_delta: float = 0.1
    max_rolling_recharge_recency: float = Field(14, ge=0)

class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    churn_probability: float
    high_risk: bool
    decision_threshold: float
    top_reasons: list[str]

@app.post("/predict", response_model=PredictResponse)
def predict(payload: CustomerInput):
    row = pd.DataFrame([payload.model_dump()])[features]
    probability = float(pipeline.predict_proba(row)[:, 1][0])
    reasons = []
    if payload.avg_days_since_recharge > 10:
        reasons.append("Recharge recency is elevated")
    if payload.support_tickets >= 3:
        reasons.append("Frequent support issues")
    if payload.avg_monthly_data_delta < -0.2:
        reasons.append("Usage is trending down")
    if payload.contract_type == "Monthly":
        reasons.append("Monthly contract has lower commitment")
    if payload.autopay_type == "Manual Recharge":
        reasons.append("Manual recharge creates payment friction")
    if not reasons:
        reasons.append("Risk reflects the combined customer behavior pattern")
    return {
        "model_name": model_name,
        "model_version": model_version,
        "churn_probability": round(probability, 4),
        "high_risk": probability >= decision_threshold,
        "decision_threshold": decision_threshold,
        "top_reasons": reasons[:3],
    }
