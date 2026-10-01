from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

artifact = joblib.load(Path("models/churn_model.joblib"))
pipeline = artifact["pipeline"]
features = artifact["features"]
threshold = artifact["decision_threshold"]

app = FastAPI(title="Indian Subscription Churn API", version="1.1.0")

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
    avg_monthly_data_delta: float = .1
    max_rolling_recharge_recency: float = Field(14, ge=0)

@app.get("/health")
def health():
    return {"status":"ok","model":artifact["model_name"],"decision_threshold":threshold}

@app.post("/predict")
def predict(payload: CustomerInput):
    row = pd.DataFrame([payload.model_dump()])[features]
    probability = float(pipeline.predict_proba(row)[:,1][0])
    reasons = []
    if payload.avg_days_since_recharge > 10:
        reasons.append("Recharge recency is elevated")
    if payload.support_tickets >= 3:
        reasons.append("Frequent support issues")
    if payload.avg_monthly_data_delta < -.2:
        reasons.append("Usage is trending down")
    if payload.contract_type == "Monthly":
        reasons.append("Monthly contract has lower commitment")
    if payload.autopay_type == "Manual Recharge":
        reasons.append("Manual recharge creates payment friction")
    if not reasons:
        reasons.append("Risk reflects the combined customer behavior pattern")
    return {
        "churn_probability": round(probability,4),
        "high_risk": probability >= threshold,
        "decision_threshold": threshold,
        "top_reasons": reasons[:3],
    }
