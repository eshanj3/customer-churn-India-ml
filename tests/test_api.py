import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

@pytest.fixture(scope="module")
def client():
    if not (ROOT / "models/churn_model.joblib").exists():
        pytest.skip("Model artifact not built")
    import api
    return TestClient(api.app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]

def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "classification" in body and "ranking" in body

def test_roi(client):
    r = client.get("/roi")
    assert r.status_code == 200
    body = r.json()
    assert body["curve"] and "optimal" in body

def test_explainability(client):
    r = client.get("/explainability?top_n=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["drivers"]) <= 5
    assert body["method"] == "SHAP"

def test_predict(client):
    payload = {
        "city": "Indore",
        "city_tier": 2,
        "plan_name": "Jio 299",
        "autopay_type": "Manual Recharge",
        "tenure_months": 10,
        "tenure_bucket": "7-12",
        "support_tickets": 4,
        "contract_type": "Monthly",
        "data_cap_gb": 6,
        "avg_data_gb": 4,
        "avg_calls": 200,
        "avg_days_since_recharge": 16,
        "max_days_since_recharge": 24,
        "avg_recharge_amount_inr": 300,
        "recharge_frequency": 6,
        "avg_rolling_3m_data_gb": 3.8,
        "usage_increase_months": 1,
        "avg_monthly_data_delta": -0.6,
        "max_rolling_recharge_recency": 24,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["high_risk"], bool)
    assert body["top_reasons"]

def test_predict_validation_error(client):
    r = client.post("/predict", json={"city_tier": 9})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"
