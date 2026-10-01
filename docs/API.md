# FastAPI Contract

Every model-serving response includes model_version so the UI can verify that model health, metrics, ROI, SHAP and prediction responses belong to the same trained artifact.

## GET /health

Example response:

    {
      "status": "ok",
      "model": "LogisticRegression",
      "model_version": "2026.10-75a1e2145375",
      "decision_threshold": 0.57
    }

## GET /metrics

    {
      "model_name": "LogisticRegression",
      "model_version": "2026.10-75a1e2145375",
      "selection_metric": "cv_pr_auc_mean",
      "dataset": {
        "test_size": 1200,
        "test_churn_rate": 0.2341666667
      },
      "classification": {
        "precision": 0.3676092545,
        "recall": 0.5088967972,
        "f1": 0.4268656716
      },
      "ranking": {
        "roc_auc": 0.6668783569,
        "pr_auc": 0.3716107082
      },
      "decision_threshold": 0.57,
      "roi": {
        "targeted_share": 0.3447916667,
        "captured_churners": 119,
        "gross_expected_value_inr": 12495.0,
        "intervention_cost_inr": 9930.0,
        "net_expected_value_inr": 2565.0
      }
    }

## GET /roi

No request body. Returns the business assumptions, optimal validation threshold and the complete threshold curve from 0.05 to 0.90.

## GET /explainability?top_n=10

Returns SHAP global feature importance. mean_abs_shap measures importance magnitude; it is not a causal or individual-level directional statement.

## POST /predict

Request fields:
city, city_tier, plan_name, autopay_type, tenure_months, tenure_bucket, support_tickets, contract_type, data_cap_gb, avg_data_gb, avg_calls, avg_days_since_recharge, max_days_since_recharge, avg_recharge_amount_inr, recharge_frequency, avg_rolling_3m_data_gb, usage_increase_months, avg_monthly_data_delta, max_rolling_recharge_recency.

Example request:

    {
      "city": "Indore",
      "city_tier": 2,
      "plan_name": "Jio 299",
      "autopay_type": "Manual Recharge",
      "tenure_months": 10,
      "tenure_bucket": "7-12",
      "support_tickets": 4,
      "contract_type": "Monthly",
      "data_cap_gb": 6.0,
      "avg_data_gb": 4.0,
      "avg_calls": 200.0,
      "avg_days_since_recharge": 16.0,
      "max_days_since_recharge": 24.0,
      "avg_recharge_amount_inr": 300.0,
      "recharge_frequency": 6.0,
      "avg_rolling_3m_data_gb": 3.8,
      "usage_increase_months": 1.0,
      "avg_monthly_data_delta": -0.6,
      "max_rolling_recharge_recency": 24.0
    }

Example response:

    {
      "model_name": "LogisticRegression",
      "model_version": "2026.10-75a1e2145375",
      "churn_probability": 0.8934,
      "high_risk": true,
      "decision_threshold": 0.57,
      "top_reasons": [
        "Recharge recency is elevated",
        "Frequent support issues",
        "Usage is trending down"
      ]
    }
