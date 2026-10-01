# Customer Churn Prediction for an Indian Subscription Business

A hiring-grade ML portfolio project combining SQL feature engineering, imbalanced classification, validation-safe model selection, ROI threshold optimization, SHAP explainability, FastAPI serving, and a Streamlit dashboard.

GitHub: https://github.com/eshanj3/customer-churn-India-ml  
Figma: https://www.figma.com/design/ORVtXABGbmEPXOWJIv7s82

> The dataset is intentionally seeded synthetic data. Performance numbers are reproducible project results, not evidence of real-world telecom performance.

## Current verified run

| Item | Result |
|---|---:|
| Model | Logistic Regression |
| Test customers | 1,200 |
| Test churn rate | 23.42% |
| Precision | 36.76% |
| Recall | 50.89% |
| F1 | 42.69% |
| ROC-AUC | 66.69% |
| PR-AUC | 37.16% |
| Validation ROI threshold | 0.57 |
| Validation targeted share | 34.48% |
| Validation captured churners | 119 |
| Validation net expected value | ₹2,565 |
| Model version | 2026.10-75a1e2145375 |

Model selection is based on five-fold cross-validation PR-AUC, with CV recall and F1 as tie-breakers. The validation set is used to optimize the retention threshold. The untouched test set is used only for the final reported evaluation.

## Architecture

Raw customer and monthly usage data flow through SQLite SQL window features into a train/validation/test ML pipeline. Model candidates are compared, the operating threshold is optimized on validation data, the selected pipeline is refit on train plus validation, and the untouched test set is evaluated once. The final artifact is versioned and served through FastAPI.

## API

| Endpoint | Purpose | UI |
|---|---|---|
| GET /health | API and model status | Overview, Model Health |
| GET /metrics | Holdout metrics and selected threshold | Overview, Model Health |
| GET /roi | Assumptions, optimal threshold and threshold curve | ROI Optimizer |
| GET /explainability?top_n=10 | Global SHAP drivers | Explainability |
| POST /predict | Individual churn probability and reasons | Customer Scoring |

Full request/response details are in docs/API.md.

## SQL feature engineering

sql/features.sql uses SQLite LAG and rolling windows to derive tenure buckets, usage deltas, rolling three-month data usage, recharge recency, recharge frequency, usage-increase months, and other behavioral aggregates. The Python model consumes the SQL-generated feature table.

## Modeling

Compared candidates:
- Logistic Regression with class-balanced training
- Random Forest with class-balanced training
- XGBoost with positive-class weighting

Model selection is performed from training data using five-fold CV. Test metrics are never used to choose the model.

## ROI optimization

The threshold sweep evaluates 0.05 to 0.90 using:

Net Expected Value = Captured Churners × Save Probability × Monthly Margin − Targeted Customers × Intervention Cost

Current documented assumptions:
- monthly contribution margin: ₹300
- intervention cost: ₹30
- successful-retention probability if targeted: 35%

The current validation optimum is threshold 0.57, targeting 34.48% of validation customers and capturing 119 observed churners for ₹2,565 validation net expected value.

## Explainability

SHAP creates global feature-importance outputs at reports/shap_top_features.csv and assets/shap_summary.png. The API maps technical feature names to human-readable labels. mean_abs_shap describes importance magnitude and should not be interpreted as causality or individual directional effect.

## EDA

Run python src/eda.py to generate the churn distribution, churn-by-plan, churn-by-tenure visuals and reports/eda_summary.json.

## Local run

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/generate_data.py
    python src/eda.py
    python src/sql_features.py
    python src/train.py
    python src/explain.py
    pytest -q
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Run the dashboard separately:

    streamlit run src/app.py

Swagger: http://localhost:8000/docs

## Deployment

render.yaml rebuilds the reproducible data, EDA, model and SHAP artifacts during deployment and starts FastAPI with /health as the health check.

.github/workflows/ci.yml runs the same pipeline and tests on pushes and pull requests.

## Repository structure

    assets/
    data/
    models/
    reports/
    sql/
    src/
    tests/
    docs/
    .github/workflows/
    config.yaml
    model_card.md
    RESUME_BULLETS.md
    render.yaml
