# Customer Churn Prediction for an Indian Subscription Business — predict and prioritize churners before they leave.

Live App: [LOVABLE_URL_PENDING] | GitHub: https://github.com/eshanj3/customer-churn-India-ml

![ROI threshold hero](assets/hero.png)

## Top 3 Findings

1. The validated Logistic Regression baseline achieved **65.57% ROC-AUC**, **24.50% PR-AUC**, **65.28% recall**, and **34.33% F1** on a 1,200-customer untouched test set.
2. The top-20% risk audience contained **34.7% of observed churners**, showing how probability ranking can concentrate retention activity.
3. Under the documented initial assumptions, the top-20% retention scenario produced **₹7,035 gross expected value**, **₹19,200 intervention cost**, and **-₹12,165 net expected value**. The project therefore optimizes the intervention threshold for economics instead of hiding an unfavorable business case.

> These figures come from the validated project evaluation described in the repository. Run `python src/train.py` to regenerate the evaluation artifacts after changing the generator, model, or business assumptions.

## Business problem

An Indian telecom/OTT subscription business wants to identify customers who are likely to churn early enough to justify a retention intervention. The objective is not simply to maximize classification accuracy: missed churners have an opportunity cost, while unnecessary offers waste retention budget.

## Data

The repository uses a **seeded synthetic dataset** so the complete project can be reproduced without redistributing a third-party Kaggle dataset. It contains Jio/Airtel/Vi-style plans, OTT plans, UPI/card/manual recharge behavior, and Tier-1/2/3 Indian cities.

The generator creates 6,000 customers with six monthly usage observations each.

## SQL feature engineering

Raw customer and monthly usage data are loaded into SQLite and transformed in SQL using:

- tenure buckets
- `LAG()` usage deltas
- rolling three-month usage averages
- rolling recharge-recency averages
- recharge frequency
- average and maximum recharge recency
- monthly usage trend
- support-ticket counts
- city tier

The model never calculates these behavioral aggregates directly from the raw CSVs in Python.

## Modeling

The project compares:

- Logistic Regression with class-balanced training
- Random Forest with class-balanced training
- XGBoost

A stratified train/validation/test split is used, followed by five-fold cross-validation on the training partition.

The selected model is Logistic Regression because the documented evaluation emphasizes recall and PR-AUC while preserving a transparent and deployable model.

## Metrics and business cost

The positive class is churn.

**Recall matters because a missed churner represents a customer the retention team never gets an opportunity to save. Precision matters because false positives consume intervention budget. PR-AUC is particularly informative for an imbalanced churn problem.**

Final validated holdout:

| Metric | Result |
|---|---:|
| Test customers | 1,200 |
| Test churn rate | 16.08% |
| Precision | 23.29% |
| Recall | 65.28% |
| F1 | 34.33% |
| ROC-AUC | 65.57% |
| PR-AUC | 24.50% |

## ROI threshold optimization

The operating threshold is not assumed to be 0.50.

Validation-set probabilities are swept from 0.05 to 0.90. For each threshold the pipeline calculates:

`Net Expected Value = Captured Churners × Save Probability × Monthly Margin − Targeted Customers × Intervention Cost`

The default assumptions are:

- monthly contribution margin: ₹300
- retention intervention cost: ₹80
- probability a targeted churner is successfully retained: 35%

The optimal threshold is selected **only on validation data**. The test set is then evaluated once using that frozen threshold.

Generated artifacts:

- `reports/roi_thresholds.csv`
- `reports/roi_optimal.json`
- `reports/test_metrics.json`

## Explainability

SHAP is used to calculate global feature importance for the fitted Logistic Regression pipeline.

Run:

```bash
python src/explain.py
```

Outputs:

- `reports/shap_top_features.csv`
- `assets/shap_summary.png`

Business-readable prediction reasons are also returned by the FastAPI endpoint.

## Leakage prevention

- The churn label is excluded from feature calculations.
- Behavioral features use only the defined observation window.
- Train/validation/test splitting happens before model fitting.
- The final test set is untouched during model and threshold selection.
- Preprocessing is fitted inside an sklearn Pipeline.
- ROI threshold selection uses validation predictions only.
- No future churn outcome is used as a model feature.

## Deployment

### FastAPI on Render

The repository includes `render.yaml`.

Build command:

```bash
pip install -r requirements.txt && python src/generate_data.py && python src/train.py && python src/explain.py
```

Start command:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Health endpoint:

```text
/health
```

Prediction endpoint:

```text
POST /predict
```

### Lovable UI

A polished frontend is being built in the Lovable project **Churn Guardian**. The frontend is designed to call the FastAPI `/predict` endpoint and show churn probability, the frozen intervention threshold, top reasons, and retention economics.

## Local Streamlit UI

```bash
streamlit run src/app.py
```

## Reproducibility

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

python src/generate_data.py
python src/sql_features.py
python src/train.py
python src/explain.py
pytest -q

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Model card

See [model_card.md](model_card.md).

## Limitations

This is a methodology and portfolio demonstration using synthetic data. The reported performance should not be interpreted as expected performance on a real telecom subscriber population. Business value depends on the assumed margin, intervention cost, and successful-retention probability. Real deployment would require calibration, drift monitoring, subgroup analysis, experiment-based uplift measurement, and governance around customer targeting.

## Repository structure

```text
assets/             Evaluation/SHAP visuals
data/               Generated datasets
models/             Serialized trained pipeline
reports/            Metrics, model comparison, ROI and SHAP outputs
sql/                Schema and SQL feature engineering
src/                Data generation, SQL, training, ROI, SHAP, API, UI
tests/              Preprocessing tests
config.yaml         Model and business assumptions
model_card.md       Model documentation
RESUME_BULLETS.md   Resume variants
render.yaml         Render deployment configuration
```
