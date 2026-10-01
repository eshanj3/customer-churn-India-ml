# Model Card

## Model

Current selected model: Logistic Regression with class-balanced training.

Selection method: five-fold stratified cross-validation, ranked by mean PR-AUC with recall and F1 as tie-breakers.

## Data

Seeded synthetic India-oriented subscription data representing telecom/OTT-style plans, Indian city tiers, payment methods and six months of monthly usage.

- 6,000 customers
- 36,000 monthly usage records
- 23.42% generated churn rate in the current seed-42 run

## Evaluation

The model and decision threshold are selected without using the test set. The selected pipeline is refit on train plus validation, then the untouched 1,200-customer test set is evaluated once.

Current test results:
- Precision: 36.76%
- Recall: 50.89%
- F1: 42.69%
- ROC-AUC: 66.69%
- PR-AUC: 37.16%

## Business threshold

The intervention threshold is optimized from validation predictions using explicit retention economics.

Current assumptions:
- monthly margin: ₹300
- intervention cost: ₹30
- successful-retention probability: 35%

Current validation-selected threshold: 0.57.

## Leakage prevention

- Churn is excluded from model inputs.
- Behavioral aggregates are computed from the defined observation window.
- Preprocessing is contained in sklearn pipelines.
- Model selection uses training/CV data only.
- ROI threshold selection uses validation predictions only.
- The final test set is untouched until final evaluation.

## Explainability

SHAP is used for global feature-importance reporting. FastAPI also returns rule-based customer-level reasons derived from observable behavior. Neither should be treated as causal evidence.

## Intended use

Retention prioritization and portfolio demonstration.

## Limitations

The dataset is synthetic and cannot establish production performance. Real deployment should include calibration, drift monitoring, subgroup evaluation, intervention fairness checks and controlled retention experiments.

## Human oversight

Predictions should support retention review and experiments. They should not automatically deny service or infer protected or sensitive attributes.
