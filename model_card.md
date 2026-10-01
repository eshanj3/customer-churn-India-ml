# Model Card

## Model
Logistic Regression with class-balanced training and an sklearn preprocessing pipeline.

## Data
Seeded synthetic India-oriented subscription data. The generator models telecom/OTT-style plans, UPI/card/manual recharge, and Tier-1/2/3 cities.

## Evaluation
Final holdout reported by the validated training run:

- 1,200 test customers
- 16.08% test churn rate
- ROC-AUC: 65.57%
- PR-AUC: 24.50%
- Recall: 65.28%
- F1: 34.33%

## Business threshold
The intervention threshold is selected using validation-set predictions and an explicit retention economics function. The test set is not used to choose the threshold.

## Leakage prevention
Raw monthly behavior is converted to features with SQL window functions. The churn label is excluded from feature calculations. The final test set remains untouched until final evaluation. Preprocessing is fitted within sklearn pipelines.

## Intended use
Retention prioritization and portfolio demonstration.

## Limitations and bias
The dataset is synthetic and cannot establish real-world performance. City tier, plan, payment method, and tenure may correlate with socioeconomic or geographic factors. Before production use, monitor calibration, subgroup performance, drift, and intervention fairness.

## Human oversight
Predictions should prioritize retention conversations or offers. They should not automatically deny service or infer sensitive attributes.
