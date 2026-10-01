# Resume Bullets

GitHub: https://github.com/eshanj3/customer-churn-India-ml  
Figma: https://www.figma.com/design/ORVtXABGbmEPXOWJIv7s82

- Built a leakage-controlled customer churn pipeline for a synthetic Indian subscription business, achieving 66.69% ROC-AUC and 37.16% PR-AUC on a 1,200-customer untouched holdout using SQL window features and class-balanced Logistic Regression.

- Engineered behavioral features with SQLite LAG and rolling windows, compared Logistic Regression, Random Forest and XGBoost using five-fold CV, and selected the final model by validation-safe CV PR-AUC rather than test-set performance.

- Optimized the retention decision threshold against intervention economics, targeting 34.48% of validation customers, capturing 119 churners, and generating ₹2,565 validation net expected value under explicit business assumptions.

- Productionized the model with FastAPI endpoints for health, metrics, ROI, SHAP explainability and prediction, plus a Streamlit dashboard and GitHub Actions CI for reproducible training and testing.
