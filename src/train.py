from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sql_features import build_feature_table
from roi import optimize_threshold

SEED = 42
TARGET = "churn"

def make_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

def metrics(y, probability, threshold=.5):
    pred = (probability >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
    }

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df = build_feature_table()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, "customer_id"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=cfg["model"]["validation_size"],
        stratify=y_train_full, random_state=SEED
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=350, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04,
                                 subsample=.85, colsample_bytree=.85,
                                 eval_metric="logloss", random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True, random_state=SEED)
    rows, candidates = [], {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"roc_auc":"roc_auc","pr_auc":"average_precision","recall":"recall","f1":"f1"},
            n_jobs=-1,
        )
        pipe.fit(X_train, y_train)
        test_p = pipe.predict_proba(X_test)[:,1]
        val_p = pipe.predict_proba(X_val)[:,1]
        m = metrics(y_test, test_p)
        rows.append({
            "model": name,
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
            "cv_recall_mean": scores["test_recall"].mean(),
            "cv_f1_mean": scores["test_f1"].mean(),
            **{f"test_{k}": v for k,v in m.items()},
        })
        candidates[name] = (pipe, val_p, test_p)

    comparison = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    comparison.to_csv("reports/model_comparison.csv", index=False)

    selected = comparison.sort_values(["test_recall","test_pr_auc"], ascending=False).iloc[0]["model"]
    pipe, val_p, test_p = candidates[selected]

    roi_table, best = optimize_threshold(y_val.to_numpy(), val_p)
    final = metrics(y_test, test_p, float(best["threshold"]))

    payload = {
        "model_name": selected,
        "test_size": int(len(X_test)),
        "test_churn_rate": float(y_test.mean()),
        **final,
        "validation_optimal_threshold": float(best["threshold"]),
        "validation_targeted_share": float(best["targeted_share"]),
        "validation_captured_churners": int(best["captured_churners"]),
        "validation_gross_expected_value_inr": float(best["gross_expected_value_inr"]),
        "validation_intervention_cost_inr": float(best["intervention_cost_inr"]),
        "validation_net_expected_value_inr": float(best["net_expected_value_inr"]),
    }
    Path("reports/test_metrics.json").write_text(json.dumps(payload, indent=2))
    joblib.dump({
        "pipeline": pipe,
        "features": X.columns.tolist(),
        "model_name": selected,
        "decision_threshold": float(best["threshold"]),
    }, "models/churn_model.joblib")
    plt.figure(figsize=(8, 4.5))
    plt.plot(roi_table["threshold"], roi_table["net_expected_value_inr"], linewidth=2)
    plt.axvline(float(best["threshold"]), linestyle="--", label=f"Optimal = {best['threshold']:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Net expected value (₹)")
    plt.title("Retention ROI by Churn Decision Threshold")
    plt.legend()
    plt.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    plt.savefig("assets/hero.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()
