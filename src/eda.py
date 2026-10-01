from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

def main():
    customers = pd.read_csv("data/customers.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    numeric = customers.select_dtypes(include="number")
    summary = {
        "rows": int(len(customers)),
        "columns": int(len(customers.columns)),
        "churn_rate": float(customers["churn"].mean()),
        "missing_cells": int(customers.isna().sum().sum()),
        "duplicate_customer_ids": int(customers["customer_id"].duplicated().sum()),
        "numeric_summary": numeric.describe().round(3).to_dict(),
    }
    Path("reports/eda_summary.json").write_text(json.dumps(summary, indent=2))
    customers.groupby("plan_name", as_index=False)["churn"].mean().rename(
        columns={"churn": "churn_rate"}
    ).sort_values("churn_rate", ascending=False).to_csv(
        "reports/churn_by_plan.csv", index=False
    )
    churn_counts = customers["churn"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Retained", "Churned"],
        [int(churn_counts.get(0, 0)), int(churn_counts.get(1, 0))]
    )
    plt.ylabel("Customers")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig("assets/eda_churn_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()
    plan_rates = customers.groupby("plan_name")["churn"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(plan_rates.index, plan_rates.values)
    plt.ylabel("Churn rate")
    plt.title("Churn Rate by Plan")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("assets/eda_churn_by_plan.png", dpi=160, bbox_inches="tight")
    plt.close()
    tenure_bins = pd.cut(
        customers["tenure_months"],
        bins=[0, 6, 12, 24, 48, 72],
        include_lowest=True
    )
    tenure_rates = customers.groupby(tenure_bins, observed=True)["churn"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(tenure_rates.index.astype(str), tenure_rates.values)
    plt.ylabel("Churn rate")
    plt.title("Churn Rate by Tenure Bucket")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("assets/eda_churn_by_tenure.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
