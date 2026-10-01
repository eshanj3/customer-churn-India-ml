from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 6000
MONTHS = 6
PLANS = {
    "Jio 299": 299, "Airtel 349": 349, "Vi 399": 399,
    "Hotstar Mobile": 149, "OTT Combo 499": 499,
}
CITIES = ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune",
          "Jaipur","Indore","Lucknow","Patna","Bhopal"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    rng = np.random.default_rng(SEED)
    ids = np.arange(100000, 100000 + N_CUSTOMERS)
    city = rng.choice(CITIES, N_CUSTOMERS, p=[.14,.13,.11,.08,.09,.11,.08,.08,.09,.09])
    plan = rng.choice(list(PLANS), N_CUSTOMERS, p=[.24,.22,.18,.16,.20])
    autopay = rng.choice(["UPI Autopay","Card Autopay","Manual Recharge"],
                         N_CUSTOMERS, p=[.48,.17,.35])
    tenure = np.clip(rng.gamma(2.6, 10, N_CUSTOMERS).astype(int), 1, 72)
    support = np.clip(rng.poisson(1.5, N_CUSTOMERS), 0, 10)
    contract = rng.choice(["Monthly","Quarterly","Annual"], N_CUSTOMERS, p=[.63,.20,.17])
    data_cap = rng.choice([2,6,12,25,50], N_CUSTOMERS, p=[.12,.28,.27,.22,.11])
    base = np.array([PLANS[p] for p in plan], dtype=float)

    monthly = []
    for month in range(1, MONTHS + 1):
        monthly.append(pd.DataFrame({
            "customer_id": ids,
            "month_index": month,
            "data_gb": np.maximum(.2, rng.normal(data_cap, np.maximum(.3, data_cap*.18))),
            "calls_count": np.maximum(0, rng.poisson(240, N_CUSTOMERS)),
            "days_since_recharge": np.clip(rng.normal(7+.8*month, 3, N_CUSTOMERS), .1, 45),
            "recharge_amount_inr": np.maximum(99, base*rng.normal(1, .04, N_CUSTOMERS)),
        }))
    monthly_df = pd.concat(monthly, ignore_index=True)
    trend = monthly_df.groupby("customer_id").apply(
        lambda g: np.polyfit(g["month_index"], g["data_gb"], 1)[0],
        include_groups=False,
    ).rename("data_usage_trend")
    usage = monthly_df.groupby("customer_id").agg(
        avg_data_gb=("data_gb","mean"),
        avg_calls=("calls_count","mean"),
        avg_days_since_recharge=("days_since_recharge","mean"),
        avg_recharge=("recharge_amount_inr","mean"),
        max_days_since_recharge=("days_since_recharge","max"),
    )
    frame = pd.DataFrame({"customer_id": ids}).set_index("customer_id").join([trend, usage]).reset_index()
    frame = frame.merge(pd.DataFrame({
        "customer_id": ids, "city": city, "plan_name": plan,
        "autopay_type": autopay, "tenure_months": tenure,
        "support_tickets": support, "contract_type": contract,
        "data_cap_gb": data_cap,
    }), on="customer_id")
    logit = (-1.45 + .055*frame.avg_days_since_recharge
             - .020*frame.tenure_months
             - .35*(frame.autopay_type=="UPI Autopay")
             - .15*(frame.autopay_type=="Card Autopay")
             + .22*frame.support_tickets
             - .035*frame.avg_data_gb
             - .45*frame.data_usage_trend
             + .55*(frame.contract_type=="Monthly")
             + .35*(frame.contract_type=="Quarterly")
             + .12*frame.city.isin(["Patna","Bhopal"])
             + .15*(frame.plan_name=="Hotstar Mobile")
             + rng.normal(0,.38,N_CUSTOMERS))
    frame["churn"] = rng.binomial(1, sigmoid(logit))
    customers = frame[["customer_id","city","plan_name","autopay_type","tenure_months",
                       "support_tickets","contract_type","data_cap_gb","churn"]]
    Path("data").mkdir(exist_ok=True)
    customers.to_csv("data/customers.csv", index=False)
    monthly_df.to_csv("data/customer_monthly_usage.csv", index=False)
    print(f"Generated {len(customers):,} customers and {len(monthly_df):,} monthly records.")
    print(f"Churn rate: {customers.churn.mean():.3f}")

if __name__ == "__main__":
    main()
