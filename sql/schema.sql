CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    city TEXT NOT NULL,
    plan_name TEXT NOT NULL,
    autopay_type TEXT NOT NULL,
    tenure_months INTEGER NOT NULL,
    support_tickets INTEGER NOT NULL,
    contract_type TEXT NOT NULL,
    data_cap_gb REAL NOT NULL,
    churn INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS customer_monthly_usage (
    customer_id INTEGER NOT NULL,
    month_index INTEGER NOT NULL,
    data_gb REAL NOT NULL,
    calls_count INTEGER NOT NULL,
    days_since_recharge REAL NOT NULL,
    recharge_amount_inr REAL NOT NULL,
    PRIMARY KEY (customer_id, month_index)
);
