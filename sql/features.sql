WITH ordered AS (
    SELECT
        customer_id,
        month_index,
        data_gb,
        calls_count,
        days_since_recharge,
        recharge_amount_inr,
        LAG(data_gb) OVER (
            PARTITION BY customer_id ORDER BY month_index
        ) AS prev_data_gb,
        AVG(data_gb) OVER (
            PARTITION BY customer_id ORDER BY month_index
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m_data_gb,
        AVG(days_since_recharge) OVER (
            PARTITION BY customer_id ORDER BY month_index
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3m_recharge_recency
    FROM customer_monthly_usage
),
agg AS (
    SELECT
        customer_id,
        AVG(data_gb) AS avg_data_gb,
        AVG(calls_count) AS avg_calls,
        AVG(days_since_recharge) AS avg_days_since_recharge,
        MAX(days_since_recharge) AS max_days_since_recharge,
        AVG(recharge_amount_inr) AS avg_recharge_amount_inr,
        COUNT(*) AS recharge_frequency,
        AVG(rolling_3m_data_gb) AS avg_rolling_3m_data_gb,
        SUM(CASE WHEN month_index >= 4 AND data_gb > prev_data_gb
                 THEN 1 ELSE 0 END) AS usage_increase_months,
        AVG(data_gb - COALESCE(prev_data_gb, data_gb))
            AS avg_monthly_data_delta,
        MAX(rolling_3m_recharge_recency)
            AS max_rolling_recharge_recency
    FROM ordered
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.city,
    CASE
        WHEN c.city IN ('Mumbai','Delhi','Bengaluru','Hyderabad','Pune')
            THEN 1
        WHEN c.city IN ('Jaipur','Lucknow','Indore')
            THEN 2
        ELSE 3
    END AS city_tier,
    c.plan_name,
    c.autopay_type,
    c.tenure_months,
    CASE
        WHEN c.tenure_months <= 6 THEN '0-6'
        WHEN c.tenure_months <= 12 THEN '7-12'
        WHEN c.tenure_months <= 24 THEN '13-24'
        WHEN c.tenure_months <= 48 THEN '25-48'
        ELSE '49-72'
    END AS tenure_bucket,
    c.support_tickets,
    c.contract_type,
    c.data_cap_gb,
    a.avg_data_gb,
    a.avg_calls,
    a.avg_days_since_recharge,
    a.max_days_since_recharge,
    a.avg_recharge_amount_inr,
    a.recharge_frequency,
    a.avg_rolling_3m_data_gb,
    a.usage_increase_months,
    a.avg_monthly_data_delta,
    a.max_rolling_recharge_recency,
    c.churn
FROM customers c
JOIN agg a USING (customer_id);
