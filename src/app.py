from pathlib import Path
import json
import requests
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
)

st.title("Customer Churn Intelligence")
st.caption("Predict churn, prioritize retention, and connect model risk to retention economics.")

metric_path = Path("reports/test_metrics.json")
roi_path = Path("reports/roi_optimal.json")

if metric_path.exists():
    metrics = json.loads(metric_path.read_text())
    cols = st.columns(4)
    cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
    cols[1].metric("PR-AUC", f"{metrics['pr_auc']:.2%}")
    cols[2].metric("Recall", f"{metrics['recall']:.2%}")
    cols[3].metric("F1", f"{metrics['f1']:.2%}")

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("Customer profile")
    city = st.selectbox("City", ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune","Jaipur","Lucknow","Indore","Patna","Bhopal"], index=7)
    tier = 1 if city in ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune"] else (2 if city in ["Jaipur","Lucknow","Indore"] else 3)
    plan = st.selectbox("Plan", ["Jio 299","Airtel 349","Vi 399","Hotstar Mobile","OTT Combo 499"])
    autopay = st.selectbox("Payment", ["UPI Autopay","Card Autopay","Manual Recharge"])
    tenure = st.slider("Tenure (months)", 1, 72, 10)
    support = st.slider("Support tickets", 0, 10, 1)
    contract = st.selectbox("Contract", ["Monthly","Quarterly","Annual"])
    avg_data = st.number_input("Average monthly data (GB)", 0.1, 60.0, 4.5)
    days = st.number_input("Average days since recharge", 0.0, 45.0, 8.0)
    delta = st.number_input("Average monthly usage change (GB)", -20.0, 20.0, 0.1)

with right:
    st.subheader("Prediction")
    api_url = st.text_input("FastAPI URL", "http://localhost:8000")
    if st.button("Predict churn risk", type="primary"):
        bucket = "0-6" if tenure <= 6 else "7-12" if tenure <= 12 else "13-24" if tenure <= 24 else "25-48" if tenure <= 48 else "49-72"
        payload = {
            "city": city, "city_tier": tier, "plan_name": plan,
            "autopay_type": autopay, "tenure_months": tenure,
            "tenure_bucket": bucket, "support_tickets": support,
            "contract_type": contract, "data_cap_gb": max(avg_data, 1.0),
            "avg_data_gb": avg_data, "avg_calls": 200,
            "avg_days_since_recharge": days,
            "max_days_since_recharge": days + 5,
            "avg_recharge_amount_inr": 300,
            "recharge_frequency": 6,
            "avg_rolling_3m_data_gb": avg_data,
            "usage_increase_months": 2,
            "avg_monthly_data_delta": delta,
            "max_rolling_recharge_recency": days + 5,
        }
        try:
            response = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.metric("Churn probability", f"{result['churn_probability']:.1%}")
            if result["high_risk"]:
                st.error("High-risk retention audience")
            else:
                st.success("Below the intervention threshold")
            st.write("**Top reasons**")
            for reason in result["top_reasons"]:
                st.write(f"• {reason}")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

if roi_path.exists():
    roi = json.loads(roi_path.read_text())
    st.divider()
    st.subheader("Retention economics")
    cols = st.columns(3)
    cols[0].metric("Validation threshold", f"{roi['threshold']:.2f}")
    cols[1].metric("Targeted share", f"{roi['targeted_share']:.1%}")
    cols[2].metric("Net expected value", f"₹{roi['net_expected_value_inr']:,.0f}")
    st.caption("Threshold is optimized on validation predictions using the business assumptions in config.yaml.")
