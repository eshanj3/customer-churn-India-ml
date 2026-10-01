from __future__ import annotations
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Guardian", page_icon="📉", layout="wide", initial_sidebar_state="expanded")
API_DEFAULT = "http://localhost:8000"

def api_get(api_url: str, path: str, params: dict | None = None):
    response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def api_post(api_url: str, path: str, payload: dict):
    response = requests.post(f"{api_url.rstrip('/')}{path}", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

st.markdown("## CHURN GUARDIAN")
st.caption("Indian subscription churn intelligence — predict risk, explain drivers, and connect retention to economics.")

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Overview", "Customer Scoring", "ROI Optimizer", "Explainability", "Model Health"], label_visibility="collapsed")
    st.divider()
    api_url = st.text_input("FastAPI URL", API_DEFAULT)

try:
    health = api_get(api_url, "/health")
    api_ok = True
except requests.RequestException:
    health = None
    api_ok = False

if api_ok:
    st.success(f"● API healthy · {health['model']} · v{health['model_version']}")
else:
    st.warning("API unavailable. Start FastAPI to enable live predictions and model endpoints.")

if page == "Overview":
    st.title("Churn Risk Overview")
    st.caption("Prioritize retention using predicted risk and expected value.")
    try:
        metrics = api_get(api_url, "/metrics")
        roi = api_get(api_url, "/roi")
    except requests.RequestException as exc:
        st.error(f"Could not load model analytics: {exc}")
        metrics = roi = None
    if metrics:
        cols = st.columns(4)
        cols[0].metric("Test customers", f"{metrics['dataset']['test_size']:,}")
        cols[1].metric("Observed churn", f"{metrics['dataset']['test_churn_rate']:.1%}")
        cols[2].metric("ROC-AUC", f"{metrics['ranking']['roc_auc']:.1%}")
        cols[3].metric("PR-AUC", f"{metrics['ranking']['pr_auc']:.1%}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk audience")
            st.metric("Validation threshold", f"{metrics['decision_threshold']:.2f}")
            st.metric("Churners captured", f"{metrics['roi']['captured_churners']:,}")
            st.caption(f"Targeted share: {metrics['roi']['targeted_share']:.1%}")
        with c2:
            st.subheader("Retention economics")
            st.metric("Gross expected value", f"₹{metrics['roi']['gross_expected_value_inr']:,.0f}")
            st.metric("Intervention cost", f"₹{metrics['roi']['intervention_cost_inr']:,.0f}")
            st.metric("Net expected value", f"₹{metrics['roi']['net_expected_value_inr']:,.0f}")
        st.caption(f"Model version: {metrics['model_version']} · selection metric: {metrics['selection_metric']}")
    if roi:
        curve = pd.DataFrame(roi["curve"])
        st.subheader("ROI by decision threshold")
        st.line_chart(curve.set_index("threshold")["net_expected_value_inr"])

elif page == "Customer Scoring":
    st.title("Customer Scoring")
    st.caption("Score an individual customer and review the model's top risk signals.")
    left, right = st.columns([1, 1])
    with left:
        city = st.selectbox("City", ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune","Jaipur","Lucknow","Indore","Patna","Bhopal"], index=7)
        tier = 1 if city in ["Mumbai","Delhi","Bengaluru","Hyderabad","Pune"] else (2 if city in ["Jaipur","Lucknow","Indore"] else 3)
        plan = st.selectbox("Plan", ["Jio 299","Airtel 349","Vi 399","Hotstar Mobile","OTT Combo 499"])
        autopay = st.selectbox("Payment", ["UPI Autopay","Card Autopay","Manual Recharge"])
        tenure = st.slider("Tenure (months)", 1, 72, 10)
        support = st.slider("Support tickets", 0, 10, 1)
        contract = st.selectbox("Contract", ["Monthly","Quarterly","Annual"])
        data_cap = st.number_input("Data cap (GB)", 0.0, 100.0, 6.0)
        avg_data = st.number_input("Average monthly data (GB)", 0.0, 100.0, 4.5)
        avg_calls = st.number_input("Average calls", 0.0, 2000.0, 200.0)
        days = st.number_input("Average days since recharge", 0.0, 60.0, 8.0)
        max_days = st.number_input("Maximum days since recharge", 0.0, 90.0, 14.0)
    with right:
        avg_recharge = st.number_input("Average recharge amount (₹)", 0.0, 5000.0, 300.0)
        recharge_frequency = st.number_input("Recharge frequency", 0.0, 31.0, 6.0)
        rolling_data = st.number_input("Rolling 3M average data (GB)", 0.0, 100.0, avg_data)
        usage_increase = st.number_input("Months with increasing usage", 0.0, 6.0, 2.0)
        delta = st.number_input("Average monthly data change (GB)", -30.0, 30.0, 0.1)
        rolling_recharge = st.number_input("Max rolling recharge recency", 0.0, 90.0, max_days)
        tenure_bucket = "0-6" if tenure <= 6 else "7-12" if tenure <= 12 else "13-24" if tenure <= 24 else "25-48" if tenure <= 48 else "49-72"
        if st.button("Predict churn risk", type="primary", use_container_width=True):
            payload = {
                "city": city, "city_tier": tier, "plan_name": plan,
                "autopay_type": autopay, "tenure_months": tenure,
                "tenure_bucket": tenure_bucket, "support_tickets": support,
                "contract_type": contract, "data_cap_gb": data_cap,
                "avg_data_gb": avg_data, "avg_calls": avg_calls,
                "avg_days_since_recharge": days, "max_days_since_recharge": max_days,
                "avg_recharge_amount_inr": avg_recharge,
                "recharge_frequency": recharge_frequency,
                "avg_rolling_3m_data_gb": rolling_data,
                "usage_increase_months": usage_increase,
                "avg_monthly_data_delta": delta,
                "max_rolling_recharge_recency": rolling_recharge,
            }
            try:
                st.session_state["prediction"] = api_post(api_url, "/predict", payload)
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
        result = st.session_state.get("prediction")
        if result:
            st.metric("Churn probability", f"{result['churn_probability']:.1%}")
            if result["high_risk"]:
                st.error("HIGH RISK — route for retention review")
            else:
                st.success("Below intervention threshold")
            st.caption(f"Threshold: {result['decision_threshold']:.2f} · Model v{result['model_version']}")
            st.subheader("Top reasons")
            for reason in result["top_reasons"]:
                st.write(f"• {reason}")

elif page == "ROI Optimizer":
    st.title("ROI Optimizer")
    st.caption("Understand how the decision threshold changes audience size and retention economics.")
    try:
        roi = api_get(api_url, "/roi")
    except requests.RequestException as exc:
        st.error(f"Could not load ROI data: {exc}")
        roi = None
    if roi:
        a, o = roi["assumptions"], roi["optimal"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected threshold", f"{o['threshold']:.2f}")
        c2.metric("Targeted share", f"{o['targeted_share']:.1%}")
        c3.metric("Net expected value", f"₹{o['net_expected_value_inr']:,.0f}")
        st.subheader("Business assumptions")
        st.write(f"Monthly margin: ₹{a['monthly_margin_inr']:,.0f} · Intervention cost: ₹{a['intervention_cost_inr']:,.0f} · Save probability: {a['save_probability_if_targeted']:.0%}")
        curve = pd.DataFrame(roi["curve"])
        st.subheader("Expected value by threshold")
        st.line_chart(curve.set_index("threshold")["net_expected_value_inr"])
        st.subheader("Scenario table")
        st.dataframe(curve, use_container_width=True, hide_index=True)

elif page == "Explainability":
    st.title("Explainability")
    st.caption("Global SHAP feature importance plus customer-level prediction reasons.")
    try:
        data = api_get(api_url, "/explainability", params={"top_n": 12})
    except requests.RequestException as exc:
        st.error(f"Could not load explainability data: {exc}")
        data = None
    if data:
        st.caption(f"Method: {data['method']} · Model v{data['model_version']}")
        df = pd.DataFrame(data["drivers"])
        st.bar_chart(df.set_index("display_name")["mean_abs_shap"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info("Mean absolute SHAP measures importance magnitude. It does not by itself establish causality or direction for a specific customer.")

else:
    st.title("Model Health")
    st.caption("Operational status, evaluation metrics, and reproducibility metadata.")
    if health:
        c1, c2, c3 = st.columns(3)
        c1.metric("API", health["status"].upper())
        c2.metric("Model", health["model"])
        c3.metric("Model version", health["model_version"])
    try:
        metrics = api_get(api_url, "/metrics")
    except requests.RequestException as exc:
        st.error(f"Could not load model metrics: {exc}")
        metrics = None
    if metrics:
        cols = st.columns(5)
        for col, key, label in zip(cols, ["precision","recall","f1"], ["Precision","Recall","F1"]):
            col.metric(label, f"{metrics['classification'][key]:.1%}")
        cols2 = st.columns(2)
        cols2[0].metric("ROC-AUC", f"{metrics['ranking']['roc_auc']:.1%}")
        cols2[1].metric("PR-AUC", f"{metrics['ranking']['pr_auc']:.1%}")
        st.subheader("Evaluation metadata")
        st.json({
            "selection_metric": metrics["selection_metric"],
            "test_size": metrics["dataset"]["test_size"],
            "test_churn_rate": metrics["dataset"]["test_churn_rate"],
            "decision_threshold": metrics["decision_threshold"],
            "model_version": metrics["model_version"],
        })
