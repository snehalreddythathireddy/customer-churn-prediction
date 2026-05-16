import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #f0f0f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 80%, #001a33 0%, transparent 60%);
    min-height: 100vh;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.hero {
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}

.hero-tag {
    display: inline-block;
    background: linear-gradient(90deg, #7B2FFF22, #00D4FF22);
    border: 1px solid #7B2FFF55;
    color: #a78bfa;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(40px, 6vw, 80px);
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #00D4FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 16px;
}

.hero-sub {
    color: #888;
    font-size: 18px;
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto 40px;
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 50px;
    flex-wrap: wrap;
}

.stat-box {
    text-align: center;
    padding: 16px 24px;
    background: #ffffff08;
    border: 1px solid #ffffff12;
    border-radius: 12px;
    min-width: 120px;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #00D4FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 11px;
    color: #666;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

.card {
    background: #ffffff06;
    border: 1px solid #ffffff10;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}

.card:hover { border-color: #7B2FFF44; }

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 20px;
}

.result-high {
    background: linear-gradient(135deg, #ff000015, #ff000005);
    border: 1px solid #ff000044;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-mid {
    background: linear-gradient(135deg, #ff990015, #ff990005);
    border: 1px solid #ff990044;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-low {
    background: linear-gradient(135deg, #00ff8815, #00ff8805);
    border: 1px solid #00ff8844;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-prob {
    font-family: 'Syne', sans-serif;
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
    margin: 16px 0;
}

.rec-item {
    background: #ffffff08;
    border-left: 3px solid #7B2FFF;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 14px;
    color: #ccc;
}

div[data-testid="stSlider"] > div { padding: 0; }

.stSelectbox > div > div {
    background: #ffffff08 !important;
    border: 1px solid #ffffff15 !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
}

label { color: #aaa !important; font-size: 13px !important; }

.stButton > button {
    background: linear-gradient(135deg, #7B2FFF, #00D4FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    margin-top: 10px !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #ffffff10 !important; }
</style>
""", unsafe_allow_html=True)

model = joblib.load('notebooks/churn_model.pkl')
feature_columns = joblib.load('notebooks/feature_columns.pkl')

st.markdown("""
<div class="hero">
    <div class="hero-tag">⚡ Powered by XGBoost + SHAP</div>
    <div class="hero-title">ChurnSense AI</div>
    <div class="hero-sub">Predict customer churn in seconds with explainable machine learning</div>
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">0.83</div><div class="stat-label">ROC-AUC</div></div>
        <div class="stat-box"><div class="stat-num">79%</div><div class="stat-label">Recall</div></div>
        <div class="stat-box"><div class="stat-num">7K+</div><div class="stat-label">Trained On</div></div>
        <div class="stat-box"><div class="stat-num">30</div><div class="stat-label">Features</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><div class="card-title">📋 Contract & Billing</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">💰 Charges & Service</div>', unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-title">👥 Demographics</div>', unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

st.button("⚡ ANALYZE CHURN RISK", key="predict")

if st.session_state.get("predict"):
    input_dict = {col: 0 for col in feature_columns}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    if partner == "Yes": input_dict['Partner_Yes'] = 1
    if dependents == "Yes": input_dict['Dependents_Yes'] = 1
    if paperless_billing == "Yes": input_dict['PaperlessBilling_Yes'] = 1
    contract_map = {"One year": "Contract_One year", "Two year": "Contract_Two year"}
    if contract in contract_map: input_dict[contract_map[contract]] = 1
    internet_map = {"Fiber optic": "InternetService_Fiber optic", "No": "InternetService_No"}
    if internet_service in internet_map: input_dict[internet_map[internet_service]] = 1
    payment_map = {
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check",
        "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)"
    }
    if payment_method in payment_map: input_dict[payment_map[payment_method]] = 1

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prob > 0.7:
        css_class = "result-high"
        emoji = "🔴"
        label = "HIGH CHURN RISK"
        color = "#ff4444"
        recs = ["Assign a dedicated retention specialist immediately", "Offer a discounted 2-year contract upgrade", "Investigate service quality complaints"]
    elif prob > 0.4:
        css_class = "result-mid"
        emoji = "🟡"
        label = "MEDIUM CHURN RISK"
        color = "#ffaa00"
        recs = ["Send a personalised loyalty offer", "Schedule a satisfaction check-in call", "Offer a free service upgrade trial"]
    else:
        css_class = "result-low"
        emoji = "🟢"
        label = "LOW CHURN RISK"
        color = "#00ff88"
        recs = ["Customer is satisfied — consider upselling", "Enroll in loyalty rewards programme", "Great candidate for referral programme"]

    recs_html = "".join([f'<div class="rec-item">→ {r}</div>' for r in recs])

    st.markdown(f"""
    <div class="{css_class}">
        <div style="font-size:13px;letter-spacing:3px;text-transform:uppercase;color:{color};opacity:0.8">{emoji} {label}</div>
        <div class="result-prob" style="color:{color}">{prob:.0%}</div>
        <div style="color:#888;font-size:14px;margin-bottom:20px">churn probability</div>
        <div style="text-align:left;max-width:400px;margin:0 auto">
            <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#666;margin-bottom:10px">Recommended Actions</div>
            {recs_html}
        </div>
    </div>
    """, unsafe_allow_html=True)