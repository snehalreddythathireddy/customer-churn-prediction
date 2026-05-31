"""
app.py — Telco Customer Churn Prediction
Streamlit interactive demo for real-time churn risk assessment.

Run: streamlit run app.py
"""

import os, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title { font-size:2.2rem; font-weight:700; color:#1565C0; margin-bottom:0; }
.sub-title  { font-size:1rem;  color:#546E7A;  margin-top:0; }
.risk-high   { background:#FFEBEE; border-left:5px solid #E53935;
               border-radius:10px; padding:16px 20px; text-align:center; }
.risk-medium { background:#FFF8E1; border-left:5px solid #F9A825;
               border-radius:10px; padding:16px 20px; text-align:center; }
.risk-low    { background:#E8F5E9; border-left:5px solid #2E7D32;
               border-radius:10px; padding:16px 20px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdir = os.path.join(os.path.dirname(__file__), "models")
    try:
        model         = joblib.load(f"{mdir}/best_model.pkl")
        scaler        = joblib.load(f"{mdir}/scaler.pkl")
        feature_names = joblib.load(f"{mdir}/feature_names.pkl")
        model_name    = joblib.load(f"{mdir}/best_model_name.pkl")
        return model, scaler, feature_names, model_name
    except FileNotFoundError:
        return None, None, None, None

model, scaler, feature_names, model_name = load_artifacts()
@st.cache_data
def get_train_data():
    df2 = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df2.drop(columns=['customerID'], inplace=True)
    df2['TotalCharges'] = pd.to_numeric(df2['TotalCharges'], errors='coerce')
    df2['TotalCharges'].fillna(df2['TotalCharges'].median(), inplace=True)
    df2['Churn'] = df2['Churn'].map({'Yes':1,'No':0})
    binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling',
                   'MultipleLines','OnlineSecurity','OnlineBackup',
                   'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in binary_cols:
        df2[col] = df2[col].map({'Yes':1,'No':0,'No phone service':0,'No internet service':0})
    df2['gender'] = df2['gender'].map({'Male':1,'Female':0})
    df2 = pd.get_dummies(df2, columns=['Contract','PaymentMethod','InternetService'])
    bool_cols = df2.select_dtypes(include='bool').columns
    df2[bool_cols] = df2[bool_cols].astype(int)
    X2 = df2.drop(columns=['Churn'])
    from sklearn.model_selection import train_test_split
    X_tr, _ = train_test_split(X2, test_size=0.2, random_state=42)
    X_tr = X_tr[feature_names]
    X_tr = X_tr.copy()
    X_tr[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_tr[['tenure','MonthlyCharges','TotalCharges']])
    return X_tr

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📡 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">IBM Telco Dataset · 7,043 customers · Real-time risk assessment</p>',
            unsafe_allow_html=True)

if model_name:
    st.success(f"✅ Active model: **{model_name}** — loaded from /models")
else:
    st.error("⚠️ No trained model found. Run all cells in `notebooks/02_model.ipynb` first.")
    st.stop()

# ── Sidebar: Customer Input ───────────────────────────────────────────────────
st.sidebar.header("👤 Customer Profile")

with st.sidebar:
    st.subheader("Demographics")
    gender         = st.selectbox("Gender",         ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner",        ["Yes", "No"])
    dependents     = st.selectbox("Dependents",     ["Yes", "No"])

    st.subheader("Account")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type",
                                   ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                      float(tenure * monthly_charges), 10.0)

    st.subheader("Services")
    phone_service    = st.selectbox("Phone Service",   ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service",["Fiber optic", "DSL", "No"])
    online_security  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
    device_prot      = st.selectbox("Device Protection",["No","Yes","No internet service"])
    tech_support     = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",["No", "Yes", "No internet service"])

    predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ── Build input row ───────────────────────────────────────────────────────────
def yn(v):   return 1 if v == "Yes" else 0
def svc(v):  return 0 if v in ("No","No phone service","No internet service") else 1

def build_row():
    base = {
        "gender": 1 if gender=="Male" else 0,
        "SeniorCitizen": yn(senior_citizen),
        "Partner": yn(partner), "Dependents": yn(dependents),
        "tenure": tenure, "PhoneService": yn(phone_service),
        "MultipleLines": svc(multiple_lines),
        "OnlineSecurity": svc(online_security), "OnlineBackup": svc(online_backup),
        "DeviceProtection": svc(device_prot),   "TechSupport": svc(tech_support),
        "StreamingTV": svc(streaming_tv),        "StreamingMovies": svc(streaming_movies),
        "PaperlessBilling": yn(paperless),
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }
    for c in ["Month-to-month","One year","Two year"]:
        base[f"Contract_{c}"] = 1 if contract==c else 0
    for p in ["Electronic check","Mailed check",
              "Bank transfer (automatic)","Credit card (automatic)"]:
        base[f"PaymentMethod_{p}"] = 1 if payment_method==p else 0
    for s in ["DSL","Fiber optic","No"]:
        base[f"InternetService_{s}"] = 1 if internet_service==s else 0

    row = pd.DataFrame([base])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    return row[feature_names]

def scale_row(row):
    r = row.copy()
    r[["tenure","MonthlyCharges","TotalCharges"]] = \
        scaler.transform(r[["tenure","MonthlyCharges","TotalCharges"]])
    return r

def risk_info(prob):
    if prob >= 0.65:
        return "🔴 HIGH RISK",   "risk-high",   "#E53935"
    elif prob >= 0.35:
        return "🟡 MEDIUM RISK", "risk-medium", "#F9A825"
    else:
        return "🟢 LOW RISK",    "risk-low",    "#2E7D32"

# ── Gauge ─────────────────────────────────────────────────────────────────────
def gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={"text":"Churn Probability (%)","font":{"size":16}},
        gauge={
            "axis":  {"range":[0,100]},
            "bar":   {"color":color,"thickness":0.25},
            "steps": [{"range":[0,35], "color":"#E8F5E9"},
                      {"range":[35,65],"color":"#FFF8E1"},
                      {"range":[65,100],"color":"#FFEBEE"}],
            "threshold":{"line":{"color":"#1565C0","width":3},
                         "thickness":0.8,"value":26.5},
        },
        number={"suffix":"%","font":{"size":28}},
    ))
    fig.update_layout(height=260, margin=dict(t=30,b=10,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ── SHAP waterfall ────────────────────────────────────────────────────────────
def get_shap_explanation(model, model_name, row_sc, X_train_sc):
    model_type = type(model).__name__.lower()
    if 'xgb' in model_type or 'forest' in model_type:
        exp = shap.TreeExplainer(model)
        raw = exp.shap_values(row_sc)
        sv  = raw[1][0] if isinstance(raw, list) else raw[0]
        ev  = exp.expected_value[1] if hasattr(exp.expected_value, '__len__') else exp.expected_value
    else:
        exp = shap.LinearExplainer(model, X_train_sc)
        raw = exp.shap_values(row_sc)
        sv  = raw[0] if hasattr(raw, '__len__') else raw
        ev  = exp.expected_value

    explanation = shap.Explanation(
        values=sv, base_values=ev,
        data=row_sc.values[0],
        feature_names=list(row_sc.columns)
    )
    fig, _ = plt.subplots(figsize=(9, 5))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.tight_layout()
    return fig
# ── Main panel ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 About the Model"])

with tab1:
    if predict_btn:
        row    = build_row()
        row_sc = scale_row(row)
        prob   = model.predict_proba(row_sc)[0][1]
        pred   = int(prob >= 0.5)
        label, css, color = risk_info(prob)

        st.markdown("---")
        c1, c2, c3 = st.columns([1.3, 1, 1.2])

        with c1:
            st.plotly_chart(gauge(prob, color), use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="{css}">
                <h2>{label}</h2>
                <h3>{prob*100:.1f}% churn probability</h3>
                <p>{"⚠️ Likely to churn" if pred else "✅ Likely to stay"}</p>
                <small style="color:#78909C">Dataset avg: ~26.5%</small>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown("**💡 Retention Suggestions**")
            tips = []
            if contract == "Month-to-month":
                tips.append("Offer a discounted **1-year contract**")
            if internet_service == "Fiber optic" and online_security == "No":
                tips.append("Bundle **Online Security** for free (3 months)")
            if tech_support == "No" and internet_service != "No":
                tips.append("Offer free **Tech Support** trial")
            if monthly_charges > 70:
                tips.append("Provide a **loyalty discount** or bundle deal")
            if tenure < 12:
                tips.append("Send a **welcome loyalty reward** — new customers at high risk")
            if not tips:
                tips.append("Customer profile looks stable — continue standard engagement")
            for t in tips:
                st.markdown(f"- {t}")

       
       # SHAP
        st.markdown("---")
        st.markdown("#### 🔍 Why this prediction? (SHAP Explanation)")
        try:
            X_train_sc = get_train_data()
            fig = get_shap_explanation(model, model_name, row_sc, X_train_sc)
            st.pyplot(fig, use_container_width=True)
            st.caption("Red bars → increase churn risk  |  Blue bars → decrease churn risk")
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    elif not predict_btn:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#90A4AE;">
            <h2>👈 Fill in the customer details in the sidebar</h2>
            <p>Then click <b>Predict Churn Risk</b></p>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### About This Project
    **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features

    **Models trained & compared:**
    | Model | Notes |
    |---|---|
    | Logistic Regression | Baseline, class-weight balanced |
    | Random Forest | 200 trees, depth 10 |
    | XGBoost | 300 rounds, lr=0.05, scale_pos_weight=3 |

    **Key EDA Findings:**
    - 26.6% overall churn rate (imbalanced dataset)
    - Month-to-month contracts churn at ~42%
    - Fiber optic users churn most (~42%)
    - Electronic check payment has highest churn (~45%)
    - Customers with low tenure churn heavily in first 10 months

    **Explainability:** SHAP TreeExplainer — beeswarm, bar chart, and waterfall plots

    Run `notebooks/02_model.ipynb` to retrain models and regenerate artifacts.
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#90A4AE;font-size:0.8rem;'>"
    "IBM Telco Churn · Python · Scikit-learn · XGBoost · SHAP · Streamlit"
    "</div>", unsafe_allow_html=True
)
