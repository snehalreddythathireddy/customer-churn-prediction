import streamlit as st
import pandas as pd
import joblib

model = joblib.load('notebooks/churn_model.pkl')
feature_columns = joblib.load('notebooks/feature_columns.pkl')

st.title("📉 Telco Customer Churn Predictor")
st.write("Fill in the customer details to predict churn probability.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

if st.button("Predict Churn"):
    input_dict = {col: 0 for col in feature_columns}
    
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    
    for col in ['Partner_Yes', 'Dependents_Yes', 'PaperlessBilling_Yes']:
        key = col.split('_')[0]
        val = locals()[key.lower().replace('paperlessbilling', 'paperless_billing').replace('partner', 'partner').replace('dependents', 'dependents')]
        input_dict[col] = 1 if val == "Yes" else 0

    contract_map = {"Month-to-month": None, "One year": "Contract_One year", "Two year": "Contract_Two year"}
    if contract_map[contract]:
        input_dict[contract_map[contract]] = 1

    internet_map = {"DSL": None, "Fiber optic": "InternetService_Fiber optic", "No": "InternetService_No"}
    if internet_map[internet_service]:
        input_dict[internet_map[internet_service]] = 1

    payment_map = {
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check",
        "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
        "Credit card (automatic)": None
    }
    if payment_map[payment_method]:
        input_dict[payment_map[payment_method]] = 1

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prob > 0.5:
        st.error(f"⚠️ High Churn Risk: {prob:.1%} probability")
    else:
        st.success(f"✅ Low Churn Risk: {prob:.1%} probability")
    
    st.progress(float(prob))