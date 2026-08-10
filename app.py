import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_card_fraud_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

st.divider()

# Input fields
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=1000.0
)

v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
v5 = st.number_input("V5", value=0.0)

# Prediction button
if st.button("🔍 Predict Fraud"):

    input_data = pd.DataFrame({
        "Time": [time],
        "V1": [v1],
        "V2": [v2],
        "V3": [v3],
        "V4": [v4],
        "V5": [v5],
        "Amount": [amount]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION")
        st.write(f"Fraud probability: **{probability * 100:.2f}%**")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")
        st.write(f"Fraud probability: **{probability * 100:.2f}%**")