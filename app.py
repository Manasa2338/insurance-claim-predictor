import streamlit as st
import pickle
import pandas as pd

# Load saved objects
model = pickle.load(open("claim_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.set_page_config(page_title="Insurance Claim Prediction", layout="centered")

st.title("🚗 Car Insurance Claim Prediction")
st.write("Predict whether a customer is likely to file an insurance claim.")

# User Inputs
gender = st.selectbox("Gender (1 = Male, 0 = Female)", [0, 1])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
annual_mileage = st.number_input("Annual Mileage", min_value=0)
speeding_violations = st.number_input("Speeding Violations", min_value=0)
duis = st.number_input("DUIs", min_value=0)
vehicle_year_before_2015 = st.selectbox("Vehicle Manufactured Before 2015", [0, 1])

# Create input dataframe
input_data = pd.DataFrame([[gender, credit_score, annual_mileage,
                            speeding_violations, duis,
                            vehicle_year_before_2015]],
                          columns=[
                              'gender',
                              'credit_score',
                              'annual_mileage',
                              'speeding_violations',
                              'duis',
                              'vehicle_year_before 2015'
                          ])

# Align with training features
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Claim"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Claim (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Claim (Probability: {probability:.2f})")
