import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
saved_data = joblib.load('healthcare_risk_model.pkl')
model = saved_data['model']
scaler = saved_data['scaler']

st.title("Healthcare Risk Stratification App")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=50)
length_of_stay = st.number_input("Length of Stay (days)", min_value=0, value=5)
treatment_cost = st.number_input("Treatment Cost", min_value=0.0, value=5000.0)
abnormal_lab_count = st.number_input("Abnormal Lab Count", min_value=0, max_value=10, value=0)

if st.button("Predict"):
    # Create input dataframe
    input_data = pd.DataFrame([[
        age, length_of_stay, treatment_cost, abnormal_lab_count
    ]], columns=['Age', 'LengthOfStay', 'TreatmentCost', 'AbnormalLabCount'])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"Risk Prediction: {risk_level}")
    st.info(f"Risk Probability: {probability:.2%}")

    

