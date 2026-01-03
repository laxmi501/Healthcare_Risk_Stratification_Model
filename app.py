import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    try:
        saved_data = joblib.load('healthcare_risk_model.pkl')
        return saved_data['model'], saved_data['scaler']
    except:
        st.error("Model not found! Please run the analysis first.")
        return None, None

model, scaler = load_model()

st.set_page_config(page_title="Healthcare Risk Stratification", layout="wide")

st.title("Healthcare Risk Stratification App")
st.markdown("Predict patient risk levels based on clinical data")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.slider("Age", 0, 100, 50)
        length_of_stay = st.slider("Length of Stay (days)", 0, 100, 5)
        treatment_cost = st.number_input("Treatment Cost ($)", min_value=0, value=5000, step=100)
        abnormal_lab_count = st.slider("Abnormal Lab Count", 0, 10, 2)
        
    with col2:
        st.subheader("Prediction Results")
        if st.button("Predict Risk", type="primary"):
            # Prepare input
            input_data = pd.DataFrame([[
                age, length_of_stay, treatment_cost, abnormal_lab_count
            ]], columns=['Age', 'LengthOfStay', 'TreatmentCost', 'AbnormalLabCount'])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            if prediction == 1:
                st.error(f"HIGH RISK Patient")
                st.progress(probability)
            else:
                st.success(f"LOW RISK Patient") 
                st.progress(probability)
            
            st.metric("Risk Probability", f"{probability:.1%}")
            
            # Risk factors analysis
            st.subheader("Risk Factors Analysis")
            risk_factors = []
            if age > 65: risk_factors.append(f"Age > 65 ({age})")
            if length_of_stay > 10: risk_factors.append(f"Long stay >10 days ({length_of_stay})")
            if treatment_cost > 8000: risk_factors.append(f"High cost >$8000 (${treatment_cost})")
            if abnormal_lab_count > 3: risk_factors.append(f"Multiple abnormal labs ({abnormal_lab_count})")
            
            if risk_factors:
                st.warning("Identified Risk Factors:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No major risk factors identified")
    
    # Add some statistics
    st.sidebar.header("Model Information")
    st.sidebar.info("""
    This model predicts patient risk based on:
    - Age
    - Length of Stay  
    - Treatment Cost
    - Abnormal Lab Count
    """)
    
    st.sidebar.success("Model Status: Loaded")
else:
    st.error("Please run the analysis script first to train the model!")