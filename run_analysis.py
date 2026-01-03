import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Starting Healthcare Risk Stratification Analysis...")

# Load data
try:
    patients = pd.read_csv("patients.csv")
    diagnoses = pd.read_csv("diagnoses.csv") 
    outcomes = pd.read_csv("outcomes.csv")
    labs = pd.read_csv("labs.csv")
    print(" Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Data preprocessing
print("Preprocessing data...")

# Merge datasets
patients = patients.merge(diagnoses, on='DiagnosisID')
patients = patients.merge(outcomes, on='OutcomeID')

# Fix date processing
patients['AdmissionDate'] = pd.to_datetime(patients['AdmissionDate'], format='%d-%m-%Y', errors='coerce')
patients['DischargeDate'] = pd.to_datetime(patients['DischargeDate'], format='%d-%m-%Y', errors='coerce')
patients['LengthOfStay'] = (patients['DischargeDate'] - patients['AdmissionDate']).dt.days
patients['LengthOfStay'] = patients['LengthOfStay'].clip(lower=0)

# Outcome encoding
patients['OutcomeEncoded'] = patients['OutcomeName'].map({'Recovered': 0, 'Complicated': 1, 'Deceased': 1})

# Lab analysis
abnormal_conditions = {
    'Blood Pressure': lambda x: x > 140,
    'Blood Sugar': lambda x: x > 126,
    'Cholesterol': lambda x: x > 200,
    'Hemoglobin': lambda x: x < 13,
    'Creatinine': lambda x: x > 1.2,
    'Vitamin D': lambda x: x < 20
}

def count_abnormal_labs(patient_id):
    patient_labs = labs[labs['PatientID'] == patient_id]
    count = 0
    for test_name, condition in abnormal_conditions.items():
        test_results = patient_labs[patient_labs['TestName'] == test_name]
        if not test_results.empty:
            count += test_results['Result'].apply(condition).sum()
    return count

patients['AbnormalLabCount'] = patients['PatientID'].apply(count_abnormal_labs)

print("Data preprocessing completed!")

# Model training
print("Training machine learning model...")

# Prepare features
features = patients[['Age', 'LengthOfStay', 'TreatmentCost', 'AbnormalLabCount']]
target = patients['OutcomeEncoded']

# Handle missing values
features = features.fillna(features.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model training completed!")

# Model evaluation
print("Evaluating model performance...")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*50)
print("CLASSIFICATION REPORT:")
print("="*50)
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nModel AUC Score: {roc_auc:.4f}")

# Save model
joblib.dump({'model': model, 'scaler': scaler}, 'healthcare_risk_model.pkl')
print("Model saved as 'healthcare_risk_model.pkl'")

# Data insights
print("\n" + "="*50)
print("DATA INSIGHTS:")
print("="*50)
print(f"Total Patients: {len(patients)}")
print(f"High Risk Patients: {patients['OutcomeEncoded'].sum()}")
print(f"Recovery Rate: {(patients['OutcomeEncoded'] == 0).sum() / len(patients) * 100:.1f}%")

print("\nAnalysis completed successfully!")