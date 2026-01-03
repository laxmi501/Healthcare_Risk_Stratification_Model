Healthcare-Risk-Stratification-Model
Healthcare Risk Stratification App

An AI-powered web application that predicts patient risk levels based on clinical data such as age, length of stay, treatment cost, and abnormal lab counts. This project uses machine learning models to assist healthcare professionals in identifying high-risk patients and improving preventive care decisions.

About the Project

This interactive Streamlit-based app enables users to input key patient parameters and instantly receive a predicted risk score (e.g., Low, Medium, or High). The system leverages a trained machine learning model that analyzes patterns from healthcare datasets to forecast patient outcomes and risk probabilities.

Features:-

Interactive UI built with Streamlit

Predicts patient risk using:

Age

Length of Stay

Treatment Cost

Abnormal Lab Count

Displays model status and prediction results in real time

Modular backend for easy retraining with new healthcare data

Tech Stack Category Tools Used:-

Frontend/UI:- Streamlit
Backend/ML:- Python, Scikit-learn, Pandas, NumPy
Visualization:- Matplotlib, Seaborn
Model Deployment:- Streamlit Cloud / Localhost
Model Workflow:-

Data Preprocessing: Cleaned and standardized patient records.

Feature Selection: Focused on key health indicators.

Model Training: Trained ML models (e.g., Logistic Regression, Random Forest).

Evaluation: Used metrics such as accuracy, precision, recall, and AUC score.

Deployment: Integrated final model into a Streamlit dashboard for user interaction.

User Interface

The interface provides:-

A sidebar with model details and status updates.

Sliders and input fields for entering patient information.

A “Predict Risk” button to generate real-time predictions.

Output Screenshort
![image alt](https://github.com/laxmi501/Healthcare_Risk_Stratification_Model/blob/308a3f498a07895edd51572294c439571a2c8878/Output%20Screenshort.png)
Project link:- 
https://healthcareriskstratificationmodel-amrf7qkjjci2fxapnfljqr.streamlit.app/
