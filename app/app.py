import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import base64
import os

def generate_report(user_input, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt="User Input Data:", ln=True)
    for col in user_input.columns:
        pdf.cell(200, 10, txt=f"{col}: {user_input[col].values[0]}", ln=True)

    pdf.ln(10)
    result = "Likely to have Diabetes" if prediction == 1 else "Not likely to have Diabetes"
    pdf.cell(200, 10, txt=f"Prediction Result: {result}", ln=True)

    pdf_path = "diabetes_report.pdf"
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    os.remove(pdf_path)

    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="diabetes_report.pdf">📄 Download Report</a>'
    return href

model = pickle.load(open("model/diabetes_model.pkl", "rb"))

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🩺 Diabetes Prediction App")
st.markdown("Welcome to the Diabetes Risk Predictor. Enter your health details below to check your risk.")

df = pd.read_csv("small_diabetes_data.csv")

if st.sidebar.checkbox("Show Data Insights"):
    st.subheader("🔍 Dataset Overview")
    st.write(df.head())

    st.subheader("📊 Diabetes Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='diabetes', ax=ax)
    st.pyplot(fig)

    st.subheader("📈 BMI vs Glucose Level")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='bmi', y='blood_glucose_level', hue='diabetes', ax=ax2)
    st.pyplot(fig2)

    st.subheader("🧮 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app predicts the likelihood of diabetes based on user health details.

    ✅ Built using a Random Forest model  
    📊 Includes interactive charts and visualizations  
    ⚠️ This tool is for educational purposes only.  
    For medical advice, please consult a healthcare professional.
    """
)

with st.expander("📘 How to Use This App"):
    st.markdown("""
    1. Enter your health information in the fields provided.  
    2. Click **Predict** to check your diabetes risk.  
    3. Explore health data insights and trends by checking the box below.  
    4. This app uses a trained Random Forest model for predictions.
    """)

st.sidebar.header("📝 Patient Health Info")

gender = st.sidebar.selectbox(
    "Gender", [("Male", 1), ("Female", 0)],
    format_func=lambda x: x[0]
)[1]

age = st.sidebar.number_input("Age (in years)", 1, 120, 35)

hypertension = st.sidebar.selectbox(
    "Hypertension", [("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]

heart_disease = st.sidebar.selectbox(
    "Heart Disease", [("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]

smoking_history = st.sidebar.selectbox(
    "Smoking History", ['never', 'former', 'current', 'ever', 'not current']
)

bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 24.0)
hba1c = st.sidebar.number_input("HbA1c (%)", 3.0, 15.0, 5.8)
glucose = st.sidebar.number_input("Blood Glucose Level", 50.0, 300.0, 110.0)

if st.button("Predict"):
    smoking_encoded = [0] * 5
    smoking_options = ['current', 'ever', 'former', 'never', 'not current']
    if smoking_history in smoking_options:
        smoking_encoded[smoking_options.index(smoking_history)] = 1
        
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = 1 if hypertension == 'Yes' else 0
    heart_disease_encoded = 1 if heart_disease == 'Yes' else 0

    features = [
    gender_encoded, age, hypertension_encoded, heart_disease_encoded,
    bmi, HbA1c_level, blood_glucose_level] + smoking_encoded

    user_input = pd.DataFrame([features], columns=[
    'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'smoking_history_current',
    'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', 'smoking_history_not current'
])

    prediction = model.predict(user_input)[0]

    if prediction == 1:
        st.error("🚨 You might be diabetic. Please consult a doctor.")
    else:
        st.success("✅ You are not diabetic.")

    st.markdown(generate_report(user_input, prediction), unsafe_allow_html=True)

# Feature Importance
feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'current', 'ever',
    'former', 'never', 'not current'
]

importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

st.subheader("📊 Model Feature Importance")
st.dataframe(importance_df, use_container_width=True)
st.bar_chart(importance_df.set_index("Feature"))

st.subheader("📈 Health Insights from Dataset")

if st.checkbox("Show health insights and visualizations"):
    df = pd.read_csv('small_diabetes_data.csv')
    st.markdown("**Age Distribution**")
    st.bar_chart(df['age'].value_counts().sort_index())

    st.markdown("**BMI Distribution**")
    st.line_chart(df['bmi'].sort_values().reset_index(drop=True))

    st.markdown("**Diabetes Rate by Smoking History**")
    smoking_group = df.groupby('smoking_history')['diabetes'].mean()
    st.bar_chart(smoking_group)

    st.markdown("**Average Glucose Level by Diabetes Outcome**")
    glucose_group = df.groupby('diabetes')['blood_glucose_level'].mean()
    st.bar_chart(glucose_group)

    st.markdown("*These patterns help users see how health factors relate to diabetes.*")

st.markdown("---")
st.markdown(
    "<center><small>Made with ❤️ by Pavani Karanam | Random Forest Classifier | 2025</small></center>",
    unsafe_allow_html=True
)

