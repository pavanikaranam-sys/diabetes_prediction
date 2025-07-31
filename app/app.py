import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/diabetes_model.pkl", "rb")) 
# ✅ Set the Streamlit page configuration FIRST
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("Welcome to the Diabetes Risk Predictor. Enter your health details below to check your risk.")

# Load data
df = pd.read_csv("small_diabetes_data.csv")

# Sidebar toggle for analytics
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


# Sidebar inputs

# Sidebar info
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

# Instruction block at the top
with st.expander("📘 How to Use This App"):
    st.markdown("""
    1. Enter your health information in the fields provided.  
    2. Click **Predict** to check your diabetes risk.  
    3. Explore health data insights and trends by checking the box below.  
    4. This app uses a trained Random Forest model for predictions.
    """)

st.sidebar.header("📝 Patient Health Info")

gender = st.sidebar.selectbox(
    "Gender",
    [("Male", 1), ("Female", 0)],
    format_func=lambda x: x[0],
    help="Select the biological gender"
)[1]

age = st.sidebar.number_input(
    "Age (in years)", min_value=1, max_value=120, value=35,
    help="Enter age between 1 and 120"
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0],
    help="Do you have high blood pressure?"
)[1]

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0],
    help="Do you have any known heart disease?"
)[1]

smoking_history = st.sidebar.selectbox(
    "Smoking History",
    ['never', 'former', 'current', 'ever', 'not current'],
    help="Select your smoking history"
)

bmi = st.sidebar.number_input(
    "BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.0,
    help="Normal BMI range is 18.5–24.9"
)

hba1c = st.sidebar.number_input(
    "HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.8,
    help="Normal HbA1c is below 5.7%"
)

glucose = st.sidebar.number_input(
    "Blood Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=110.0,
    help="Normal fasting glucose is under 100 mg/dL"
)


# Predict button
if st.button("Predict"):
    # One-hot encode smoking history
    smoking_encoded = [0] * 5
    smoking_options = ['current', 'ever', 'former', 'never', 'not current']
    if smoking_history in smoking_options:
        smoking_encoded[smoking_options.index(smoking_history)] = 1

    # Create input features
    features = [
        gender,
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose
    ] + smoking_encoded

    # Make prediction
    prediction = model.predict([features])[0]

    # Display result
    if prediction == 1:
        st.error("🚨 You might be diabetic. Please consult a doctor.")
    else:
        st.success("✅ You are not diabetic.")
# Feature names
feature_names = [
    'Gender', 'Age', 'Hypertension', 'Heart Disease', 'BMI',
    'HbA1c', 'Glucose', 'Smoking: current', 'Smoking: ever',
    'Smoking: former', 'Smoking: never', 'Smoking: not current'
]

# Load model using pickle
model = pickle.load(open('model/diabetes_model.pkl', 'rb'))

# Get feature importances from Random Forest
importances = model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)

# Show top features
st.subheader("📊 Model Feature Importance")
st.dataframe(importance_df, use_container_width=True)

# Optional: Plot as a bar chart
st.bar_chart(importance_df.set_index("Feature"))

st.subheader("📈 Health Insights from Dataset")

if st.checkbox("Show health insights and visualizations"):
    # Load dataset
    df = pd.read_csv('small_diabetes_data.csv')

    # Distribution of age
    st.markdown("**Age Distribution**")
    st.bar_chart(df['age'].value_counts().sort_index())

    # BMI distribution
    st.markdown("**BMI Distribution**")
    st.line_chart(df['bmi'].sort_values().reset_index(drop=True))

    # Diabetes rate by smoking history
    st.markdown("**Diabetes Rate by Smoking History**")
    smoking_group = df.groupby('smoking_history')['diabetes'].mean()
    st.bar_chart(smoking_group)

    # Glucose vs Diabetes
    st.markdown("**Average Glucose Level by Diabetes Outcome**")
    glucose_group = df.groupby('diabetes')['blood_glucose_level'].mean()
    st.bar_chart(glucose_group)

    st.markdown("*These patterns help users see how health factors relate to diabetes.*")

st.markdown("---")
st.markdown(
    "<center><small>Made with ❤️ by Pavani Karanam | Random Forest Classifier | 2025</small></center>",
    unsafe_allow_html=True
)
