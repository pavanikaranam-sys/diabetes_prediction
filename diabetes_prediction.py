import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up Streamlit page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Title
st.title("🩺 Diabetes Prediction Web App")

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("/home/user/Desktop/p/diabetes/diabetes_pred.csv")

    # Label Encoding
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])

    # One-hot encoding
    d1 = pd.get_dummies(data['smoking_history'], dtype=int)
    d1 = d1.drop('No Info', axis=1)
    data = pd.concat([data, d1], axis=1)
    data = data.drop('smoking_history', axis=1)

    # Remove outliers
    def remove_outliers(df, column):
        Q1, Q3 = df[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lf = Q1 - 1.5 * IQR
        uf = Q3 + 1.5 * IQR
        return df[(df[column] >= lf) & (df[column] <= uf)]

    data = remove_outliers(data, 'blood_glucose_level')
    data = remove_outliers(data, 'bmi')

    return data

data = load_and_prepare_data()

# Train model
x = data.drop('diabetes', axis=1)
y = data['diabetes']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(x_train, y_train)

# Evaluation (optional view)
with st.expander("📊 Model Evaluation Metrics"):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", acc)
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# Input form
st.header("🔍 Predict Diabetes")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 10.0, 5.5)
glucose = st.number_input("Blood Glucose Level", 80, 300, 140)
smoke = st.selectbox("Smoking History", ['never', 'current', 'former', 'ever', 'not current'])

# Prepare input
input_data = {
    'gender': 0 if gender == "Female" else 1,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'bmi': bmi,
    'HbA1c_level': hba1c,
    'blood_glucose_level': glucose,
    'current': 1 if smoke == 'current' else 0,
    'ever': 1 if smoke == 'ever' else 0,
    'former': 1 if smoke == 'former' else 0,
    'never': 1 if smoke == 'never' else 0,
    'not current': 1 if smoke == 'not current' else 0,
}

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ The person is likely to have diabetes once consult a doctor.")
    else:
        st.success("✅ The person is unlikely to have diabetes.")

