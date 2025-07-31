# 🩺️Diabetes Prediction

## 📌 Project Overview

This machine learning project aims to detect diabetes prediction . The dataset is preprocessed, visualized, and cleaned, and then used to train a classification model that predicts whether a person diabetic or not.

## ✅ Steps Performed

1. **Data Collection**  
   Loaded the dataset (`diabetes_pred.csv`) containing diabetic features and labels.

2. **Data Cleaning**  
   - Checked for null values  
   - Handled outliers using the IQR method  

3. **Data Visualization**  
   - Used box plots and histograms to explore distributions 

4. **Data Preprocessing**    
   - Split the dataset into training and testing sets

5. **Model Selection & Training**  
   - Used **Logistic Regression** with `class_weight='balanced'`  
   - Evaluated the model with classification metrics and a confusion matrix

## 📊 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Live checking

      To check your diabetes report live, here is the link : https://diabetes-prediction-app-production-d7c2.up.railway.app/
