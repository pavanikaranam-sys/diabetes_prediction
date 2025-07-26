from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')  # Make sure this path is correct

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        # Manual one-hot encoding for smoking history
        smoking_encoded = {
            'current': 0,
            'ever': 0,
            'former': 0,
            'never': 0,
            'not current': 0
        }
        if smoking_history in smoking_encoded:
            smoking_encoded[smoking_history] = 1

        features = [
            gender,
            age,
            hypertension,
            heart_disease,
            bmi,
            hba1c_level,
            blood_glucose_level,
            smoking_encoded['current'],
            smoking_encoded['ever'],
            smoking_encoded['former'],
            smoking_encoded['never'],
            smoking_encoded['not current']
        ]

        prediction = model.predict([features])[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
   
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
