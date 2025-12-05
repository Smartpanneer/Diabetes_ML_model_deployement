from flask import Flask, request, jsonify
import pickle
import numpy as np
import os




app = Flask(__name__)

# Load the saved model and scaler
with open('Logistic_Regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Diabetes Prediction API'})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Extract features (8 features for diabetes dataset)
        features = np.array([[
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1]),
            'result': 'Diabetes' if prediction == 1 else 'No Diabetes'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400







if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

   

