from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the saved model and scaler
print("Loading model and scaler...")
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
print("✓ Model loaded")

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
print("✓ Scaler loaded")

# HTML interface
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        h1 { color: #333; margin-bottom: 10px; text-align: center; }
        .subtitle { color: #666; text-align: center; margin-bottom: 30px; font-size: 14px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 500; font-size: 14px; }
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 14px;
        }
        input:focus { outline: none; border-color: #667eea; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover { transform: translateY(-2px); }
        .test-data { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .test-data button { margin-top: 10px; padding: 10px; font-size: 12px; background-color: #9c27b0; }
        .result { margin-top: 30px; padding: 20px; border-radius: 5px; display: none; }
        .result.show { display: block; }
        .result.positive { background-color: #ffebee; border-left: 5px solid #f44336; }
        .result.negative { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
        .result-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        .result.positive .result-title { color: #c62828; }
        .result.negative .result-title { color: #2e7d32; }
        .probability-bar { margin: 15px 0; }
        .bar-label { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 13px; }
        .bar { height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden; }
        .bar-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
        .error { background-color: #ffebee; color: #c62828; padding: 15px; border-radius: 5px; margin-top: 20px; display: none; }
        .error.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Diabetes Prediction</h1>
        <p class="subtitle">Enter health data to predict diabetes risk</p>
        
        <div class="test-data">
            <strong>Quick Test:</strong><br>
            <button onclick="fillTestData('positive')">Test Positive</button>
            <button onclick="fillTestData('negative')">Test Negative</button>
        </div>
        
        <form id="predictionForm">
            <div class="form-row">
                <div class="form-group">
                    <label>Pregnancies</label>
                    <input type="number" id="pregnancies" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Glucose (mg/dL)</label>
                    <input type="number" id="glucose" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>Blood Pressure (mmHg)</label>
                    <input type="number" id="bloodpressure" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Skin Thickness (mm)</label>
                    <input type="number" id="skinthickness" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>Insulin (mu U/ml)</label>
                    <input type="number" id="insulin" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>BMI (kg/m²)</label>
                    <input type="number" id="bmi" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>Diabetes Pedigree</label>
                    <input type="number" id="diabetespedigreefunction" step="0.001" required>
                </div>
                <div class="form-group">
                    <label>Age (years)</label>
                    <input type="number" id="age" step="1" required>
                </div>
            </div>
            
            <button type="submit">🔮 Predict</button>
        </form>
        
        <div class="error" id="error"></div>
        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div class="result-content" id="resultContent"></div>
        </div>
    </div>

    <script>
        function fillTestData(type) {
            if (type === 'positive') {
                document.getElementById('pregnancies').value = 6;
                document.getElementById('glucose').value = 148;
                document.getElementById('bloodpressure').value = 72;
                document.getElementById('skinthickness').value = 35;
                document.getElementById('insulin').value = 0;
                document.getElementById('bmi').value = 33.6;
                document.getElementById('diabetespedigreefunction').value = 0.627;
                document.getElementById('age').value = 50;
            } else {
                document.getElementById('pregnancies').value = 1;
                document.getElementById('glucose').value = 85;
                document.getElementById('bloodpressure').value = 66;
                document.getElementById('skinthickness').value = 29;
                document.getElementById('insulin').value = 0;
                document.getElementById('bmi').value = 26.6;
                document.getElementById('diabetespedigreefunction').value = 0.351;
                document.getElementById('age').value = 31;
            }
        }

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const data = {
                Pregnancies: parseFloat(document.getElementById('pregnancies').value),
                Glucose: parseFloat(document.getElementById('glucose').value),
                BloodPressure: parseFloat(document.getElementById('bloodpressure').value),
                SkinThickness: parseFloat(document.getElementById('skinthickness').value),
                Insulin: parseFloat(document.getElementById('insulin').value),
                BMI: parseFloat(document.getElementById('bmi').value),
                DiabetesPedigreeFunction: parseFloat(document.getElementById('diabetespedigreefunction').value),
                Age: parseFloat(document.getElementById('age').value)
            };

            document.getElementById('error').classList.remove('show');
            document.getElementById('result').classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.error || 'Prediction failed');
                }

                const resultDiv = document.getElementById('result');
                const resultClass = result.prediction === 1 ? 'positive' : 'negative';
                const resultTitle = result.result === 'Diabetes' ? '⚠️ Diabetes Detected' : '✅ No Diabetes';
                
                resultDiv.className = `result show ${resultClass}`;
                document.getElementById('resultTitle').textContent = resultTitle;
                
                const probNo = (result.probability_no_diabetes * 100).toFixed(2);
                const probYes = (result.probability_diabetes * 100).toFixed(2);
                
                document.getElementById('resultContent').innerHTML = `
                    <p><strong>Prediction:</strong> ${result.result}</p>
                    <div class="probability-bar">
                        <div class="bar-label"><span>No Diabetes</span><span>${probNo}%</span></div>
                        <div class="bar"><div class="bar-fill" style="width: ${probNo}%"></div></div>
                    </div>
                    <div class="probability-bar">
                        <div class="bar-label"><span>Diabetes</span><span>${probYes}%</span></div>
                        <div class="bar"><div class="bar-fill" style="width: ${probYes}%"></div></div>
                    </div>
                `;
            } catch (error) {
                document.getElementById('error').textContent = `Error: ${error.message}`;
                document.getElementById('error').classList.add('show');
            }
        });

        window.addEventListener('load', () => fillTestData('negative'));
    </script>
</body>
</html>'''

@app.route('/', methods=['GET'])
def home():
    return HTML_PAGE

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = np.array([[
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0)),
            float(data.get('Age', 0))
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1]),
            'result': 'Diabetes' if prediction == 1 else 'No Diabetes'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
