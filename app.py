from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load the saved model and scaler
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# HTML Content embedded
HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
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
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
        }
        
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
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
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 20px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        
        .result.show {
            display: block;
        }
        
        .result.positive {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        
        .result.negative {
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .result.positive .result-title {
            color: #c62828;
        }
        
        .result.negative .result-title {
            color: #2e7d32;
        }
        
        .result-content {
            font-size: 14px;
            line-height: 1.6;
        }
        
        .probability-bar {
            margin: 15px 0;
        }
        
        .bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 13px;
        }
        
        .bar {
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background-color: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .test-data {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 12px;
        }
        
        .test-data button {
            margin-top: 10px;
            padding: 10px;
            font-size: 12px;
            background-color: #9c27b0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Diabetes Prediction Model</h1>
        <p class="subtitle">Enter your health data to predict diabetes risk</p>
        
        <div class="test-data">
            <strong>Quick Test:</strong>
            <button onclick="fillTestData('positive')">Test Positive Case</button>
            <button onclick="fillTestData('negative')">Test Negative Case</button>
        </div>
        
        <form id="predictionForm">
            <div class="form-row">
                <div class="form-group">
                    <label for="pregnancies">Pregnancies</label>
                    <input type="number" id="pregnancies" name="pregnancies" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="glucose">Glucose (mg/dL)</label>
                    <input type="number" id="glucose" name="glucose" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="bloodpressure">Blood Pressure (mmHg)</label>
                    <input type="number" id="bloodpressure" name="bloodpressure" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="skinthickness">Skin Thickness (mm)</label>
                    <input type="number" id="skinthickness" name="skinthickness" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="insulin">Insulin (mu U/ml)</label>
                    <input type="number" id="insulin" name="insulin" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="bmi">BMI (kg/m²)</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" required>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="diabetespedigreefunction">Diabetes Pedigree Function</label>
                    <input type="number" id="diabetespedigreefunction" name="diabetespedigreefunction" step="0.001" required>
                </div>
                <div class="form-group">
                    <label for="age">Age (years)</label>
                    <input type="number" id="age" name="age" step="1" required>
                </div>
            </div>
            
            <button type="submit">🔮 Predict</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px; color: #666;">Making prediction...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div class="result-content" id="resultContent"></div>
        </div>
    </div>

    <script>
        // Fill test data
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
            } else if (type === 'negative') {
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

        // Form submission
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

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').classList.remove('show');
            document.getElementById('result').classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.error || 'Prediction failed');
                }

                // Display result
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
                        <div class="bar-label">
                            <span>No Diabetes</span>
                            <span>${probNo}%</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill" style="width: ${probNo}%"></div>
                        </div>
                    </div>
                    <div class="probability-bar">
                        <div class="bar-label">
                            <span>Diabetes</span>
                            <span>${probYes}%</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill" style="width: ${probYes}%"></div>
                        </div>
                    </div>
                `;
            } catch (error) {
                document.getElementById('error').textContent = `Error: ${error.message}`;
                document.getElementById('error').classList.add('show');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        // Set default values
        window.addEventListener('load', () => {
            fillTestData('negative');
        });
    </script>
</body>
</html>
'''

# Home route - Serve HTML interface
@app.route('/', methods=['GET'])
def home():
    return HTML_CONTENT

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
    app.run(debug=True)