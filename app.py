from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder=os.path.abspath('templates'))
CORS(app)

# Load model and scaler
print("Loading ML model and scaler...")
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("✓ Model loaded")

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    print("✓ Scaler loaded")

# Routes
@app.route('/', methods=['GET'])
def home():
    """Serve the HTML interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'Diabetes Prediction API'}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make diabetes predictions"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
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
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1]),
            'result': 'Diabetes' if prediction == 1 else 'No Diabetes'
        }), 200
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
