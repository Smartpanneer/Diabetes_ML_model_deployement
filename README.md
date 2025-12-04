# Diabetes ML Model Deployment

This is a Flask-based web application that predicts whether a person has diabetes using a Logistic Regression model trained on the Pima Indians Diabetes Dataset.

## Files

- `app.py` - Flask application with prediction endpoint
- `dataset.ipynb` - Jupyter notebook with data exploration and model training
- `logistic_regression_model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - Feature scaler for data normalization
- `diabetes.csv` - Dataset
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app locally:
```bash
python app.py
```

## API Endpoints

### GET `/`
Health check endpoint

Response:
```json
{"message": "Diabetes Prediction API"}
```

### POST `/predict`
Make diabetes predictions

Request body:
```json
{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}
```

Response:
```json
{
    "prediction": 1,
    "probability_no_diabetes": 0.25,
    "probability_diabetes": 0.75,
    "result": "Diabetes"
}
```

## Deployment on Render

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python app.py`
6. Deploy!

## Model Accuracy

- Logistic Regression: 75.32%
- Gradient Boosting: 74.68%
- Random Forest: 74.03%

## Features Used

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
