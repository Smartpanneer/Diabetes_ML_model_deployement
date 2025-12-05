import requests
import json

# Test the API locally
BASE_URL = 'http://localhost:5000'

# Test 1: Health check
print("=" * 50)
print("Test 1: Health Check (GET /)")
print("=" * 50)
try:
    response = requests.get(f'{BASE_URL}/')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Prediction - Person with Diabetes
print("\n" + "=" * 50)
print("Test 2: Predict Diabetes (Positive Case)")
print("=" * 50)
data_positive = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}
try:
    response = requests.post(f'{BASE_URL}/predict', json=data_positive)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Prediction - Person without Diabetes
print("\n" + "=" * 50)
print("Test 3: Predict No Diabetes (Negative Case)")
print("=" * 50)
data_negative = {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31
}
try:
    response = requests.post(f'{BASE_URL}/predict', json=data_negative)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("✓ All tests completed!")
print("=" * 50)
