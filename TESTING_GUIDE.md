# Testing Your Diabetes ML App Locally

## Step 1: Stop any running Flask instances
Press `Ctrl+C` in the terminal where the app is running.

## Step 2: Start the Flask App
Open PowerShell/Terminal and run:
```bash
cd D:\visual_studio\Diabetes_ml_deployement
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

## Step 3: Access the Web Interface
Open your browser and go to:
```
http://localhost:5000
```
or
```
http://127.0.0.1:5000
```

You'll see a beautiful UI where you can:
- Fill in health data manually
- Use "Test Positive Case" or "Test Negative Case" buttons
- Click "Predict" to get results
- View prediction probability bars

## Step 4: Test with API (Alternative Method)
If you want to test with Python requests instead:

1. Open another terminal/PowerShell
2. Run:
```bash
cd D:\visual_studio\Diabetes_ml_deployement
pip install requests
python test_api.py
```

This will test 3 scenarios:
- Health check (GET /)
- Prediction for positive case
- Prediction for negative case

## Step 5: Test with cURL (Windows PowerShell)
Run this in PowerShell:

```powershell
$body = @{
    Pregnancies = 6
    Glucose = 148
    BloodPressure = 72
    SkinThickness = 35
    Insulin = 0
    BMI = 33.6
    DiabetesPedigreeFunction = 0.627
    Age = 50
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

## What You'll See

### In Browser (HTML UI):
- Beautiful purple gradient interface
- Input fields for all 8 health parameters
- Quick test buttons
- Color-coded results (red for diabetes, green for no diabetes)
- Probability bars showing confidence

### In Terminal (API Response):
```json
{
    "prediction": 1,
    "probability_no_diabetes": 0.25,
    "probability_diabetes": 0.75,
    "result": "Diabetes"
}
```

## Troubleshooting

### Error: "No module named 'flask'"
```bash
pip install flask
```

### Error: "No such file or directory: 'logistic_regression_model.pkl'"
Make sure you:
1. Saved the model in the notebook (ran the pickle save cell)
2. The .pkl files are in the same directory as app.py
3. File names are exactly: `logistic_regression_model.pkl` and `scaler.pkl`

### Error: "Port 5000 already in use"
```bash
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Next Steps: Deploy to Render
Once everything works locally:
1. Push to GitHub
2. Create Render account
3. Deploy Web Service
4. Share your public URL!

---
Happy Testing! 🎉
