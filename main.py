from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
import numpy as np

app = FastAPI()

# Load models
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')

with open('model_meta.json') as f:
    meta = json.load(f)

feature_cols = meta['feature_cols']
threshold = meta['threshold']

class InputData(BaseModel):
    rain_lag_1hr: float
    rain_lag_3hr: float
    rain_lag_6hr: float
    rain_lag_24hr: float
    rain_roll_3: float
    rain_roll_6: float
    rain_roll_24: float
    soil_root_roll_6: float
    soil_root_roll_24: float
    temp_roll_6: float
    temp_roll_24: float

@app.get("/")
def home():
    return {"status": "Rainfall Anomaly Detection API is running!"}

@app.post("/predict")
def predict(data: InputData):
    features = [[
        data.rain_lag_1hr, data.rain_lag_3hr, data.rain_lag_6hr, data.rain_lag_24hr,
        data.rain_roll_3, data.rain_roll_6, data.rain_roll_24,
        data.soil_root_roll_6, data.soil_root_roll_24,
        data.temp_roll_6, data.temp_roll_24
    ]]

    rf_pred = rf_model.predict(features)[0]
    gb_pred = gb_model.predict(features)[0]
    rpi = 0.6 * rf_pred + 0.4 * gb_pred

    if rpi > threshold:
        anomaly = "Heavy Rain ⚠️"
        anomaly_code = 1
    elif rpi < -threshold:
        anomaly = "Low Rain 🔵"
        anomaly_code = -1
    else:
        anomaly = "Normal ✅"
        anomaly_code = 0

    return {
        "rpi": round(float(rpi), 4),
        "anomaly": anomaly,
        "anomaly_code": anomaly_code,
        "threshold": round(threshold, 4)
    }