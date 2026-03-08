from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json
import numpy as np
import pandas as pd
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')

with open('model_meta.json') as f:
    meta = json.load(f)

feature_cols = meta['feature_cols']
threshold = meta['threshold']

# Load dataset
df = pd.read_csv('FINAL_DATASET_WITH_ROLLING.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'], dayfirst=True, errors='coerce')

# Coimbatore coordinates
LAT = 11.0168
LON = 76.9558

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

def fetch_openmeteo():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=precipitation,temperature_2m,soil_moisture_0_to_7cm"
        f"&past_days=2&forecast_days=1"
        f"&timezone=Asia/Kolkata"
    )
    response = requests.get(url)
    data = response.json()
    hourly = data['hourly']
    df_live = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'rain': hourly['precipitation'],
        'temp': hourly['temperature_2m'],
        'soil': hourly['soil_moisture_0_to_7cm']
    })
    df_live = df_live.dropna()
    return df_live

def compute_features(df_live):
    latest = df_live.iloc[-1]
    rain = df_live['rain'].reset_index(drop=True)
    temp = df_live['temp'].reset_index(drop=True)
    soil = df_live['soil'].reset_index(drop=True)
    n = len(rain)

    def safe_get(series, idx):
        return float(series.iloc[idx]) if abs(idx) <= len(series) else 0.0

    features = {
        'rain_lag_1hr':   safe_get(rain, -2),
        'rain_lag_3hr':   safe_get(rain, -4),
        'rain_lag_6hr':   safe_get(rain, -7),
        'rain_lag_24hr':  safe_get(rain, -25),
        'rain_roll_3':    float(rain.iloc[-min(3,n):].mean()),
        'rain_roll_6':    float(rain.iloc[-min(6,n):].mean()),
        'rain_roll_24':   float(rain.iloc[-min(24,n):].mean()),
        'soil_root_roll_6':  float(soil.iloc[-min(6,n):].mean()),
        'soil_root_roll_24': float(soil.iloc[-min(24,n):].mean()),
        'temp_roll_6':    float(temp.iloc[-min(6,n):].mean()),
        'temp_roll_24':   float(temp.iloc[-min(24,n):].mean()),
    }
    return features, latest

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

@app.get("/live")
def get_live():
    try:
        df_live = fetch_openmeteo()
        features, latest = compute_features(df_live)
        feat_array = [[features[f] for f in feature_cols]]
        rf_pred = rf_model.predict(feat_array)[0]
        gb_pred = gb_model.predict(feat_array)[0]
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
            "time": str(latest['time']),
            "avg_rain": round(float(latest['rain']), 2),
            "avg_temp": round(float(latest['temp']), 2),
            "rpi": round(float(rpi), 4),
            "anomaly": anomaly,
            "anomaly_code": anomaly_code,
            "source": "Live - Open-Meteo"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/calendar/{year}/{month}")
def get_calendar(year: int, month: int):
    filtered = df[(df['YEAR'] == year) & (df['MO'] == month)]
    daily = filtered.groupby('DY').agg(
        avg_rain=('PRECTOTCORR', 'mean'),
        avg_temp=('T2M', 'mean'),
        avg_rpi=('RPI', 'mean')
    ).reset_index()
    result = []
    for _, row in daily.iterrows():
        rpi = row['avg_rpi']
        if rpi > 0.8365:
            anomaly = "heavy"
        elif rpi < -0.8365:
            anomaly = "low"
        else:
            anomaly = "normal"
        result.append({
            "day": int(row['DY']),
            "avg_rain": round(float(row['avg_rain']), 2),
            "avg_temp": round(float(row['avg_temp']), 2),
            "avg_rpi": round(float(rpi), 4),
            "anomaly": anomaly
        })
    return {"year": year, "month": month, "data": result}

@app.get("/date/{year}/{month}/{day}")
def get_date(year: int, month: int, day: int):
    filtered = df[(df['YEAR'] == year) & (df['MO'] == month) & (df['DY'] == day)]
    if filtered.empty:
        return {"error": "No data for this date"}
    avg = filtered.agg(
        avg_rain=('PRECTOTCORR', 'mean'),
        avg_temp=('T2M', 'mean'),
        avg_rpi=('RPI', 'mean')
    )
    rpi = float(avg['avg_rpi'])
    if rpi > 0.8365:
        anomaly = "Heavy Rain ⚠️"
    elif rpi < -0.8365:
        anomaly = "Low Rain 🔵"
    else:
        anomaly = "Normal ✅"
    return {
        "date": f"{day}/{month}/{year}",
        "avg_rain": round(float(avg['avg_rain']), 2),
        "avg_temp": round(float(avg['avg_temp']), 2),
        "avg_rpi": round(rpi, 4),
        "anomaly": anomaly
    }

@app.get("/trend/{year}/{month}")
def get_trend(year: int, month: int):
    results = []
    for m in range(month, month + 3):
        actual_month = ((m - 1) % 12) + 1
        actual_year = year + (m - 1) // 12
        filtered = df[(df['YEAR'] == actual_year) & (df['MO'] == actual_month)]
        if not filtered.empty:
            daily = filtered.groupby('DY')['RPI'].mean().reset_index()
            for _, row in daily.iterrows():
                results.append({
                    "label": f"{actual_year}-{actual_month:02d}-{int(row['DY']):02d}",
                    "rpi": round(float(row['RPI']), 4)
                })
    return {"trend": results}

@app.get("/latest")
def get_latest():
    latest = df.sort_values(['YEAR', 'MO', 'DY', 'HR']).iloc[-1]
    rpi = float(latest['RPI'])
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
        "date": f"{int(latest['DY'])}/{int(latest['MO'])}/{int(latest['YEAR'])}",
        "hour": int(latest['HR']),
        "avg_rain": round(float(latest['PRECTOTCORR']), 2),
        "avg_temp": round(float(latest['T2M']), 2),
        "rpi": round(rpi, 4),
        "anomaly": anomaly,
        "anomaly_code": anomaly_code
    }