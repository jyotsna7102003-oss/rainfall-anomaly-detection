from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
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

rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')

with open('model_meta.json') as f:
    meta = json.load(f)

feature_cols = meta['feature_cols']
threshold = meta['threshold']

df = pd.read_csv('FINAL_DATASET_WITH_ROLLING.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'], dayfirst=True, errors='coerce')

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


def fetch_openmeteo(forecast_days=1):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=precipitation,temperature_2m,soil_moisture_0_to_7cm"
        f"&past_days=2&forecast_days={forecast_days}"
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
    df_live['rain'] = df_live['rain'].fillna(0.0)
    df_live['temp'] = df_live['temp'].ffill().fillna(25.0)
    df_live['soil'] = df_live['soil'].ffill().fillna(0.5)
    return df_live


def compute_features(df_live):
    latest = df_live.iloc[-1]
    rain = df_live['rain'].reset_index(drop=True)
    temp = df_live['temp'].reset_index(drop=True)
    soil = df_live['soil'].reset_index(drop=True)
    n = len(rain)

    def safe_get(series, idx):
        try:
            return float(series.iloc[idx])
        except:
            return 0.0

    features = {
        'rain_lag_1hr':      safe_get(rain, -2),
        'rain_lag_3hr':      safe_get(rain, -4),
        'rain_lag_6hr':      safe_get(rain, -7),
        'rain_lag_24hr':     safe_get(rain, -25),
        'rain_roll_3':       float(rain.iloc[-min(3, n):].mean()),
        'rain_roll_6':       float(rain.iloc[-min(6, n):].mean()),
        'rain_roll_24':      float(rain.iloc[-min(24, n):].mean()),
        'soil_root_roll_6':  float(soil.iloc[-min(6, n):].mean()),
        'soil_root_roll_24': float(soil.iloc[-min(24, n):].mean()),
        'temp_roll_6':       float(temp.iloc[-min(6, n):].mean()),
        'temp_roll_24':      float(temp.iloc[-min(24, n):].mean()),
    }
    return features, latest


def get_anomaly(rpi):
    if rpi > threshold:
        return "Heavy Rain ⚠️", 1
    elif rpi < -threshold:
        return "Low Rain 🔵", -1
    return "Normal ✅", 0


def predict_rpi(features):
    feat_array = [[features[f] for f in feature_cols]]
    rpi = 0.6 * rf_model.predict(feat_array)[0] + 0.4 * gb_model.predict(feat_array)[0]
    return float(rpi)


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
    anomaly, anomaly_code = get_anomaly(rpi)
    return {
        "rpi": round(float(rpi), 4),
        "anomaly": anomaly,
        "anomaly_code": anomaly_code,
        "threshold": round(threshold, 4)
    }


@app.get("/live")
def get_live():
    try:
        df_live = fetch_openmeteo(forecast_days=1)
        features, latest = compute_features(df_live)
        rpi = predict_rpi(features)
        anomaly, anomaly_code = get_anomaly(rpi)
        return {
            "time": str(latest['time']),
            "avg_rain": round(float(latest['rain']), 2),
            "avg_temp": round(float(latest['temp']), 2),
            "rpi": round(rpi, 4),
            "anomaly": anomaly,
            "anomaly_code": anomaly_code,
            "source": "Live - Open-Meteo"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def get_forecast():
    try:
        today = datetime.now()
        results = []

        df_fc = fetch_openmeteo(forecast_days=7)
        df_fc['date'] = df_fc['time'].dt.date

        for date, group in df_fc.groupby('date'):
            if pd.Timestamp(date) < pd.Timestamp(today.date()):
                continue
            group = group.reset_index(drop=True)
            n = len(group)
            rain = group['rain']
            temp = group['temp']
            soil = group['soil']

            def safe_get(s, i):
                try:
                    return float(s.iloc[i])
                except:
                    return 0.0

            features = {
                'rain_lag_1hr':      safe_get(rain, -2),
                'rain_lag_3hr':      safe_get(rain, -4),
                'rain_lag_6hr':      safe_get(rain, -7),
                'rain_lag_24hr':     safe_get(rain, -25),
                'rain_roll_3':       float(rain.iloc[-min(3, n):].mean()),
                'rain_roll_6':       float(rain.iloc[-min(6, n):].mean()),
                'rain_roll_24':      float(rain.iloc[-min(24, n):].mean()),
                'soil_root_roll_6':  float(soil.iloc[-min(6, n):].mean()),
                'soil_root_roll_24': float(soil.iloc[-min(24, n):].mean()),
                'temp_roll_6':       float(temp.iloc[-min(6, n):].mean()),
                'temp_roll_24':      float(temp.iloc[-min(24, n):].mean()),
            }

            rpi = predict_rpi(features)
            anomaly, anomaly_code = get_anomaly(rpi)

            if anomaly_code != 0:
                results.append({
                    "date": str(date),
                    "anomaly": anomaly,
                    "anomaly_code": anomaly_code,
                    "rpi": round(rpi, 4),
                    "avg_rain": round(float(rain.mean()), 2),
                    "avg_temp": round(float(temp.mean()), 2),
                    "source": "7-day forecast"
                })

        for i in range(3):
            future_date = today + timedelta(days=30 * (i + 1))
            future_month = future_date.month
            future_year = future_date.year

            monthly = df[df['MO'] == future_month]
            if monthly.empty:
                continue

            avg_rain = float(monthly['PRECTOTCORR'].mean())
            avg_temp = float(monthly['T2M'].mean())
            avg_soil = float(monthly['GWETROOT'].mean())

            features = {
                'rain_lag_1hr':      avg_rain,
                'rain_lag_3hr':      avg_rain,
                'rain_lag_6hr':      avg_rain,
                'rain_lag_24hr':     avg_rain,
                'rain_roll_3':       avg_rain,
                'rain_roll_6':       avg_rain,
                'rain_roll_24':      avg_rain,
                'soil_root_roll_6':  avg_soil,
                'soil_root_roll_24': avg_soil,
                'temp_roll_6':       avg_temp,
                'temp_roll_24':      avg_temp,
            }

            rpi = predict_rpi(features)
            anomaly, anomaly_code = get_anomaly(rpi)

            results.append({
                "date": f"{future_year}-{future_month:02d}-01",
                "anomaly": anomaly,
                "anomaly_code": anomaly_code,
                "rpi": round(rpi, 4),
                "avg_rain": round(avg_rain, 2),
                "avg_temp": round(avg_temp, 2),
                "source": f"Historical avg for month {future_month}"
            })

        return {"forecast": results, "generated_at": str(today)}

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
    anomaly, _ = get_anomaly(rpi)
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
    anomaly, anomaly_code = get_anomaly(rpi)
    return {
        "date": f"{int(latest['DY'])}/{int(latest['MO'])}/{int(latest['YEAR'])}",
        "hour": int(latest['HR']),
        "avg_rain": round(float(latest['PRECTOTCORR']), 2),
        "avg_temp": round(float(latest['T2M']), 2),
        "rpi": round(rpi, 4),
        "anomaly": anomaly,
        "anomaly_code": anomaly_code
    }