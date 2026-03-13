from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import joblib, json
import numpy as np
import pandas as pd

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


def get_features_from_row(row):
    """Build feature dict from a dataset row using rolling columns"""
    features = {}
    for f in feature_cols:
        if f in row.index:
            features[f] = float(row[f]) if pd.notna(row[f]) else 0.0
        else:
            features[f] = 0.0
    return features


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
        # Use latest row from dataset — no external API needed
        latest = df.sort_values(['YEAR', 'MO', 'DY', 'HR']).iloc[-1]
        features = get_features_from_row(latest)
        rpi = predict_rpi(features)
        anomaly, anomaly_code = get_anomaly(rpi)
        return {
            "time": f"{int(latest['YEAR'])}-{int(latest['MO']):02d}-{int(latest['DY']):02d} {int(latest['HR']):02d}:00",
            "avg_rain": round(float(latest['PRECTOTCORR']), 2),
            "avg_temp": round(float(latest['T2M']), 2),
            "rpi": round(rpi, 4),
            "anomaly": anomaly,
            "anomaly_code": anomaly_code,
            "source": "Dataset - Latest Reading"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def get_forecast():
    try:
        today = datetime.now()
        results = []

        # Use last 7 days from dataset
        sorted_df = df.sort_values(['YEAR', 'MO', 'DY', 'HR'])
        last_7_days = sorted_df.groupby(['YEAR', 'MO', 'DY']).last().tail(7).reset_index()

        for _, row in last_7_days.iterrows():
            features = get_features_from_row(row)
            rpi = predict_rpi(features)
            anomaly, anomaly_code = get_anomaly(rpi)
            if anomaly_code != 0:
                results.append({
                    "date": f"{int(row['YEAR'])}-{int(row['MO']):02d}-{int(row['DY']):02d}",
                    "anomaly": anomaly,
                    "anomaly_code": anomaly_code,
                    "rpi": round(rpi, 4),
                    "avg_rain": round(float(row['PRECTOTCORR']), 2),
                    "avg_temp": round(float(row['T2M']), 2),
                    "source": "Dataset recent"
                })

        # Next 3 months forecast using historical averages
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


@app.get("/forecast2026")
def get_forecast_2026():
    try:
        results = []
        for month in range(1, 13):
            monthly = df[df['MO'] == month]
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
                "month": month,
                "anomaly": anomaly,
                "anomaly_code": anomaly_code,
                "rpi": round(float(rpi), 4),
                "avg_rain": round(avg_rain, 2),
                "avg_temp": round(avg_temp, 2),
            })
        return {"year": 2026, "predictions": results}
    except Exception as e:
        return {"error": str(e)}


@app.get("/calendar/{year}/{month}")
def get_calendar(year: int, month: int):
    try:
        filtered = df[(df['YEAR'] == year) & (df['MO'] == month)]
        if filtered.empty:
            return {"year": year, "month": month, "data": []}
        daily = filtered.groupby('DY').agg(
            avg_rain=('PRECTOTCORR', 'mean'),
            avg_temp=('T2M', 'mean'),
            avg_rpi=('RPI', 'mean')
        ).reset_index()
        result = []
        for _, row in daily.iterrows():
            rpi = row['avg_rpi']
            if rpi > threshold:
                anomaly = "heavy"
            elif rpi < -threshold:
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
    except Exception as e:
        return {"error": str(e)}


@app.get("/date/{year}/{month}/{day}")
def get_date(year: int, month: int, day: int):
    try:
        filtered = df[
            (df['YEAR'] == year) &
            (df['MO'] == month) &
            (df['DY'] == day)
        ]
        if filtered.empty:
            return {"error": "No data for this date"}
        avg_rain = float(filtered['PRECTOTCORR'].mean())
        avg_temp = float(filtered['T2M'].mean())
        avg_rpi = float(filtered['RPI'].mean())
        anomaly, anomaly_code = get_anomaly(avg_rpi)
        return {
            "date": f"{day}/{month}/{year}",
            "avg_rain": round(avg_rain, 2),
            "avg_temp": round(avg_temp, 2),
            "avg_rpi": round(avg_rpi, 4),
            "anomaly": anomaly,
            "anomaly_code": anomaly_code
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/trend/{year}/{month}")
def get_trend(year: int, month: int):
    try:
        results = []
        for m in range(month, month + 3):
            actual_month = ((m - 1) % 12) + 1
            actual_year = year + (m - 1) // 12
            filtered = df[
                (df['YEAR'] == actual_year) &
                (df['MO'] == actual_month)
            ]
            if not filtered.empty:
                daily = filtered.groupby('DY').agg(
                    avg_rain=('PRECTOTCORR', 'mean'),
                    avg_rpi=('RPI', 'mean')
                ).reset_index()
                for _, row in daily.iterrows():
                    results.append({
                        "label": f"{actual_year}-{actual_month:02d}-{int(row['DY']):02d}",
                        "rpi": round(float(row['avg_rpi']), 4),
                        "avg_rain": round(float(row['avg_rain']), 2)
                    })
        return {"trend": results}
    except Exception as e:
        return {"error": str(e)}


@app.get("/latest")
def get_latest():
    try:
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
    except Exception as e:
        return {"error": str(e)}