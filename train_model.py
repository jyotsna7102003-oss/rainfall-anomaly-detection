import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv('FINAL_DATASET_WITH_ROLLING.csv')
# ------------------------------
# 2️⃣ Select lag + rolling features
# ------------------------------
feature_cols = [
    'rain_lag_1hr', 'rain_lag_3hr', 'rain_lag_6hr', 'rain_lag_24hr',
    'rain_roll_3', 'rain_roll_6', 'rain_roll_24',
    'soil_root_roll_6', 'soil_root_roll_24',
    'temp_roll_6', 'temp_roll_24'
]

X = df[feature_cols]
y = df['RPI']

# ------------------------------
# 3️⃣ Chronological split (better for climate)
# ------------------------------
train_df = df[df['YEAR'] <= 2022].copy()
test_df  = df[df['YEAR'] >= 2023].copy()

X_train = train_df[feature_cols]
y_train = train_df['RPI']
X_test  = test_df[feature_cols]
y_test  = test_df['RPI']

# ------------------------------
# 4️⃣ Sample 20% of training for fast tuning
# ------------------------------
X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train, train_size=0.2, random_state=42
)

# ------------------------------
# 5️⃣ Optuna objective
# ------------------------------
def objective(trial):

    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int('rf_n_estimators', 50, 200),
        max_depth=trial.suggest_int('rf_max_depth', 3, 10),
        n_jobs=-1,
        random_state=42
    )

    gb = GradientBoostingRegressor(
        n_estimators=trial.suggest_int('gb_n_estimators', 50, 200),
        learning_rate=trial.suggest_float('gb_learning_rate', 0.05, 0.2),
        max_depth=trial.suggest_int('gb_max_depth', 3, 6),
        random_state=42
    )

    rf.fit(X_sample, y_sample)
    gb.fit(X_sample, y_sample)

    y_pred = 0.6*rf.predict(X_sample) + 0.4*gb.predict(X_sample)

    return np.sqrt(mean_squared_error(y_sample, y_pred))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

best = study.best_params
print("Best hyperparameters:", best)

# ------------------------------
# 6️⃣ Train final models
# ------------------------------
rf_best = RandomForestRegressor(
    n_estimators=best['rf_n_estimators'],
    max_depth=best['rf_max_depth'],
    n_jobs=-1,
    random_state=42
)

gb_best = GradientBoostingRegressor(
    n_estimators=best['gb_n_estimators'],
    learning_rate=best['gb_learning_rate'],
    max_depth=best['gb_max_depth'],
    random_state=42
)

rf_best.fit(X_train, y_train)
gb_best.fit(X_train, y_train)

# ------------------------------
# 7️⃣ Hybrid Predictions
# ------------------------------
y_pred_train = 0.6*rf_best.predict(X_train) + 0.4*gb_best.predict(X_train)
y_pred_test  = 0.6*rf_best.predict(X_test)  + 0.4*gb_best.predict(X_test)

# ------------------------------
# 8️⃣ Performance Evaluation
# ------------------------------
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

mse, rmse, mae, r2 = evaluate(y_test, y_pred_test)

print("\nHybrid RF + GB Results")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ======================================================
# 🌧 9️⃣ Rainfall Anomaly Detection (±1.2σ)
# ======================================================

train_df["Predicted_RPI"] = y_pred_train
test_df["Predicted_RPI"]  = y_pred_test

train_df["Residual"] = train_df["RPI"] - train_df["Predicted_RPI"]
test_df["Residual"]  = test_df["RPI"] - test_df["Predicted_RPI"]

threshold = 1.2 * train_df["Residual"].std()
print(f"\nAdaptive Threshold (±1.2σ): {threshold:.4f}")

def classify(res):
    if res > threshold:
        return 1
    elif res < -threshold:
        return -1
    return 0

test_df["Anomaly"] = test_df["Residual"].apply(classify)
test_df["Positive_Anomaly"] = (test_df["Anomaly"] == 1).astype(int)
test_df["Negative_Anomaly"] = (test_df["Anomaly"] == -1).astype(int)

print("\nAnomaly Summary (Test)")
print("Normal:", (test_df["Anomaly"] == 0).sum())
print("Heavy Rain:", (test_df["Anomaly"] == 1).sum())
print("Low Rain:", (test_df["Anomaly"] == -1).sum())




joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(gb_best, 'gb_model.pkl')

meta = {"feature_cols": feature_cols, "threshold": float(threshold)}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f)

print("✅ Models saved successfully!")