# app.py

import os
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

# --- Part 1: Your Functions from the Notebook ---
# (I've copied these directly from your .ipynb file)

def create_labels(df: pd.DataFrame, rain_thr_mm_h=10.0, humidity_thr_pct=80.0, pressure_drop_thr_hpa=5.0, wind_thr_kmh=50.0) -> pd.DataFrame:
    df = df.copy()
    df["rainfall_intensity"] = df["precipitation"]
    df["pressure_change_3h"] = df["pressure"].diff(3).fillna(0)
    score = (
        (df["rainfall_intensity"] > rain_thr_mm_h).astype(int) +
        (df["humidity"] > humidity_thr_pct).astype(int) +
        (df["pressure_change_3h"] < -pressure_drop_thr_hpa).astype(int) +
        (df["wind_speed"] > wind_thr_kmh).astype(int)
    )
    df["cloud_burst_risk"] = (score >= 3).astype(int)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime']) # Ensure datetime is in the correct format
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    for w in [3, 6]:
        df[f"temp_mean_{w}h"] = df["temperature"].rolling(w, min_periods=1).mean()
        df[f"humidity_mean_{w}h"] = df["humidity"].rolling(w, min_periods=1).mean()
        df[f"rain_sum_{w}h"] = df["precipitation"].rolling(w, min_periods=1).sum()
        df[f"pressure_std_{w}h"] = df["pressure"].rolling(w, min_periods=1).std().fillna(0)
    df["temp_change_1h"] = df["temperature"].diff(1).fillna(0)
    df["humidity_change_1h"] = df["humidity"].diff(1).fillna(0)
    df["wind_speed_change_1h"] = df["wind_speed"].diff(1).fillna(0)
    t = df["temperature"].clip(-30, 55)
    h = (df["humidity"].clip(0, 100) / 100.0)
    df["heat_index"] = t + 0.33 * h * 6.105 * np.exp(17.27 * t / (237.7 + t)) - 0.5 * df["wind_speed"] - 4
    df["precip_efficiency"] = df["precipitation"] / (df["humidity"] + 1e-6)
    return df

def fetch_open_meteo_hourly(lat: float, lon: float, days_back: int) -> pd.DataFrame:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m",
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]), "temperature": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []), "precipitation": hourly.get("precipitation", []),
        "pressure": hourly.get("pressure_msl", []), "wind_speed": hourly.get("wind_speed_10m", []),
        "wind_direction": hourly.get("wind_direction_10m", []),
    })
    return df

# --- Part 2: Model Training and Loading ---

MODEL_FILE = 'cloudburst_model.pkl'
FEATURES = [
    'temperature', 'humidity', 'precipitation', 'pressure', 'wind_speed',
    'wind_direction', 'hour', 'dayofyear', 'month', 'temp_mean_3h',
    'humidity_mean_3h', 'rain_sum_3h', 'pressure_std_3h', 'temp_mean_6h',
    'humidity_mean_6h', 'rain_sum_6h', 'pressure_std_6h', 'temp_change_1h',
    'humidity_change_1h', 'wind_speed_change_1h', 'heat_index',
    'precip_efficiency'
]

def train_and_save_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    
    print("🧠 Training new model...")
    # Fetch a larger dataset for better training
    df_train = fetch_open_meteo_hourly(30.3165, 78.0322, days_back=90)
    df_train = create_labels(df_train)
    df_train = engineer_features(df_train)
    
    # Handle missing values
    df_train = df_train.dropna(subset=['cloud_burst_risk'])
    X = df_train[FEATURES]
    y = df_train['cloud_burst_risk']
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_imputed, y)
    
    # Save the imputer and model together
    joblib.dump({'model': model, 'imputer': imputer}, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")
    return {'model': model, 'imputer': imputer}

# Load the model or train if it doesn't exist
if not os.path.exists(MODEL_FILE):
    print(f"⚠️ Model file {MODEL_FILE} not found. Training a new model...")
    pipeline = train_and_save_model()
else:
    print(f"🚀 Loading existing model from {MODEL_FILE}")
    pipeline = joblib.load(MODEL_FILE)

model = pipeline['model']
imputer = pipeline['imputer']


# --- Part 3: Flask Web Server ---

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Cloudburst Prediction API</h1><p>Send a POST request to /predict</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Latitude and Longitude are required.'}), 400
    
    try:
        # Fetch the last 24 hours of data to calculate rolling features
        df_live = fetch_open_meteo_hourly(lat, lon, days_back=1)
        if df_live.empty:
            return jsonify({'error': 'Could not fetch live weather data.'}), 500
        
        # Process features and get the most recent data point
        df_processed = engineer_features(df_live)
        latest_data = df_processed[FEATURES].tail(1)
        
        # Handle potential NaNs in live data using the trained imputer
        latest_data_imputed = imputer.transform(latest_data)

        # Make prediction
        prediction = model.predict(latest_data_imputed)
        probability = model.predict_proba(latest_data_imputed)
        
        is_risk = bool(prediction[0])
        risk_prob = f"{probability[0][1] * 100:.2f}%"

        return jsonify({
            'location': {'lat': lat, 'lon': lon},
            'cloudburst_risk': is_risk,
            'risk_probability': risk_prob
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    # The server will run on port 5000 by default.
    # For deployment, a Gunicorn server will be used instead of this.
    app.run(debug=True)