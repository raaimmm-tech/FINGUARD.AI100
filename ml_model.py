from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["amount_ratio", "is_night", "device_new", "recipient_new", "tx_count_last_hour", "shared_device_count", "region_changed"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["amount_ratio"] = pd.to_numeric(df.get("amount_ratio", 1), errors="coerce").fillna(1).clip(0, 100)
    features["is_night"] = df.get("is_night", df["datetime"].dt.hour.between(0, 5)).astype(int)
    features["device_new"] = df["device_new"].astype(int)
    features["recipient_new"] = df["recipient_new"].astype(int)
    features["tx_count_last_hour"] = pd.to_numeric(df.get("tx_count_last_hour", 1), errors="coerce").fillna(1).clip(0, 100)
    features["shared_device_count"] = pd.to_numeric(df.get("shared_device_count", 1), errors="coerce").fillna(1).clip(0, 100)
    home_region = df.get("home_region", df["region"])
    features["region_changed"] = (df["region"].astype(str) != home_region.astype(str)).astype(int)
    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)


def add_ml_scores(df: pd.DataFrame, contamination: float = 0.10, random_state: int = 42) -> pd.DataFrame:
    result = df.copy()
    if len(result) < 20:
        result["ml_anomaly_score"] = 0.0
        result["ml_prediction"] = 0
        result["final_score"] = result["risk_score"].round(1)
        return result
    features = prepare_features(result)
    scaled = StandardScaler().fit_transform(features)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=150)
    predictions = model.fit_predict(scaled)
    raw_scores = -model.score_samples(scaled)
    minimum, maximum = float(raw_scores.min()), float(raw_scores.max())
    ml_scores = np.zeros(len(result)) if maximum <= minimum else (raw_scores - minimum) / (maximum - minimum) * 100
    result["ml_anomaly_score"] = np.round(ml_scores, 1)
    result["ml_prediction"] = (predictions == -1).astype(int)
    result["final_score"] = np.round(result["risk_score"] * 0.6 + result["ml_anomaly_score"] * 0.4, 1).clip(0, 100)
    return result
