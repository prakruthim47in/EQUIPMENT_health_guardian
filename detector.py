"""
Anomaly detection for the sensor streams.
Uses IsolationForest on rolling-window features (mean, std) per machine.
Simple, fast to train, and easy to explain to judges — no deep learning needed.
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["vibration_mm_s", "temperature_c", "rpm"]


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values(["machine", "t"]).copy()
    for col in FEATURES:
        df[f"{col}_roll_mean"] = df.groupby("machine")[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = df.groupby("machine")[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )
    return df


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.12) -> pd.DataFrame:
    """Fit one IsolationForest per machine on its own history and flag anomalies.

    Returns the dataframe with an added 'is_anomaly' boolean column and
    'anomaly_score' column (lower = more anomalous).
    """
    df = add_rolling_features(df)
    feature_cols = FEATURES + [f"{c}_roll_mean" for c in FEATURES] + [f"{c}_roll_std" for c in FEATURES]

    out_frames = []
    for machine, g in df.groupby("machine"):
        g = g.copy()
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
        X = g[feature_cols].values
        model.fit(X)
        g["anomaly_score"] = model.decision_function(X)
        g["is_anomaly"] = model.predict(X) == -1
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def latest_status(df_with_anomalies: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row per machine, with a simple health status."""
    latest = df_with_anomalies.sort_values("t").groupby("machine").tail(1).copy()

    def status(row):
        if row["is_anomaly"]:
            return "⚠️ ALERT"
        return "✅ Healthy"

    latest["status"] = latest.apply(status, axis=1)
    return latest.reset_index(drop=True)
