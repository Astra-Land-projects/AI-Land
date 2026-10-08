"""
Anomaly Detection in Electricity Usage
Uses Isolation Forest, an unsupervised algorithm that detects outliers
without needing labeled "normal" vs "anomaly" data - useful when you
don't know in advance what an anomaly looks like.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def load_data(path="electricity_usage.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded dataset: {len(df)} hourly readings")
    return df


def add_features(df):
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def detect_anomalies(df, contamination=0.02):
    """contamination = expected fraction of anomalies in the data (2% here)."""
    features = df[["usage_kwh", "hour_of_day", "day_of_week"]]

    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(features)
    # IsolationForest returns -1 for anomalies, 1 for normal points
    df["is_anomaly"] = (df["anomaly"] == -1).astype(int)

    return df, model


def plot_results(df, save_path="anomaly_plot.png"):
    plt.figure(figsize=(14, 5))
    normal = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    plt.plot(normal["timestamp"], normal["usage_kwh"], label="Normal", color="steelblue", linewidth=0.8)
    plt.scatter(anomalies["timestamp"], anomalies["usage_kwh"], color="red", label="Anomaly", zorder=5, s=40)

    plt.xlabel("Time")
    plt.ylabel("Usage (kWh)")
    plt.title("Electricity Usage - Detected Anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    df = load_data()
    df = add_features(df)
    df, model = detect_anomalies(df)

    n_anomalies = df["is_anomaly"].sum()
    print(f"\nDetected {n_anomalies} anomalies out of {len(df)} readings ({n_anomalies/len(df):.1%})")

    print("\nDetected anomalies:")
    anomalies = df[df["is_anomaly"] == 1][["timestamp", "usage_kwh"]]
    print(anomalies.to_string(index=False))

    plot_results(df)


if __name__ == "__main__":
    main()