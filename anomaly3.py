"""
Generate synthetic electricity usage data (readings every hour for 30 days)
with a few unusual/anomalous spikes injected - simulating things like
equipment malfunction, meter tampering, or unusual consumption spikes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

hours = 24 * 30  # 30 days of hourly readings
timestamps = pd.date_range("2026-01-01", periods=hours, freq="h")

# Normal daily usage pattern: low at night, peaks in morning and evening
hour_of_day = np.array(timestamps.hour)
base_pattern = 2 + 3 * np.clip(np.sin((hour_of_day - 6) * np.pi / 12), 0, None)
noise = np.random.normal(0, 0.3, hours)
usage_kwh = np.clip(base_pattern + noise, 0.2, None)

df = pd.DataFrame({
    "timestamp": timestamps,
    "usage_kwh": usage_kwh.round(2),
})

# Inject a handful of anomalies (unusually high or low readings)
anomaly_indices = np.random.choice(hours, size=15, replace=False)
for idx in anomaly_indices:
    if np.random.rand() > 0.3:
        df.loc[idx, "usage_kwh"] = round(np.random.uniform(15, 25), 2)  # spike
    else:
        df.loc[idx, "usage_kwh"] = 0.0  # sudden drop to zero (possible outage/fault)

df.to_csv("electricity_usage.csv", index=False)
print("Dataset created: electricity_usage.csv")
print(df.head())
print(f"\nTotal readings: {len(df)}")
print(f"Injected anomalies: {len(anomaly_indices)}")