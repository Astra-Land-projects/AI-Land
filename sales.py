"""
Generate a synthetic daily sales dataset with trend + weekly seasonality,
for time series forecasting practice.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_days = 365
dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

day_index = np.arange(n_days)
trend = 100 + 0.3 * day_index                                    # slow upward trend
weekly_pattern = 20 * np.sin(2 * np.pi * day_index / 7)           # weekly seasonality (weekend boost)
yearly_pattern = 15 * np.sin(2 * np.pi * day_index / 365)         # mild yearly seasonality
noise = np.random.normal(0, 8, n_days)

sales = trend + weekly_pattern + yearly_pattern + noise
sales = np.clip(sales, 10, None).round(1)

df = pd.DataFrame({"date": dates, "sales": sales})
df.to_csv("daily_sales.csv", index=False)
print("Dataset created: daily_sales.csv")
print(df.head())
print(f"\nTotal days: {len(df)}")