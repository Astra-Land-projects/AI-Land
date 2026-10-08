"""
Time Series Forecasting - Daily Sales
Forecasts future sales using feature-engineered regression (day index,
day of week, month) - a lightweight approach that works well without
specialized time series libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def load_data(path="daily_sales.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded dataset: {len(df)} days of sales data")
    return df


def add_features(df):
    df = df.copy()
    df["day_index"] = np.arange(len(df))
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def train_model(df):
    # Train on the first 80% of days, test on the most recent 20%
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    feature_cols = ["day_index", "day_of_week", "month"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train_df[feature_cols], train_df["sales"])

    test_predictions = model.predict(test_df[feature_cols])
    mae = mean_absolute_error(test_df["sales"], test_predictions)
    print(f"\nTest set Mean Absolute Error: {mae:.1f} units")

    return model, feature_cols, train_df, test_df, test_predictions


def forecast_future(model, feature_cols, df, days_ahead=30):
    last_index = df["day_index"].max()
    last_date = df["date"].max()

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead)
    future_df = pd.DataFrame({
        "date": future_dates,
        "day_index": np.arange(last_index + 1, last_index + 1 + days_ahead),
        "day_of_week": future_dates.dayofweek,
        "month": future_dates.month,
    })

    future_df["forecast"] = model.predict(future_df[feature_cols])
    return future_df


def plot_results(df, test_df, test_predictions, future_df, save_path="sales_forecast.png"):
    plt.figure(figsize=(14, 5))

    plt.plot(df["date"], df["sales"], label="Historical sales", color="steelblue", alpha=0.7)
    plt.plot(test_df["date"], test_predictions, label="Test predictions", color="orange", linewidth=2)
    plt.plot(future_df["date"], future_df["forecast"], label="Future forecast", color="green", linewidth=2, linestyle="--")

    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Daily Sales - Historical Data and Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    df = load_data()
    df = add_features(df)

    model, feature_cols, train_df, test_df, test_predictions = train_model(df)

    future_df = forecast_future(model, feature_cols, df, days_ahead=30)

    print("\nNext 7 days forecast:")
    print(future_df[["date", "forecast"]].head(7).to_string(index=False))

    plot_results(df, test_df, test_predictions, future_df)


if __name__ == "__main__":
    main()