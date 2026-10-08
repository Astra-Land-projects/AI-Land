"""
House Price Prediction
Compares Linear Regression and Random Forest models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(path="house_prices.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} samples")
    print(df.describe())
    return df


def prepare_data(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.1f}k")
    print(f"Model accuracy (R²): {r2:.3f}  (closer to 1 is better)")

    return model, mae, r2


def predict_new_house(model, scaler, use_scaled, feature_names):
    print("\n" + "=" * 40)
    print("Predict price for a new house")
    print("=" * 40)

    area = float(input("Area (square meters): "))
    rooms = int(input("Number of bedrooms: "))
    age = int(input("Building age (years): "))
    floor = int(input("Floor number: "))
    distance_center = float(input("Distance from city center (km): "))
    has_parking = int(input("Has parking? (1 for yes, 0 for no): "))

    new_data = pd.DataFrame([{
        "area": area,
        "rooms": rooms,
        "age": age,
        "floor": floor,
        "distance_center": distance_center,
        "has_parking": has_parking,
    }])[feature_names]

    if use_scaled:
        new_data = scaler.transform(new_data)

    predicted_price = model.predict(new_data)[0]
    print(f"\n💰 Predicted price: ${predicted_price:,.1f}k")


def main():
    df = load_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    feature_names = list(X_train.columns)

    lr_model, lr_mae, lr_r2 = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test,
        LinearRegression(), "Linear Regression"
    )

    rf_model, rf_mae, rf_r2 = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"
    )

    print("\n" + "=" * 40)
    if rf_r2 > lr_r2:
        print("Random Forest performed better ✅")
        best_model, use_scaled = rf_model, False
    else:
        print("Linear Regression performed better ✅")
        best_model, use_scaled = lr_model, True

    if hasattr(rf_model, "feature_importances_"):
        print("\nFeature importance in price (based on Random Forest):")
        importances = sorted(
            zip(feature_names, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, importance in importances:
            print(f"  {name}: {importance:.1%}")

    while True:
        choice = input("\nWould you like to predict the price of a new house? (yes/no): ").strip().lower()
        if choice == "yes":
            predict_new_house(best_model, scaler, use_scaled, feature_names)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()