"""
Fraud Detection
Detects fraudulent bank transactions. Since fraud is rare (imbalanced
data), this project also demonstrates handling class imbalance -
an important real-world ML skill.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_data(path="fraud_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} transactions")
    print(df["is_fraud"].value_counts())
    print(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    return df


def prepare_data(df):
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    # class_weight="balanced" tells the model to pay more attention to
    # the rare fraud cases, instead of just predicting "not fraud" always
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report (0=normal, 1=fraud):")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions))

    auc = roc_auc_score(y_test, probabilities)
    print(f"\nROC-AUC score: {auc:.3f}  (closer to 1.0 is better)")

    return model


def check_transaction(model, feature_names):
    print("\n" + "=" * 40)
    print("Check a new transaction")
    print("=" * 40)

    amount = float(input("Transaction amount ($): "))
    hour_of_day = int(input("Hour of day (0-23): "))
    transactions_last_hour = int(input("Transactions made in the last hour: "))
    distance_from_home_km = float(input("Distance from home (km): "))
    is_new_device = int(input("Made from a new/unrecognized device? (1 for yes, 0 for no): "))

    new_data = pd.DataFrame([{
        "amount": amount,
        "hour_of_day": hour_of_day,
        "transactions_last_hour": transactions_last_hour,
        "distance_from_home_km": distance_from_home_km,
        "is_new_device": is_new_device,
    }])[feature_names]

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    print(f"\nResult: {'⚠️  LIKELY FRAUD' if prediction == 1 else '✅ LOOKS NORMAL'}")
    print(f"Fraud probability: {probability:.1%}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    feature_names = list(X_train.columns)

    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nMost important signals for detecting fraud:")
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, importance in importances:
        print(f"  {name}: {importance:.1%}")

    while True:
        choice = input("\nWould you like to check a new transaction? (yes/no): ").strip().lower()
        if choice == "yes":
            check_transaction(model, feature_names)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()