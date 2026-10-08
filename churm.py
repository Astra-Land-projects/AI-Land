"""
Customer Churn Prediction
Predicts whether a customer will churn (cancel their subscription)
based on their usage and account details. Compares Logistic Regression
and Random Forest classifiers.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CONTRACT_LABELS = {0: "monthly", 1: "yearly", 2: "two-year"}


def load_data(path="churn_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} customers")
    print(df["churn"].value_counts())
    return df


def prepare_data(df):
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.1%}")
    print("\nClassification report (0=stayed, 1=churned):")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions))

    return model, accuracy


def predict_new_customer(model, scaler, use_scaled, feature_names):
    print("\n" + "=" * 40)
    print("Predict churn for a new customer")
    print("=" * 40)

    tenure_months = int(input("How many months has this customer stayed? "))
    monthly_charge = float(input("Monthly charge ($): "))
    support_calls = int(input("Number of support calls made: "))
    print("Contract type: 0=monthly, 1=yearly, 2=two-year")
    contract_type = int(input("Contract type (0/1/2): "))
    has_online_backup = int(input("Has online backup? (1 for yes, 0 for no): "))
    satisfaction_score = int(input("Satisfaction score (1-5): "))

    new_data = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "has_online_backup": has_online_backup,
        "satisfaction_score": satisfaction_score,
    }])[feature_names]

    if use_scaled:
        new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    print(f"\nPrediction: {'WILL CHURN' if prediction == 1 else 'WILL STAY'}")
    print(f"Confidence: stay={probability[0]:.1%}, churn={probability[1]:.1%}")


def main():
    df = load_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    feature_names = list(X_train.columns)

    lr_model, lr_acc = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test,
        LogisticRegression(max_iter=1000), "Logistic Regression"
    )

    rf_model, rf_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"
    )

    print("\n" + "=" * 40)
    if rf_acc >= lr_acc:
        print("Random Forest performed better (or equal) ✅")
        best_model, use_scaled = rf_model, False
    else:
        print("Logistic Regression performed better ✅")
        best_model, use_scaled = lr_model, True

    if hasattr(rf_model, "feature_importances_"):
        print("\nFeature importance in churn (based on Random Forest):")
        importances = sorted(
            zip(feature_names, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, importance in importances:
            print(f"  {name}: {importance:.1%}")

    while True:
        choice = input("\nWould you like to predict churn for a new customer? (yes/no): ").strip().lower()
        if choice == "yes":
            predict_new_customer(best_model, scaler, use_scaled, feature_names)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()