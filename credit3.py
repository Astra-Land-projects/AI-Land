"""
Credit Score / Loan Default Predictor
Predicts whether a loan applicant is likely to default (fail to repay)
based on their financial profile. A classic use case in fintech.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path="credit_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} applicants")
    print(df["defaulted"].value_counts())
    return df


def prepare_data(df):
    X = df.drop("defaulted", axis=1)
    y = df["defaulted"]

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
    print("\nClassification report (0=repaid, 1=defaulted):")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions))

    return model, accuracy


def predict_applicant(model, scaler, use_scaled, feature_names):
    print("\n" + "=" * 40)
    print("Evaluate a new loan applicant")
    print("=" * 40)

    age = int(input("Age: "))
    annual_income = float(input("Annual income ($): "))
    loan_amount = float(input("Requested loan amount ($): "))
    credit_history_years = int(input("Years of credit history: "))
    existing_debts = float(input("Existing debts ($): "))
    missed_payments = int(input("Number of missed payments in the past: "))

    new_data = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_history_years": credit_history_years,
        "existing_debts": existing_debts,
        "missed_payments": missed_payments,
    }])[feature_names]

    if use_scaled:
        new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    print(f"\nResult: {'⚠️  HIGH RISK OF DEFAULT' if prediction == 1 else '✅ LOW RISK - LIKELY TO REPAY'}")
    print(f"Confidence: repay={probability[0]:.1%}, default={probability[1]:.1%}")


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
        print("\nMost important factors in default risk:")
        importances = sorted(
            zip(feature_names, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, importance in importances:
            print(f"  {name}: {importance:.1%}")
"""
Credit Score / Loan Default Predictor
Predicts whether a loan applicant is likely to default (fail to repay)
based on their financial profile. A classic use case in fintech.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path="credit_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} applicants")
    print(df["defaulted"].value_counts())
    return df


def prepare_data(df):
    X = df.drop("defaulted", axis=1)
    y = df["defaulted"]

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
    print("\nClassification report (0=repaid, 1=defaulted):")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions))

    return model, accuracy


def predict_applicant(model, scaler, use_scaled, feature_names):
    print("\n" + "=" * 40)
    print("Evaluate a new loan applicant")
    print("=" * 40)

    age = int(input("Age: "))
    annual_income = float(input("Annual income ($): "))
    loan_amount = float(input("Requested loan amount ($): "))
    credit_history_years = int(input("Years of credit history: "))
    existing_debts = float(input("Existing debts ($): "))
    missed_payments = int(input("Number of missed payments in the past: "))

    new_data = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_history_years": credit_history_years,
        "existing_debts": existing_debts,
        "missed_payments": missed_payments,
    }])[feature_names]

    if use_scaled:
        new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0]

    print(f"\nResult: {'⚠️  HIGH RISK OF DEFAULT' if prediction == 1 else '✅ LOW RISK - LIKELY TO REPAY'}")
    print(f"Confidence: repay={probability[0]:.1%}, default={probability[1]:.1%}")


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
        print("\nMost important factors in default risk:")
        importances = sorted(
            zip(feature_names, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, importance in importances:
            print(f"  {name}: {importance:.1%}")

    while True:
        choice = input("\nWould you like to evaluate a new applicant? (yes/no): ").strip().lower()
        if choice == "yes":
            predict_applicant(best_model, scaler, use_scaled, feature_names)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()

    while True:
        choice = input("\nWould you like to evaluate a new applicant? (yes/no): ").strip().lower()
        if choice == "yes":
            predict_applicant(best_model, scaler, use_scaled, feature_names)
        else:
            print("Goodbye! 👋")
            break


if __name__ == "__main__":
    main()