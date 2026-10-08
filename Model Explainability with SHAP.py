"""
Model Explainability with SHAP
Most projects stop at "here's the model's accuracy." This one answers a
much more important question that companies actually care about:
"WHY did the model make THIS specific prediction for THIS specific person?"

Uses SHAP (SHapley Additive exPlanations) - the industry-standard technique
for explaining individual predictions from any ML model. This matters a lot
in regulated industries (finance, healthcare, hiring) where "the model said so"
is not a legally or ethically acceptable answer.

Requires: pip install shap scikit-learn pandas matplotlib
Requires: credit_dataset.csv (from the earlier Credit Score project) in the
          same folder. If missing, run generate_credit_dataset.py first.
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(path="credit_dataset.csv"):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} loan applicants")
    return df


def train_model(df):
    X = df.drop("defaulted", axis=1)
    y = df["defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Model trained. Test accuracy: {model.score(X_test, y_test):.1%}")
    return model, X_train, X_test, y_test


def explain_global(model, X_train, save_path="shap_summary.png"):
    """Shows which features matter most ACROSS ALL predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # For binary classification, shap_values is a list [class_0, class_1] - use class 1 (default)
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(values_to_plot, X_train, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nGlobal explanation saved to: {save_path}")
    print("(Shows which features matter most across ALL applicants)")
    plt.close()

    return explainer


def explain_single_prediction(model, explainer, X_test, y_test, index=0, save_path="shap_single.png"):
    """Shows WHY the model made ONE specific prediction."""
    applicant = X_test.iloc[[index]]
    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0][1]

    print(f"\n{'='*50}")
    print(f"Explaining prediction for applicant #{index}")
    print(f"{'='*50}")
    print(applicant.to_string(index=False))
    print(f"\nPrediction: {'WILL DEFAULT' if prediction == 1 else 'WILL REPAY'}")
    print(f"Default probability: {probability:.1%}")
    print(f"Actual outcome (from data): {'defaulted' if y_test.iloc[index] == 1 else 'repaid'}")

    shap_values = explainer.shap_values(applicant)
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

    print("\nFeature contributions to THIS prediction:")
    for feature, value, contribution in zip(applicant.columns, applicant.values[0], values_to_plot[0]):
        direction = "increases" if contribution > 0 else "decreases"
        print(f"  {feature} = {value}: {direction} default risk by {abs(contribution):.3f}")

    plt.figure()
    shap.force_plot(
        expected_value, values_to_plot[0], applicant.iloc[0],
        matplotlib=True, show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nVisual explanation saved to: {save_path}")
    plt.close()


def main():
    df = load_data()
    model, X_train, X_test, y_test = train_model(df)

    explainer = explain_global(model, X_train)

    # Explain a couple of individual predictions
    explain_single_prediction(model, explainer, X_test, y_test, index=0, save_path="shap_applicant_0.png")
    explain_single_prediction(model, explainer, X_test, y_test, index=1, save_path="shap_applicant_1.png")


if __name__ == "__main__":
    main()