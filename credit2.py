"""
Generate a synthetic dataset for credit risk prediction:
whether a loan applicant is likely to default (fail to repay) or not.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 700

age = np.random.randint(18, 70, n_samples)
annual_income = np.random.uniform(15000, 150000, n_samples)
loan_amount = np.random.uniform(1000, 50000, n_samples)
credit_history_years = np.random.randint(0, 30, n_samples)
existing_debts = np.random.uniform(0, 40000, n_samples)
missed_payments = np.random.randint(0, 6, n_samples)

# Debt-to-income ratio is a strong real-world predictor of default risk
debt_to_income = (existing_debts + loan_amount) / annual_income

default_score = (
    2.0 * debt_to_income
    + 0.5 * missed_payments
    - 0.03 * credit_history_years
    - 0.00002 * annual_income
    + np.random.normal(0, 0.8, n_samples)
)

default_probability = 1 / (1 + np.exp(-(default_score - 1.5)))
defaulted = (default_probability > 0.5).astype(int)

df = pd.DataFrame({
    "age": age,
    "annual_income": annual_income.round(0),
    "loan_amount": loan_amount.round(0),
    "credit_history_years": credit_history_years,
    "existing_debts": existing_debts.round(0),
    "missed_payments": missed_payments,
    "defaulted": defaulted,  # 1 = did not repay the loan, 0 = repaid successfully
})

df.to_csv("credit_dataset.csv", index=False)
print("Dataset created: credit_dataset.csv")
print(df.head())
print("\nDefault distribution:")
print(df["defaulted"].value_counts())