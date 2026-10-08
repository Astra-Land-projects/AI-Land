"""
Generate a synthetic but realistic customer churn dataset.
Churn = whether a customer stopped using the service (canceled subscription).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 600

tenure_months = np.random.randint(1, 72, n_samples)          # how long they've been a customer
monthly_charge = np.random.uniform(15, 120, n_samples)        # monthly subscription cost
support_calls = np.random.randint(0, 10, n_samples)            # number of support calls made
contract_type = np.random.choice([0, 1, 2], n_samples)         # 0=monthly, 1=yearly, 2=two-year
has_online_backup = np.random.randint(0, 2, n_samples)
satisfaction_score = np.random.randint(1, 6, n_samples)        # 1 (low) to 5 (high)

# Churn probability logic: short tenure, high charges, many support calls,
# monthly contracts, and low satisfaction increase churn likelihood
churn_score = (
    -0.05 * tenure_months
    + 0.03 * monthly_charge
    + 0.4 * support_calls
    - 0.8 * contract_type
    - 0.3 * has_online_backup
    - 0.6 * satisfaction_score
    + np.random.normal(0, 1.5, n_samples)
)

churn_probability = 1 / (1 + np.exp(-churn_score))  # sigmoid to get 0-1 range
churn = (churn_probability > 0.5).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charge": monthly_charge.round(2),
    "support_calls": support_calls,
    "contract_type": contract_type,     # 0=monthly, 1=yearly, 2=two-year
    "has_online_backup": has_online_backup,
    "satisfaction_score": satisfaction_score,
    "churn": churn,                     # 0=stayed, 1=churned
})

df.to_csv("churn_dataset.csv", index=False)
print("Dataset created: churn_dataset.csv")
print(df.head())
print("\nChurn distribution:")
print(df["churn"].value_counts())