"""
Generate a synthetic bank transaction dataset for fraud detection.
Fraudulent transactions are rare (imbalanced), just like in the real world.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_normal = 950
n_fraud = 50

# --- Normal transactions ---
normal = pd.DataFrame({
    "amount": np.random.uniform(5, 500, n_normal),
    "hour_of_day": np.random.randint(7, 23, n_normal),          # mostly daytime
    "transactions_last_hour": np.random.randint(0, 3, n_normal),  # low frequency
    "distance_from_home_km": np.random.uniform(0, 20, n_normal),  # close to home
    "is_new_device": np.random.choice([0, 1], n_normal, p=[0.9, 0.1]),
    "is_fraud": 0,
})

# --- Fraudulent transactions (different, riskier patterns) ---
fraud = pd.DataFrame({
    "amount": np.random.uniform(200, 3000, n_fraud),             # larger amounts
    "hour_of_day": np.random.randint(0, 6, n_fraud),              # late night
    "transactions_last_hour": np.random.randint(3, 10, n_fraud),  # rapid-fire
    "distance_from_home_km": np.random.uniform(50, 2000, n_fraud),  # far away
    "is_new_device": np.random.choice([0, 1], n_fraud, p=[0.2, 0.8]),
    "is_fraud": 1,
})

df = pd.concat([normal, fraud], ignore_index=True)
df["amount"] = df["amount"].round(2)
df["distance_from_home_km"] = df["distance_from_home_km"].round(1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

df.to_csv("fraud_dataset.csv", index=False)
print("Dataset created: fraud_dataset.csv")
print(df.head())
print(f"\nTotal transactions: {len(df)}")
print(df["is_fraud"].value_counts())
print(f"Fraud rate: {df['is_fraud'].mean():.1%}")