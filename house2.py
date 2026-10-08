"""
Generate a synthetic but realistic house price dataset.
(If you have a real dataset, you can swap this out - e.g. California Housing dataset.)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 500

area = np.random.randint(50, 350, n_samples)              # square meters
rooms = np.random.randint(1, 6, n_samples)                 # bedrooms
age = np.random.randint(0, 40, n_samples)                  # building age (years)
floor = np.random.randint(0, 15, n_samples)                 # floor number
distance_center = np.random.uniform(0.5, 30, n_samples)     # distance from city center (km)
has_parking = np.random.randint(0, 2, n_samples)            # has parking? (0/1)

# Assumed price formula (in thousands of dollars) + some random noise
price = (
    area * 4.5
    + rooms * 12
    - age * 0.8
    + floor * 1.5
    - distance_center * 2
    + has_parking * 9
    + np.random.normal(0, 15, n_samples)
)
price = np.clip(price, 30, None)  # minimum reasonable price

df = pd.DataFrame({
    "area": area,
    "rooms": rooms,
    "age": age,
    "floor": floor,
    "distance_center": distance_center.round(2),
    "has_parking": has_parking,
    "price": price.round(1),
})

df.to_csv("house_prices.csv", index=False)
print("Dataset created: house_prices.csv")
print(df.head())