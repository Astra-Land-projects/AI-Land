"""
Data Drift Detector
A model that scored 95% accuracy on launch day can quietly become
unreliable months later, because real-world data changes over time
(customer behavior shifts, market conditions change, etc.) - a problem
called "data drift." Almost no one covers this in learning projects,
but in production ML (MLOps) it's one of the most important things to
monitor.

This tool statistically compares a "reference" dataset (what the model
was trained on) against new incoming data, and flags which features have
drifted significantly using the Kolmogorov-Smirnov test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def generate_reference_and_new_data():
    """
    Simulates the original training data ("reference") and new data
    collected months later ("current") - where customer behavior has
    shifted over time (a realistic drift scenario).
    """
    np.random.seed(42)
    n = 500

    reference = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_charge": np.random.uniform(15, 120, n),
        "support_calls": np.random.randint(0, 10, n),
        "satisfaction_score": np.random.randint(1, 6, n),
    })

    # Simulate drift: newer customers pay more (price increase) and are
    # calling support more often (product quality issue) - realistic scenario
    np.random.seed(99)
    current = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),               # unchanged
        "monthly_charge": np.random.uniform(40, 160, n),              # drifted: prices went up
        "support_calls": np.random.randint(2, 15, n),                 # drifted: more complaints
        "satisfaction_score": np.random.randint(1, 6, n),            # unchanged
    })

    return reference, current


def detect_drift(reference, current, alpha=0.05):
    """
    Runs a Kolmogorov-Smirnov test on each feature to check whether its
    distribution has changed significantly between reference and current data.
    A p-value below alpha means the difference is statistically significant.
    """
    results = []

    for column in reference.columns:
        statistic, p_value = stats.ks_2samp(reference[column], current[column])
        drifted = p_value < alpha

        results.append({
            "feature": column,
            "ks_statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "drifted": drifted,
        })

    return pd.DataFrame(results)


def plot_distributions(reference, current, drift_report, save_path="drift_report.png"):
    features = reference.columns
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))

    for ax, feature in zip(axes, features):
        ax.hist(reference[feature], bins=20, alpha=0.5, label="Reference (training data)", color="steelblue")
        ax.hist(current[feature], bins=20, alpha=0.5, label="Current (new data)", color="orange")

        drifted = drift_report.loc[drift_report["feature"] == feature, "drifted"].values[0]
        title_color = "red" if drifted else "black"
        status = "DRIFT DETECTED" if drifted else "No significant drift"
        ax.set_title(f"{feature}\n{status}", color=title_color, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nDistribution comparison plot saved to: {save_path}")
    plt.close()


def main():
    print("=" * 50)
    print("Data Drift Detector")
    print("=" * 50)

    reference, current = generate_reference_and_new_data()
    print(f"\nReference dataset (original training data): {len(reference)} rows")
    print(f"Current dataset (new incoming data): {len(current)} rows")

    drift_report = detect_drift(reference, current)

    print("\nDrift analysis results:")
    print(drift_report.to_string(index=False))

    drifted_features = drift_report[drift_report["drifted"]]["feature"].tolist()

    print("\n" + "=" * 50)
    if drifted_features:
        print(f"⚠️  DRIFT DETECTED in {len(drifted_features)} feature(s): {', '.join(drifted_features)}")
        print("Recommendation: Consider retraining the model with more recent data.")
    else:
        print("✅ No significant drift detected. The model should still be reliable.")

    plot_distributions(reference, current, drift_report)


if __name__ == "__main__":
    main()