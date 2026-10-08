"""
Handwritten Digit Recognition (0-9)
Uses scikit-learn's built-in digits dataset (8x8 pixel images) -
lightweight, no download or GPU needed. Compares two simple classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    digits = load_digits()
    print(f"Loaded dataset: {len(digits.data)} images, each {digits.images[0].shape[0]}x{digits.images[0].shape[1]} pixels")
    print(f"Classes: {sorted(set(digits.target))}")
    return digits


def show_sample_images(digits, n=10, save_path="sample_digits.png"):
    fig, axes = plt.subplots(1, n, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(str(digits.target[i]))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Sample images saved to {save_path}")
    plt.close()


def prepare_data(digits):
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    return model, accuracy


def show_predictions(model, scaler, digits, X_test_raw, y_test, save_path="predictions.png"):
    X_test_scaled = scaler.transform(X_test_raw)
    predictions = model.predict(X_test_scaled)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        image = X_test_raw[i].reshape(8, 8)
        ax.imshow(image, cmap="gray")
        correct = predictions[i] == y_test[i]
        color = "green" if correct else "red"
        ax.set_title(f"Pred: {predictions[i]} (True: {y_test[i]})", color=color)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPrediction examples saved to {save_path}")
    plt.close()


def main():
    digits = load_data()
    show_sample_images(digits)

    X_train, X_test, y_train, y_test, scaler = prepare_data(digits)

    lr_model, lr_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        LogisticRegression(max_iter=1000), "Logistic Regression"
    )

    svm_model, svm_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        SVC(kernel="rbf", gamma=0.001), "Support Vector Machine (SVM)"
    )

    print("\n" + "=" * 40)
    if svm_acc >= lr_acc:
        print("SVM performed better (or equal) ✅")
        best_model = svm_model
    else:
        print("Logistic Regression performed better ✅")
        best_model = lr_model

    # Get raw (unscaled) test images for visualization
    X_raw = digits.data
    y_all = digits.target
    _, X_test_raw, _, y_test_raw = train_test_split(
        X_raw, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    show_predictions(best_model, scaler, digits, X_test_raw, y_test_raw)

    print("\nDone! Check sample_digits.png and predictions.png to see the results visually.")


if __name__ == "__main__":
    main()