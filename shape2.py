"""
Real CNN (Convolutional Neural Network) - Shape Classifier
Classifies images of circles, squares, and triangles using an actual
CNN architecture (convolutional + pooling layers) built with TensorFlow/Keras.
Small images (32x32) and a shallow network keep this fast to train on a CPU.

Requires: pip install tensorflow numpy pillow matplotlib
Run generate_shapes_dataset.py first to create the training data.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def load_data():
    images = np.load("shapes_dataset/images.npy")
    labels = np.load("shapes_dataset/labels.npy")
    with open("shapes_dataset/class_names.txt") as f:
        class_names = f.read().splitlines()

    # Normalize pixel values to 0-1 and add the "channel" dimension CNNs expect
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)  # shape becomes (N, 32, 32, 1)

    print(f"Loaded {len(images)} images, shape: {images.shape}")
    print(f"Classes: {class_names}")

    return images, labels, class_names


def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First convolutional block: learns simple features like edges
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block: learns more complex shapes
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Flatten and classify
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train loss")
    ax2.plot(history.history["val_loss"], label="Validation loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_predictions(model, X_test, y_test, class_names, save_path="cnn_predictions.png"):
    predictions = model.predict(X_test[:10])
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        correct = predicted_labels[i] == y_test[i]
        color = "green" if correct else "red"
        ax.set_title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[y_test[i]]}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction examples saved to {save_path}")
    plt.close()


def main():
    images, labels, class_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_cnn(input_shape=images.shape[1:], num_classes=len(class_names))
    model.summary()

    print("\nTraining the CNN...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.15,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.1%}")

    plot_training_history(history)
    plot_predictions(model, X_test, y_test, class_names)

    model.save("shape_classifier.keras")
    print("\nModel saved to shape_classifier.keras")


if __name__ == "__main__":
    main()