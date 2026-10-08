"""
Generate a small synthetic image dataset of simple shapes (circle, square,
triangle) for training a real Convolutional Neural Network (CNN).
Synthetic shapes keep this lightweight - no large downloads needed,
and training finishes in under a minute on a CPU.
"""

import numpy as np
from PIL import Image, ImageDraw
import os

np.random.seed(42)

IMAGE_SIZE = 32
N_PER_CLASS = 200
OUTPUT_DIR = "shapes_dataset"


def draw_circle(draw, size):
    margin = np.random.randint(3, 8)
    draw.ellipse([margin, margin, size - margin, size - margin], fill="black")


def draw_square(draw, size):
    margin = np.random.randint(3, 8)
    draw.rectangle([margin, margin, size - margin, size - margin], fill="black")


def draw_triangle(draw, size):
    margin = np.random.randint(3, 8)
    points = [
        (size // 2, margin),
        (margin, size - margin),
        (size - margin, size - margin),
    ]
    draw.polygon(points, fill="black")


SHAPE_FUNCTIONS = {
    "circle": draw_circle,
    "square": draw_square,
    "triangle": draw_triangle,
}


def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = []
    labels = []

    for label, (shape_name, draw_fn) in enumerate(SHAPE_FUNCTIONS.items()):
        for i in range(N_PER_CLASS):
            img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color="white")
            draw = ImageDraw.Draw(img)
            draw_fn(draw, IMAGE_SIZE)

            # Add slight random noise to make it more realistic / less trivial
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 10, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

            images.append(arr)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    np.save(os.path.join(OUTPUT_DIR, "images.npy"), images)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

    class_names = list(SHAPE_FUNCTIONS.keys())
    with open(os.path.join(OUTPUT_DIR, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))

    print(f"Dataset created in '{OUTPUT_DIR}/': {len(images)} images, {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print(f"Classes: {class_names}")

    # Save a few sample images for a visual check
    from PIL import Image as PILImage
    sample_grid = PILImage.new("L", (IMAGE_SIZE * 6, IMAGE_SIZE), color="white")
    for i in range(6):
        sample_grid.paste(PILImage.fromarray(images[i]), (i * IMAGE_SIZE, 0))
    sample_grid.save(os.path.join(OUTPUT_DIR, "sample_preview.png"))
    print(f"Sample preview saved to {OUTPUT_DIR}/sample_preview.png")


if __name__ == "__main__":
    generate_dataset()