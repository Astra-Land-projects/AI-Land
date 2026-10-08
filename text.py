"""
OCR - Extract Text from Images
Uses Tesseract OCR engine to read text from images (scanned documents,
photos of text, screenshots, etc.).

REQUIRES: Tesseract OCR must be installed on your system (separate from Python):
  Windows: https://github.com/UB-Mannheim/tesseract/wiki (download installer)
  macOS:   brew install tesseract
  Linux:   sudo apt install tesseract-ocr

Also requires: pip install pytesseract pillow

Usage: python ocr_reader.py your_image.jpg
"""

import sys
from PIL import Image
import pytesseract

# If Tesseract is not in your system PATH (common on Windows), uncomment
# and set the path to where you installed it:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(image_path):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error: could not open image '{image_path}': {e}")
        return None

    text = pytesseract.image_to_string(image)
    return text


def extract_text_with_boxes(image_path, output_path="ocr_boxes.jpg"):
    """Draws bounding boxes around each detected word and saves the result."""
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    if image is None:
        return

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        if int(data["conf"][i]) > 30 and data["text"][i].strip():
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Image with word boxes saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_reader.py <path_to_image>")
        print("Example: python ocr_reader.py scanned_document.jpg")
        return

    image_path = sys.argv[1]
    text = extract_text(image_path)

    if text is None:
        return

    print("=" * 50)
    print("EXTRACTED TEXT")
    print("=" * 50)
    print(text if text.strip() else "(No text detected)")

    if text.strip():
        try:
            extract_text_with_boxes(image_path)
        except Exception as e:
            print(f"\n(Skipped drawing boxes: {e})")


if __name__ == "__main__":
    main()