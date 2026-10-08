"""
Face Detection with OpenCV
Detects faces in an image using a pre-trained Haar Cascade classifier
(built into OpenCV, no training needed, very lightweight).

Usage: python face_detector.py your_photo.jpg
"""

import sys
import cv2


def detect_faces(image_path, output_path="detected_faces.jpg"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image at '{image_path}'")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's built-in pre-trained face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print(f"Found {len(faces)} face(s) in the image.")

    # Draw rectangles around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <path_to_image>")
        print("Example: python face_detector.py my_photo.jpg")
        return

    image_path = sys.argv[1]
    detect_faces(image_path)


if __name__ == "__main__":
    main()