import cv2
import numpy as np
from PIL import Image

def detect_bubbles(image: Image.Image):
    """
    Detects speech bubbles in a manga image using contour detection.
    Returns a list of bounding boxes for the detected bubbles.
    """
    # Convert PIL image to OpenCV format, and then to grayscale
    cv_image = np.array(image.convert('L'))

    # Invert the image (speech bubbles are white, contours are found on white objects)
    # and apply a threshold.
    _, mask = cv2.threshold(cv_image, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = image.width * image.height
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter based on area and aspect ratio to remove noise
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        if area > img_area * 0.003 and area < img_area * 0.25 and aspect_ratio < 5 and aspect_ratio > 0.2:
             # Check if the region is mostly white, which is characteristic of a speech bubble
            bubble_candidate = cv_image[y:y+h, x:x+w]
            if np.mean(bubble_candidate) > 220:
                boxes.append((x, y, x + w, y + h))

    return boxes
