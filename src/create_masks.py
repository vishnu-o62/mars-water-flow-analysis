# src/create_masks.py

import os
import cv2
import numpy as np

# Input and output folders
# image_folder = "/data/raw/hirise"
# mask_folder = "/data/masks"
# Current project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_folder = os.path.join(BASE_DIR, "data", "raw", "hirise")
mask_folder = os.path.join(BASE_DIR, "data", "masks")


# Create masks folder if it doesn't exist
os.makedirs(mask_folder, exist_ok=True)

# Process all images
for filename in os.listdir(image_folder):

    if filename.endswith(".jpg") or filename.endswith(".png"):

        image_path = os.path.join(image_folder, filename)

        # Read image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological closing to connect channels
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Convert to binary mask
        mask[mask > 0] = 255

        # Save mask
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(mask_folder, mask_name)

        cv2.imwrite(mask_path, mask)

        print(f"Mask created: {mask_name}")

print("All masks generated successfully.")