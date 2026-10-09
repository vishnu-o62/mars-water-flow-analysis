# src/visualize.py

import os
import cv2
import matplotlib.pyplot as plt

# ===== Load original image =====
image_folder = "data/raw/hirise"

# Select first image automatically
image_name = os.listdir(image_folder)[2]

image_path = os.path.join(image_folder, image_name)

print("Original image:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("Could not load original image!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ===== Load predicted mask =====
mask_path = "outputs/ESP_028535_1990_predicted.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Could not load predicted mask!")
    exit()

# ===== Display =====
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Mars Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Water Channels")
plt.axis("off")

plt.tight_layout()
plt.show()