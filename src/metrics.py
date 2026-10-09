# src/metrics.py

import os
import cv2
import torch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from unet_model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = build_model()

model.load_state_dict(
    torch.load(
        "models_saved/unet_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# Select first image
image_folder = "data/raw/hirise"

# image_name = os.listdir(image_folder)[1]
image_name = "PSP_007925_1990.jpg"

image_path = os.path.join(
    image_folder,
    image_name
)

# Corresponding mask
mask_name = image_name.replace(".jpg", ".png")

# mask_path = os.path.join(
#     "data/dataset/masks",
#     mask_name
# )
mask_path = os.path.join(
    "data/sam2_masks",
    mask_name
)

# Read image
image = cv2.imread(image_path)

image = cv2.resize(
    image,
    (256,256)
)

image = image / 255.0

image_tensor = torch.tensor(
    image
).permute(2,0,1)

image_tensor = image_tensor.unsqueeze(0).float().to(device)

# Predict
with torch.no_grad():

    prediction = model(image_tensor)

prediction = torch.sigmoid(prediction)

prediction = prediction.squeeze().cpu().numpy()

prediction = (prediction > 0.5).astype(np.uint8)

# Ground truth
mask = cv2.imread(
    mask_path,
    cv2.IMREAD_GRAYSCALE
)

mask = cv2.resize(
    mask,
    (256,256)
)

mask = (mask > 127).astype(np.uint8)

# Flatten arrays
pred_flat = prediction.flatten()

mask_flat = mask.flatten()

# Metrics
accuracy = accuracy_score(mask_flat, pred_flat)

precision = precision_score(
    mask_flat,
    pred_flat,
    zero_division=0
)

recall = recall_score(
    mask_flat,
    pred_flat,
    zero_division=0
)

f1 = f1_score(
    mask_flat,
    pred_flat,
    zero_division=0
)

# IoU
intersection = np.logical_and(
    mask_flat,
    pred_flat
).sum()

union = np.logical_or(
    mask_flat,
    pred_flat
).sum()

iou = intersection / union

print("\nResults")
print("-------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"IoU       : {iou:.4f}")


print("\n----------------------------------------")
print("Input Image       :", image_name)
print("Image Path        :", image_path)

print("Ground Truth Mask :", mask_name)
print("Mask Path         :", mask_path)
print("----------------------------------------")
