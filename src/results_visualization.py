# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ---------------------------------------------------
# # Image name
# #  ---------------------------------------------------
# # image_name = "t7"
#
# import sys
#
#
# # Image name
#
#
# if len(sys.argv) > 1:
#     image_name = sys.argv[1].replace(".jpg", "")
# else:
#     image_name = "t7"
#
# # ---------------------------------------------------
# # Paths
# # ---------------------------------------------------
# image_path = f"data/raw/hirise_original/{image_name}.jpg"
# gt_path = f"data/sam2_masks/{image_name}.png"
# pred_path = f"outputs/{image_name}_predicted.png"
#
# # ---------------------------------------------------
# # Load images
# # ---------------------------------------------------
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# gt_mask = cv2.imread(gt_path, 0)
# pred_mask = cv2.imread(pred_path, 0)
#
# # ---------------------------------------------------
# # Resize masks to original image size
# # ---------------------------------------------------
# h, w = image.shape[:2]
#
# gt_mask = cv2.resize(gt_mask, (w, h))
# pred_mask = cv2.resize(pred_mask, (w, h))
#
# # Binary masks
# gt_mask = (gt_mask > 127)
# pred_mask = (pred_mask > 127)
#
# # ---------------------------------------------------
# # True Positive, False Positive, False Negative
# # ---------------------------------------------------
# tp = np.logical_and(pred_mask, gt_mask)
# fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
# fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
#
# # ---------------------------------------------------
# # Create colored maps
# # ---------------------------------------------------
# tp_img = np.zeros((h, w, 3), dtype=np.uint8)
# fp_img = np.zeros((h, w, 3), dtype=np.uint8)
# fn_img = np.zeros((h, w, 3), dtype=np.uint8)
#
# tp_img[tp] = [0,255,0]      # Green
# fp_img[fp] = [255,0,0]      # Red
# fn_img[fn] = [0,0,255]      # Blue
#
# # ---------------------------------------------------
# # Metrics
# # ---------------------------------------------------
# intersection = np.sum(tp)
#
# precision = intersection / (np.sum(pred_mask)+1e-6)
#
# recall = intersection / (np.sum(gt_mask)+1e-6)
#
# f1 = 2 * precision * recall / (precision + recall + 1e-6)
#
# iou = intersection / (
#         np.sum(np.logical_or(pred_mask, gt_mask))
#         + 1e-6
# )
#
# accuracy = (
#         np.sum(pred_mask == gt_mask)
#         / gt_mask.size
# )
#
# # ---------------------------------------------------
# # Plot
# # ---------------------------------------------------
# plt.figure(figsize=(20,10))
#
# plt.subplot(2,3,1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(2,3,2)
# plt.imshow(gt_mask, cmap="gray")
# plt.title("Ground Truth (SAM2)")
# plt.axis("off")
#
# plt.subplot(2,3,3)
# plt.imshow(pred_mask, cmap="gray")
# plt.title("Predicted Mask")
# plt.axis("off")
#
# plt.subplot(2,3,4)
# plt.imshow(tp_img)
# plt.title("True Positives (Green)")
# plt.axis("off")
#
# plt.subplot(2,3,5)
# plt.imshow(fp_img)
# plt.title("False Positives (Red)")
# plt.axis("off")
#
# plt.subplot(2,3,6)
# plt.imshow(fn_img)
# plt.title("False Negatives (Blue)")
# plt.axis("off")
#
# plt.suptitle(
# f"""
# Accuracy : {accuracy:.4f}
#
# Precision : {precision:.4f}
#
# Recall : {recall:.4f}
#
# F1 Score : {f1:.4f}
#
# IoU : {iou:.4f}
# """,
# fontsize=14
# )
#
# plt.tight_layout()
#
# plt.show()

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Image name
# ---------------------------------------------------
if len(sys.argv) > 1:
    image_name = sys.argv[1]
else:
    image_name = "ESP_031871_1985.jpg"

# ---------------------------------------------------
# Handle extension correctly
# ---------------------------------------------------
if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
    image_file = image_name
    image_base = os.path.splitext(image_name)[0]
else:
    image_file = image_name + ".jpg"
    image_base = image_name

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
image_path = os.path.join("data/raw/hirise", image_file)

gt_path = os.path.join(
    "data/sam2_masks",
    image_base + ".png"
)

# pred_path = os.path.join(
#     "outputs",
#     image_base + "_predicted.png"
# )
pred_path = os.path.join(
    "outputs",
    "prediction",
    image_base + "_predicted.png"
)

# ---------------------------------------------------
# Load Original Image
# ---------------------------------------------------
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    raise FileNotFoundError(
        f"\nCannot open original image:\n{image_path}"
    )

# Convert image for matplotlib
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------
# Load Masks
# ---------------------------------------------------
gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

if gt_mask is None:
    raise FileNotFoundError(
        f"\nGround Truth mask not found:\n{gt_path}"
    )

pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

if pred_mask is None:
    raise FileNotFoundError(
        f"\nPrediction mask not found:\n{pred_path}"
    )

# ---------------------------------------------------
# Resize masks
# ---------------------------------------------------
h, w = image.shape[:2]

gt_mask = cv2.resize(
    gt_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

pred_mask = cv2.resize(
    pred_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ---------------------------------------------------
# Binary masks
# ---------------------------------------------------
gt_mask = gt_mask > 127
pred_mask = pred_mask > 127

# ---------------------------------------------------
# Calculate TP / FP / FN
# ---------------------------------------------------
tp = np.logical_and(pred_mask, gt_mask)

fp = np.logical_and(
    pred_mask,
    np.logical_not(gt_mask)
)

fn = np.logical_and(
    np.logical_not(pred_mask),
    gt_mask
)

# ---------------------------------------------------
# Colored Results
# ---------------------------------------------------
tp_img = np.zeros((h, w, 3), dtype=np.uint8)
fp_img = np.zeros((h, w, 3), dtype=np.uint8)
fn_img = np.zeros((h, w, 3), dtype=np.uint8)

tp_img[tp] = [0, 255, 0]      # Green
fp_img[fp] = [255, 0, 0]      # Red
fn_img[fn] = [0, 0, 255]      # Blue

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
intersection = np.sum(tp)
union = np.sum(np.logical_or(pred_mask, gt_mask))

precision = intersection / (np.sum(pred_mask) + 1e-6)

recall = intersection / (np.sum(gt_mask) + 1e-6)

f1 = (
    2 * precision * recall
    / (precision + recall + 1e-6)
)

iou = intersection / (union + 1e-6)

accuracy = np.sum(
    pred_mask == gt_mask
) / gt_mask.size

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gt_mask, cmap="gray")
plt.title("Ground Truth (SAM2)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(tp_img)
plt.title("True Positives (Green)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(fp_img)
plt.title("False Positives (Red)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(fn_img)
plt.title("False Negatives (Blue)")
plt.axis("off")

plt.suptitle(
    f"""
Segmentation Performance

Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
IoU       : {iou:.4f}
""",
    fontsize=14
)

plt.tight_layout()
plt.show()