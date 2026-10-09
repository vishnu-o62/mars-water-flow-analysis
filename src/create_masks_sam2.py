# import os
# import sys
# import cv2
# import numpy as np
# import torch
#
# # ----------------------------------------------------
# # Add outer sam2 folder to Python path
# # ----------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# SAM2_DIR = os.path.join(BASE_DIR, "sam2")
#
# if SAM2_DIR not in sys.path:
#     sys.path.insert(0, SAM2_DIR)
#
# # ----------------------------------------------------
# # Imports
# # ----------------------------------------------------
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
#
# # ----------------------------------------------------
# # Paths
# # ----------------------------------------------------
# checkpoint_path = "checkpoints/sam2_hiera_tiny.pt"
#
# # IMPORTANT:
# # This is the Hydra config name, NOT a file path
# config_name = "configs/sam2/sam2_hiera_t"
#
# image_folder = "data/raw/hirise/t1.jpg"
# output_folder = "data/sam2_masks"
#
# os.makedirs(output_folder, exist_ok=True)
#
# # ----------------------------------------------------
# # Device
# # ----------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# print("Using device:", device)
#
# # ----------------------------------------------------
# # Load model
# # ----------------------------------------------------
# sam2_model = build_sam2(
#     config_file=config_name,
#     ckpt_path=checkpoint_path,
#     device=device
# )
#
# # ----------------------------------------------------
# # Automatic mask generator
# # ----------------------------------------------------
# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2_model
# )
#
# # ----------------------------------------------------
# # Process images
# # ----------------------------------------------------
# for image_name in os.listdir(image_folder):
#
#     if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
#
#         print("Processing:", image_name)
#
#         image_path = os.path.join(
#             image_folder,
#             image_name
#         )
#
#         image = cv2.imread(image_path)
#
#         if image is None:
#             print("Could not read:", image_name)
#             continue
#
#         image = cv2.cvtColor(
#             image,
#             cv2.COLOR_BGR2RGB
#         )
#
#         masks = mask_generator.generate(image)
#
#         final_mask = np.zeros(
#             image.shape[:2],
#             dtype=np.uint8
#         )
#
#         for mask in masks:
#             final_mask[mask["segmentation"]] = 255
#
#         save_name = (
#             os.path.splitext(image_name)[0]
#             + ".png"
#         )
#
#         save_path = os.path.join(
#             output_folder,
#             save_name
#         )
#
#         cv2.imwrite(
#             save_path,
#             final_mask
#         )
#
#         print("Saved:", save_name)
#
# print("\nAll masks generated successfully!")


import os
import sys
import cv2
import numpy as np
import torch

# ----------------------------------------------------
# Add outer sam2 folder to Python path
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAM2_DIR = os.path.join(BASE_DIR, "sam2")

if SAM2_DIR not in sys.path:
    sys.path.insert(0, SAM2_DIR)

# ----------------------------------------------------
# Imports
# ----------------------------------------------------
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
checkpoint_path = "checkpoints/sam2_hiera_tiny.pt"

# Hydra config name
config_name = "configs/sam2/sam2_hiera_t"

# Give the full path of the image
image_path = r"data/raw/hirise/t3.jpg"

output_folder = "data/sam2_masks"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------------------------------
# Device
# ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# ----------------------------------------------------
# Load model
# ----------------------------------------------------
sam2_model = build_sam2(
    config_file=config_name,
    ckpt_path=checkpoint_path,
    device=device
)

# ----------------------------------------------------
# Automatic mask generator
# ----------------------------------------------------
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

# ----------------------------------------------------
# Process Single Image
# ----------------------------------------------------
print("Processing:", os.path.basename(image_path))

image = cv2.imread(image_path)

if image is None:
    print("Could not read image!")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(image)

# Create final binary mask
final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

for mask in masks:
    final_mask[mask["segmentation"]] = 255

# Save mask
image_name = os.path.basename(image_path)
save_name = os.path.splitext(image_name)[0] + ".png"
save_path = os.path.join(output_folder, save_name)

cv2.imwrite(save_path, final_mask)

print("Saved:", save_path)
print("\nMask generated successfully!")