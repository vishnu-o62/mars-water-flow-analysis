# import os
# import cv2
# import torch
# import numpy as np
#
# from unet_model import build_model
#
# # ==========================================================
# # Device
# # ==========================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ==========================================================
# # Load Model
# # ==========================================================
# model = build_model()
# model.load_state_dict(torch.load("models_saved/unet_model.pth", map_location=device))
# model.to(device)
# model.eval()
#
# # ==========================================================
# # Paths
# # ==========================================================
# image_folder = "data/raw/hirise_original"
# output_folder = "outputs/prediction_2"
# os.makedirs(output_folder, exist_ok=True)
#
# image_files = sorted([
#     f for f in os.listdir(image_folder)
#     if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ])
#
# print(f"\nFound {len(image_files)} images.\n")
#
# # -------------------------
# # Noise Reduction Parameters
# # -------------------------
# THRESHOLD = 0.65          # Higher confidence threshold
# MEDIAN_KERNEL = 5         # Removes salt-and-pepper noise
# MIN_COMPONENT_AREA = 150  # Remove tiny blobs
# MORPH_KERNEL = 3          #Morphological cleanup
#
# THRESHOLD = 0.50      # instead of 0.65
# MEDIAN_KERNEL = 3     # instead of 5
# MIN_COMPONENT_AREA = 40
# MORPH_KERNEL = 3
# kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
#
# with torch.no_grad():
#     for image_file in image_files:
#
#         image_name = os.path.splitext(image_file)[0]
#         image_path = os.path.join(image_folder, image_file)
#
#         print(f"Processing: {image_name}")
#
#         image = cv2.imread(image_path)
#
#         if image is None:
#             print(f"Could not read {image_file}")
#             continue
#
#         h, w = image.shape[:2]
#
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         rgb = cv2.resize(rgb, (256, 256))
#         rgb = rgb.astype(np.float32) / 255.0
#         rgb = np.transpose(rgb, (2, 0, 1))
#
#         tensor = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).to(device)
#
#         pred = model(tensor)
#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()
#
#         pred = (pred > THRESHOLD).astype(np.uint8) * 255
#
#         pred = cv2.medianBlur(pred, MEDIAN_KERNEL)
#         pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
#         pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
#
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
#
#         cleaned = np.zeros_like(pred)
#
#         for i in range(1, num_labels):
#             area = stats[i, cv2.CC_STAT_AREA]
#             if area >= MIN_COMPONENT_AREA:
#                 cleaned[labels == i] = 255
#
#         cleaned = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_NEAREST)
#
#         out = os.path.join(output_folder, f"{image_name}_predicted.png")
#         cv2.imwrite(out, cleaned)
#
#         print(f"Saved -> {out}")
#
# print("\n===================================")
# print("Prediction V2 completed successfully.")
# print("===================================")


import os
import cv2
import torch
import numpy as np
from unet_model import build_model

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Load Model
# ==========================================================
model = build_model()
model.load_state_dict(torch.load("models_saved/unet_model.pth", map_location=device))
model.to(device)
model.eval()

# ==========================================================
# Paths
# ==========================================================
image_folder = "data/raw/hirise/"
output_folder = "outputs/prediction_v2"
os.makedirs(output_folder, exist_ok=True)

# image_files = sorted([
#     f for f in os.listdir(image_folder)
#     if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ])
image_files = [
    "ESP_028535_1990.jpg"
]

print(f"\nFound {len(image_files)} images.")

# -----------------------------
# Tunable Parameters
# -----------------------------
BASE_THRESHOLD = 0.50
MIN_COMPONENT_AREA = 20
CIRCULARITY_LIMIT = 0.75

kernel = np.ones((3,3), np.uint8)

with torch.no_grad():

    for image_file in image_files:

        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_folder, image_file)

        print(f"\nProcessing : {image_name}")

        image = cv2.imread(image_path)

        if image is None:
            print("Cannot read:", image_file)
            continue

        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256,256))
        rgb = rgb.astype(np.float32)/255.0
        rgb = np.transpose(rgb,(2,0,1))

        tensor = torch.from_numpy(rgb).float().unsqueeze(0).to(device)

        pred = model(tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # ================================
        # Save probability map (NEW)
        # ================================
        prob = (pred * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_folder, f"{image_name}_probability.png"),
            prob
        )

        # Gentle smoothing
        pred = cv2.GaussianBlur(pred,(3,3),0)

        # Adaptive threshold
        adaptive = max(BASE_THRESHOLD, float(pred.mean()+0.05))
        mask = (pred >= adaptive).astype(np.uint8)*255

        # Fill tiny holes only
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove tiny isolated blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
        cleaned = np.zeros_like(mask)

        for i in range(1,num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_COMPONENT_AREA:
                cleaned[labels==i]=255

        # Remove tiny round blobs but keep elongated valleys
        contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final = np.zeros_like(cleaned)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_COMPONENT_AREA:
                continue

            peri = cv2.arcLength(cnt,True)
            if peri == 0:
                continue

            circularity = 4*np.pi*area/(peri*peri)
            x,y,bw,bh = cv2.boundingRect(cnt)
            aspect = max(bw,bh)/(min(bw,bh)+1e-5)

            # Keep elongated features OR non-circular larger regions
            if aspect > 2.0 or circularity < CIRCULARITY_LIMIT or area > 100:
                cv2.drawContours(final,[cnt],-1,255,-1)

        final = cv2.resize(final,(w,h),interpolation=cv2.INTER_NEAREST)

        out = os.path.join(output_folder,f"{image_name}_predicted.png")
        cv2.imwrite(out,final)
        print(f"Saved -> {out}")

print("\n===================================")
print("Prediction V2 Finished")
print("Outputs : outputs/prediction_v2/")
print("===================================")

