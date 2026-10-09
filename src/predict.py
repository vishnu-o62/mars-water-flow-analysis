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

model.load_state_dict(
    torch.load("models_saved/unet_model.pth", map_location=device)
)

model.to(device)
model.eval()

# ==========================================================
# Paths
# ==========================================================

image_folder = "data/raw/hirise_original"
image_files = os.listdir(image_folder)
output_folder = "outputs/prediction"

os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# Get all JPG images
# ==========================================================

# image_files = sorted([
#     f for f in os.listdir(image_folder)
#     if f.lower().endswith(".jpg")
# ])
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nFound {len(image_files)} images.\n")

# ==========================================================
# Predict Every Image
# ==========================================================

with torch.no_grad():

    for image_file in image_files:

        image_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(image_folder, image_file)

        print(f"Processing: {image_name}")

        # ------------------------------------------
        # Read Image
        # ------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_file}")
            continue

        original_h, original_w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_rgb = cv2.resize(image_rgb, (256, 256))

        image_rgb = image_rgb.astype(np.float32) / 255.0

        image_rgb = np.transpose(image_rgb, (2, 0, 1))

        image_tensor = torch.tensor(
            image_rgb,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model(image_tensor)

        prediction = torch.sigmoid(prediction)

        prediction = prediction.squeeze().cpu().numpy()

        prediction = (prediction > 0.5).astype(np.uint8) * 255

        prediction = cv2.resize(
            prediction,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        # ------------------------------------------
        # Save Output
        # ------------------------------------------

        output_path = os.path.join(
            output_folder,
            f"{image_name}_predicted.png"
        )

        cv2.imwrite(output_path, prediction)

        print(f"Saved -> {output_path}")

print("\n===================================")
print("Prediction completed successfully.")
print("===================================")


