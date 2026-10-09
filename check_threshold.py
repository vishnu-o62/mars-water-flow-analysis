import cv2
import torch
import numpy as np
import sys

from src.unet_model import build_model


# ==========================================================
# INPUT
# ==========================================================

if len(sys.argv) < 2:
    raise ValueError(
        "Usage:\n"
        "python check_threshold.py <image_path>"
    )

image_path = sys.argv[1]


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("U-NET THRESHOLD DIAGNOSTIC")
print("=" * 70)

print("\nInput image:")
print(image_path)

print("\nDevice:")
print(device)


# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Unable to read image:\n{image_path}"
    )

original_height, original_width = image.shape[:2]

print("\nOriginal image shape:")
print(image.shape)


# ==========================================================
# SAME PREPROCESSING AS PIPELINE 1
# ==========================================================

resized_image = cv2.resize(
    image,
    (256, 256)
)

resized_image = (
    resized_image.astype(np.float32) / 255.0
)


image_tensor = torch.tensor(
    resized_image
)

image_tensor = image_tensor.permute(
    2,
    0,
    1
)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = (
    image_tensor
    .float()
    .to(device)
)


# ==========================================================
# LOAD SAME U-NET MODEL
# ==========================================================

model_path = "models_saved/unet_model.pth"

print("\nLoading U-Net model:")
print(model_path)

model = build_model()

model.to(device)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("U-Net loaded successfully.")


# ==========================================================
# PREDICTION
# ==========================================================

print("\nRunning U-Net prediction...")

with torch.no_grad():

    prediction = model(
        image_tensor
    )

prediction = torch.sigmoid(
    prediction
)

prediction = (
    prediction
    .squeeze()
    .cpu()
    .numpy()
)


# ==========================================================
# PREDICTION STATISTICS
# ==========================================================

print("\nPrediction Statistics")
print("-" * 40)

print(
    "Minimum :",
    prediction.min()
)

print(
    "Maximum :",
    prediction.max()
)

print(
    "Mean    :",
    prediction.mean()
)


# ==========================================================
# THRESHOLD COMPARISON
# ==========================================================

print("\nPixels Above Thresholds")
print("-" * 40)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    pixels = np.sum(
        prediction > threshold
    )

    print(
        f"Threshold {threshold:.1f} : {pixels} pixels"
    )


# ==========================================================
# CREATE 0.5 THRESHOLD MASK
# ==========================================================

threshold = 0.5

print("\nCreating threshold 0.5 mask...")

prediction_binary = (
    prediction > threshold
).astype(
    np.uint8
)

prediction_mask = (
    prediction_binary * 255
)


# ==========================================================
# RESIZE BACK TO ORIGINAL IMAGE SIZE
# ==========================================================

prediction_mask = cv2.resize(
    prediction_mask,
    (
        original_width,
        original_height
    ),
    interpolation=cv2.INTER_NEAREST
)


# ==========================================================
# SAVE 0.5 MASK
# ==========================================================

output_path = (
    "outputs/"
    "ESP_037300_1825_threshold05.png"
)

success = cv2.imwrite(
    output_path,
    prediction_mask
)

if not success:
    raise Exception(
        "Failed to save threshold 0.5 mask."
    )


# ==========================================================
# FINAL PIXEL STATISTICS
# ==========================================================

final_binary = (
    prediction_mask > 127
).astype(
    np.uint8
)

foreground_pixels = np.sum(
    final_binary
)

total_pixels = (
    final_binary.size
)

background_pixels = (
    total_pixels -
    foreground_pixels
)


print("\nThreshold 0.5 Mask Statistics")
print("-" * 40)

print(
    "Foreground Pixels :",
    foreground_pixels
)

print(
    "Background Pixels :",
    background_pixels
)

print(
    "Total Pixels      :",
    total_pixels
)


# ==========================================================
# OUTPUT
# ==========================================================

print("\nThreshold 0.5 mask saved:")
print(output_path)

print("\n" + "=" * 70)
print("THRESHOLD DIAGNOSTIC COMPLETED")
print("=" * 70)