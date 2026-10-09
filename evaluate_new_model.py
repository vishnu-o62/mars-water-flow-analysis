import cv2
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from src.unet_model import build_model


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = r"data\raw\hirise_original\h3.jpg"
MASK_PATH = r"data\sam2_masks\h3.png"
MODEL_PATH = r"models_saved\unet_model_multidata.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading new model...")

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image = cv2.resize(
    image,
    (256, 256),
    interpolation=cv2.INTER_AREA
)

image = image.astype(np.float32) / 255.0

tensor = torch.from_numpy(image)
tensor = tensor.permute(2, 0, 1)
tensor = tensor.unsqueeze(0).float().to(device)


# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():
    output = model(tensor)

prediction = torch.sigmoid(output)

prediction = prediction.squeeze().cpu().numpy()


# ============================================================
# LOAD SAM2 REFERENCE MASK
# ============================================================

reference = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    raise FileNotFoundError(
        f"Could not read reference mask: {MASK_PATH}"
    )

reference = cv2.resize(
    reference,
    (256, 256),
    interpolation=cv2.INTER_NEAREST
)

reference = reference > 127


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n==============================================")
print("NEW MODEL vs SAM2 REFERENCE")
print("==============================================")

print("Image          :", IMAGE_PATH)
print("Reference mask :", MASK_PATH)
print("Model          :", MODEL_PATH)

print("\nPrediction statistics:")
print("Min            :", prediction.min())
print("Max            :", prediction.max())
print("Mean           :", prediction.mean())

print("\nSAM2 reference foreground pixels:",
      np.sum(reference))


# ============================================================
# THRESHOLD TEST
# ============================================================

thresholds = [
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10
]

print("\n==============================================")
print("THRESHOLD RESULTS")
print("==============================================")

print(
    f"{'Threshold':<12}"
    f"{'FG Pixels':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'IoU':<12}"
)

best_f1 = -1
best_threshold = None

for threshold in thresholds:

    predicted = prediction > threshold

    y_true = reference.flatten().astype(np.uint8)
    y_pred = predicted.flatten().astype(np.uint8)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    intersection = np.logical_and(
        reference,
        predicted
    ).sum()

    union = np.logical_or(
        reference,
        predicted
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 0
    )

    foreground = np.sum(predicted)

    print(
        f"{threshold:<12.2f}"
        f"{foreground:<12}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{iou:<12.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n==============================================")
print("BEST RESULT")
print("==============================================")

print("Best threshold :", best_threshold)
print("Best F1        :", round(best_f1, 4))

print("\nDone.")