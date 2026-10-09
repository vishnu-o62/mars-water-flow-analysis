import cv2
import torch
import numpy as np

from src.unet_model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(
    torch.load(
        "models_saved/unet_model_multidata.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

image_path = r"data\raw\hirise_original\h5.jpg"

image = cv2.imread(image_path)

image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0

tensor = torch.tensor(image)
tensor = tensor.permute(2, 0, 1)
tensor = tensor.unsqueeze(0).float().to(device)

with torch.no_grad():
    prediction = model(tensor)

prediction = torch.sigmoid(prediction)
prediction = prediction.squeeze().cpu().numpy()

print("\n========================================")
print("NEW MODEL THRESHOLD DIAGNOSTIC")
print("========================================")

print("Min :", prediction.min())
print("Max :", prediction.max())
print("Mean:", prediction.mean())

for threshold in [0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
    foreground = np.sum(prediction > threshold)
    percentage = (foreground / prediction.size) * 100

    print(
        f"Threshold {threshold:.2f} "
        f"-> Foreground: {foreground} "
        f"({percentage:.2f}%)"
    )

print("========================================")