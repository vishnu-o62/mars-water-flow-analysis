# src/train.py
import os

import torch
from torch.utils.data import DataLoader
from dataset_loader import MarsDataset
from unet_model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# image_folder = os.path.join(BASE_DIR, "data", "raw", "hirise")
# mask_folder = os.path.join(BASE_DIR, "data", "masks")

dataset = MarsDataset(
    os.path.join(BASE_DIR, "data", "raw", "hirise"),
    os.path.join(BASE_DIR, "data", "sam2_masks")
)
# print("=" * 50)
# print(f"Total training images: {len(dataset)}")
# print("=" * 50)

#
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=False
# )

model = build_model().to(device)

# criterion = torch.nn.BCEWithLogitsLoss()


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        smooth = 1e-6

        intersection = (pred * target).sum()

        dice = (
            2.0 * intersection + smooth
        ) / (
            pred.sum() + target.sum() + smooth
        )

        return 1 - dice



bce_loss = torch.nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # loss = criterion(outputs, masks)

        loss = (
                bce_loss(outputs, masks)
                + dice_loss(outputs, masks)
        )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss:{running_loss:.4f}")

# torch.save(
#     model.state_dict(),
#     "../models_saved/unet_model.pth"
# )

os.makedirs("models_saved", exist_ok=True)

torch.save(
    model.state_dict(),
    "models_saved/unet_model.pth"
)
print("Model saved.")
