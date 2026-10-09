# # src/dataset_loader.py
#
# import os
# import cv2
# import torch
# from torch.utils.data import Dataset
#
#
# class MarsDataset(Dataset):
#     def __init__(self, image_dir, mask_dir):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.images = os.listdir(image_dir)
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#
#         image_path = os.path.join(self.image_dir, self.images[index])
#
#         mask_name = self.images[index].replace(".jpg", ".png")
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (256, 256))
#         image = image / 255.0
#
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, (256, 256))
#         mask = mask / 255.0
#
#         image = torch.tensor(image).permute(2, 0, 1).float()
#         mask = torch.tensor(mask).unsqueeze(0).float()
#
#         return image, mask

# src/dataset_loader.py

import os
import cv2
import torch
from torch.utils.data import Dataset


class MarsDataset(Dataset):
    def __init__(self, image_path, mask_path):

        # If directories are provided
        if os.path.isdir(image_path):

            self.single_image = False
            self.image_dir = image_path
            self.mask_dir = mask_path
            self.images = os.listdir(image_path)

        # If a single image is provided
        else:

            self.single_image = True
            self.image_path = image_path
            self.mask_path = mask_path

    def __len__(self):

        if self.single_image:
            return 1

        return len(self.images)

    def __getitem__(self, index):

        if self.single_image:

            image_path = self.image_path
            mask_path = self.mask_path

        else:

            image_path = os.path.join(
                self.image_dir,
                self.images[index]
            )

            mask_name = self.images[index].replace(".jpg", ".png")

            mask_path = os.path.join(
                self.mask_dir,
                mask_name
            )

        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask