# src/unet_model.py

import segmentation_models_pytorch as smp


def build_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    return model