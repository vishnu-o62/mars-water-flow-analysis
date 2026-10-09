
import os
import sys
import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset_loader import MarsDataset
from unet_model import build_model


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================================
# Add SAM2 to Python Path
# ==========================================================

SAM2_DIR = os.path.join(BASE_DIR, "sam2")

if SAM2_DIR not in sys.path:
    sys.path.insert(0, SAM2_DIR)


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Mars Water Detection - Pipeline 1")
print("=" * 70)
print("Device :", device)


# ==========================================================
# COMMAND LINE
# ==========================================================

if len(sys.argv) < 2:

    raise ValueError(
        "Usage:\n"
        "  TRAIN           : python pipeline.py <image_path> train\n"
        "  PREDICT         : python pipeline.py <image_path> predict\n"
        "  CHECK DATASET   : python pipeline.py <image_path> check_dataset\n"
        "  PREDICT MULTI   : python pipeline.py <image_path> predict_multi"
    )


image_path = os.path.abspath(sys.argv[1])


# Default mode
mode = "predict"

if len(sys.argv) >= 3:
    mode = sys.argv[2].lower()


if mode not in ["train", "predict", "check_dataset", "predict_multi"]:

    raise ValueError(
        "Invalid mode.\n"
        "Use 'train', 'predict', 'check_dataset', or 'predict_multi'."
    )


print("\nMode :", mode.upper())

print("\nInput image:")
print(image_path)


# ==========================================================
# VERIFY IMAGE
# ==========================================================

if not os.path.exists(image_path):

    raise FileNotFoundError(
        f"Input image not found:\n{image_path}"
    )


image = cv2.imread(image_path)

if image is None:

    raise Exception(
        "Unable to read input image."
    )


# ==========================================================
# AUTOMATIC PATHS
# ==========================================================

image_name = os.path.basename(image_path)

image_stem = os.path.splitext(
    image_name
)[0]


# ----------------------------------------------------------
# SAM2 MASK
# ----------------------------------------------------------

mask_folder = os.path.join(
    BASE_DIR,
    "data",
    "sam2_masks"
)

os.makedirs(
    mask_folder,
    exist_ok=True
)


mask_path = os.path.join(
    mask_folder,
    image_stem + ".png"
)


# ----------------------------------------------------------
# OUTPUT FOLDER
# ----------------------------------------------------------

output_folder = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ----------------------------------------------------------
# MODEL FOLDER
# ----------------------------------------------------------

model_folder = os.path.join(
    BASE_DIR,
    "models_saved"
)

os.makedirs(
    model_folder,
    exist_ok=True
)


model_path = os.path.join(
    model_folder,
    "unet_model.pth"
)

# ==========================================================
# MULTI-IMAGE TRAINING PATHS
# ==========================================================

training_root = os.path.join(
    BASE_DIR,
    "data",
    "training_unet"
)

training_image_folder = os.path.join(
    training_root,
    "images"
)

training_mask_folder = os.path.join(
    training_root,
    "masks"
)

multidata_model_path = os.path.join(
    model_folder,
    "unet_model_multidata.pth"
)


# ----------------------------------------------------------
# PREDICTION PATH
# ----------------------------------------------------------

prediction_path = os.path.join(
    output_folder,
    image_stem + "_predicted.png"
)


# ==========================================================
# FUNCTION: GENERATE SAM2 MASK
# ==========================================================

def generate_sam2_mask():

    print("\n" + "=" * 70)
    print("SAM2 MASK GENERATION")
    print("=" * 70)

    checkpoint_path = os.path.join(
        BASE_DIR,
        "checkpoints",
        "sam2_hiera_tiny.pt"
    )

    config_name = "configs/sam2/sam2_hiera_t"

    print("\nLoading SAM2 Model...")

    sam2_model = build_sam2(
        config_file=config_name,
        ckpt_path=checkpoint_path,
        device=device
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model
    )

    print("SAM2 Loaded Successfully")

    print("\nGenerating segmentation mask...")

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    masks = mask_generator.generate(
        rgb
    )

    print(
        "Masks Found :",
        len(masks)
    )

    final_mask = np.zeros(
        rgb.shape[:2],
        dtype=np.uint8
    )

    for mask in masks:

        final_mask[
            mask["segmentation"]
        ] = 255

    cv2.imwrite(
        mask_path,
        final_mask
    )

    print("\nSAM2 Mask Saved:")
    print(mask_path)

    return mask_path



# ==========================================================
# FUNCTION: PREPARE MULTI-IMAGE TRAINING DATASET
# ==========================================================

def prepare_multidata_training():

    print("\n" + "=" * 70)
    print("PREPARING MULTI-IMAGE U-NET DATASET")
    print("=" * 70)

    import shutil

    os.makedirs(
        training_image_folder,
        exist_ok=True
    )

    os.makedirs(
        training_mask_folder,
        exist_ok=True
    )

    image_folders = [
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "hirise"
        ),
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "hirise_original"
        )
    ]

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    added_pairs = 0
    skipped_images = 0
    used_names = set()

    # Clear only files previously created in the dedicated training folder.
    # Original input folders and SAM2 masks are never modified.
    for folder in (
        training_image_folder,
        training_mask_folder
    ):
        for old_name in os.listdir(folder):
            old_path = os.path.join(folder, old_name)
            if os.path.isfile(old_path):
                os.remove(old_path)

    for folder in image_folders:

        if not os.path.exists(folder):
            print("\nWARNING: Folder not found:")
            print(folder)
            continue

        print("\nChecking folder:")
        print(folder)

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(valid_extensions):
                continue

            image_path_local = os.path.join(
                folder,
                filename
            )

            image_stem_local = os.path.splitext(
                filename
            )[0]

            mask_path_local = os.path.join(
                mask_folder,
                image_stem_local + ".png"
            )

            if not os.path.exists(mask_path_local):
                print("SKIPPED - No mask:", filename)
                skipped_images += 1
                continue

            if image_stem_local in used_names:
                print("SKIPPED - Duplicate:", filename)
                skipped_images += 1
                continue

            # Verify that both files can actually be read.
            test_image = cv2.imread(image_path_local)
            test_mask = cv2.imread(
                mask_path_local,
                cv2.IMREAD_GRAYSCALE
            )

            if test_image is None:
                print("SKIPPED - Image unreadable:", filename)
                skipped_images += 1
                continue

            if test_mask is None:
                print("SKIPPED - Mask unreadable:", image_stem_local + ".png")
                skipped_images += 1
                continue

            used_names.add(image_stem_local)

            # Normalize every staged training image to .jpg.
            # MarsDataset directory mode maps <image_stem>.jpg -> <image_stem>.png,
            # so this keeps image/mask pairing reliable for .jpeg/.png/etc. sources.
            staged_image_name = image_stem_local + ".jpg"

            shutil.copy2(
                image_path_local,
                os.path.join(
                    training_image_folder,
                    staged_image_name
                )
            )

            shutil.copy2(
                mask_path_local,
                os.path.join(
                    training_mask_folder,
                    image_stem_local + ".png"
                )
            )

            added_pairs += 1

            print(
                "ADDED:",
                staged_image_name,
                "->",
                image_stem_local + ".png"
            )

    print("\n" + "-" * 70)
    print(
        "Training image-mask pairs :",
        added_pairs
    )
    print(
        "Skipped images            :",
        skipped_images
    )
    print("-" * 70)

    if added_pairs == 0:
        raise RuntimeError(
            "No valid image-mask pairs found."
        )

    return (
        training_image_folder,
        training_mask_folder
    )


# ==========================================================
# FUNCTION: CHECK MULTI-IMAGE DATASET
# ==========================================================

def check_multidata_dataset():

    print("\n" + "=" * 70)
    print("CHECKING MULTI-IMAGE U-NET DATASET")
    print("=" * 70)

    image_folders = [
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "hirise"
        ),
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "hirise_original"
        )
    ]

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    total_images = 0
    valid_pairs = 0
    missing_masks = 0
    duplicate_images = 0
    unreadable_images = 0
    unreadable_masks = 0

    used_names = set()

    for folder in image_folders:

        print("\n" + "-" * 70)
        print("Checking folder:")
        print(folder)
        print("-" * 70)

        if not os.path.exists(folder):
            print("WARNING: Folder not found")
            continue

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(valid_extensions):
                continue

            total_images += 1

            image_stem_local = os.path.splitext(
                filename
            )[0]

            image_path_local = os.path.join(
                folder,
                filename
            )

            mask_path_local = os.path.join(
                mask_folder,
                image_stem_local + ".png"
            )

            if image_stem_local in used_names:
                print("DUPLICATE:", filename)
                duplicate_images += 1
                continue

            if not os.path.exists(mask_path_local):
                print("NO MASK:", filename)
                missing_masks += 1
                continue

            test_image = cv2.imread(image_path_local)
            test_mask = cv2.imread(
                mask_path_local,
                cv2.IMREAD_GRAYSCALE
            )

            if test_image is None:
                print("UNREADABLE IMAGE:", filename)
                unreadable_images += 1
                continue

            if test_mask is None:
                print(
                    "UNREADABLE MASK:",
                    image_stem_local + ".png"
                )
                unreadable_masks += 1
                continue

            used_names.add(image_stem_local)

            valid_pairs += 1

            print(
                "VALID:",
                filename,
                "->",
                image_stem_local + ".png"
            )

    print("\n" + "=" * 70)
    print("DATASET CHECK SUMMARY")
    print("=" * 70)

    print(
        "Total input images       :",
        total_images
    )
    print(
        "Valid image-mask pairs   :",
        valid_pairs
    )
    print(
        "Missing masks            :",
        missing_masks
    )
    print(
        "Duplicate image names    :",
        duplicate_images
    )
    print(
        "Unreadable images        :",
        unreadable_images
    )
    print(
        "Unreadable masks         :",
        unreadable_masks
    )

    print("=" * 70)

    if valid_pairs == 0:
        print(
            "\nERROR: No valid image-mask pairs found."
        )
        return False

    print(
        "\nDataset check completed successfully."
    )
    print(
        "NO TRAINING WAS PERFORMED."
    )

    return True


# ==========================================================
# FUNCTION: TRAIN U-NET
# ==========================================================

def train_unet():

    print("\n" + "=" * 70)
    print("U-NET MULTI-IMAGE TRAINING")
    print("=" * 70)

    training_images, training_masks = (
        prepare_multidata_training()
    )

    print("\nPreparing Dataset...")

    dataset = MarsDataset(
        training_images,
        training_masks
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    print("Dataset Ready")

    print(
        "Number of Images :",
        len(dataset)
    )

    print("\nBuilding U-Net...")

    model = build_model()

    model.to(device)

    print(
        "Model Created Successfully"
    )

    class DiceLoss(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(
            self,
            pred,
            target
        ):

            pred = torch.sigmoid(pred)

            smooth = 1e-6

            intersection = (
                pred * target
            ).sum()

            dice = (
                2 * intersection + smooth
            ) / (
                pred.sum()
                + target.sum()
                + smooth
            )

            return 1 - dice

    bce_loss = torch.nn.BCEWithLogitsLoss()

    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00001
    )

    epochs = 200

    print("\n" + "=" * 70)
    print("MULTI-IMAGE TRAINING STARTED")
    print("=" * 70)

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, masks in loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(
                images
            )

            loss = (
                bce_loss(
                    outputs,
                    masks
                )
                +
                dice_loss(
                    outputs,
                    masks
                )
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1:03d}/{epochs}"
            f"  Loss : {running_loss:.4f}"
        )

    print(
        "\nMulti-image training completed successfully."
    )

    torch.save(
        model.state_dict(),
        multidata_model_path
    )

    print(
        "\nNew Multi-Image Model Saved:"
    )

    print(
        multidata_model_path
    )

    return model


# ==========================================================
# FUNCTION: LOAD U-NET
# ==========================================================

def load_unet():

    print("\n" + "=" * 70)
    print("LOADING SAVED U-NET")
    print("=" * 70)

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "Saved U-Net model not found:\n"
            f"{model_path}\n\n"
            "Run Pipeline 1 in TRAIN mode first."
        )


    model = build_model()

    model.to(device)


    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )


    model.eval()

    print(
        "Saved U-Net Loaded Successfully"
    )

    print(model_path)

    return model


# ==========================================================
# FUNCTION: LOAD MULTI-IMAGE U-NET
# ==========================================================

def load_multidata_unet():

    print("\n" + "=" * 70)
    print("LOADING MULTI-IMAGE U-NET")
    print("=" * 70)

    if not os.path.exists(
        multidata_model_path
    ):
        raise FileNotFoundError(
            "Multi-image U-Net model not found:\n"
            f"{multidata_model_path}\n\n"
            "Run TRAIN mode first."
        )

    model = build_model()

    model.to(device)

    model.load_state_dict(
        torch.load(
            multidata_model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        "Multi-Image U-Net Loaded Successfully"
    )

    print(
        multidata_model_path
    )

    return model


# ==========================================================
# FUNCTION: PREDICT WITH MULTI-IMAGE MODEL
# ==========================================================

def predict_multi(model):

    print("\n" + "=" * 70)
    print("MULTI-IMAGE U-NET PREDICTION")
    print("=" * 70)

    original_image = image.copy()

    original_height = (
        original_image.shape[0]
    )

    original_width = (
        original_image.shape[1]
    )

    resized_image = cv2.resize(
        image,
        (256, 256)
    )

    resized_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )

    image_tensor = torch.tensor(
        resized_image
    )

    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = (
        image_tensor
        .float()
        .to(device)
    )

    print(
        "\nRunning Multi-Image U-Net prediction..."
    )

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

    print("\nPrediction Statistics")
    print("--------------------------------")

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

    threshold = 0.6

    prediction_binary = (
        prediction > threshold
    ).astype(
        np.uint8
    )

    prediction_mask = (
        prediction_binary * 255
    )

    prediction_mask = cv2.resize(
        prediction_mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    multidata_prediction_path = os.path.join(
        output_folder,
        image_stem + "_multidata_predicted.png"
    )

    cv2.imwrite(
        multidata_prediction_path,
        prediction_mask
    )

    print(
        "\nMulti-image prediction saved:"
    )

    print(
        multidata_prediction_path
    )

    print(
        "Foreground Pixels :",
        int(np.count_nonzero(prediction_mask))
    )

    return multidata_prediction_path


# ==========================================================
# FUNCTION: PREDICT
# ==========================================================

def predict(model):

    print("\n" + "=" * 70)
    print("U-NET SEGMENTATION PREDICTION")
    print("=" * 70)


    # ------------------------------------------------------
    # Original Image
    # ------------------------------------------------------

    original_image = image.copy()

    original_height = (
        original_image.shape[0]
    )

    original_width = (
        original_image.shape[1]
    )


    # ------------------------------------------------------
    # Resize for U-Net
    # ------------------------------------------------------

    resized_image = cv2.resize(
        image,
        (256, 256)
    )


    resized_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )


    # ------------------------------------------------------
    # Convert to Tensor
    # ------------------------------------------------------

    image_tensor = torch.tensor(
        resized_image
    )


    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )


    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = (
        image_tensor
        .float()
        .to(device)
    )


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # Prediction Statistics
    # ------------------------------------------------------

    print("\nPrediction Statistics")
    print("--------------------------------")

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


    # ------------------------------------------------------
    # Probability → Binary Mask
    # ------------------------------------------------------

    threshold = 0.6

    prediction_binary = (
        prediction > threshold
    ).astype(
        np.uint8
    )


    prediction_mask = (
        prediction_binary * 255
    )


    # ------------------------------------------------------
    # Resize Back to Original Size
    # ------------------------------------------------------

    prediction_mask = cv2.resize(
        prediction_mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # ------------------------------------------------------
    # Re-create Binary Mask
    # ------------------------------------------------------

    prediction_binary = (
        prediction_mask > 127
    ).astype(
        np.uint8
    )


    # ------------------------------------------------------
    # Save Prediction
    # ------------------------------------------------------

    cv2.imwrite(
        prediction_path,
        prediction_mask
    )


    print(
        "\nPrediction Saved Successfully:"
    )

    print(prediction_path)


    # ------------------------------------------------------
    # Pixel Statistics
    # ------------------------------------------------------

    foreground_pixels = np.sum(
        prediction_binary
    )

    background_pixels = (
        prediction_binary.size
        -
        foreground_pixels
    )


    print(
        "\nPrediction Pixel Statistics"
    )

    print("--------------------------------")

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
        prediction_binary.size
    )


    print(
        "\nOriginal Image Shape :",
        original_image.shape
    )

    print(
        "Prediction Shape     :",
        prediction_binary.shape
    )


    print(
        "\nPrediction Completed Successfully."
    )

    return prediction_path


# ==========================================================
# FUNCTION: EVALUATION
# ==========================================================

def evaluate_prediction():

    if not os.path.exists(mask_path):

        print(
            "\nEvaluation skipped."
        )

        print(
            "SAM2 ground-truth mask not found."
        )

        return


    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)


    # ------------------------------------------------------
    # Load Ground Truth
    # ------------------------------------------------------

    ground_truth = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if ground_truth is None:

        print(
            "Unable to load ground-truth mask."
        )

        return


    ground_truth = cv2.resize(
        ground_truth,
        (256, 256)
    )


    ground_truth = (
        ground_truth > 127
    ).astype(
        np.uint8
    )


    # ------------------------------------------------------
    # Load Prediction
    # ------------------------------------------------------

    prediction_mask = cv2.imread(
        prediction_path,
        cv2.IMREAD_GRAYSCALE
    )


    if prediction_mask is None:

        print(
            "Unable to load prediction."
        )

        return


    evaluation_prediction = cv2.resize(
        prediction_mask,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )


    evaluation_prediction = (
        evaluation_prediction > 127
    ).astype(
        np.uint8
    )


    # ------------------------------------------------------
    # Flatten
    # ------------------------------------------------------

    pred_flat = (
        evaluation_prediction.flatten()
    )

    gt_flat = (
        ground_truth.flatten()
    )


    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        gt_flat,
        pred_flat
    )

    precision = precision_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )

    recall = recall_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )

    f1 = f1_score(
        gt_flat,
        pred_flat,
        zero_division=0
    )


    # ------------------------------------------------------
    # IoU
    # ------------------------------------------------------

    intersection = np.logical_and(
        gt_flat,
        pred_flat
    ).sum()


    union = np.logical_or(
        gt_flat,
        pred_flat
    ).sum()


    if union == 0:

        iou = 1.0

    else:

        iou = (
            intersection /
            union
        )


    # ------------------------------------------------------
    # Pixel Statistics
    # ------------------------------------------------------

    gt_foreground = np.sum(
        ground_truth
    )

    pred_foreground = np.sum(
        evaluation_prediction
    )


    gt_background = (
        ground_truth.size
        -
        gt_foreground
    )

    pred_background = (
        evaluation_prediction.size
        -
        pred_foreground
    )


    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("\nGround Truth Statistics")
    print("--------------------------------")

    print(
        "Foreground Pixels :",
        gt_foreground
    )

    print(
        "Background Pixels :",
        gt_background
    )


    print("\nPrediction Statistics")
    print("--------------------------------")

    print(
        "Foreground Pixels :",
        pred_foreground
    )

    print(
        "Background Pixels :",
        pred_background
    )


    print("\nOverlap Statistics")
    print("--------------------------------")

    print(
        "Intersection :",
        intersection
    )

    print(
        "Union        :",
        union
    )


    print("\nEvaluation Results")
    print("--------------------------------")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"IoU       : {iou:.4f}"
    )


    print(
        "\nEvaluation Completed Successfully."
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # CHECK DATASET MODE
    # ======================================================

    if mode == "check_dataset":

        print("\n")
        print("=" * 70)
        print("DATASET CHECK MODE SELECTED")
        print("=" * 70)

        check_multidata_dataset()

        print("\n" + "=" * 70)
        print("DATASET CHECK FINISHED")
        print("=" * 70)

        sys.exit(0)

    # ======================================================
    # TRAIN MODE
    # ======================================================

    elif mode == "train":

        print("\n")
        print("=" * 70)
        print("MULTI-IMAGE TRAIN MODE SELECTED")
        print("=" * 70)

        print(
            "\nExisting unet_model.pth will NOT be overwritten."
        )

        print(
            "New model will be saved as:"
        )

        print(
            multidata_model_path
        )

        train_unet()

        print("\n" + "=" * 70)
        print("MULTI-IMAGE TRAINING FINISHED")
        print("=" * 70)

    # ======================================================
    # MULTI-IMAGE PREDICT MODE
    # ======================================================

    elif mode == "predict_multi":

        print("\n")
        print("=" * 70)
        print("MULTI-IMAGE PREDICT MODE SELECTED")
        print("=" * 70)

        model = load_multidata_unet()

        predict_multi(
            model
        )

        print("\n" + "=" * 70)
        print("MULTI-IMAGE PREDICTION FINISHED")
        print("=" * 70)

    # ======================================================
    # ORIGINAL PREDICT MODE
    # ======================================================

    elif mode == "predict":

        print("\n")
        print("=" * 70)
        print("PREDICT MODE SELECTED")
        print("=" * 70)

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                "\nNo trained U-Net model found.\n\n"
                "Expected model:\n"
                f"{model_path}\n\n"
                "Please train the model first."
            )

        print(
            "\nExisting trained model found."
        )

        print(
            "Training will NOT be performed."
        )

        model = load_unet()

        predict(
            model
        )

    print("\n" + "=" * 70)
    print("PIPELINE 1 FINISHED SUCCESSFULLY")
    print("=" * 70)

    print("\nInput Image:")
    print(image_path)

    if mode == "predict":
        print("\nPrediction:")
        print(prediction_path)

        print("\nModel:")
        print(model_path)

    elif mode == "predict_multi":
        print("\nModel:")
        print(multidata_model_path)

    elif mode == "train":
        print("\nModel:")
        print(multidata_model_path)

    elif mode == "check_dataset":
        print("\nDataset:")
        print("hirise + hirise_original")

    print("\n" + "=" * 70)
