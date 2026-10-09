import os
import csv
import sys
import subprocess
import json
import re
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ==========================================================
# PIPELINE 3 - COMPLETE MARS FLOW ANALYSIS
#
# Pipeline 1:
#   SAM2 → U-Net Training / Prediction
#
# Pipeline 2:
#   Mask Cleaning
#   → Valley Detection
#   → Skeleton / Centerline
#   → Flow Direction
#   → Flow Reconstruction
#
# Pipeline 3:
#   Automatic orchestration of Pipeline 1 + Pipeline 2
#
# IMPORTANT:
#   - Pipeline 1 is NOT modified.
#   - Pipeline 2 is NOT modified.
#   - Existing prediction masks can be supplied.
#   - Missing SAM2 reference masks are generated automatically.
#   - Existing SAM2 masks are reused.
#   - SAM2 is NOT shown in the final visualization.
# ==========================================================


# ==========================================================
# 1. BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)


# ==========================================================
# SAM2 SUPPORT - REFERENCE MASK GENERATION
# ==========================================================
# Pipeline 3 uses SAM2 only to create a reference mask for
# evaluation when a reference mask does not already exist.
# This does NOT train or modify the U-Net.
SAM2_DIR = os.path.join(BASE_DIR, "sam2")

if SAM2_DIR not in sys.path:
    sys.path.insert(0, SAM2_DIR)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# ==========================================================
# 2. CHECK INPUT
# ==========================================================

if len(sys.argv) < 2 or len(sys.argv) > 3:

    print("\nUsage:")
    print(
        "python src/pipeline3.py "
        "<image_path> "
        "[prediction_mask_path]"
    )

    print("\nAutomatic mode example:")
    print(
        "python src/pipeline3.py "
        "data/raw/hirise_original/h3.jpg"
    )

    print("\nSupplied existing mask example:")
    print(
        "python src/pipeline3.py "
        "data/raw/hirise_original/h3.jpg "
        "outputs/prediction/h3_predicted.png"
    )

    sys.exit(1)


# ==========================================================
# 3. INPUT IMAGE
# ==========================================================

image_path = os.path.abspath(
    sys.argv[1]
)


if not os.path.exists(image_path):

    raise FileNotFoundError(
        f"\nInput image not found:\n"
        f"{image_path}"
    )


# ==========================================================
# 4. IMAGE INFORMATION
# ==========================================================

image_name_with_ext = os.path.basename(
    image_path
)

image_name = os.path.splitext(
    image_name_with_ext
)[0]

# Start timing the complete Pipeline 3 processing.
pipeline3_start_time = time.perf_counter()


# ==========================================================
# 5. PIPELINE FILES
# ==========================================================

PIPELINE1 = os.path.join(
    SRC_DIR,
    "pipeline.py"
)

PIPELINE2 = os.path.join(
    SRC_DIR,
    "pipeline2.py"
)


if not os.path.exists(PIPELINE1):

    raise FileNotFoundError(
        f"\nPipeline 1 not found:\n"
        f"{PIPELINE1}"
    )


if not os.path.exists(PIPELINE2):

    raise FileNotFoundError(
        f"\nPipeline 2 not found:\n"
        f"{PIPELINE2}"
    )


# ==========================================================
# 6. MODEL PATH
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models_saved",
    "unet_model.pth"
)


# ==========================================================
# 7. OUTPUT PATHS
# ==========================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ----------------------------------------------------------
# SAM2 MASK
# ----------------------------------------------------------

SAM2_MASK_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sam2_masks",
    image_name + ".png"
)


# ----------------------------------------------------------
# DEFAULT U-NET PREDICTION
# ----------------------------------------------------------

DEFAULT_PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    image_name + "_predicted.png"
)


# ----------------------------------------------------------
# SUPPLIED PREDICTION MASK
# ----------------------------------------------------------

SUPPLIED_PREDICTION_PATH = None


if len(sys.argv) == 3:

    SUPPLIED_PREDICTION_PATH = os.path.abspath(
        sys.argv[2]
    )

    if not os.path.exists(
        SUPPLIED_PREDICTION_PATH
    ):

        raise FileNotFoundError(
            f"\nSupplied prediction mask not found:\n"
            f"{SUPPLIED_PREDICTION_PATH}"
        )


# ----------------------------------------------------------
# FINAL PREDICTION PATH
# ----------------------------------------------------------

if SUPPLIED_PREDICTION_PATH is not None:

    PREDICTION_PATH = (
        SUPPLIED_PREDICTION_PATH
    )

else:

    PREDICTION_PATH = (
        DEFAULT_PREDICTION_PATH
    )


# ==========================================================
# 8. PIPELINE 2 OUTPUT PATHS
# ==========================================================

PIPELINE2_ROOT = os.path.join(
    OUTPUT_DIR,
    "pipeline2_improved"
)


CLEANED_MASK_PATH = os.path.join(
    PIPELINE2_ROOT,
    "cleaned_mask",
    image_name + "_cleaned.png"
)


VALLEY_PATH = os.path.join(
    PIPELINE2_ROOT,
    "valley_detection",
    image_name + "_valleys.png"
)


SKELETON_PATH = os.path.join(
    PIPELINE2_ROOT,
    "skeleton",
    image_name + "_skeleton.png"
)


DIRECTION_PATH = os.path.join(
    PIPELINE2_ROOT,
    "flow_direction",
    image_name + "_flow_direction.png"
)


RECONSTRUCTION_PATH = os.path.join(
    PIPELINE2_ROOT,
    "flow_reconstruction",
    image_name + "_flow_reconstruction.png"
)

# Pipeline 2 JSON status file.
PIPELINE2_JSON_PATH = os.path.join(
    PIPELINE2_ROOT,
    "json",
    image_name + "_pipeline2_improved.json"
)


# ==========================================================
# 9. PIPELINE 3 HEADER
# ==========================================================

print("\n")

print("=" * 75)

print(
    "       PIPELINE 3 - COMPLETE MARS FLOW ANALYSIS"
)

print("=" * 75)


print("\nInput Image:")
print(image_path)


print("\nImage Name:")
print(image_name)


print("\nU-Net Model:")
print(MODEL_PATH)


# ==========================================================
# 10. PREDICTION MODE
# ==========================================================

if SUPPLIED_PREDICTION_PATH is not None:

    print("\nPrediction Mode:")
    print("SUPPLIED EXISTING MASK")

    print("\nSupplied Prediction Mask:")
    print(SUPPLIED_PREDICTION_PATH)

else:

    print("\nPrediction Mode:")
    print("AUTOMATIC U-NET PREDICTION")


# ==========================================================
# 11. CHECK TRAINED U-NET MODEL
# ==========================================================

print("\n")

print("=" * 75)

print(
    "STEP 1 - CHECKING TRAINED U-NET MODEL"
)

print("=" * 75)


# ==========================================================
# CASE 1 - MODEL DOES NOT EXIST
# ==========================================================

if not os.path.exists(
    MODEL_PATH
):

    print(
        "\nNo trained U-Net model found."
    )

    print(
        "\nThis is the first model-training run."
    )

    print(
        "Pipeline 1 will train the U-Net."
    )

    print(
        "\nSAM2 → U-Net Training → Model Saving"
    )


    pipeline1_train = subprocess.run(
        [
            sys.executable,
            PIPELINE1,
            image_path,
            "train"
        ],
        cwd=BASE_DIR
    )


    if pipeline1_train.returncode != 0:

        raise RuntimeError(
            "\nPipeline 1 TRAIN failed.\n"
            "Pipeline 3 stopped."
        )


    # ------------------------------------------------------
    # Verify model
    # ------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nPipeline 1 finished, "
            "but U-Net model was not saved:\n"
            + MODEL_PATH
        )


    print(
        "\n✓ U-Net training completed."
    )

    print(
        "✓ U-Net model saved."
    )


# ==========================================================
# CASE 2 - MODEL EXISTS
# ==========================================================

else:

    print(
        "\n✓ Existing trained U-Net model found."
    )

    print(
        "✓ Training will NOT be performed."
    )

    print(
        "✓ Existing model will be reused."
    )


# ==========================================================
# 12. RUN / SKIP PIPELINE 1 PREDICTION
# ==========================================================

print("\n")

print("=" * 75)


if SUPPLIED_PREDICTION_PATH is not None:

    print(
        "STEP 1B - USING SUPPLIED PREDICTION MASK"
    )

else:

    print(
        "STEP 1B - RUNNING PIPELINE 1 PREDICTION"
    )


print("=" * 75)


# ==========================================================
# SUPPLIED MASK MODE
# ==========================================================

if SUPPLIED_PREDICTION_PATH is not None:

    print(
        "\n✓ Pipeline 1 prediction skipped."
    )

    print(
        "✓ Existing prediction mask will be used."
    )

    print(
        "\nPrediction Mask:"
    )

    print(
        PREDICTION_PATH
    )


# ==========================================================
# AUTOMATIC MODE
# ==========================================================

else:

    print(
        "\nLoading trained U-Net "
        "and generating prediction..."
    )


    pipeline1_predict = subprocess.run(
        [
            sys.executable,
            PIPELINE1,
            image_path,
            "predict"
        ],
        cwd=BASE_DIR
    )


    if pipeline1_predict.returncode != 0:

        raise RuntimeError(
            "\nPipeline 1 PREDICT failed.\n"
            "Pipeline 3 stopped."
        )


# ==========================================================
# 13. VERIFY PREDICTION MASK
# ==========================================================

if not os.path.exists(
    PREDICTION_PATH
):

    raise FileNotFoundError(
        "\nPrediction mask was not found:\n"
        + PREDICTION_PATH
    )


print(
    "\n✓ Prediction mask verified."
)

print(
    "\nPrediction mask:"
)

print(
    PREDICTION_PATH
)


# ==========================================================
# 14. RUN PIPELINE 2
# ==========================================================

print("\n")

print("=" * 75)

print(
    "STEP 2 - RUNNING PIPELINE 2"
)

print("=" * 75)


print(
    "\nCleaning → Valley Detection → "
    "Skeleton → Flow Direction → "
    "Flow Reconstruction\n"
)


pipeline2_result = subprocess.run(
    [
        sys.executable,
        PIPELINE2,
        image_path,
        PREDICTION_PATH
    ],
    cwd=BASE_DIR
)


if pipeline2_result.returncode != 0:

    raise RuntimeError(
        "\nPipeline 2 failed.\n"
        "Pipeline 3 stopped."
    )


print(
    "\n✓ Pipeline 2 completed successfully."
)


# ==========================================================
# 15. PIPELINE 2 OUTPUT PATHS
# ==========================================================

pipeline2_outputs = {

    "Cleaned Mask":
        CLEANED_MASK_PATH,

    "Valley Detection":
        VALLEY_PATH,

    "Skeleton":
        SKELETON_PATH,

    "Flow Direction":
        DIRECTION_PATH,

    "Flow Reconstruction":
        RECONSTRUCTION_PATH
}


# ==========================================================
# 16. VERIFY PIPELINE 2 OUTPUTS
# ==========================================================

print("\n")

print("=" * 75)

print(
    "CHECKING PIPELINE 2 OUTPUTS"
)

print("=" * 75)


all_outputs_found = True


for output_name, output_path in (
    pipeline2_outputs.items()
):

    if os.path.exists(
        output_path
    ):

        print(
            f"\n✓ {output_name}:"
        )

        print(
            output_path
        )

    else:

        print(
            f"\nWARNING: {output_name} "
            "not found:"
        )

        print(
            output_path
        )

        all_outputs_found = False


if not all_outputs_found:

    raise FileNotFoundError(
        "\nOne or more Pipeline 2 outputs "
        "were not created."
    )


# ==========================================================
# 17. LOAD IMAGE FUNCTION
# ==========================================================

def load_image(
    path,
    grayscale=False
):

    if not os.path.exists(
        path
    ):

        return None


    if grayscale:

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

    else:

        image = cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )


        if image is not None:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


    return image


# ==========================================================
# 17A. ENSURE SAM2 REFERENCE MASK EXISTS
# ==========================================================
# New/unseen images normally do not have a SAM2 mask yet.
# Generate it automatically only when it is missing.
# Existing SAM2 masks are reused and are never overwritten.

def ensure_sam2_reference_mask():
    if os.path.exists(SAM2_MASK_PATH):
        print("\n✓ Existing SAM2 reference mask found:")
        print(SAM2_MASK_PATH)
        return

    print("\n" + "=" * 75)
    print("GENERATING SAM2 REFERENCE MASK")
    print("=" * 75)
    print("\nNew image detected: no existing SAM2 reference mask found.")
    print("Generating SAM2 reference mask for evaluation...")
    print("Device:", DEVICE)

    checkpoint_path = os.path.join(
        BASE_DIR,
        "checkpoints",
        "sam2_hiera_tiny.pt"
    )
    config_name = "configs/sam2/sam2_hiera_t"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "\nSAM2 checkpoint not found:\n" + checkpoint_path
        )

    sam2_model = None

    try:
        print("\nLoading SAM2 Model...")
        sam2_model = build_sam2(
            config_file=config_name,
            ckpt_path=checkpoint_path,
            device=DEVICE
        )

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model
        )

        print("SAM2 Loaded Successfully")
        print("\nGenerating segmentation mask...")

        rgb = cv2.cvtColor(
            cv2.imread(image_path),
            cv2.COLOR_BGR2RGB
        )

        if rgb is None:
            raise RuntimeError(
                "Unable to read input image for SAM2 generation."
            )

        masks = mask_generator.generate(rgb)
        print("Masks Found :", len(masks))

        final_mask = np.zeros(
            rgb.shape[:2],
            dtype=np.uint8
        )

        for mask in masks:
            final_mask[mask["segmentation"]] = 255

        os.makedirs(
            os.path.dirname(SAM2_MASK_PATH),
            exist_ok=True
        )

        if not cv2.imwrite(SAM2_MASK_PATH, final_mask):
            raise RuntimeError(
                "Failed to save SAM2 reference mask:\n"
                + SAM2_MASK_PATH
            )

        print("\n✓ SAM2 Reference Mask Generated:")
        print(SAM2_MASK_PATH)

    finally:
        # Release SAM2 resources before continuing with visualization.
        try:
            del mask_generator
        except Exception:
            pass
        try:
            del sam2_model
        except Exception:
            pass
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# Generate/reuse the reference mask before evaluation.
ensure_sam2_reference_mask()


# ==========================================================
# 18. LOAD VISUALIZATION / EVALUATION IMAGES
#
# SAM2 is NOT displayed in the final visualization.
# It is loaded only as the reference mask for evaluation metrics.
# ==========================================================

original = load_image(
    image_path
)

sam2_mask = load_image(
    SAM2_MASK_PATH,
    grayscale=True
)


predicted = load_image(
    PREDICTION_PATH,
    grayscale=True
)


cleaned = load_image(
    CLEANED_MASK_PATH,
    grayscale=True
)


valleys = load_image(
    VALLEY_PATH
)


skeleton = load_image(
    SKELETON_PATH,
    grayscale=True
)


direction = load_image(
    DIRECTION_PATH
)


reconstruction = load_image(
    RECONSTRUCTION_PATH
)


# ==========================================================
# 19. CHECK ORIGINAL IMAGE
# ==========================================================

if original is None:

    raise RuntimeError(
        "\nUnable to load original image "
        "for visualization."
    )


original_height, original_width = (
    original.shape[:2]
)


# ==========================================================
# 20. RESIZE VISUALIZATION IMAGES
# ==========================================================

def resize_to_original(
    image
):

    if image is None:

        return None


    if image.shape[:2] == (
        original_height,
        original_width
    ):

        return image


    return cv2.resize(
        image,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


predicted = resize_to_original(
    predicted
)

sam2_mask = resize_to_original(
    sam2_mask
)


cleaned = resize_to_original(
    cleaned
)


valleys = resize_to_original(
    valleys
)


skeleton = resize_to_original(
    skeleton
)


direction = resize_to_original(
    direction
)


reconstruction = resize_to_original(
    reconstruction
)


# ==========================================================
# 21. CHECK VISUALIZATION IMAGES
# ==========================================================

visualization_images = {

    "U-Net Predicted Mask":
        predicted,

    "Cleaned Prediction Mask":
        cleaned,

    "Valley Detection":
        valleys,

    "Skeleton / Centerline":
        skeleton,

    "Flow Direction":
        direction,

    "Flow Reconstruction":
        reconstruction
}


for image_title, image_data in (
    visualization_images.items()
):

    if image_data is None:

        raise RuntimeError(
            "\nUnable to load visualization image:\n"
            + image_title
        )


# ==========================================================
# 22. CREATE FINAL COMBINED VISUALIZATION
#
# 3 ROWS × 3 COLUMNS
#
# SAM2 is intentionally NOT displayed.
#
# Row 1:
#   Original | U-Net Prediction | Cleaned Mask
#
# Row 2:
#   Valley Detection | Skeleton | Flow Direction / Status
#
# Row 3:
#   Flow Reconstruction / Status | Additional Information |
#   Detection Summary
#
# Evaluation metrics are calculated and saved separately, but
# intentionally not displayed in the final visualization.
# ==========================================================

print("\n")

print("=" * 75)

print(
    "STEP 3 - CREATING COMBINED VISUALIZATION"
)

print("=" * 75)


fig, axes = plt.subplots(
    3,
    3,
    figsize=(18, 12),
    constrained_layout=True
)


# ==========================================================
# MAIN TITLE
# ==========================================================

fig.suptitle(
    "AI-Based Identification of Past Water Flow on Mars",
    fontsize=18,
    fontweight="bold"
)


# ==========================================================
# HELPER - ADD PANEL SUBTITLE
# ==========================================================

def add_panel_subtitle(
    ax,
    text
):
    ax.text(
        0.5,
        -0.08,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold"
    )


# ==========================================================
# PANEL 1 - ORIGINAL MARS IMAGE
# ==========================================================

axes[0, 0].imshow(
    original
)

axes[0, 0].set_title(
    "1. Original Mars Image",
    fontsize=13,
    fontweight="bold"
)

add_panel_subtitle(
    axes[0, 0],
    "INPUT IMAGE"
)

axes[0, 0].axis(
    "off"
)


# ==========================================================
# PANEL 2 - U-NET PREDICTION
# ==========================================================

axes[0, 1].imshow(
    predicted,
    cmap="gray"
)

axes[0, 1].set_title(
    "2. U-Net Predicted Mask\n(Raw U-Net Prediction)",
    fontsize=13,
    fontweight="bold"
)

add_panel_subtitle(
    axes[0, 1],
    "OUTPUT FROM TRAINED U-NET"
)

axes[0, 1].axis(
    "off"
)


# ==========================================================
# PANEL 3 - CLEANED MASK
# ==========================================================

axes[0, 2].imshow(
    cleaned,
    cmap="gray"
)

axes[0, 2].set_title(
    "3. Cleaned Prediction Mask\n(After Post-processing)",
    fontsize=13,
    fontweight="bold"
)

add_panel_subtitle(
    axes[0, 2],
    "REFINED BINARY MASK"
)

axes[0, 2].axis(
    "off"
)


# ==========================================================
# PANEL 4 - VALLEY DETECTION
# ==========================================================

axes[1, 0].imshow(
    valleys
)

axes[1, 0].set_title(
    "4. Valley Detection",
    fontsize=13,
    fontweight="bold"
)

add_panel_subtitle(
    axes[1, 0],
    "DETECTED VALLEY / CANDIDATE REGIONS"
)

axes[1, 0].axis(
    "off"
)


# ==========================================================
# PANEL 5 - SKELETON / CENTERLINE
# ==========================================================

axes[1, 1].imshow(
    skeleton,
    cmap="gray"
)

axes[1, 1].set_title(
    "5. Skeleton / Centerline",
    fontsize=13,
    fontweight="bold"
)

add_panel_subtitle(
    axes[1, 1],
    "EXTRACTED FLOW CENTERLINE"
)

axes[1, 1].axis(
    "off"
)


# ==========================================================
# 21A. EVALUATION METRICS + ADDITIONAL INFORMATION
# ==========================================================

def _to_binary_mask(image):
    """Convert a grayscale mask to a boolean binary mask."""
    if image is None:
        return None

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image > 127


def calculate_segmentation_metrics(prediction, reference):
    """
    Calculate segmentation metrics using the same preprocessing
    used by Pipeline 1: both masks are resized to 256x256 and
    thresholded at 127 before comparison.

    SAM2 is a reference mask, not manual ground truth.
    """
    if prediction is None or reference is None:
        return None

    try:
        if prediction.ndim == 3:
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
        if reference.ndim == 3:
            reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

        pred = cv2.resize(prediction, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
        ref = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_NEAREST) > 127

        tp = np.logical_and(pred, ref).sum()
        tn = np.logical_and(~pred, ~ref).sum()
        fp = np.logical_and(pred, ~ref).sum()
        fn = np.logical_and(~pred, ref).sum()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 1.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        return {
            "accuracy": accuracy * 100.0,
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
            "iou": iou * 100.0,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "evaluation_size": "256 x 256"
        }
    except Exception as exc:
        print(f"\nWarning: evaluation failed: {exc}")
        return None


def _find_first_numeric_for_keys(obj, key_names):
    """
    Recursively find the first numeric value whose JSON key
    matches one of the supplied names.
    """
    normalized_targets = {_normalize_key(k) for k in key_names}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if _normalize_key(key) in normalized_targets:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value

                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass

            found = _find_first_numeric_for_keys(value, key_names)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_numeric_for_keys(item, key_names)
            if found is not None:
                return found

    return None


def get_path_information():
    """
    Get the best reconstructed flow information from Pipeline 2.

    Pipeline 2 stores one record per reconstructed valley inside
    ``flow_results``. The main flow length is stored as
    ``length_pixels`` and the corresponding confidence as
    ``confidence``.

    Best Path Length     = longest reconstructed flow length (pixels)
    Best Path Confidence = confidence of that same longest flow
    """

    best_path_length = None
    best_path_confidence = None

    if not os.path.exists(PIPELINE2_JSON_PATH):
        return best_path_length, best_path_confidence

    try:
        with open(PIPELINE2_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Pipeline 2 stores the reconstructed flows here.
        flow_results = data.get("flow_results", []) if isinstance(data, dict) else []

        if isinstance(flow_results, list):
            valid_flows = []

            for flow in flow_results:
                if not isinstance(flow, dict):
                    continue

                # Current Pipeline 2 field names.
                length_value = flow.get("length_pixels")
                confidence_value = flow.get("confidence")

                # Compatibility with older flow-result formats.
                if length_value is None:
                    length_value = flow.get("length")

                if confidence_value is None:
                    confidence_value = flow.get("path_confidence")

                try:
                    if length_value is None:
                        continue

                    length_value = float(length_value)

                    if confidence_value is not None:
                        confidence_value = float(confidence_value)

                    valid_flows.append({
                        "length": length_value,
                        "confidence": confidence_value
                    })

                except (TypeError, ValueError):
                    continue

            # Select the longest reconstructed flow.
            if valid_flows:
                best_path = max(
                    valid_flows,
                    key=lambda x: x["length"]
                )

                best_path_length = best_path["length"]
                best_path_confidence = best_path["confidence"]

    except Exception as exc:
        print(
            f"\nWarning: could not read best path information: {exc}"
        )

    return best_path_length, best_path_confidence


# Calculate evaluation metrics only if the SAM2 reference mask exists.
evaluation_metrics = calculate_segmentation_metrics(
    predicted,
    sam2_mask
)

best_path_length, best_path_confidence = get_path_information()

# Processing time is measured after Pipeline 1 + Pipeline 2 complete.
processing_time = time.perf_counter() - pipeline3_start_time


# ==========================================================
# PANEL 6 / 7 STATUS HELPERS
# ==========================================================
#
# When Pipeline 2 has no valid result, do NOT show the
# unchanged original image by itself. Instead, show the
# original image as the background and place a clear
# "not available" status overlay on top.
#
# This is visualization-only. Pipeline 1 and Pipeline 2
# processing are not modified.
# ==========================================================

def _normalize_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_numeric_status(obj, key_names):
    """Recursively find the first numeric value for matching JSON keys."""
    normalized_targets = {_normalize_key(k) for k in key_names}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if _normalize_key(key) in normalized_targets:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value
                if isinstance(value, list):
                    return len(value)
            found = _find_numeric_status(value, key_names)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_numeric_status(item, key_names)
            if found is not None:
                return found

    return None




def _find_flow_paths_reconstructed(data):
    """Return Pipeline 2's TOTAL reconstructed-flow count only.

    Pipeline 2 reports per-valley ``accepted_paths`` values as well as the
    overall ``Flow paths reconstructed`` value. Never use a per-valley count
    as the overall flow-path count.
    """
    value = _find_numeric_status(
        data,
        [
            # Pipeline 2 stores the overall number of reconstructed flows
            # under this summary key.
            "reconstructed_valleys",
            "flow_paths_reconstructed",
            "flow paths reconstructed",
            "reconstructed_flow_paths",
        ]
    )
    if value is not None:
        return int(value)

    # Handle JSON structures where the summary key is nested under an
    # unexpected object but serialized with the exact human-readable label.
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                normalized = _normalize_key(key)
                if normalized in {
                    "flowpathsreconstructed",
                    "reconstructedflowpaths",
                } and isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = walk(item)
                if found is not None:
                    return found
        return None

    return walk(data)


def _image_has_overlay(base_image, output_image):
    """
    Fallback check for Pipeline 2 images that are normally the
    original image when no annotation/path was generated.
    """
    if base_image is None or output_image is None:
        return False

    if base_image.shape[:2] != output_image.shape[:2]:
        return False

    a = base_image.astype(np.int16)
    b = output_image.astype(np.int16)

    difference = np.abs(a - b)

    # Count pixels with a meaningful annotation-level difference.
    changed_pixels = np.any(difference > 20, axis=2).sum()
    total_pixels = base_image.shape[0] * base_image.shape[1]

    return changed_pixels > max(20, int(total_pixels * 0.00001))


def get_pipeline2_status():
    """
    Determine whether Pipeline 2 produced:
      - at least one accepted valley
      - at least one reconstructed flow path

    JSON is preferred. If its structure differs or is unavailable,
    the visualization images are used as a safe fallback.
    """
    accepted_valleys = None
    accepted_paths = None

    if os.path.exists(PIPELINE2_JSON_PATH):
        try:
            with open(PIPELINE2_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            accepted_valleys = _find_numeric_status(
                data,
                [
                    "accepted_valleys",
                    "valleys_accepted",
                    "accepted valleys",
                ]
            )

            # IMPORTANT: use the Pipeline 2 TOTAL number of reconstructed
            # flow paths, not the first per-valley "accepted_paths" value.
            # A recursive search for "accepted_paths" can incorrectly return
            # the first valley's count (for example, 1) even when 5 valleys
            # produced 5 reconstructed flow paths.
            accepted_paths = _find_flow_paths_reconstructed(data)
        except Exception:
            # Use image comparison fallback below.
            pass

    # Fallback based on the actual Pipeline 2 visualization images.
    if accepted_valleys is None:
        accepted_valleys = 1 if _image_has_overlay(original, valleys) else 0

    if accepted_paths is None:
        accepted_paths = 1 if _image_has_overlay(original, reconstruction) else 0

    return int(accepted_valleys), int(accepted_paths)


accepted_valleys_count, accepted_paths_count = get_pipeline2_status()


def show_no_result_panel(ax, background, message_lines):
    """
    Show the original image as the panel background with a
    clear dark translucent status box and a prohibition icon.
    """
    ax.imshow(background)

    # Dark translucent message box.
    ax.text(
        0.50,
        0.50,
        "\n".join(message_lines),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.75",
            facecolor="black",
            edgecolor="none",
            alpha=0.72
        ),
        zorder=5
    )

    # Prohibition icon above the message box.
    icon_center = (0.50, 0.70)
    icon_radius = 0.075

    circle = Circle(
        icon_center,
        icon_radius,
        transform=ax.transAxes,
        fill=False,
        edgecolor="white",
        linewidth=4,
        zorder=6
    )
    ax.add_patch(circle)

    ax.plot(
        [
            icon_center[0] - icon_radius * 0.70,
            icon_center[0] + icon_radius * 0.70
        ],
        [
            icon_center[1] + icon_radius * 0.70,
            icon_center[1] - icon_radius * 0.70
        ],
        transform=ax.transAxes,
        color="white",
        linewidth=4,
        solid_capstyle="round",
        zorder=6
    )


# ==========================================================
# PANEL 6 - FLOW DIRECTION
# ==========================================================

axes[1, 2].set_title(
    "6. Flow Direction",
    fontsize=13,
    fontweight="bold"
)

if accepted_valleys_count > 0:
    axes[1, 2].imshow(direction)
else:
    show_no_result_panel(
        axes[1, 2],
        original,
        [
            "No accepted valley",
            "Flow direction unavailable"
        ]
    )

add_panel_subtitle(
    axes[1, 2],
    "ESTIMATED FLOW DIRECTION"
)

axes[1, 2].axis("off")


# ==========================================================
# PANEL 7 - FLOW RECONSTRUCTION
# ==========================================================

axes[2, 0].set_title(
    "7. Flow Reconstruction",
    fontsize=13,
    fontweight="bold"
)

if accepted_paths_count > 0:
    axes[2, 0].imshow(reconstruction)
else:
    show_no_result_panel(
        axes[2, 0],
        original,
        [
            "No flow path reconstructed"
        ]
    )

add_panel_subtitle(
    axes[2, 0],
    "RECONSTRUCTED FLOW PATHS"
)

axes[2, 0].axis("off")


# ==========================================================
# PANEL 8 - ADDITIONAL INFORMATION
# ==========================================================

axes[2, 1].axis("off")

axes[2, 1].set_title(
    "8. Additional Information",
    fontsize=13,
    fontweight="bold"
)

model_used_text = os.path.basename(MODEL_PATH)

# ----------------------------------------------------------
# SAM2 REFERENCE MASK STATUS
# ----------------------------------------------------------
sam2_status_text = (
    "Available"
    if sam2_mask is not None
    else "Not Available"
)

# ----------------------------------------------------------
# U-NET PREDICTION STATUS
# ----------------------------------------------------------
# A prediction file can exist even when it contains no foreground
# pixels. Therefore, status is based on actual mask content.
if SUPPLIED_PREDICTION_PATH is not None:
    prediction_status_text = "Supplied existing mask"
elif predicted is not None and np.count_nonzero(predicted) > 0:
    prediction_status_text = "Generated"
else:
    prediction_status_text = "No features"

# ----------------------------------------------------------
# CLEANED MASK STATUS
# ----------------------------------------------------------
# File existence alone does not mean a usable feature was found.
if cleaned is not None and np.count_nonzero(cleaned) > 0:
    cleaned_status_text = "Features present"
else:
    cleaned_status_text = "No features"

# ----------------------------------------------------------
# VALLEY DETECTION STATUS
# ----------------------------------------------------------
if accepted_valleys_count > 0:
    valley_status_text = f"{accepted_valleys_count} detected"
else:
    valley_status_text = "No valleys detected"

# ----------------------------------------------------------
# SKELETON / CENTERLINE STATUS
# ----------------------------------------------------------
if skeleton is not None and np.count_nonzero(skeleton) > 0:
    skeleton_status_text = "Centerline detected"
else:
    skeleton_status_text = "No centerline"

# ----------------------------------------------------------
# FLOW DIRECTION STATUS
# ----------------------------------------------------------
# Flow direction is meaningful only when an accepted valley exists.
if accepted_valleys_count > 0 and direction is not None:
    direction_status_text = "Available"
else:
    direction_status_text = "Not Available"

# ----------------------------------------------------------
# FLOW RECONSTRUCTION STATUS
# ----------------------------------------------------------
# Reconstruction is meaningful only when at least one flow path
# was actually reconstructed.
if accepted_paths_count > 0 and reconstruction is not None:
    reconstruction_status_text = (
        f"{accepted_paths_count} path(s) reconstructed"
    )
else:
    reconstruction_status_text = "Not Available"

best_path_length_text = (
    f"{best_path_length:,.2f} px"
    if best_path_length is not None
    else "N/A"
)

best_path_confidence_text = (
    f"{best_path_confidence:.2f}%"
    if best_path_confidence is not None
    else "N/A"
)

additional_info_lines = [
    ("Image Name", image_name_with_ext),
    (
        "Image Size",
        f"{original_width} × {original_height}"
    ),
    ("Model Used", model_used_text),
    ("SAM2 Mask", sam2_status_text),
    ("Prediction Mask", prediction_status_text),
    ("Cleaned Mask", cleaned_status_text),
    ("Valley Detection", valley_status_text),
    ("Skeleton / Centerline", skeleton_status_text),
    ("Flow Direction", direction_status_text),
    ("Flow Reconstruction", reconstruction_status_text),
    (
        "Number of Valleys",
        str(accepted_valleys_count)
    ),
    (
        "Number of Paths",
        str(accepted_paths_count)
    ),
    (
        "Best Path Length",
        best_path_length_text
    ),
    (
        "Best Path Confidence",
        best_path_confidence_text
    ),
    (
        "Processing Time",
        f"{processing_time:.1f} s"
    )
]

additional_info_text = ""

for label, value in additional_info_lines:
    additional_info_text += (
        f"{label:<20}: {value}\n"
    )

axes[2, 1].text(
    0.05,
    0.93,
    additional_info_text.rstrip(),
    transform=axes[2, 1].transAxes,
    ha="left",
    va="top",
    fontsize=8.9,
    family="monospace",
    linespacing=1.28
)


# ==========================================================
# PANEL 9 - DETECTION SUMMARY
# ==========================================================

axes[2, 2].axis("off")

axes[2, 2].set_title(
    "9. Detection Summary",
    fontsize=13,
    fontweight="bold"
)

summary_lines = [
    ("Valleys Detected", str(accepted_valleys_count)),
    ("Flow Paths", str(accepted_paths_count)),
    ("Best Path Length", best_path_length_text),
    ("Best Path Confidence", best_path_confidence_text),
    ("Skeleton Pixels", f"{int(np.count_nonzero(skeleton)):,}"),
    ("Processing Time", f"{processing_time:.1f} s"),
    ("Analysis Status", "Completed")
]

summary_text = ""
for label, value in summary_lines:
    summary_text += f"{label:<20}: {value}\n"

axes[2, 2].text(
    0.08,
    0.82,
    summary_text.rstrip(),
    transform=axes[2, 2].transAxes,
    ha="left",
    va="top",
    fontsize=11,
    family="monospace",
    linespacing=1.65,
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="lightgray",
        edgecolor="gray",
        alpha=0.30
    )
)

axes[2, 2].text(
    0.50,
    0.14,
    "Evaluation metrics are saved separately\n"
    "for research-paper analysis.",
    transform=axes[2, 2].transAxes,
    ha="center",
    va="center",
    fontsize=9.5,
    linespacing=1.35
)


# ==========================================================
# 23. SAVE PIPELINE 3 VISUALIZATION
# ==========================================================

PIPELINE3_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "pipeline3"
)


os.makedirs(
    PIPELINE3_OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# 23A. SAVE EVALUATION METRICS SEPARATELY
# ==========================================================

EVALUATION_DIR = os.path.join(
    PIPELINE3_OUTPUT_DIR,
    "evaluation"
)

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


evaluation_json_path = os.path.join(
    EVALUATION_DIR,
    image_name + "_metrics.json"
)


evaluation_record = {
    "image_name": image_name_with_ext,
    "image_size": f"{original_width} x {original_height}",
    "reference_mask": "SAM2-generated mask",
    "reference_type": "reference mask (not manual ground truth)",
    "evaluation_size": "256 x 256",
    "metrics": evaluation_metrics
}


with open(
    evaluation_json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        evaluation_record,
        f,
        indent=4
    )


# Also maintain a cumulative CSV for all evaluated images.
evaluation_csv_path = os.path.join(
    EVALUATION_DIR,
    "evaluation_metrics.csv"
)

if evaluation_metrics is not None:

    csv_fields = [
        "image_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluation_size"
    ]

    csv_row = {
        "image_name": image_name_with_ext,
        **{k: evaluation_metrics.get(k) for k in csv_fields if k != "image_name"}
    }

    existing_rows = []
    if os.path.exists(evaluation_csv_path):
        try:
            with open(evaluation_csv_path, "r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        except Exception:
            existing_rows = []

    existing_rows = [
        row for row in existing_rows
        if row.get("image_name") != image_name_with_ext
    ]
    existing_rows.append(csv_row)

    with open(
        evaluation_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(existing_rows)

print("\n✓ Evaluation metrics saved:")
print(evaluation_json_path)
if evaluation_metrics is not None:
    print(evaluation_csv_path)


COMBINED_VISUALIZATION_PATH = os.path.join(
    PIPELINE3_OUTPUT_DIR,
    image_name +
    "_pipeline3_visualization.png"
)


plt.savefig(
    COMBINED_VISUALIZATION_PATH,
    dpi=150,
    bbox_inches="tight"
)


print(
    "\n✓ Combined visualization saved:"
)


print(
    COMBINED_VISUALIZATION_PATH
)


# ==========================================================
# 24. SHOW FINAL VISUALIZATION
# ==========================================================

# print(
#     "\nOpening final visualization window..."
# )
#
#
# plt.show()


# ==========================================================
# 25. FINAL SUMMARY
# ==========================================================

print("\n")

print("=" * 75)

print(
    "        PIPELINE 3 COMPLETED SUCCESSFULLY"
)

print("=" * 75)


# ==========================================================
# INPUT
# ==========================================================

print("\nINPUT")

print("-" * 75)


print(
    "Original Image :",
    image_path
)


# ==========================================================
# MODEL
# ==========================================================

print("\nMODEL")

print("-" * 75)


print(
    "U-Net Model    :",
    MODEL_PATH
)


print(
    "Training       :",
    "Only when model was missing"
)


# ==========================================================
# PIPELINE 1
# ==========================================================

print("\nPIPELINE 1")

print("-" * 75)


print(
    "SAM2 Mask      :",
    SAM2_MASK_PATH
)


print(
    "Predicted Mask :",
    PREDICTION_PATH
)


if SUPPLIED_PREDICTION_PATH is not None:

    print(
        "Prediction     :",
        "Supplied existing mask"
    )

else:

    print(
        "Prediction     :",
        "Generated by Pipeline 1"
    )


# ==========================================================
# PIPELINE 2
# ==========================================================

print("\nPIPELINE 2")

print("-" * 75)


for output_name, output_path in (
    pipeline2_outputs.items()
):

    print(
        f"{output_name:<22}:"
        f" {output_path}"
    )


# ==========================================================
# PIPELINE 3
# ==========================================================

print("\nPIPELINE 3")

print("-" * 75)


print(
    "Combined Visualization :",
    COMBINED_VISUALIZATION_PATH
)

print(
    "Processing Time         :",
    f"{processing_time:.1f} s"
)


print("\n")

print("=" * 75)

print(
    "FULL MARS FLOW ANALYSIS FINISHED"
)

print("=" * 75)
