# ==========================================================
# Automated Valley Detection
# Fixed Threshold + Improved Cleaning and Filtering
# ==========================================================

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize


# ==========================================================
# 1. INPUT IMAGE
# ==========================================================

image_name = "h4"       # Enter filename WITHOUT extension

input_folder = "data/raw/hirise_original"

supported_formats = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp"
]

raw_path = None


# Search for the image in any supported format
for ext in supported_formats:

    candidate = os.path.join(
        input_folder,
        image_name + ext
    )

    if os.path.exists(candidate):

        raw_path = candidate
        break


if raw_path is None:

    raise FileNotFoundError(
        f"No image found for '{image_name}' "
        f"in {input_folder}"
    )


# ==========================================================
# 2. TEMPORARY JPG CONVERSION
# ==========================================================

temp_jpg = os.path.join(
    input_folder,
    f"{image_name}_temp.jpg"
)

img = cv2.imread(raw_path)

if img is None:

    raise FileNotFoundError(raw_path)

cv2.imwrite(
    temp_jpg,
    img
)

raw_path = temp_jpg


# ==========================================================
# 3. PREDICTED MASK PATH
# ==========================================================

pred_path = os.path.join(
    "outputs",
    "prediction",
    f"{image_name}_predicted.png"
)


# ==========================================================
# 4. OUTPUT FOLDER
# ==========================================================

output_folder = "outputs/valley_detection"

os.makedirs(
    output_folder,
    exist_ok=True
)


binary_path = os.path.join(
    output_folder,
    f"{image_name}_binary.png"
)

clean_path = os.path.join(
    output_folder,
    f"{image_name}_clean.png"
)

valley_path = os.path.join(
    output_folder,
    f"{image_name}_valleys.png"
)

skeleton_path = os.path.join(
    output_folder,
    f"{image_name}_skeleton_green.png"
)

result_path = os.path.join(
    output_folder,
    f"{image_name}_results.png"
)


# ==========================================================
# 5. LOAD IMAGES
# ==========================================================

raw = cv2.imread(
    raw_path
)

pred = cv2.imread(
    pred_path,
    cv2.IMREAD_GRAYSCALE
)


if raw is None:

    raise FileNotFoundError(
        raw_path
    )


if pred is None:

    raise FileNotFoundError(
        pred_path
    )


# ==========================================================
# 6. FIXED THRESHOLDING
# ==========================================================
#
# We are intentionally using the original fixed threshold.
#
# White  = detected prediction
# Black  = background
#
# This gives consistent interpretation across images.
# ==========================================================

THRESHOLD_VALUE = 127


_, binary = cv2.threshold(
    pred,
    THRESHOLD_VALUE,
    255,
    cv2.THRESH_BINARY
)


# Save binary mask

cv2.imwrite(
    binary_path,
    binary
)


# ==========================================================
# 7. MORPHOLOGICAL CLEANING
# ==========================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


# ----------------------------------------------------------
# Opening
# ----------------------------------------------------------
# Removes small isolated noise.

clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)


# ----------------------------------------------------------
# Closing
# ----------------------------------------------------------
# Fills small gaps and connects nearby regions.

clean = cv2.morphologyEx(
    clean,
    cv2.MORPH_CLOSE,
    kernel
)


# ==========================================================
# 8. CONNECTED COMPONENT FILTERING
# ==========================================================
#
# Remove small isolated connected regions.
# ==========================================================

NUM_LABELS, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )
)


COMPONENT_MIN_AREA = 150


component_clean = np.zeros_like(
    clean
)


for label in range(
    1,
    NUM_LABELS
):

    component_area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if component_area >= COMPONENT_MIN_AREA:

        component_clean[
            labels == label
        ] = 255


clean = component_clean


# Save clean mask

cv2.imwrite(
    clean_path,
    clean
)


# ==========================================================
# 9. FIND CONTOURS
# ==========================================================

contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ==========================================================
# 10. VALLEY FILTERING CONDITIONS
# ==========================================================

MIN_AREA = 150

MIN_ASPECT_RATIO = 2.5

MAX_CIRCULARITY = 0.45

MIN_CONFIDENCE = 40


filtered = []


print(
    "\n=============================================="
)

print(
    "        Candidate Valley Analysis"
)

print(
    "==============================================\n"
)


# ==========================================================
# 11. CONTOUR ANALYSIS
# ==========================================================

for cnt in contours:

    # ------------------------------------------------------
    # Area
    # ------------------------------------------------------

    area = cv2.contourArea(
        cnt
    )

    if area < MIN_AREA:

        continue


    # ------------------------------------------------------
    # Perimeter
    # ------------------------------------------------------

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if perimeter == 0:

        continue


    # ------------------------------------------------------
    # Bounding Box
    # ------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        cnt
    )


    # ------------------------------------------------------
    # Border Filtering
    # ------------------------------------------------------

    margin = 5


    if (
        x <= margin or
        y <= margin or
        x + w >= clean.shape[1] - margin or
        y + h >= clean.shape[0] - margin
    ):

        continue


    # ------------------------------------------------------
    # Aspect Ratio
    # ------------------------------------------------------

    aspect_ratio = (
        max(w, h)
        /
        max(min(w, h), 1)
    )


    # ------------------------------------------------------
    # Circularity
    # ------------------------------------------------------

    circularity = (
        4 * np.pi * area
    ) / (
        perimeter * perimeter
    )


    # ======================================================
    # 12. CONFIDENCE SCORE
    # ======================================================

    confidence = 0


    # ------------------------------------------------------
    # Aspect ratio contribution
    # ------------------------------------------------------

    if aspect_ratio >= 5:

        confidence += 40

    elif aspect_ratio >= 3:

        confidence += 25


    # ------------------------------------------------------
    # Circularity contribution
    # ------------------------------------------------------

    if circularity <= 0.20:

        confidence += 40

    elif circularity <= 0.35:

        confidence += 25


    # ------------------------------------------------------
    # Area contribution
    # ------------------------------------------------------

    if area >= 300:

        confidence += 20


    confidence = min(
        confidence,
        100
    )


    # ------------------------------------------------------
    # Print candidate information
    # ------------------------------------------------------

    print(
        f"Area={area:7.1f} | "
        f"Aspect={aspect_ratio:5.2f} | "
        f"Circularity={circularity:5.2f} | "
        f"Confidence={confidence:3d}% | "
        f"Bounding Box={w}x{h}"
    )


    # ======================================================
    # 13. FINAL VALLEY SELECTION
    # ======================================================
    #
    # Candidate must satisfy:
    #
    # 1. Minimum area
    # 2. Minimum aspect ratio
    # 3. Maximum circularity
    # 4. Minimum confidence
    #

    if (
        area >= MIN_AREA and
        aspect_ratio >= MIN_ASPECT_RATIO and
        circularity <= MAX_CIRCULARITY and
        confidence >= MIN_CONFIDENCE
    ):

        filtered.append({

            "contour": cnt,

            "area": area,

            "perimeter": perimeter,

            "x": x,

            "y": y,

            "w": w,

            "h": h,

            "aspect_ratio": aspect_ratio,

            "circularity": circularity,

            "confidence": confidence

        })


# ==========================================================
# 14. DETECTION SUMMARY
# ==========================================================

print(
    "\n----------------------------------------------"
)

print(
    f"Valley Candidates Selected : "
    f"{len(filtered)}"
)

print(
    "----------------------------------------------"
)


# ==========================================================
# 15. DRAW DETECTED VALLEYS
# ==========================================================

overlay = raw.copy()


for i, valley in enumerate(
    filtered
):

    cnt = valley["contour"]

    x = valley["x"]
    y = valley["y"]
    w = valley["w"]
    h = valley["h"]


    # ------------------------------------------------------
    # Dynamic contour thickness
    # ------------------------------------------------------

    if w > 250 or h > 250:

        thickness = 3

    elif w > 120 or h > 120:

        thickness = 2

    else:

        thickness = 1


    # ------------------------------------------------------
    # Draw detected valley contour
    # ------------------------------------------------------

    cv2.drawContours(
        overlay,
        [cnt],
        -1,
        (0, 255, 0),
        thickness
    )


    # ------------------------------------------------------
    # Draw bounding box
    # ------------------------------------------------------

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )


    # ------------------------------------------------------
    # Valley label
    # ------------------------------------------------------

    cv2.putText(
        overlay,
        f"V{i + 1}",
        (
            x,
            max(y - 5, 15)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    valley_path,
    overlay
)


# ==========================================================
# 16. SKELETON GENERATION
# ==========================================================
#
# IMPORTANT:
# The skeleton remains binary internally.
# ==========================================================

binary_bool = clean > 0


skeleton = skeletonize(
    binary_bool
)


# Convert to uint8 binary image

skeleton = (
    skeleton * 255
).astype(np.uint8)


# ==========================================================
# 17. GREEN SKELETON VISUALIZATION
# ==========================================================
#
# Black background
# Green skeleton lines
# ==========================================================

skeleton_color = np.zeros(
    (
        skeleton.shape[0],
        skeleton.shape[1],
        3
    ),
    dtype=np.uint8
)


# OpenCV uses BGR
# (0, 255, 0) = GREEN

skeleton_color[
    skeleton > 0
] = (
    0,
    255,
    0
)


# Save green skeleton image

cv2.imwrite(
    skeleton_path,
    skeleton_color
)


# ==========================================================
# 18. VALLEY LENGTH ESTIMATION
# ==========================================================

PIXEL_SIZE = 0.25


for valley in filtered:

    valley_mask = np.zeros_like(
        clean
    )


    cv2.drawContours(
        valley_mask,
        [valley["contour"]],
        -1,
        255,
        thickness=cv2.FILLED
    )


    # ------------------------------------------------------
    # Use BINARY skeleton for calculation
    # ------------------------------------------------------

    valley_skeleton = cv2.bitwise_and(
        skeleton,
        valley_mask
    )


    length_pixels = cv2.countNonZero(
        valley_skeleton
    )


    valley["length"] = (
        length_pixels *
        PIXEL_SIZE
    )


# ==========================================================
# 19. VALLEY STATISTICS
# ==========================================================

print(
    "\n=============================================="
)

print(
    "           Valley Detection Results"
)

print(
    "==============================================\n"
)


print(
    f"Detected Valleys : "
    f"{len(filtered)}\n"
)


for i, valley in enumerate(
    filtered
):

    print(
        f"Valley {i + 1}"
    )


    print(
        "--------------------------------------"
    )


    # Area conversion

    area_m2 = (
        valley["area"] *
        (PIXEL_SIZE ** 2)
    )


    # Perimeter conversion

    perimeter_m = (
        valley["perimeter"] *
        PIXEL_SIZE
    )


    print(
        f"Area         : "
        f"{area_m2:.2f} m²"
    )


    print(
        f"Perimeter    : "
        f"{perimeter_m:.2f} m"
    )


    print(
        f"Bounding Box : "
        f"({valley['x']},"
        f"{valley['y']}) "
        f"{valley['w']}x"
        f"{valley['h']}"
    )


    print(
        f"Aspect Ratio : "
        f"{valley['aspect_ratio']:.2f}"
    )


    print(
        f"Circularity  : "
        f"{valley['circularity']:.2f}"
    )


    print(
        f"Confidence   : "
        f"{valley['confidence']}%"
    )


    print(
        f"Length       : "
        f"{valley['length']:.2f} m"
    )


    print()


# ==========================================================
# 20. FINAL VISUALIZATION
# ==========================================================

plt.figure(
    figsize=(16, 10)
)


# ==========================================================
# Original Image
# ==========================================================

plt.subplot(
    2,
    3,
    1
)

plt.imshow(
    cv2.cvtColor(
        raw,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Original Image"
)

plt.axis(
    "off"
)


# ==========================================================
# Predicted Mask
# ==========================================================

plt.subplot(
    2,
    3,
    2
)

plt.imshow(
    pred,
    cmap="gray"
)

plt.title(
    "Predicted Mask"
)

plt.axis(
    "off"
)


# ==========================================================
# Binary Mask
# ==========================================================

plt.subplot(
    2,
    3,
    3
)

plt.imshow(
    binary,
    cmap="gray"
)

plt.title(
    "Binary Mask"
)

plt.axis(
    "off"
)


# ==========================================================
# Clean Mask
# ==========================================================

plt.subplot(
    2,
    3,
    4
)

plt.imshow(
    clean,
    cmap="gray"
)

plt.title(
    "Clean Mask"
)

plt.axis(
    "off"
)


# ==========================================================
# Detected Valleys
# ==========================================================

plt.subplot(
    2,
    3,
    5
)

plt.imshow(
    cv2.cvtColor(
        raw,
        cv2.COLOR_BGR2RGB
    )
)


for i, valley in enumerate(
    filtered
):

    cnt = valley["contour"].squeeze()


    if cnt.ndim == 2:

        plt.plot(
            cnt[:, 0],
            cnt[:, 1],
            color="lime",
            linewidth=1
        )


    plt.text(
        valley["x"],
        max(
            valley["y"] - 5,
            5
        ),
        f"V{i + 1}",
        color="yellow",
        fontsize=8,
        weight="bold"
    )


plt.title(
    f"Detected Valleys ({len(filtered)})"
)

plt.axis(
    "off"
)


# ==========================================================
# GREEN SKELETON
# ==========================================================

plt.subplot(
    2,
    3,
    6
)

plt.imshow(
    cv2.cvtColor(
        skeleton_color,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Skeleton"
)

plt.axis(
    "off"
)


# ==========================================================
# Layout
# ==========================================================

plt.tight_layout()


# ==========================================================
# 21. SAVE FINAL RESULT
# ==========================================================

plt.savefig(
    result_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ==========================================================
# 22. FINAL SUMMARY
# ==========================================================

print(
    f"\nImage Name          : "
    f"{image_name}"
)

print(
    f"Total Contours      : "
    f"{len(contours)}"
)

print(
    f"Detected Valleys    : "
    f"{len(filtered)}"
)


print(
    "\nSaved Files"
)

print(
    "----------------------------------------------"
)

print(
    binary_path
)

print(
    clean_path
)

print(
    valley_path
)

print(
    skeleton_path
)

print(
    result_path
)

print(
    "----------------------------------------------"
)


print(
    "\n=============================================="
)

print(
    " Valley Detection Completed Successfully"
)

print(
    "=============================================="
)


# ==========================================================
# 23. REMOVE TEMPORARY JPG
# ==========================================================

if os.path.exists(
    temp_jpg
):

    os.remove(
        temp_jpg
)