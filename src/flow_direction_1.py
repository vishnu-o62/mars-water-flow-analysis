# new version updated one on 09-07-26

import os
from unittest import result

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
# import os
#
# print("Running:", os.path.abspath(__file__))
# ==========================================================
# Draw Dense Flow Arrows Inside Valley
# ==========================================================

def draw_flow_field(image,
                    contour,
                    start,
                    end,
                    spacing=12):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, -1)

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = np.sqrt(dx*dx + dy*dy)

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    arrow_length = 8

    for y in range(0, image.shape[0], spacing):

        for x in range(0, image.shape[1], spacing):

            if mask[y, x] == 0:
                continue

            p1 = (x, y)

            p2 = (
                int(x + ux * arrow_length),
                int(y + uy * arrow_length)
            )

            cv2.arrowedLine(
                image,
                p1,
                p2,
                (0,255,255),
                1,
                tipLength=0.4
            )

# ==========================================================
# Configuration
# ==========================================================
#
# IMAGE_NAME = "tile_y512_x3072"
#
# RAW_IMAGE = f"data/raw/hirise_original/{IMAGE_NAME}.jpg"
#
# PREDICTED_MASK = f"outputs/prediction/{IMAGE_NAME}_predicted.png"

import os
import glob

IMAGE_NAME = "ESP_028535_1990"

RAW_FOLDER = "data/raw/hirise"
OUTPUT_FOLDER = "flow_outputs"

PREDICTED_MASK = f"outputs/prediction/{IMAGE_NAME}_predicted.png"

# Supported image formats
SUPPORTED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp"
]

RAW_IMAGE = None

for ext in SUPPORTED_EXTENSIONS:
    candidate = os.path.join(RAW_FOLDER, IMAGE_NAME + ext)
    if os.path.exists(candidate):
        RAW_IMAGE = candidate
        break

if RAW_IMAGE is None:
    print("\nAvailable images:")
    for file in glob.glob(os.path.join(RAW_FOLDER, "*")):
        print(" -", os.path.basename(file))

    raise FileNotFoundError(
        f"\nImage '{IMAGE_NAME}' not found in {RAW_FOLDER}\n"
        f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )

OUTPUT_FOLDER = "flow_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_IMAGE = os.path.join(
    OUTPUT_FOLDER,
    f"{IMAGE_NAME}_flow_direction.png"
)

# ==========================================================
# Valley Filtering Parameters
# ==========================================================

MIN_AREA = 150
MIN_ASPECT_RATIO = 2.5
MAX_CIRCULARITY = 0.45

# ==========================================================
# Helper Function
# Find Skeleton Endpoints
# ==========================================================

def find_skeleton_endpoints(skeleton):

    endpoints = []

    rows, cols = skeleton.shape

    for y in range(1, rows - 1):

        for x in range(1, cols - 1):

            if skeleton[y, x] == 0:
                continue

            neighbors = np.sum(
                skeleton[y-1:y+2, x-1:x+2] > 0
            ) - 1

            if neighbors == 1:
                endpoints.append((x, y))

    return endpoints


# ==========================================================
# Helper Function
# Euclidean Distance
# ==========================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )

# ==========================================================
# Angle to Compass Direction
# ==========================================================

def angle_to_direction(angle):

    directions = [

        "East",
        "North-East",
        "North",
        "North-West",
        "West",
        "South-West",
        "South",
        "South-East"

    ]

    angle = angle % 360

    index = int((angle + 22.5) / 45) % 8

    return directions[index]

# ==========================================================
# Load Images
# ==========================================================

raw = cv2.imread(RAW_IMAGE)

prediction = cv2.imread(
    PREDICTED_MASK,
    cv2.IMREAD_GRAYSCALE
)

if raw is None:
    raise FileNotFoundError(RAW_IMAGE)

if prediction is None:
    raise FileNotFoundError(PREDICTED_MASK)

print("\n========================================")
print("      Water Flow Direction Estimation")
print("========================================")

print(f"\nLoaded Image : {IMAGE_NAME}")

# ==========================================================
# Threshold Prediction
# ==========================================================

_, binary = cv2.threshold(
    prediction,
    127,
    255,
    cv2.THRESH_BINARY
)

# ==========================================================
# Morphological Cleaning
# ==========================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

clean = cv2.morphologyEx(
    clean,
    cv2.MORPH_CLOSE,
    kernel
)

print("\nMorphological cleaning completed.")

# ==========================================================
# Skeleton Generation
# ==========================================================

binary_bool = clean > 0

skeleton = skeletonize(binary_bool)

skeleton = (skeleton.astype(np.uint8)) * 255

print("Skeleton generated successfully.")

# ==========================================================
# Find Contours
# ==========================================================

contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"\nTotal Contours Found : {len(contours)}")

# ==========================================================
# Smart Valley Filtering
# ==========================================================

filtered = []

print("\n========================================")
print("Candidate Valley Analysis")
print("========================================")

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < MIN_AREA:
        continue

    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = max(w, h) / (min(w, h) + 1)

    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    print(
        f"Area={area:7.1f} | "
        f"Aspect={aspect_ratio:5.2f} | "
        f"Circularity={circularity:5.2f}"
    )

    if (
        area >= MIN_AREA and
        aspect_ratio >= MIN_ASPECT_RATIO and
        circularity <= MAX_CIRCULARITY
    ):

        confidence = 0

        # Aspect Ratio Score
        if aspect_ratio >= 5:
            confidence += 40
        elif aspect_ratio >= 3:
            confidence += 25

        # Circularity Score
        if circularity <= 0.20:
            confidence += 40
        elif circularity <= 0.35:
            confidence += 25

        # Area Score
        if area >= 300:
            confidence += 20

        confidence = min(confidence, 100)

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

print("\n----------------------------------------")
print(f"Selected Valleys : {len(filtered)}")
print("----------------------------------------")

# ==========================================================
# Draw Candidate Valleys
# ==========================================================

overlay = raw.copy()

for i, valley in enumerate(filtered):

    cnt = valley["contour"]

    x = valley["x"]
    y = valley["y"]
    w = valley["w"]
    h = valley["h"]

    # Valley contour
    cv2.drawContours(
        overlay,
        [cnt],
        -1,
        (0,255,0),
        2
    )

    # Bounding box
    cv2.rectangle(
        overlay,
        (x, y),
        (x+w, y+h),
        (255,0,0),
        2
    )

    # Valley label
    cv2.putText(
        overlay,
        f"V{i+1}",
        (x, max(20, y-5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,0),
        2
    )

# ==========================================================
# Display Candidate Valleys
# ==========================================================

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.imshow(clean, cmap="gray")
plt.title("Clean Prediction Mask")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Candidate Valleys ({len(filtered)})")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n========================================")
print("Block 2 Completed Successfully")
print("========================================")

# ==========================================================
# Adaptive Confidence Filtering
# ==========================================================

print("\n========================================")
print("Adaptive Confidence Filtering")
print("========================================")

candidate_count = len(filtered)

if candidate_count > 30:
    confidence_threshold = 80
elif candidate_count > 20:
    confidence_threshold = 70
elif candidate_count > 10:
    confidence_threshold = 60
else:
    confidence_threshold = 40

accepted=[]
rejected=[]

for idx,valley in enumerate(filtered,1):
    if valley["confidence"]>=confidence_threshold:
        accepted.append(valley)
        print(f"Valley {idx:02d} : {valley['confidence']:3d}% -> Accepted")
    else:
        rejected.append(valley)
        print(f"Valley {idx:02d} : {valley['confidence']:3d}% -> Rejected (Below Threshold)")

accepted=sorted(accepted,key=lambda x:(x['confidence'],x['area']),reverse=True)

MAX_VALLEYS=20
if len(accepted)>MAX_VALLEYS:
    print(f"Keeping Top {MAX_VALLEYS} valleys by confidence.")
    accepted=accepted[:MAX_VALLEYS]

filtered=accepted

print("\n----------------------------------------")
print(f"Candidate Valleys      : {candidate_count}")
print(f"Confidence Threshold   : {confidence_threshold}%")
print(f"Accepted Valleys       : {len(filtered)}")
print(f"Rejected Valleys       : {len(rejected)}")
print("----------------------------------------")

# ==========================================================
# Final Selected Valleys
# ==========================================================

overlay = raw.copy()

for i, valley in enumerate(filtered):

    cnt = valley["contour"]

    x = valley["x"]
    y = valley["y"]
    w = valley["w"]
    h = valley["h"]

    cv2.drawContours(
        overlay,
        [cnt],
        -1,
        (0,255,0),
        2
    )

    cv2.rectangle(
        overlay,
        (x,y),
        (x+w,y+h),
        (255,0,0),
        2
    )

    cv2.putText(
        overlay,
        f"V{i+1}",
        (x,max(20,y-5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,0),
        2
    )

plt.figure(figsize=(14,6))

plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))

plt.title(f"Final Selected Valleys ({len(filtered)})")

plt.axis("off")

plt.tight_layout()

plt.show()


valley_images = []
# ==========================================================
# Visualization
# ==========================================================
#
# plt.figure(figsize=(18,6))
#
# plt.subplot(1,3,1)
# plt.imshow(clean, cmap="gray")
# plt.title("Clean Mask")
# plt.axis("off")
#
# plt.subplot(1,3,2)
# plt.imshow(skeleton, cmap="gray")
# plt.title("Global Skeleton")
# plt.axis("off")
#
# plt.subplot(1,3,3)
# plt.imshow(
#     cv2.cvtColor(
#         flow_overlay,
#         cv2.COLOR_BGR2RGB
#     )
# )
# plt.title("Skeleton Endpoints")
# plt.axis("off")
#
# plt.tight_layout()
#
# plt.show()
#
# print("\n========================================")
# print("Block 3 Completed Successfully")
# print("========================================")

# ==========================================================
# Skeleton Analysis for Each Valley
# ==========================================================

print("\n========================================")
print("Skeleton Endpoint Detection")
print("========================================")

flow_data = []

# Copy overlay from Block 2
flow_overlay = overlay.copy()

for i, valley in enumerate(filtered):

    cnt = valley["contour"]

    # ------------------------------------------------------
    # Create mask for this valley only
    # ------------------------------------------------------

    valley_mask = np.zeros_like(clean)

    cv2.drawContours(
        valley_mask,
        [cnt],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ------------------------------------------------------
    # Skeleton for this valley
    # ------------------------------------------------------

    valley_skeleton = cv2.bitwise_and(
        skeleton,
        valley_mask
    )

    # ------------------------------------------------------
    # Find endpoints
    # ------------------------------------------------------

    endpoints = find_skeleton_endpoints(
        valley_skeleton
    )

    print(f"\nValley {i+1}")
    print("--------------------------------")

    if len(endpoints) == 0:

        print("No endpoints found.")

        continue

    print(f"Endpoints Found : {len(endpoints)}")

    # ------------------------------------------------------
    # Draw Skeleton
    # ------------------------------------------------------

    flow_overlay[valley_skeleton > 0] = (
        255,
        255,
        255
    )

    # ------------------------------------------------------
    # Draw Endpoints
    # ------------------------------------------------------

    for pt in endpoints:

        cv2.circle(

            flow_overlay,

            pt,

            5,

            (0,255,0),

            -1

        )

    # ------------------------------------------------------
    # Choose longest endpoint pair
    # ------------------------------------------------------

    max_distance = 0

    start = None
    end = None

    if len(endpoints) >= 2:

        for p1 in endpoints:

            for p2 in endpoints:

                d = distance(
                    p1,
                    p2
                )

                if d > max_distance:

                    max_distance = d

                    start = p1

                    end = p2

        cv2.circle(
            flow_overlay,
            start,
            7,
            (255,0,0),
            -1
        )

        cv2.circle(
            flow_overlay,
            end,
            7,
            (0,0,255),
            -1
        )

        flow_data.append({

            "valley": i+1,

            "start": start,

            "end": end,

            "length": max_distance,

            "mask": valley_mask,

            "skeleton": valley_skeleton

        })

        print("Start :", start)

        print("End   :", end)

        print(
            f"Length : {max_distance:.2f} pixels"
        )

print("\n----------------------------------------")
print("Skeleton analysis completed.")
print("----------------------------------------")


# ==========================================================
# Water Flow Direction Estimation
# ==========================================================

print("\n========================================")
print("Estimated Water Flow Direction")
print("========================================")

for valley in flow_data:

    start = valley["start"]

    end = valley["end"]

    # ------------------------------------------------------
    # Draw Arrow
    # ------------------------------------------------------

    # cv2.arrowedLine(
    #
    #     flow_overlay,
    #
    #     start,
    #
    #     end,
    #
    #     (255,0,0),
    #
    #     3,
    #
    #     tipLength=0.10
    #
    # )
    #
    # cv2.arrowedLine(
    #     result,
    #     start,
    #     end,
    #     (255, 0, 0),  # Blue
    #     2,  # Thickness
    #     line_type=cv2.LINE_AA,
    #     tipLength=0.15
    # )

    draw_flow_field(
        flow_overlay,
        filtered[valley["valley"] - 1]["contour"],
        start,
        end,
        spacing=12
    )
    # ----------------------------------------
    # Crop Valley Region
    # ----------------------------------------

    cnt = filtered[valley["valley"] - 1]["contour"]

    x, y, w, h = cv2.boundingRect(cnt)

    padding = 30

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(raw.shape[1], x + w + padding)
    y2 = min(raw.shape[0], y + h + padding)

    crop = raw[y1:y2, x1:x2].copy()

    # ----------------------------------------
    # Shift Arrow Coordinates
    # ----------------------------------------

    start = (
        valley["start"][0] - x1,
        valley["start"][1] - y1
    )

    end = (
        valley["end"][0] - x1,
        valley["end"][1] - y1
    )

    # cv2.arrowedLine(
    #
    #     crop,
    #
    #     start,
    #
    #     end,
    #
    #     (255, 0, 0),
    #
    #     2,
    #
    #     tipLength=0.15
    #
    # )

    # Determine arrow thickness from crop size
    arrow_thickness = max(1, min(crop.shape[:2]) // 120)

    cv2.arrowedLine(
        crop,
        start,
        end,
        (255, 0, 0),
        arrow_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.10
    )

    circle_radius = max(2, arrow_thickness + 1)

    cv2.circle(crop, start, circle_radius, (0, 255, 0), -1)
    cv2.circle(crop, end, circle_radius, (0, 0, 255), -1)

    cv2.circle(
        crop,
        start,
        5,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        crop,
        end,
        5,
        (0, 0, 255),
        -1
    )

    valley_images.append(crop)

    # ------------------------------------------------------
    # Flow Angle
    # ------------------------------------------------------

    dx = end[0] - start[0]

    dy = start[1] - end[1]

    angle = np.degrees(

        np.arctan2(

            dy,

            dx

        )

    )

    if angle < 0:

        angle += 360

    direction = angle_to_direction(angle)

    # ------------------------------------------------------
    # Valley Center
    # ------------------------------------------------------

    center_x = int(

        (start[0] + end[0]) / 2

    )

    center_y = int(

        (start[1] + end[1]) / 2

    )

    # ------------------------------------------------------
    # Draw Direction Label
    # ------------------------------------------------------

    cv2.putText(

        flow_overlay,

        direction,

        (center_x, center_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255,255,255),

        2

    )

    print()

    print(f"Valley {valley['valley']}")

    print("----------------------------")

    print(f"Estimated Flow : {direction}")

    print(f"Angle          : {angle:.2f}°")

    print(f"Length         : {valley['length']:.2f} pixels")

print("\nFlow estimation completed.")

# ==========================================================
# Save Result
# ==========================================================

cv2.imwrite(

    OUTPUT_IMAGE,

    flow_overlay

)

print()

print("Saved Result :")

print(OUTPUT_IMAGE)
# valley_images = []
# ==========================================================
# Final Visualization
# ==========================================================

# plt.figure(figsize=(16,8))
#
# plt.imshow(
#
#     cv2.cvtColor(
#
#         flow_overlay,
#
#         cv2.COLOR_BGR2RGB
#
#     )
#
# )
#
# plt.title(
#
#     "Estimated Water Flow Direction"
#
# )
#
# plt.axis("off")
#
# plt.tight_layout()
#
# plt.show()
#
# print()
# ==================================================
# Valley Flow Visualization
# ==================================================

# n = len(valley_images)
#
# plt.figure(figsize=(6*n,6))
#
# for i, img in enumerate(valley_images):
#
#     plt.subplot(1,n,i+1)
#
#     plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#
#     plt.title(f"Valley {i+1}")
#
#     plt.axis("off")
#
# plt.tight_layout()
#
# plt.show()

# ==================================================
# Valley Flow Visualization
# ==================================================

import math

n = len(valley_images)

COLS = 6                    # Number of columns
ROWS = math.ceil(n / COLS)  # Automatically compute rows

plt.figure(figsize=(18, ROWS * 4))

for i, img in enumerate(valley_images):

    plt.subplot(ROWS, COLS, i + 1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(
        f"Valley {i+1}",
        fontsize=12,
        fontweight="bold"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# for 50 images
# import math
#
# n = len(valley_images)
#
# COLS = 5
# ROWS = math.ceil(n / COLS)
#
# TITLE_SIZE = 9
#
# plt.figure(figsize=(20, ROWS * 3.8))
#
# for i, img in enumerate(valley_images):
#
#     plt.subplot(ROWS, COLS, i + 1)
#
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     plt.title(
#         f"Valley {i+1}\n"
#         f"Confidence : {filtered[flow_data[i]['valley']-1]['confidence']}%\n"
#         f"Length : {flow_data[i]['length']:.1f}px",
#         fontsize=TITLE_SIZE,
#         fontweight="bold"
#     )
#
#     plt.axis("off")
#
# plt.tight_layout()
#
# plt.show()

# ==================================================
# Top Confidence Valley Visualization (Top 5)
# ==================================================

top_candidates = []

for valley in flow_data:
    valley_info = filtered[valley["valley"] - 1]
    top_candidates.append({
        "flow": valley,
        "confidence": valley_info["confidence"],
        "length": valley["length"],
        "contour": valley_info["contour"]
    })

top_candidates = sorted(
    top_candidates,
    key=lambda x: (x["confidence"], x["length"]),
    reverse=True
)

TOP_VALLEYS = 5
top_candidates = top_candidates[:TOP_VALLEYS]

top_images = []

for item in top_candidates:

    valley = item["flow"]
    valley_info = filtered[valley["valley"]-1]
    cnt = item["contour"]

    x, y, w, h = cv2.boundingRect(cnt)
    pad = 30

    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(raw.shape[1], x+w+pad)
    y2 = min(raw.shape[0], y+h+pad)

    crop = raw[y1:y2, x1:x2].copy()

    start = (valley["start"][0]-x1, valley["start"][1]-y1)
    end   = (valley["end"][0]-x1, valley["end"][1]-y1)

    cv2.arrowedLine(crop, start, end, (255,0,0), 2, tipLength=0.15)
    cv2.circle(crop, start, 5, (0,255,0), -1)
    cv2.circle(crop, end, 5, (0,0,255), -1)

    # cv2.putText(
    #     crop,
    #     f"Conf:{item['confidence']}%  Len:{item['length']:.0f}",
    #     (10,25),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.65,
    #     (0,255,255),
    #     2
    # )

    top_images.append(crop)

if top_images:
    FONT_SIZE = 10  # Change this to any size you like

    plt.figure(figsize=(5 * len(top_images), 5))

    for i, img in enumerate(top_images):
        plt.subplot(1, len(top_images), i + 1)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.title(
            f"Top {i + 1}\n"
            f"Confidence : {top_candidates[i]['confidence']}%\n"
            f"Length : {top_candidates[i]['length']:.1f} pixels",
            fontsize=FONT_SIZE,
            fontweight='bold'
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()

    # ==================================================
    # Top 5 Valleys on Original Image
    # ==================================================

    top_overlay = raw.copy()

    for rank, item in enumerate(top_candidates, start=1):
        valley = item["flow"]
        valley_info = filtered[valley["valley"] - 1]
        cnt = valley_info["contour"]

        cv2.drawContours(
            top_overlay,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        start = valley["start"]
        end = valley["end"]

        cv2.arrowedLine(
            top_overlay,
            start,
            end,
            (255, 0, 0),
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.15
        )

        cv2.circle(top_overlay, start, 5, (0, 255, 0), -1)
        cv2.circle(top_overlay, end, 5, (0, 0, 255), -1)

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.putText(
            top_overlay,
            f"Top {rank} ({item['confidence']}%)",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

    plt.figure(figsize=(14, 10))

    plt.imshow(cv2.cvtColor(top_overlay, cv2.COLOR_BGR2RGB))

    plt.title(
        "Top 5 Highest Confidence Valleys on Original Image",
        fontsize=16,
        fontweight="bold"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()
    # plt.figure(figsize=(5*len(top_images),5))
    # for i,img in enumerate(top_images):
    #     plt.subplot(1,len(top_images),i+1)
    #     plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #     plt.title(f"Top {i+1}")
    #     plt.axis("off")
    # plt.tight_layout()
    # plt.show()


print("========================================")

print("Flow Direction Module Completed")

print("========================================")