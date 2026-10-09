"""
=============================================================
Mars Water Flow Reconstruction
-------------------------------------------------------------
Project:
AI-Based Identification of Past Water Flow Patterns
on Mars Using NASA Imagery

Author : Vishnu Jadhav

Description
-----------
Reconstructs probable ancient water flow paths using the
predicted valley mask and extracted skeleton.

Pipeline

Prediction Mask
        │
        ▼
Skeleton
        │
        ▼
Graph Construction
        │
        ▼
Flow Path Tracing
        │
        ▼
Flow Reconstruction
        │
        ▼
Overlay + Statistics

=============================================================
"""

import os
import cv2
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from collections import defaultdict

from scipy.interpolate import splprep
from scipy.interpolate import splev

from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.measure import regionprops

# from src.skeleton import IMAGE_FOLDER, MASK_FOLDER

# ==========================================================
# Configuration
# ==========================================================
#
# IMAGE_FOLDER = "data/raw/hirise"
# IMAGE_NAME = "ESP_028535_1990.jpg"
# MASK_FOLDER = "outputs/prediction"
#
# OUTPUT_ROOT = "outputs/flow_reconstruction"
#
# OUTPUT_OVERLAY = os.path.join(
#     OUTPUT_ROOT,
#     "overlay"
# )
#
# OUTPUT_VISUALIZATION = os.path.join(
#     OUTPUT_ROOT,
#     "visualization"
# )
#
# OUTPUT_JSON = os.path.join(
#     OUTPUT_ROOT,
#     "json"
# )
#
# OUTPUT_STATS = os.path.join(
#     OUTPUT_ROOT,
#     "statistics"
# )
#
# CSV_FILE = os.path.join(
#     OUTPUT_STATS,
#     "flow_statistics.csv"
# )
#
# JSON_FILE = os.path.join(
#     OUTPUT_JSON,
#     "flow_paths.json"
# )
#
# for folder in [
#
#     OUTPUT_ROOT,
#
#     OUTPUT_OVERLAY,
#
#     OUTPUT_VISUALIZATION,
#
#     OUTPUT_JSON,
#
#     OUTPUT_STATS
#
# ]:
#     os.makedirs(folder, exist_ok=True)

# ==========================================================
# Configuration
# ==========================================================

# Supported input image formats
SUPPORTED_IMAGE_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

# Change only this line whenever you want to test another image
IMAGE_PATH = "data/raw/hirise/ESP_028535_1990.jpg"

# Corresponding prediction mask
MASK_PATH = "outputs/prediction/ESP_028535_1990_predicted.png"


OUTPUT_ROOT = "outputs/flow_reconstruction"

OUTPUT_OVERLAY = os.path.join(
    OUTPUT_ROOT,
    "overlay"
)

OUTPUT_VISUALIZATION = os.path.join(
    OUTPUT_ROOT,
    "visualization"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_ROOT,
    "json"
)

OUTPUT_STATS = os.path.join(
    OUTPUT_ROOT,
    "statistics"
)

CSV_FILE = os.path.join(
    OUTPUT_STATS,
    "flow_statistics.csv"
)


# ==========================================================
# Utility
# ==========================================================

def print_header(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Read Binary Mask
# ==========================================================

def load_mask(mask_path):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise FileNotFoundError(mask_path)

    _, mask = cv2.threshold(

        mask,

        127,

        255,

        cv2.THRESH_BINARY

    )

    return mask



# ==========================================================
# Mask Cleaning
# ==========================================================

# Conservative cleaning parameters.
# These can be tuned later after comparing the outputs.
MORPH_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 100


def clean_prediction_mask(mask):
    """
    Clean the raw binary prediction mask before skeletonization.

    Pipeline:
        Raw Prediction Mask
              ↓
        Morphological Opening
              ↓
        Morphological Closing
              ↓
        Remove tiny connected components
              ↓
        Cleaned Prediction Mask
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    # Remove small isolated noise.
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Repair small gaps inside otherwise connected regions.
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Remove very small connected components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        connectivity=8
    )

    filtered = np.zeros_like(cleaned)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            filtered[labels == label_id] = 255

    return filtered


# ==========================================================
# Skeleton
# ==========================================================

def create_skeleton(mask):

    skeleton = skeletonize(mask > 0)

    skeleton = skeleton.astype(np.uint8)

    return skeleton


# ==========================================================
# Neighbour Coordinates
# ==========================================================

NEIGHBOURS = [

    (-1,-1),(-1,0),(-1,1),

    (0,-1),        (0,1),

    (1,-1),(1,0),(1,1)

]


# ==========================================================
# Count Neighbours
# ==========================================================

def neighbour_count(image,row,col):

    total = 0

    rows,cols = image.shape

    for dr,dc in NEIGHBOURS:

        rr = row + dr

        cc = col + dc

        if rr<0 or cc<0:

            continue

        if rr>=rows or cc>=cols:

            continue

        if image[rr,cc]:

            total += 1

    return total


# ==========================================================
# End Points
# ==========================================================

def detect_endpoints(skeleton):

    endpoints = []

    rows,cols = skeleton.shape

    for r in range(1,rows-1):

        for c in range(1,cols-1):

            if skeleton[r,c]==0:

                continue

            if neighbour_count(

                    skeleton,

                    r,

                    c

            )==1:

                endpoints.append(

                    (r,c)

                )

    return endpoints


# ==========================================================
# Branch Points
# ==========================================================

def detect_branchpoints(skeleton):

    branches=[]

    rows,cols=skeleton.shape

    for r in range(1,rows-1):

        for c in range(1,cols-1):

            if skeleton[r,c]==0:

                continue

            if neighbour_count(

                skeleton,

                r,

                c

            )>=3:

                branches.append(

                    (r,c)

                )

    return branches


# ==========================================================
# Build Graph
# ==========================================================

def build_graph(skeleton):

    graph = defaultdict(list)

    rows,cols = skeleton.shape

    for r in range(rows):

        for c in range(cols):

            if skeleton[r,c]==0:

                continue

            for dr,dc in NEIGHBOURS:

                rr = r+dr

                cc = c+dc

                if rr<0 or cc<0:

                    continue

                if rr>=rows or cc>=cols:

                    continue

                if skeleton[rr,cc]:

                    graph[(r,c)].append(

                        (rr,cc)

                    )

    return graph


# ==========================================================
# Euclidean Distance
# ==========================================================

def distance(p1,p2):

    return np.sqrt(

        (p1[0]-p2[0])**2 +

        (p1[1]-p2[1])**2

    )


# ==========================================================
# Longest Endpoint Pair
# ==========================================================

def longest_pair(endpoints):

    best=None

    maximum=0

    if len(endpoints)<2:

        return None,None

    for p1 in endpoints:

        for p2 in endpoints:

            d = distance(

                p1,

                p2

            )

            if d>maximum:

                maximum=d

                best=(p1,p2)

    return best

# ==========================================================
# Trace Skeleton Path (DFS)
# ==========================================================

def trace_path(graph, start, end):

    stack = [(start, [start])]
    visited = set()

    while stack:

        current, path = stack.pop()

        if current == end:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:

            if neighbor not in visited:

                stack.append(

                    (
                        neighbor,
                        path + [neighbor]
                    )

                )

    return []


# # ==========================================================
# # Trace All Flow Paths
# # ==========================================================
#
# def reconstruct_flow_paths(
#
#         skeleton,
#         graph,
#         endpoints
#
# ):
#
#     paths = []
#
#     used = set()
#
#     if len(endpoints) < 2:
#         return paths
#
#     for i in range(len(endpoints)):
#
#         for j in range(i + 1, len(endpoints)):
#
#             start = endpoints[i]
#
#             end = endpoints[j]
#
#             path = trace_path(
#
#                 graph,
#
#                 start,
#
#                 end
#
#             )
#
#             if len(path) < 20:
#                 continue
#
#             duplicate = False
#
#             for p in path:
#
#                 if p in used:
#
#                     duplicate = True
#                     break
#
#             if duplicate:
#                 continue
#
#             for p in path:
#                 used.add(p)
#
#             paths.append(path)
#
#     return paths

# ==========================================================
# Trace All Flow Paths
# ==========================================================
#
# Final path-selection pipeline:
#
# Prediction Mask
#       ↓
# Mask Cleaning
#       ↓
# Skeleton
#       ↓
# Candidate Paths
#       ↓
# Short-path filtering (< 20 skeleton pixels)
#       ↓
# 30% overlap filtering
#       ↓
# Final Flow Paths
#
# ==========================================================

def reconstruct_flow_paths(
        skeleton,
        graph,
        endpoints
):

    paths = []

    # Maximum allowed overlap before considering a path
    # a duplicate of an already accepted flow.
    OVERLAP_THRESHOLD = 0.30

    if len(endpoints) < 2:
        return paths

    for i in range(len(endpoints)):

        for j in range(i + 1, len(endpoints)):

            start = endpoints[i]
            end = endpoints[j]

            path = trace_path(
                graph,
                start,
                end
            )

            # --------------------------------------------------
            # Step 1: Reject very short paths
            # --------------------------------------------------

            if len(path) < 20:
                continue

            # --------------------------------------------------
            # Step 2: Check overlap with accepted paths
            # --------------------------------------------------

            candidate_set = set(path)

            duplicate = False

            for accepted_path in paths:

                accepted_set = set(accepted_path)

                overlap_pixels = len(
                    candidate_set.intersection(accepted_set)
                )

                # Overlap relative to the candidate path
                overlap_ratio = (
                    overlap_pixels / len(candidate_set)
                )

                # Large overlap = probably duplicate
                if overlap_ratio >= OVERLAP_THRESHOLD:

                    duplicate = True
                    break

            # --------------------------------------------------
            # Step 3: Reject highly overlapping duplicates
            # --------------------------------------------------

            if duplicate:
                continue

            # --------------------------------------------------
            # Step 4: Keep valid flow path
            # --------------------------------------------------

            paths.append(path)

    return paths

# ==========================================================
# Calculate Flow Length
# ==========================================================

def flow_length(path):

    if len(path) < 2:
        return 0

    length = 0.0

    for i in range(len(path)-1):

        p1 = path[i]
        p2 = path[i+1]

        length += np.sqrt(

            (p1[0]-p2[0])**2 +

            (p1[1]-p2[1])**2

        )

    return length


# ==========================================================
# Curvature Index
# ==========================================================

def curvature(path):

    if len(path) < 3:
        return 0

    straight = distance(

        path[0],

        path[-1]

    )

    actual = flow_length(path)

    if straight == 0:
        return 0

    return actual / straight


# ==========================================================
# Smooth Flow Path
# ==========================================================

def smooth_path(path):

    if len(path) < 8:
        return path

    y = np.array(

        [p[0] for p in path]

    )

    x = np.array(

        [p[1] for p in path]

    )

    try:

        tck, u = splprep(

            [x, y],

            s=3

        )

        unew = np.linspace(

            0,

            1,

            len(path) * 3

        )

        xnew, ynew = splev(

            unew,

            tck

        )

        smooth = []

        for xx, yy in zip(xnew, ynew):

            smooth.append(

                (

                    int(round(yy)),

                    int(round(xx))

                )

            )

        return smooth

    except Exception:

        return path


# ==========================================================
# Estimate Valley Width
# ==========================================================

def estimate_width(

        mask,

        path

):

    widths = []

    for r, c in path:

        if r < 0 or c < 0:
            continue

        if r >= mask.shape[0]:
            continue

        if c >= mask.shape[1]:
            continue

        if mask[r, c] == 0:
            continue

        left = c

        while left > 0:

            if mask[r, left] == 0:
                break

            left -= 1

        right = c

        while right < mask.shape[1]-1:

            if mask[r, right] == 0:
                break

            right += 1

        widths.append(

            right-left

        )

    if len(widths) == 0:
        return 0

    return float(

        np.mean(widths)

    )


# ==========================================================
# Confidence Score
# ==========================================================

def confidence_score(

        path,

        width,

        branches

):

    score = 50

    if len(path) > 150:
        score += 20

    elif len(path) > 80:
        score += 10

    if width > 20:
        score += 15

    elif width > 10:
        score += 8

    if branches == 0:
        score += 10
    else:
        score += max(

            0,

            10 - branches

        )

    return min(

        score,

        100

    )


# ==========================================================
# Build Flow Information
# ==========================================================

def build_flow_information(

        mask,
        skeleton

):

    endpoints = detect_endpoints(
        skeleton
    )

    branches = detect_branchpoints(
        skeleton
    )

    graph = build_graph(
        skeleton
    )

    flow_paths = reconstruct_flow_paths(

        skeleton,

        graph,

        endpoints

    )

    results = []

    for idx, path in enumerate(flow_paths):

        smooth = smooth_path(path)

        length = flow_length(smooth)

        curve = curvature(smooth)

        width = estimate_width(

            mask,

            smooth

        )

        confidence = confidence_score(

            smooth,

            width,

            len(branches)

        )

        results.append({

            "id": idx + 1,

            "path": smooth,

            "length": length,

            "curvature": curve,

            "width": width,

            "confidence": confidence

        })

    return results

# ==========================================================
# PART 3
# Visualization / Overlay / Export Utilities
# Paste below Part 2
# ==========================================================

def draw_flow_overlay(image, flow_results):
    overlay = image.copy()

    colors = [
        (0,0,255),
        (0,255,0),
        (255,0,0),
        (0,255,255),
        (255,255,0),
        (255,0,255)
    ]

    for idx, flow in enumerate(flow_results):
        pts = flow["path"]
        color = colors[idx % len(colors)]

        for i in range(len(pts)-1):
            p1 = (pts[i][1], pts[i][0])
            p2 = (pts[i+1][1], pts[i+1][0])
            cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

        if pts:
            s = (pts[0][1], pts[0][0])
            e = (pts[-1][1], pts[-1][0])
            cv2.circle(overlay, s, 5, (0,255,0), -1)
            cv2.circle(overlay, e, 5, (0,0,255), -1)

    return overlay


def save_overlay(image, flow_results, output_path):
    overlay = draw_flow_overlay(image, flow_results)
    cv2.imwrite(output_path, overlay)
    return overlay


def save_flow_json(flow_results, output_path):
    export = []
    for f in flow_results:
        export.append({
            "id": f["id"],
            "length": round(f["length"],2),
            "curvature": round(f["curvature"],3),
            "width": round(f["width"],2),
            "confidence": round(f["confidence"],2),
            "path": [[int(r), int(c)] for r,c in f["path"]]
        })
    with open(output_path,"w") as fp:
        json.dump(export, fp, indent=4)


def append_csv(csv_path, image_name, flow_results):
    header = [
        "Image","FlowID","Length",
        "Curvature","Width","Confidence"
    ]

    exists = os.path.exists(csv_path)

    with open(csv_path,"a",newline="") as fp:
        writer = csv.writer(fp)

        if not exists:
            writer.writerow(header)

        for f in flow_results:
            writer.writerow([
                image_name,
                f["id"],
                round(f["length"],2),
                round(f["curvature"],3),
                round(f["width"],2),
                round(f["confidence"],2)
            ])


def visualize_results(image, mask, skeleton, overlay, image_name):

    fig = plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(mask,cmap="gray")
    plt.title("Cleaned Prediction Mask")
    plt.axis("off")

    # plt.subplot(2,2,3)
    # plt.imshow(skeleton,cmap="gray")
    # plt.title("Skeleton")
    # plt.axis("off")

    plt.subplot(2, 2, 3)

    # Create black background
    skeleton_green = np.zeros(
        (skeleton.shape[0], skeleton.shape[1], 3),
        dtype=np.uint8
    )

    # Draw skeleton in GREEN
    skeleton_green[skeleton > 0] = (0, 255, 0)

    plt.imshow(skeleton_green)
    plt.title("Skeleton")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
    plt.title("Flow Reconstruction")
    plt.axis("off")

    plt.tight_layout()

    out = os.path.join(
        OUTPUT_VISUALIZATION,
        image_name + "_visualization.png"
    )

    plt.savefig(
        out,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()  # <-- Add this line

    plt.close(fig)

    # plt.tight_layout()
    #
    # out = os.path.join(
    #     OUTPUT_VISUALIZATION,
    #     image_name+"_visualization.png"
    # )
    #
    # plt.savefig(out,dpi=200,bbox_inches="tight")
    # # plt.show(fig)
    # plt.close(fig)


def print_flow_summary(image_name, flow_results):

    print_header(f"Flow Report : {image_name}")

    if len(flow_results)==0:
        print("No valid flow paths detected.")
        return

    for flow in flow_results:

        print(f"Flow #{flow['id']}")
        print(f"Length      : {flow['length']:.2f}")
        print(f"Curvature   : {flow['curvature']:.3f}")
        print(f"Width       : {flow['width']:.2f}")
        print(f"Confidence  : {flow['confidence']:.1f}%")
        print("-"*45)


# ==========================================================
# PART 4
# Processing Pipeline / Runner
# Paste below Part 3
# ==========================================================

def process_image(image_path, mask_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    try:
        raw_mask = load_mask(mask_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # ======================================================
    # STEP 1: Prediction Mask
    # STEP 2: Mask Cleaning
    # ======================================================

    raw_foreground = int(np.sum(raw_mask > 0))

    mask = clean_prediction_mask(raw_mask)

    cleaned_foreground = int(np.sum(mask > 0))

    print("\nMask Cleaning")
    print("---------------------------------------------")
    print(f"Raw foreground pixels     : {raw_foreground}")
    print(f"Cleaned foreground pixels : {cleaned_foreground}")
    print("---------------------------------------------")

    # Save cleaned mask for inspection.
    cleaned_mask_folder = os.path.join(
        OUTPUT_ROOT,
        "cleaned_mask"
    )

    os.makedirs(
        cleaned_mask_folder,
        exist_ok=True
    )

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    cleaned_mask_path = os.path.join(
        cleaned_mask_folder,
        name + "_cleaned.png"
    )

    cv2.imwrite(
        cleaned_mask_path,
        mask
    )

    print("Cleaned Mask :")
    print(cleaned_mask_path)

    # ======================================================
    # STEP 3: Skeleton
    # ======================================================

    skeleton = create_skeleton(mask)

    print(f"Skeleton pixels           : {int(np.sum(skeleton > 0))}")

    flow_results = build_flow_information(
        mask,
        skeleton
    )

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    overlay_path = os.path.join(
        OUTPUT_OVERLAY,
        name + "_overlay.png"
    )

    overlay = save_overlay(
        image,
        flow_results,
        overlay_path
    )

    visualize_results(
        image,
        mask,
        skeleton,
        overlay,
        name
    )

    json_path = os.path.join(
        OUTPUT_JSON,
        name + "_flow.json"
    )

    save_flow_json(
        flow_results,
        json_path
    )

    append_csv(
        CSV_FILE,
        name,
        flow_results
    )

    print_flow_summary(
        name,
        flow_results
    )


class FlowReconstruction:

    def __init__(self):
        self.image_folder = IMAGE_PATH
        self.mask_folder = MASK_PATH

    def run(self):

        print_header("Mars Water Flow Reconstruction")

        if not os.path.exists(IMAGE_PATH):
            print("\nERROR : Input image not found.")
            print(IMAGE_PATH)
            return

        extension = os.path.splitext(
            IMAGE_PATH
        )[1].lower()

        if extension not in SUPPORTED_IMAGE_FORMATS:

            print(f"\nERROR : Unsupported image format ({extension})")
            print("\nSupported formats are:")

            for fmt in SUPPORTED_IMAGE_FORMATS:
                print(" ", fmt)

            return

        if not os.path.exists(MASK_PATH):
            print("\nERROR : Prediction mask not found.")
            print(MASK_PATH)
            return

        print("\nInput Image :")
        print(IMAGE_PATH)

        print("\nPrediction Mask :")
        print(MASK_PATH)

        process_image(

            IMAGE_PATH,

            MASK_PATH

        )

        print_header("Completed")

        print("Overlay Folder :")
        print(OUTPUT_OVERLAY)

        print("\nVisualization Folder :")
        print(OUTPUT_VISUALIZATION)

        print("\nJSON Folder :")
        print(OUTPUT_JSON)

        print("\nStatistics :")
        print(CSV_FILE)

#     def run(self):
#
#         print_header("Mars Water Flow Reconstruction")
#
#         if not os.path.exists(self.image_folder):
#             print("Image folder not found.")
#             return
#
#         if not os.path.exists(self.mask_folder):
#             print("Prediction folder not found.")
#             return
#
#         images = sorted([
#             f for f in os.listdir(self.image_folder)
#             if f.lower().endswith(
#                 (".jpg", ".jpeg", ".png", ".tif", ".tiff")
#             )
#         ])
#
#         if len(images) == 0:
#             print("No images found.")
#             return
#
#         processed = 0
#
#         for image_name in images:
#
#             image_path = os.path.join(
#                 self.image_folder,
#                 image_name
#             )
#
#             base = os.path.splitext(image_name)[0]
#
#             candidates = [
#                 base + "_predicted.png",
#                 base + ".png",
#                 base + "_mask.png"
#             ]
#
#             mask_path = None
#
#             for candidate in candidates:
#                 p = os.path.join(
#                     self.mask_folder,
#                     candidate
#                 )
#                 if os.path.exists(p):
#                     mask_path = p
#                     break
#
#             if mask_path is None:
#                 print(f"[WARNING] No prediction mask for {image_name}")
#                 continue
#
#             print(f"\nProcessing: {image_name}")
#
#             process_image(
#                 image_path,
#                 mask_path
#             )
#
#             processed += 1
#
#         print_header("Completed")
#         print(f"Images Processed : {processed}")
#         print(f"Overlay Folder   : {OUTPUT_OVERLAY}")
#         print(f"Visualization    : {OUTPUT_VISUALIZATION}")
#         print(f"JSON Output      : {OUTPUT_JSON}")
#         print(f"Statistics CSV   : {CSV_FILE}")
#

def main():

    app = FlowReconstruction()
    app.run()


if __name__ == "__main__":
    main()
