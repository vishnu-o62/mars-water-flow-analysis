# ==========================================================
# PIPELINE 2 - IMPROVED / EXPLAINABLE VERSION
# Automated Mars Valley Flow-Path Analysis
# ----------------------------------------------------------
# Combines:
#   1. Valley Detection
#   2. Skeleton / Centerline Extraction
#   3. Flow Direction Estimation
#   4. Multi-Path Flow Reconstruction
#
# PIPELINE 3 READY VERSION
#
# IMPORTANT:
# The original working algorithms and established thresholds
# are preserved.
#
# Established thresholds:
#   - Minimum path length = 20 pixels
#   - Duplicate overlap threshold = 30%
# ==========================================================

import os
import sys
import csv
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, deque
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize


# ==========================================================
# 1. PROJECT PATHS / COMMAND-LINE INPUT
# ==========================================================

# Project root = folder containing src/
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ----------------------------------------------------------
# Input image
# ----------------------------------------------------------

if len(sys.argv) < 2:
    raise ValueError(
        "\nUsage:\n"
        "python src/pipeline2.py <image_path> "
        "[prediction_mask_path]\n"
    )

IMAGE_PATH = os.path.abspath(sys.argv[1])

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"Input image not found:\n{IMAGE_PATH}"
    )

# Image name without extension
IMAGE_NAME = os.path.splitext(
    os.path.basename(IMAGE_PATH)
)[0]

# ----------------------------------------------------------
# Prediction mask
#
# Pipeline 3 can provide this as argument 2.
# If it is not provided, use the standard Pipeline 1 output.
# ----------------------------------------------------------

if len(sys.argv) >= 3:

    MASK_PATH = os.path.abspath(sys.argv[2])

else:

    MASK_PATH = os.path.join(
        BASE_DIR,
        "outputs",
        IMAGE_NAME + "_predicted.png"
    )

if not os.path.isfile(MASK_PATH):
    raise FileNotFoundError(
        f"Prediction mask not found:\n{MASK_PATH}"
    )


# ==========================================================
# 2. OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "outputs",
    "pipeline2_improved"
)

OUTPUT_MASK = os.path.join(
    OUTPUT_ROOT,
    "cleaned_mask"
)

OUTPUT_VALLEYS = os.path.join(
    OUTPUT_ROOT,
    "valley_detection"
)

OUTPUT_SKELETON = os.path.join(
    OUTPUT_ROOT,
    "skeleton"
)

OUTPUT_DIRECTION = os.path.join(
    OUTPUT_ROOT,
    "flow_direction"
)

OUTPUT_RECONSTRUCTION = os.path.join(
    OUTPUT_ROOT,
    "flow_reconstruction"
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
    "pipeline2_flow_statistics.csv"
)


for folder in [
    OUTPUT_ROOT,
    OUTPUT_MASK,
    OUTPUT_VALLEYS,
    OUTPUT_SKELETON,
    OUTPUT_DIRECTION,
    OUTPUT_RECONSTRUCTION,
    OUTPUT_VISUALIZATION,
    OUTPUT_JSON,
    OUTPUT_STATS
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


# ==========================================================
# 3. SCIENTIFIC / FILTERING PARAMETERS
# ==========================================================

# Prediction-mask threshold
THRESHOLD_VALUE = 127

# Existing mask-cleaning values
MORPH_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 150

# Existing valley-detection values
MIN_AREA = 150
MIN_ASPECT_RATIO = 2.5
MAX_CIRCULARITY = 0.45
MIN_CONFIDENCE = 40

MAX_VALLEYS = 20

# Existing reconstruction values
MIN_PATH_PIXELS = 20
OVERLAP_THRESHOLD = 0.30

# Existing approximate metric conversion
PIXEL_SIZE = 0.25

# Visualization spacing
FLOW_ARROW_SPACING = 30

# Optional visualization window
SHOW_VISUALIZATION = "--show" in sys.argv


# ==========================================================
# 4. BASIC UTILITIES
# ==========================================================

NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


def print_header(title):

    print(
        "\n" + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def distance(p1, p2):

    return float(
        np.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )
    )


def neighbour_count(
    image,
    row,
    col
):

    total = 0

    rows, cols = image.shape

    for dr, dc in NEIGHBORS:

        rr = row + dr
        cc = col + dc

        if (
            0 <= rr < rows and
            0 <= cc < cols
        ):

            if image[rr, cc] > 0:
                total += 1

    return total


def count_connected_components(
    binary_image
):

    num_labels, _, _, _ = (
        cv2.connectedComponentsWithStats(
            (binary_image > 0).astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    return max(
        0,
        num_labels - 1
    )


# ==========================================================
# 5. LOAD + CLEAN PREDICTION MASK
# ==========================================================

def load_prediction_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise FileNotFoundError(
            f"Prediction mask not found: {path}"
        )

    _, mask = cv2.threshold(
        mask,
        THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY
    )

    return mask


def clean_prediction_mask(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    # Remove small isolated noise
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Repair small gaps
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Remove small connected components
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (cleaned > 0).astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    filtered = np.zeros_like(
        cleaned
    )

    for label_id in range(
        1,
        num_labels
    ):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_AREA:

            filtered[
                labels == label_id
            ] = 255

    return filtered


# ==========================================================
# 6. VALLEY DETECTION
# ==========================================================

def calculate_valley_confidence(
    area,
    aspect_ratio,
    circularity
):

    confidence = 0

    if aspect_ratio >= 5:

        confidence += 40

    elif aspect_ratio >= 3:

        confidence += 25

    if circularity <= 0.20:

        confidence += 40

    elif circularity <= 0.35:

        confidence += 25

    if area >= 300:

        confidence += 20

    return min(
        confidence,
        100
    )


def detect_valleys(clean):

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    rejected = []

    margin = 5

    for contour_id, cnt in enumerate(
        contours,
        start=1
    ):

        area = float(
            cv2.contourArea(cnt)
        )

        if area < MIN_AREA:

            rejected.append({
                "contour_id": contour_id,
                "status": "REJECTED",
                "reason":
                    f"area below minimum "
                    f"({area:.2f} < {MIN_AREA})",
                "area": round(
                    area,
                    2
                )
            })

            continue

        perimeter = float(
            cv2.arcLength(
                cnt,
                True
            )
        )

        if perimeter == 0:

            rejected.append({
                "contour_id": contour_id,
                "status": "REJECTED",
                "reason":
                    "zero contour perimeter",
                "area": round(
                    area,
                    2
                )
            })

            continue

        x, y, w, h = (
            cv2.boundingRect(cnt)
        )

        touches_border = (
            x <= margin or
            y <= margin or
            x + w >= (
                clean.shape[1] -
                margin
            ) or
            y + h >= (
                clean.shape[0] -
                margin
            )
        )

        if touches_border:

            rejected.append({
                "contour_id": contour_id,
                "status": "REJECTED",
                "reason":
                    "touches image border",
                "area": round(
                    area,
                    2
                ),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            })

            continue

        aspect_ratio = (
            max(w, h) /
            max(min(w, h), 1)
        )

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter *
            perimeter
        )

        confidence = (
            calculate_valley_confidence(
                area,
                aspect_ratio,
                circularity
            )
        )

        reasons = []

        if aspect_ratio < MIN_ASPECT_RATIO:

            reasons.append(
                f"aspect ratio below minimum "
                f"({aspect_ratio:.2f} < "
                f"{MIN_ASPECT_RATIO})"
            )

        if circularity > MAX_CIRCULARITY:

            reasons.append(
                f"circularity above maximum "
                f"({circularity:.3f} > "
                f"{MAX_CIRCULARITY})"
            )

        if confidence < MIN_CONFIDENCE:

            reasons.append(
                f"confidence below minimum "
                f"({confidence} < "
                f"{MIN_CONFIDENCE})"
            )

        record = {

            "contour_id":
                contour_id,

            "area":
                round(area, 2),

            "perimeter":
                round(perimeter, 2),

            "x":
                int(x),

            "y":
                int(y),

            "w":
                int(w),

            "h":
                int(h),

            "aspect_ratio":
                round(
                    float(aspect_ratio),
                    3
                ),

            "circularity":
                round(
                    float(circularity),
                    4
                ),

            "confidence":
                int(confidence)
        }

        if reasons:

            rejected.append({

                **record,

                "status":
                    "REJECTED",

                "reason":
                    "; ".join(reasons)

            })

        else:

            candidates.append({

                "contour": cnt,

                **record,

                "status":
                    "ACCEPTED",

                "reason":
                    "passed valley filters"

            })

    # Highest confidence first,
    # then largest area
    candidates.sort(
        key=lambda v: (
            v["confidence"],
            v["area"]
        ),
        reverse=True
    )

    # Existing MAX_VALLEYS limit
    over_limit = (
        candidates[MAX_VALLEYS:]
    )

    candidates = (
        candidates[:MAX_VALLEYS]
    )

    for extra in over_limit:

        rejected.append({

            key: value

            for key, value
            in extra.items()

            if key != "contour"

        })

        rejected[-1][
            "status"
        ] = "REJECTED"

        rejected[-1][
            "reason"
        ] = (
            f"exceeded MAX_VALLEYS "
            f"limit ({MAX_VALLEYS})"
        )

    # Assign stable IDs
    for valley_id, valley in enumerate(
        candidates,
        start=1
    ):

        valley["id"] = valley_id

    rejected.sort(
        key=lambda item:
            item.get(
                "contour_id",
                0
            )
    )

    return (
        candidates,
        rejected
    )


# ==========================================================
# 7. SKELETON / CENTERLINE
# ==========================================================

def create_skeleton(clean):

    skeleton = skeletonize(
        clean > 0
    )

    return (
        skeleton.astype(
            np.uint8
        ) * 255
    )


def detect_endpoints(skeleton):

    endpoints = []

    rows, cols = skeleton.shape

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if skeleton[r, c] == 0:
                continue

            if (
                neighbour_count(
                    skeleton,
                    r,
                    c
                ) == 1
            ):

                endpoints.append(
                    (r, c)
                )

    return endpoints


def detect_branchpoints(skeleton):

    branchpoints = []

    rows, cols = skeleton.shape

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if skeleton[r, c] == 0:
                continue

            if (
                neighbour_count(
                    skeleton,
                    r,
                    c
                ) >= 3
            ):

                branchpoints.append(
                    (r, c)
                )

    return branchpoints


def count_branch_regions(skeleton):

    branch_mask = np.zeros_like(
        skeleton
    )

    rows, cols = skeleton.shape

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if skeleton[r, c] == 0:
                continue

            if (
                neighbour_count(
                    skeleton,
                    r,
                    c
                ) >= 3
            ):

                branch_mask[
                    r, c
                ] = 255

    return (
        count_connected_components(
            branch_mask
        ),
        branch_mask
    )


# ==========================================================
# 8. GRAPH CONSTRUCTION +
#    MULTI-PATH RECONSTRUCTION
# ==========================================================

def build_graph(skeleton):

    graph = defaultdict(list)

    rows, cols = skeleton.shape

    for r in range(rows):

        for c in range(cols):

            if skeleton[r, c] == 0:
                continue

            for dr, dc in NEIGHBORS:

                rr = r + dr
                cc = c + dc

                if (
                    0 <= rr < rows and
                    0 <= cc < cols
                ):

                    if (
                        skeleton[rr, cc] > 0
                    ):

                        graph[
                            (r, c)
                        ].append(
                            (rr, cc)
                        )

    return graph


def trace_path(
    graph,
    start,
    end
):

    queue = deque([
        start
    ])

    parent = {
        start: None
    }

    while queue:

        current = queue.popleft()

        if current == end:
            break

        for neighbor in graph.get(
            current,
            []
        ):

            if neighbor not in parent:

                parent[
                    neighbor
                ] = current

                queue.append(
                    neighbor
                )

    if end not in parent:
        return []

    path = []

    current = end

    while current is not None:

        path.append(current)

        current = parent[
            current
        ]

    path.reverse()

    return path


def longest_endpoint_pair(
    endpoints
):

    if len(endpoints) < 2:

        return (
            None,
            None,
            0.0
        )

    best_start = None
    best_end = None

    maximum = 0.0

    for i in range(
        len(endpoints)
    ):

        for j in range(
            i + 1,
            len(endpoints)
        ):

            p1 = endpoints[i]
            p2 = endpoints[j]

            d = distance(
                p1,
                p2
            )

            if d > maximum:

                maximum = d

                best_start = p1
                best_end = p2

    return (
        best_start,
        best_end,
        maximum
    )


def reconstruct_paths(
    skeleton,
    endpoints
):

    graph = build_graph(
        skeleton
    )

    accepted_paths = []

    report = {

        "endpoint_pairs":
            0,

        "valid_graph_paths":
            0,

        "rejected_short":
            0,

        "rejected_overlap":
            0,

        "accepted":
            0,

        "details":
            []
    }

    if len(endpoints) < 2:

        return (
            accepted_paths,
            report
        )

    for i in range(
        len(endpoints)
    ):

        for j in range(
            i + 1,
            len(endpoints)
        ):

            start = endpoints[i]
            end = endpoints[j]

            report[
                "endpoint_pairs"
            ] += 1

            path = trace_path(
                graph,
                start,
                end
            )

            if not path:

                report[
                    "details"
                ].append({

                    "start":
                        list(start),

                    "end":
                        list(end),

                    "status":
                        "REJECTED",

                    "reason":
                        "no connected graph path"

                })

                continue

            report[
                "valid_graph_paths"
            ] += 1

            candidate_length = (
                path_length(path)
            )

            if (
                candidate_length <
                MIN_PATH_PIXELS
            ):

                report[
                    "rejected_short"
                ] += 1

                report[
                    "details"
                ].append({

                    "start":
                        list(start),

                    "end":
                        list(end),

                    "status":
                        "REJECTED",

                    "reason":
                        (
                            f"path shorter "
                            f"than minimum "
                            f"({candidate_length:.2f} "
                            f"< {MIN_PATH_PIXELS} "
                            f"pixels)"
                        ),

                    "length_pixels":
                        round(
                            candidate_length,
                            2
                        )

                })

                continue

            candidate_set = set(
                path
            )

            duplicate = False

            highest_overlap = 0.0

            for accepted_path in (
                accepted_paths
            ):

                accepted_set = set(
                    accepted_path
                )

                overlap_pixels = len(
                    candidate_set.intersection(
                        accepted_set
                    )
                )

                overlap_ratio = (
                    overlap_pixels /
                    max(
                        len(candidate_set),
                        1
                    )
                )

                highest_overlap = max(
                    highest_overlap,
                    overlap_ratio
                )

                if (
                    overlap_ratio >=
                    OVERLAP_THRESHOLD
                ):

                    duplicate = True
                    break

            if duplicate:

                report[
                    "rejected_overlap"
                ] += 1

                report[
                    "details"
                ].append({

                    "start":
                        list(start),

                    "end":
                        list(end),

                    "status":
                        "REJECTED",

                    "reason":
                        (
                            f"duplicate/"
                            f"overlapping path "
                            f"({highest_overlap * 100:.1f}% "
                            f"overlap >= "
                            f"{OVERLAP_THRESHOLD * 100:.0f}%)"
                        ),

                    "length_pixels":
                        round(
                            candidate_length,
                            2
                        ),

                    "overlap_ratio":
                        round(
                            highest_overlap,
                            4
                        )

                })

                continue

            accepted_paths.append(
                path
            )

            report[
                "accepted"
            ] += 1

            report[
                "details"
            ].append({

                "start":
                    list(start),

                "end":
                    list(end),

                "status":
                    "ACCEPTED",

                "reason":
                    "passed reconstruction filters",

                "length_pixels":
                    round(
                        candidate_length,
                        2
                    ),

                "overlap_ratio":
                    round(
                        highest_overlap,
                        4
                    )

            })

    return (
        accepted_paths,
        report
    )


# ==========================================================
# 9. FLOW DIRECTION
# ==========================================================

def angle_to_direction(
    angle
):

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

    index = int(
        (angle + 22.5) / 45
    ) % 8

    return directions[index]


def estimate_flow_direction(
    start,
    end
):

    dx = (
        end[1] -
        start[1]
    )

    dy = (
        end[0] -
        start[0]
    )

    angle = np.degrees(
        np.arctan2(
            -dy,
            dx
        )
    )

    direction = (
        angle_to_direction(
            angle
        )
    )

    return (
        float(angle % 360),
        direction
    )


# ==========================================================
# 10. PATH METRICS
# ==========================================================

def path_length(path):

    if len(path) < 2:
        return 0.0

    total = 0.0

    for i in range(
        len(path) - 1
    ):

        p1 = path[i]
        p2 = path[i + 1]

        total += np.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    return float(total)


def curvature_index(path):

    if len(path) < 3:
        return 0.0

    straight = distance(
        path[0],
        path[-1]
    )

    actual = path_length(
        path
    )

    if straight == 0:
        return 0.0

    return float(
        actual / straight
    )


def estimate_width(
    mask,
    path
):

    widths = []

    for r, c in path:

        if not (
            0 <= r < mask.shape[0]
            and
            0 <= c < mask.shape[1]
        ):

            continue

        if mask[r, c] == 0:
            continue

        left = c
        right = c

        while left > 0:

            if mask[r, left] == 0:
                break

            left -= 1

        while right < (
            mask.shape[1] - 1
        ):

            if mask[r, right] == 0:
                break

            right += 1

        widths.append(
            right - left
        )

    if not widths:
        return 0.0

    return float(
        np.mean(widths)
    )


def smooth_path(path):

    if len(path) < 8:
        return path

    y = np.array(
        [p[0] for p in path],
        dtype=float
    )

    x = np.array(
        [p[1] for p in path],
        dtype=float
    )

    try:

        tck, _ = splprep(
            [x, y],
            s=3
        )

        u_new = np.linspace(
            0,
            1,
            len(path) * 3
        )

        x_new, y_new = splev(
            u_new,
            tck
        )

        smooth = []

        for xx, yy in zip(
            x_new,
            y_new
        ):

            smooth.append(
                (
                    int(round(yy)),
                    int(round(xx))
                )
            )

        return smooth

    except Exception:

        return path


def confidence_score(
    path,
    width,
    branch_count,
    valley_confidence
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

    if branch_count == 0:

        score += 10

    else:

        score += max(
            0,
            10 - branch_count
        )

    score += int(
        0.10 *
        valley_confidence
    )

    return min(
        score,
        100
    )


# ==========================================================
# 11. DRAW FLOW FIELD / PATHS
# ==========================================================

def draw_flow_field(
    image,
    contour,
    start,
    end,
    spacing=30
):

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        -1
    )

    dx = (
        end[1] -
        start[1]
    )

    dy = (
        end[0] -
        start[0]
    )

    length = np.sqrt(
        dx * dx +
        dy * dy
    )

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    arrow_length = 10

    for y in range(
        0,
        image.shape[0],
        spacing
    ):

        for x in range(
            0,
            image.shape[1],
            spacing
        ):

            if mask[y, x] == 0:
                continue

            p1 = (
                x,
                y
            )

            p2 = (
                int(
                    x +
                    ux *
                    arrow_length
                ),
                int(
                    y +
                    uy *
                    arrow_length
                )
            )

            cv2.arrowedLine(
                image,
                p1,
                p2,
                (0, 255, 255),
                1,
                tipLength=0.4
            )


def draw_path(
    image,
    path,
    line_color=(0, 0, 255),
    thickness=2
):

    if len(path) < 2:
        return

    for i in range(
        len(path) - 1
    ):

        p1 = (
            path[i][1],
            path[i][0]
        )

        p2 = (
            path[i + 1][1],
            path[i + 1][0]
        )

        cv2.line(
            image,
            p1,
            p2,
            line_color,
            thickness,
            cv2.LINE_AA
        )


def draw_all_candidate_paths(
    image,
    paths
):

    for path in paths:

        draw_path(
            image,
            path,
            line_color=(255, 0, 255),
            thickness=1
        )


# ==========================================================
# 12. PROCESS ONE IMAGE
# ==========================================================

def process_image():

    print_header(
        "PIPELINE 2 - IMPROVED MARS FLOW ANALYSIS"
    )

    print(
        f"Input Image              : {IMAGE_PATH}"
    )

    print(
        f"Prediction Mask          : {MASK_PATH}"
    )

    # ------------------------------------------------------
    # Load original image
    # ------------------------------------------------------

    raw = cv2.imread(
        IMAGE_PATH
    )

    if raw is None:

        raise FileNotFoundError(
            f"Input image not found: "
            f"{IMAGE_PATH}"
        )

    # ------------------------------------------------------
    # Load prediction
    # ------------------------------------------------------

    prediction = load_prediction_mask(
        MASK_PATH
    )

    # ------------------------------------------------------
    # IMPORTANT:
    # Prediction mask and original image must match.
    # ------------------------------------------------------

    if prediction.shape[:2] != raw.shape[:2]:

        print(
            "\nWARNING:"
        )

        print(
            "Prediction mask size does not "
            "match the original image."
        )

        print(
            f"Original image : "
            f"{raw.shape[:2]}"
        )

        print(
            f"Prediction mask: "
            f"{prediction.shape[:2]}"
        )

        print(
            "\nResizing prediction mask "
            "to original image size..."
        )

        prediction = cv2.resize(
            prediction,
            (
                raw.shape[1],
                raw.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Save corrected prediction mask
        cv2.imwrite(
            MASK_PATH,
            prediction
        )

        print(
            "Prediction mask resized successfully."
        )

    print(
        f"Raw foreground pixels    : "
        f"{int(np.sum(prediction > 0))}"
    )

    # ------------------------------------------------------
    # Step 1 - Mask Cleaning
    # ------------------------------------------------------

    clean = clean_prediction_mask(
        prediction
    )

    print(
        f"Cleaned foreground pixels : "
        f"{int(np.sum(clean > 0))}"
    )

    cleaned_path = os.path.join(
        OUTPUT_MASK,
        IMAGE_NAME +
        "_cleaned.png"
    )

    cv2.imwrite(
        cleaned_path,
        clean
    )

    # ------------------------------------------------------
    # Step 2 - Valley Detection
    # ------------------------------------------------------

    valleys, valley_rejections = (
        detect_valleys(clean)
    )

    print_header(
        "1. VALLEY DETECTION"
    )

    print(
        f"Contours examined         : "
        f"{len(valleys) + len(valley_rejections)}"
    )

    print(
        f"Accepted valleys          : "
        f"{len(valleys)}"
    )

    print(
        f"Rejected contours         : "
        f"{len(valley_rejections)}"
    )

    for valley in valleys:

        print(
            f"V{valley['id']:02d} -> ACCEPTED | "
            f"confidence={valley['confidence']}% | "
            f"area={valley['area']:.1f} | "
            f"aspect={valley['aspect_ratio']:.2f} | "
            f"circularity={valley['circularity']:.3f}"
        )

    if valley_rejections:

        print(
            "\nRejected contour reasons:"
        )

        for item in valley_rejections:

            print(
                f"Contour "
                f"{item.get('contour_id', '?')} "
                f"-> "
                f"{item.get('reason', 'unknown reason')}"
            )

    valley_overlay = raw.copy()

    for valley in valleys:

        cnt = valley[
            "contour"
        ]

        cv2.drawContours(
            valley_overlay,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        x = valley["x"]
        y = valley["y"]
        w = valley["w"]
        h = valley["h"]

        cv2.rectangle(
            valley_overlay,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            valley_overlay,
            (
                f"V{valley['id']} "
                f"({valley['confidence']}%)"
            ),
            (
                x,
                max(20, y - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    valley_path = os.path.join(
        OUTPUT_VALLEYS,
        IMAGE_NAME +
        "_valleys.png"
    )

    cv2.imwrite(
        valley_path,
        valley_overlay
    )

    # ------------------------------------------------------
    # Step 3 - Global Skeleton
    # ------------------------------------------------------

    skeleton = create_skeleton(
        clean
    )

    skeleton_endpoints = (
        detect_endpoints(
            skeleton
        )
    )

    skeleton_branchpoints = (
        detect_branchpoints(
            skeleton
        )
    )

    skeleton_branch_regions, branch_mask = (
        count_branch_regions(
            skeleton
        )
    )

    skeleton_components = (
        count_connected_components(
            skeleton
        )
    )

    skeleton_path = os.path.join(
        OUTPUT_SKELETON,
        IMAGE_NAME +
        "_skeleton.png"
    )

    cv2.imwrite(
        skeleton_path,
        skeleton
    )

    # Extra diagnostic overlay
    skeleton_overlay = raw.copy()

    for r, c in skeleton_endpoints:

        cv2.circle(
            skeleton_overlay,
            (c, r),
            3,
            (0, 255, 0),
            -1
        )

    for r, c in skeleton_branchpoints:

        cv2.circle(
            skeleton_overlay,
            (c, r),
            2,
            (0, 0, 255),
            -1
        )

    skeleton_overlay_path = os.path.join(
        OUTPUT_SKELETON,
        IMAGE_NAME +
        "_skeleton_diagnostics.png"
    )

    cv2.imwrite(
        skeleton_overlay_path,
        skeleton_overlay
    )

    print_header(
        "2. SKELETON / CENTERLINE"
    )

    print(
        f"Skeleton pixels           : "
        f"{int(np.sum(skeleton > 0))}"
    )

    print(
        f"Skeleton components       : "
        f"{skeleton_components}"
    )

    print(
        f"Endpoints                 : "
        f"{len(skeleton_endpoints)}"
    )

    print(
        f"Branch pixels             : "
        f"{len(skeleton_branchpoints)}"
    )

    print(
        f"Branch regions            : "
        f"{skeleton_branch_regions}"
    )

    # ------------------------------------------------------
    # Step 4 - Per-valley analysis
    # ------------------------------------------------------

    reconstruction_overlay = raw.copy()

    direction_overlay = raw.copy()

    all_candidate_overlay = raw.copy()

    flow_results = []

    valley_reports = []

    for valley in valleys:

        valley_id = valley[
            "id"
        ]

        cnt = valley[
            "contour"
        ]

        print_header(
            f"VALLEY V{valley_id} ANALYSIS"
        )

        # Create valley mask
        valley_mask = np.zeros_like(
            clean
        )

        cv2.drawContours(
            valley_mask,
            [cnt],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # Extract valley skeleton
        valley_skeleton = (
            cv2.bitwise_and(
                skeleton,
                valley_mask
            )
        )

        endpoints = detect_endpoints(
            valley_skeleton
        )

        branchpoints = detect_branchpoints(
            valley_skeleton
        )

        branch_regions, _ = (
            count_branch_regions(
                valley_skeleton
            )
        )

        valley_skeleton_components = (
            count_connected_components(
                valley_skeleton
            )
        )

        print(
            f"Skeleton components       : "
            f"{valley_skeleton_components}"
        )

        print(
            f"Endpoints                 : "
            f"{len(endpoints)}"
        )

        print(
            f"Branch pixels             : "
            f"{len(branchpoints)}"
        )

        print(
            f"Branch regions            : "
            f"{branch_regions}"
        )

        valley_report = {

            "id":
                valley_id,

            "valley_confidence":
                valley["confidence"],

            "skeleton_components":
                valley_skeleton_components,

            "endpoints":
                len(endpoints),

            "branch_pixels":
                len(branchpoints),

            "branch_regions":
                branch_regions,

            "status":
                "SKIPPED",

            "reason":
                ""
        }

        if len(endpoints) < 2:

            reason = (
                f"insufficient endpoints "
                f"({len(endpoints)} found, "
                f"2 required)"
            )

            valley_report[
                "reason"
            ] = reason

            valley_reports.append(
                valley_report
            )

            print(
                f"V{valley_id} -> "
                f"SKIPPED: {reason}"
            )

            continue

        # --------------------------------------------------
        # Flow direction
        # --------------------------------------------------

        start, end, endpoint_distance = (
            longest_endpoint_pair(
                endpoints
            )
        )

        angle, direction = (
            estimate_flow_direction(
                start,
                end
            )
        )

        print(
            f"Flow direction            : "
            f"{direction} "
            f"({angle:.2f} deg)"
        )

        # --------------------------------------------------
        # Multi-path reconstruction
        # --------------------------------------------------

        paths, reconstruction_report = (
            reconstruct_paths(
                valley_skeleton,
                endpoints
            )
        )

        print(
            f"Candidate endpoint pairs  : "
            f"{reconstruction_report['endpoint_pairs']}"
        )

        print(
            f"Valid graph paths         : "
            f"{reconstruction_report['valid_graph_paths']}"
        )

        print(
            f"Rejected - short          : "
            f"{reconstruction_report['rejected_short']}"
        )

        print(
            f"Rejected - overlap        : "
            f"{reconstruction_report['rejected_overlap']}"
        )

        print(
            f"Accepted paths            : "
            f"{reconstruction_report['accepted']}"
        )

        if not paths:

            reason = (
                "no valid reconstructed path "
                "after path-length/overlap filtering"
            )

            valley_report[
                "reason"
            ] = reason

            valley_report[
                "reconstruction"
            ] = reconstruction_report

            valley_reports.append(
                valley_report
            )

            print(
                f"V{valley_id} -> "
                f"SKIPPED: {reason}"
            )

            continue

        # Longest accepted path
        main_path = max(
            paths,
            key=path_length
        )

        smooth = smooth_path(
            main_path
        )

        length_pixels = path_length(
            smooth
        )

        length_units = (
            length_pixels *
            PIXEL_SIZE
        )

        curve = curvature_index(
            smooth
        )

        width = estimate_width(
            valley_mask,
            smooth
        )

        confidence = confidence_score(
            smooth,
            width,
            branch_regions,
            valley["confidence"]
        )

        accepted_path_lengths = [

            round(
                path_length(p),
                2
            )

            for p in paths

        ]

        result = {

            "id":
                valley_id,

            "valley_confidence":
                valley["confidence"],

            "start":
                [
                    int(start[0]),
                    int(start[1])
                ],

            "end":
                [
                    int(end[0]),
                    int(end[1])
                ],

            "endpoint_distance":
                round(
                    endpoint_distance,
                    2
                ),

            "direction_angle":
                round(
                    angle,
                    2
                ),

            "flow_direction":
                direction,

            "candidate_path_count":
                reconstruction_report[
                    "endpoint_pairs"
                ],

            "valid_graph_path_count":
                reconstruction_report[
                    "valid_graph_paths"
                ],

            "accepted_path_count":
                len(paths),

            "rejected_short_count":
                reconstruction_report[
                    "rejected_short"
                ],

            "rejected_overlap_count":
                reconstruction_report[
                    "rejected_overlap"
                ],

            "accepted_path_lengths_pixels":
                accepted_path_lengths,

            "length_pixels":
                round(
                    length_pixels,
                    2
                ),

            "length_units":
                round(
                    length_units,
                    2
                ),

            "curvature":
                round(
                    curve,
                    3
                ),

            "width_pixels":
                round(
                    width,
                    2
                ),

            "branch_regions":
                branch_regions,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "path":
                [
                    [
                        int(r),
                        int(c)
                    ]

                    for r, c in smooth
                ],

            "candidate_path_details":
                reconstruction_report[
                    "details"
                ]
        }

        flow_results.append(
            result
        )

        valley_report[
            "status"
        ] = "RECONSTRUCTED"

        valley_report[
            "reason"
        ] = (
            "flow path reconstructed successfully"
        )

        valley_report[
            "flow_direction"
        ] = direction

        valley_report[
            "reconstruction"
        ] = reconstruction_report

        valley_reports.append(
            valley_report
        )

        print(
            f"V{valley_id} -> "
            f"RECONSTRUCTED | "
            f"main length="
            f"{length_pixels:.2f}px | "
            f"accepted paths="
            f"{len(paths)} | "
            f"confidence="
            f"{confidence:.1f}%"
        )

        # --------------------------------------------------
        # Draw valley contour
        # --------------------------------------------------

        cv2.drawContours(
            reconstruction_overlay,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        cv2.drawContours(
            direction_overlay,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        # All accepted candidate paths
        for candidate_path in paths:

            draw_path(
                all_candidate_overlay,
                candidate_path,
                line_color=(255, 0, 255),
                thickness=1
            )

        # Main path
        draw_path(
            reconstruction_overlay,
            smooth,
            line_color=(0, 0, 255),
            thickness=3
        )

        # Start = green
        cv2.circle(
            reconstruction_overlay,
            (start[1], start[0]),
            6,
            (0, 255, 0),
            -1
        )

        # End = red
        cv2.circle(
            reconstruction_overlay,
            (end[1], end[0]),
            6,
            (0, 0, 255),
            -1
        )

        label_text = (
            f"V{valley_id}: "
            f"{direction}"
        )

        cv2.putText(
            reconstruction_overlay,
            label_text,
            (
                valley["x"],
                max(
                    20,
                    valley["y"] - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Flow-direction field
        draw_flow_field(
            direction_overlay,
            cnt,
            start,
            end,
            spacing=FLOW_ARROW_SPACING
        )

        cv2.arrowedLine(
            direction_overlay,
            (
                start[1],
                start[0]
            ),
            (
                end[1],
                end[0]
            ),
            (255, 0, 0),
            4,
            cv2.LINE_AA,
            tipLength=0.18
        )

        cv2.circle(
            direction_overlay,
            (
                start[1],
                start[0]
            ),
            7,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            direction_overlay,
            (
                end[1],
                end[0]
            ),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            direction_overlay,
            (
                f"V{valley_id}: "
                f"{direction}"
            ),
            (
                valley["x"],
                max(
                    25,
                    valley["y"] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # ------------------------------------------------------
    # Step 5 - Save image outputs
    # ------------------------------------------------------

    direction_path = os.path.join(
        OUTPUT_DIRECTION,
        IMAGE_NAME +
        "_flow_direction.png"
    )

    reconstruction_path = os.path.join(
        OUTPUT_RECONSTRUCTION,
        IMAGE_NAME +
        "_flow_reconstruction.png"
    )

    candidates_path = os.path.join(
        OUTPUT_RECONSTRUCTION,
        IMAGE_NAME +
        "_all_accepted_candidate_paths.png"
    )

    cv2.imwrite(
        direction_path,
        direction_overlay
    )

    cv2.imwrite(
        reconstruction_path,
        reconstruction_overlay
    )

    cv2.imwrite(
        candidates_path,
        all_candidate_overlay
    )

    # ------------------------------------------------------
    # Step 6 - JSON report
    # ------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_JSON,
        IMAGE_NAME +
        "_pipeline2_improved.json"
    )

    json_data = {

        "pipeline":
            "Pipeline 2 - Improved / Explainable",

        "image":
            IMAGE_NAME,

        "input_image_path":
            IMAGE_PATH,

        "prediction_mask_path":
            MASK_PATH,

        "parameters": {

            "threshold_value":
                THRESHOLD_VALUE,

            "morph_kernel_size":
                MORPH_KERNEL_SIZE,

            "min_component_area":
                MIN_COMPONENT_AREA,

            "min_area":
                MIN_AREA,

            "min_aspect_ratio":
                MIN_ASPECT_RATIO,

            "max_circularity":
                MAX_CIRCULARITY,

            "min_confidence":
                MIN_CONFIDENCE,

            "max_valleys":
                MAX_VALLEYS,

            "min_path_pixels":
                MIN_PATH_PIXELS,

            "overlap_threshold":
                OVERLAP_THRESHOLD,

            "pixel_size":
                PIXEL_SIZE
        },

        "summary": {

            "raw_foreground_pixels":
                int(
                    np.sum(
                        prediction > 0
                    )
                ),

            "cleaned_foreground_pixels":
                int(
                    np.sum(
                        clean > 0
                    )
                ),

            "contours_examined":
                (
                    len(valleys) +
                    len(valley_rejections)
                ),

            "accepted_valleys":
                len(valleys),

            "rejected_contours":
                len(
                    valley_rejections
                ),

            "skeleton_pixels":
                int(
                    np.sum(
                        skeleton > 0
                    )
                ),

            "skeleton_components":
                skeleton_components,

            "skeleton_endpoints":
                len(
                    skeleton_endpoints
                ),

            "skeleton_branch_pixels":
                len(
                    skeleton_branchpoints
                ),

            "skeleton_branch_regions":
                skeleton_branch_regions,

            "reconstructed_valleys":
                len(flow_results),

            "skipped_valleys":
                (
                    len(valleys) -
                    len(flow_results)
                )
        },

        "valley_rejections":
            valley_rejections,

        "valley_reports":
            valley_reports,

        "flow_results":
            flow_results
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as fp:

        json.dump(
            json_data,
            fp,
            indent=4
        )

    # ------------------------------------------------------
    # Step 7 - CSV report
    # ------------------------------------------------------

    csv_exists = os.path.exists(
        CSV_FILE
    )

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as fp:

        writer = csv.writer(
            fp
        )

        if not csv_exists:

            writer.writerow([

                "Image",
                "ValleyID",
                "Status",
                "Reason",
                "ValleyConfidence",
                "Endpoints",
                "BranchPixels",
                "BranchRegions",
                "CandidatePaths",
                "ValidGraphPaths",
                "AcceptedPaths",
                "RejectedShort",
                "RejectedOverlap",
                "FlowDirection",
                "DirectionAngle",
                "LengthPixels",
                "LengthUnits",
                "Curvature",
                "WidthPixels",
                "Confidence"

            ])

        flow_by_id = {

            result["id"]:
                result

            for result
            in flow_results

        }

        for report in valley_reports:

            matching_flow = (
                flow_by_id.get(
                    report["id"]
                )
            )

            writer.writerow([

                IMAGE_NAME,

                report["id"],

                report["status"],

                report["reason"],

                report[
                    "valley_confidence"
                ],

                report[
                    "endpoints"
                ],

                report[
                    "branch_pixels"
                ],

                report[
                    "branch_regions"
                ],

                (
                    matching_flow[
                        "candidate_path_count"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "valid_graph_path_count"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "accepted_path_count"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "rejected_short_count"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "rejected_overlap_count"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "flow_direction"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "direction_angle"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "length_pixels"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "length_units"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "curvature"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "width_pixels"
                    ]
                    if matching_flow
                    else ""
                ),

                (
                    matching_flow[
                        "confidence"
                    ]
                    if matching_flow
                    else ""
                )

            ])

    # ------------------------------------------------------
    # Step 8 - Complete 6-panel visualization
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 11)
    )

    # Panel 1 - Original
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
        "1. Original Mars Image"
    )

    plt.axis("off")

    # Panel 2 - Valley Detection
    plt.subplot(
        2,
        3,
        2
    )

    plt.imshow(
        cv2.cvtColor(
            valley_overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "2. Valley Detection"
    )

    plt.axis("off")

    # Panel 3 - Skeleton
    plt.subplot(
        2,
        3,
        3
    )

    skeleton_green = np.zeros(
        (
            skeleton.shape[0],
            skeleton.shape[1],
            3
        ),
        dtype=np.uint8
    )

    skeleton_green[
        skeleton > 0
    ] = (
        0,
        255,
        0
    )

    plt.imshow(
        skeleton_green
    )

    plt.title(
        "3. Skeleton / Centerlines"
        f"\nE={len(skeleton_endpoints)}, "
        f"B={skeleton_branch_regions}"
    )

    plt.axis("off")

    # Panel 4 - Flow Direction
    plt.subplot(
        2,
        3,
        4
    )

    plt.imshow(
        cv2.cvtColor(
            direction_overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "4. Flow Direction"
    )

    plt.axis("off")

    # Panel 5 - Reconstruction
    plt.subplot(
        2,
        3,
        5
    )

    plt.imshow(
        cv2.cvtColor(
            reconstruction_overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "5. Reconstructed Flow Paths"
    )

    plt.axis("off")

    # Panel 6 - Summary
    plt.subplot(
        2,
        3,
        6
    )

    plt.axis("off")

    skipped_count = (
        len(valleys) -
        len(flow_results)
    )

    total_candidate_paths = sum(
        result[
            "candidate_path_count"
        ]

        for result
        in flow_results
    )

    total_accepted_paths = sum(
        result[
            "accepted_path_count"
        ]

        for result
        in flow_results
    )

    total_short = sum(
        result[
            "rejected_short_count"
        ]

        for result
        in flow_results
    )

    total_overlap = sum(
        result[
            "rejected_overlap_count"
        ]

        for result
        in flow_results
    )

    summary_lines = [

        "PIPELINE 2 SUMMARY",
        "",

        f"Valleys accepted : "
        f"{len(valleys)}",

        f"Valleys rejected : "
        f"{len(valley_rejections)}",

        f"Reconstructed    : "
        f"{len(flow_results)}",

        f"Skipped           : "
        f"{skipped_count}",

        "",

        "SKELETON",

        f"Pixels            : "
        f"{int(np.sum(skeleton > 0))}",

        f"Components        : "
        f"{skeleton_components}",

        f"Endpoints         : "
        f"{len(skeleton_endpoints)}",

        f"Branch regions    : "
        f"{skeleton_branch_regions}",

        "",

        "RECONSTRUCTION",

        f"Candidate paths   : "
        f"{total_candidate_paths}",

        f"Accepted paths    : "
        f"{total_accepted_paths}",

        f"Rejected short    : "
        f"{total_short}",

        f"Rejected overlap  : "
        f"{total_overlap}",

        "",

        "VALLEY STATUS"

    ]

    for report in valley_reports:

        if (
            report["status"]
            == "RECONSTRUCTED"
        ):

            summary_lines.append(
                f"V{report['id']} "
                f"-> reconstructed"
            )

        else:

            summary_lines.append(
                f"V{report['id']} "
                f"-> SKIPPED"
            )

    if valley_rejections:

        summary_lines.append("")

        summary_lines.append(
            f"{len(valley_rejections)} "
            "contours rejected "
            "during valley filtering"
        )

    if len(summary_lines) > 24:

        summary_lines = (
            summary_lines[:23]
            +
            [
                "... full report "
                "in JSON / CSV"
            ]
        )

    plt.text(
        0.02,
        0.98,
        "\n".join(
            summary_lines
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        transform=plt.gca().transAxes
    )

    plt.tight_layout()

    visualization_path = os.path.join(
        OUTPUT_VISUALIZATION,
        IMAGE_NAME +
        "_pipeline2_improved.png"
    )

    plt.savefig(
        visualization_path,
        dpi=200,
        bbox_inches="tight"
    )

    # ------------------------------------------------------
    # IMPORTANT:
    # Do NOT block automated Pipeline 3.
    # Only display when --show is supplied.
    # ------------------------------------------------------

    if SHOW_VISUALIZATION:

        plt.show()

    plt.close(fig)

    # ------------------------------------------------------
    # Step 9 - Final console report
    # ------------------------------------------------------

    print_header(
        "PIPELINE 2 IMPROVED COMPLETED"
    )

    print(
        f"Valleys accepted        : "
        f"{len(valleys)}"
    )

    print(
        f"Valleys rejected        : "
        f"{len(valley_rejections)}"
    )

    print(
        f"Flow paths reconstructed: "
        f"{len(flow_results)}"
    )

    print(
        f"Valleys skipped         : "
        f"{skipped_count}"
    )

    print(
        f"Candidate paths         : "
        f"{total_candidate_paths}"
    )

    print(
        f"Accepted paths          : "
        f"{total_accepted_paths}"
    )

    print(
        f"Rejected short paths    : "
        f"{total_short}"
    )

    print(
        f"Rejected overlap paths  : "
        f"{total_overlap}"
    )

    print_header(
        "VALLEY REJECTION / "
        "RECONSTRUCTION REPORT"
    )

    for item in valley_rejections:

        print(
            f"Contour "
            f"{item.get('contour_id', '?')} "
            f"-> REJECTED: "
            f"{item.get('reason', 'unknown reason')}"
        )

    for report in valley_reports:

        if (
            report["status"]
            == "RECONSTRUCTED"
        ):

            print(
                f"Valley {report['id']} "
                f"-> RECONSTRUCTED "
                f"({report.get('flow_direction', 'N/A')})"
            )

        else:

            print(
                f"Valley {report['id']} "
                f"-> SKIPPED: "
                f"{report['reason']}"
            )

    print(
        "\nComplete status is saved "
        "in JSON and CSV."
    )

    print(
        f"Cleaned mask           : "
        f"{cleaned_path}"
    )

    print(
        f"Valley output          : "
        f"{valley_path}"
    )

    print(
        f"Skeleton output        : "
        f"{skeleton_path}"
    )

    print(
        f"Skeleton diagnostics   : "
        f"{skeleton_overlay_path}"
    )

    print(
        f"Direction output       : "
        f"{direction_path}"
    )

    print(
        f"Reconstruction output  : "
        f"{reconstruction_path}"
    )

    print(
        f"Candidate paths output : "
        f"{candidates_path}"
    )

    print(
        f"Visualization          : "
        f"{visualization_path}"
    )

    print(
        f"JSON                   : "
        f"{json_path}"
    )

    print(
        f"CSV                    : "
        f"{CSV_FILE}"
    )

    return flow_results


# ==========================================================
# 13. MAIN
# ==========================================================

if __name__ == "__main__":

    process_image()