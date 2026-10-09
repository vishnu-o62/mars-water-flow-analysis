from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import sys
import json
import subprocess
import time

from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError



# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)
CORS(app)

# Testathon upload policy: reject oversized requests before they reach the pipeline.
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

PIPELINE3_PATH = os.path.join(
    SRC_DIR,
    "pipeline3.py"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

# Original project images used during standalone Pipeline 3 testing.
HIRISE_ORIGINAL_DIR = os.path.join(
    DATA_DIR,
    "raw",
    "hirise_original"
)

UPLOAD_DIR = os.path.join(
    DATA_DIR,
    "frontend_uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# Existing/legacy prediction folder
PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "prediction"
)

# Pipeline 2 output structure
PIPELINE2_ROOT = os.path.join(
    OUTPUT_DIR,
    "pipeline2_improved"
)

PIPELINE2_JSON_DIR = os.path.join(
    PIPELINE2_ROOT,
    "json"
)

PIPELINE2_CLEANED_DIR = os.path.join(
    PIPELINE2_ROOT,
    "cleaned_mask"
)

PIPELINE2_VALLEY_DIR = os.path.join(
    PIPELINE2_ROOT,
    "valley_detection"
)

PIPELINE2_SKELETON_DIR = os.path.join(
    PIPELINE2_ROOT,
    "skeleton"
)

PIPELINE2_DIRECTION_DIR = os.path.join(
    PIPELINE2_ROOT,
    "flow_direction"
)

PIPELINE2_RECONSTRUCTION_DIR = os.path.join(
    PIPELINE2_ROOT,
    "flow_reconstruction"
)

# Pipeline 3 output
PIPELINE3_ROOT = os.path.join(
    OUTPUT_DIR,
    "pipeline3"
)

EVALUATION_DIR = os.path.join(
    PIPELINE3_ROOT,
    "evaluation"
)


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)


# ==========================================================
# API ERROR HELPERS / GLOBAL HANDLERS
# ==========================================================

def api_error(code, message, status):
    return jsonify({
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }), status


@app.errorhandler(400)
def handle_bad_request(_error):
    return api_error("BAD_REQUEST", "The request could not be understood.", 400)


@app.errorhandler(404)
def handle_not_found(_error):
    return api_error("NOT_FOUND", "The requested API endpoint or file was not found.", 404)


@app.errorhandler(413)
def handle_too_large(_error):
    return api_error("FILE_TOO_LARGE", "The uploaded file is too large. Maximum allowed size is 5 MB.", 413)


@app.errorhandler(415)
def handle_unsupported_media(_error):
    return api_error("UNSUPPORTED_MEDIA_TYPE", "Unsupported file type. Please upload a JPG, JPEG, or PNG image.", 415)


@app.errorhandler(422)
def handle_unprocessable_entity(_error):
    return api_error("INVALID_IMAGE", "The uploaded file is not a valid or readable image.", 422)


@app.errorhandler(500)
def handle_internal_error(_error):
    return api_error("INTERNAL_SERVER_ERROR", "An unexpected server error occurred. Please try again.", 500)


def validate_uploaded_image(uploaded_file, original_filename):
    extension = os.path.splitext(original_filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return api_error(
            "UNSUPPORTED_MEDIA_TYPE",
            "Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
            415
        )

    # Reject obviously non-image MIME declarations, while allowing browsers
    # that send a generic application/octet-stream for an otherwise valid image.
    content_type = (uploaded_file.content_type or "").lower()
    if content_type and content_type not in {
        "image/jpeg", "image/png", "application/octet-stream"
    }:
        return api_error(
            "UNSUPPORTED_MEDIA_TYPE",
            "Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
            415
        )

    try:
        uploaded_file.stream.seek(0)
        with Image.open(uploaded_file.stream) as image:
            detected_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return api_error(
            "INVALID_IMAGE",
            "The uploaded file is not a valid or readable image.",
            422
        )
    finally:
        uploaded_file.stream.seek(0)

    if detected_format not in ALLOWED_PIL_FORMATS:
        return api_error(
            "UNSUPPORTED_MEDIA_TYPE",
            "Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
            415
        )

    expected_format = "JPEG" if extension in {".jpg", ".jpeg"} else "PNG"
    if detected_format != expected_format:
        return api_error(
            "INVALID_IMAGE",
            "The file extension does not match the image content.",
            422
        )

    uploaded_file.stream.seek(0)
    return None


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "ok",
        "message": "Mars Water Detection Flask API is running"
    })


# ==========================================================
# SERVE OUTPUT FILES
# ==========================================================

@app.route(
    "/api/output/<path:filename>"
)
def output_file(filename):

    return send_from_directory(
        OUTPUT_DIR,
        filename
    )


# ==========================================================
# SERVE UPLOADED FILES
# ==========================================================

@app.route(
    "/api/upload/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ==========================================================
# FIND BEST PATH INFORMATION
# ==========================================================

def find_best_path_information(data):

    best_length = None
    best_confidence = None

    if not isinstance(data, dict):
        return (
            best_length,
            best_confidence
        )

    flow_results = data.get(
        "flow_results",
        []
    )

    if not isinstance(flow_results, list):
        return (
            best_length,
            best_confidence
        )

    valid_flows = []

    for flow in flow_results:

        if not isinstance(flow, dict):
            continue

        length = flow.get(
            "length"
        )

        if length is None:
            length = flow.get(
                "length_pixels"
            )

        confidence = flow.get(
            "confidence"
        )

        if confidence is None:
            confidence = flow.get(
                "path_confidence"
            )

        try:

            if length is None:
                continue

            length = float(
                length
            )

            if confidence is not None:
                confidence = float(
                    confidence
                )

            valid_flows.append({
                "length": length,
                "confidence": confidence
            })

        except (
            TypeError,
            ValueError
        ):
            continue

    if valid_flows:

        best = max(
            valid_flows,
            key=lambda x: x["length"]
        )

        best_length = best[
            "length"
        ]

        best_confidence = best[
            "confidence"
        ]

    return (
        best_length,
        best_confidence
    )


# ==========================================================
# FIND EXISTING PREDICTION MASK
# ==========================================================

def find_existing_prediction(image_name):

    """
    Supports both prediction locations.

    IMPORTANT:
    Prefer outputs/prediction/<image>_predicted.png because this is
    the prediction-mask location used in the validated standalone
    Pipeline 3 tests (for example h5_predicted.png).

    The root outputs/<image>_predicted.png location is used only
    as a fallback when the validated prediction folder does not
    contain the mask.
    """

    candidates = [

        # FIRST: validated prediction-mask location used by
        # standalone Pipeline 3 tests.
        os.path.join(
            PREDICTION_DIR,
            image_name + "_predicted.png"
        ),

        # FALLBACK: Pipeline 3 automatic/default prediction location.
        os.path.join(
            OUTPUT_DIR,
            image_name + "_predicted.png"
        )
    ]

    for candidate in candidates:

        if os.path.exists(candidate):

            return os.path.abspath(
                candidate
            )

    return None


# ==========================================================
# FIND GENERATED PREDICTION AFTER PIPELINE 3
# ==========================================================

def find_prediction_after_pipeline3(image_name):

    """
    Pipeline 3 automatic mode normally writes:

        outputs/<image>_predicted.png

    Also check the legacy prediction directory.
    """

    candidates = [

        # FIRST: validated prediction-mask location used by
        # standalone Pipeline 3 tests.
        os.path.join(
            PREDICTION_DIR,
            image_name + "_predicted.png"
        ),

        # FALLBACK: Pipeline 3 automatic/default prediction location.
        os.path.join(
            OUTPUT_DIR,
            image_name + "_predicted.png"
        )
    ]

    for candidate in candidates:

        if os.path.exists(candidate):

            return os.path.abspath(
                candidate
            )

    return None


# ==========================================================
# ANALYZE IMAGE
# ==========================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    # ------------------------------------------------------
    # CHECK IMAGE
    # ------------------------------------------------------

    if "image" not in request.files:

        return api_error("NO_IMAGE", "No image was uploaded.", 400)


    uploaded_file = request.files[
        "image"
    ]


    if uploaded_file.filename == "":

        return api_error("NO_IMAGE", "No image was selected.", 400)


    # ------------------------------------------------------
    # SAFE FILE NAME
    # ------------------------------------------------------

    original_filename = secure_filename(
        uploaded_file.filename
    )

    if not original_filename:

        return api_error("INVALID_FILENAME", "The image filename is invalid.", 400)


    image_name = os.path.splitext(
        original_filename
    )[0]

    # ------------------------------------------------------
    # VALIDATE IMAGE CONTENT
    # ------------------------------------------------------
    validation_error = validate_uploaded_image(
        uploaded_file,
        original_filename
    )

    if validation_error is not None:
        return validation_error


    # ------------------------------------------------------
    # SAVE UPLOADED IMAGE
    # ------------------------------------------------------

    image_path = os.path.join(
        UPLOAD_DIR,
        original_filename
    )

    uploaded_file.save(
        image_path
    )

    # Use the original HiRISE dataset image when the uploaded
    # filename matches a known project image. This keeps the
    # frontend analysis identical to standalone Pipeline 3 tests.
    original_image_path = os.path.join(
        HIRISE_ORIGINAL_DIR,
        original_filename
    )

    if os.path.exists(original_image_path):
        analysis_image_path = os.path.abspath(
            original_image_path
        )
    else:
        analysis_image_path = os.path.abspath(
            image_path
        )


    print()
    print("=" * 70)
    print("FRONTEND ANALYSIS REQUEST")
    print("=" * 70)

    print(
        "Image:",
        original_filename
    )

    print(
        "Saved:",
        image_path
    )

    print(
        "Pipeline 3 input:",
        analysis_image_path
    )

    print("=" * 70)


    # ======================================================
    # PREDICTION MASK DECISION
    # ======================================================

    existing_prediction = (
        find_existing_prediction(
            image_name
        )
    )


    if existing_prediction is not None:

        prediction_mode = (
            "EXISTING PREDICTION MASK"
        )

        print()
        print("=" * 70)
        print("PREDICTION MASK FOUND")
        print("=" * 70)

        print(
            "Using existing prediction:",
            existing_prediction
        )

        print("=" * 70)

    else:

        prediction_mode = (
            "AUTOMATIC U-NET PREDICTION"
        )

        print()
        print("=" * 70)
        print("NO EXISTING PREDICTION MASK")
        print("=" * 70)

        print(
            "Pipeline 3 will automatically run the existing U-Net."
        )

        print("=" * 70)


    # ======================================================
    # RUN PIPELINE 3
    # ======================================================

    start_time = time.perf_counter()


    try:

        env = os.environ.copy()

        # Windows UTF-8 output
        env[
            "PYTHONIOENCODING"
        ] = "utf-8"


        # --------------------------------------------------
        # IMPORTANT:
        #
        # Existing mask:
        #
        # python pipeline3.py image prediction_mask
        #
        # New image:
        #
        # python pipeline3.py image
        #
        # This is the important fix.
        # --------------------------------------------------

        command = [

            sys.executable,

            PIPELINE3_PATH,

            analysis_image_path
        ]


        if existing_prediction is not None:

            command.append(
                existing_prediction
            )


        print()
        print("=" * 70)
        print("RUNNING PIPELINE 3")
        print("=" * 70)

        print(
            "Prediction Mode:",
            prediction_mode
        )

        print()

        print(
            "Command:"
        )

        print(
            " ".join(
                command
            )
        )

        print()


        result = subprocess.run(

            command,

            cwd=BASE_DIR,

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace",

            env=env
        )


    except Exception as exc:

        print()
        print("=" * 70)
        print("FLASK ERROR")
        print("=" * 70)

        print(
            str(exc)
        )

        print("=" * 70)


        return api_error(
            "PIPELINE_ERROR",
            "The image analysis pipeline could not complete. Please try again.",
            500
        )


    processing_time = (
        time.perf_counter()
        -
        start_time
    )


    # ======================================================
    # PIPELINE OUTPUT
    # ======================================================

    print()
    print("=" * 70)
    print("PIPELINE 3 STDOUT")
    print("=" * 70)

    print(
        result.stdout
    )


    if result.stderr:

        print()
        print("=" * 70)
        print("PIPELINE 3 STDERR")
        print("=" * 70)

        print(
            result.stderr
        )


    print("=" * 70)


    # ======================================================
    # PIPELINE FAILURE
    # ======================================================

    if result.returncode != 0:

        return api_error(
            "PIPELINE_ERROR",
            "The image analysis pipeline could not complete. Please try again.",
            500
        )


    # ======================================================
    # FIND FINAL PREDICTION
    # ======================================================

    prediction_path = (
        find_prediction_after_pipeline3(
            image_name
        )
    )


    if prediction_path is None:

        return api_error(
            "OUTPUT_NOT_FOUND",
            "Analysis completed, but the expected prediction output was not found.",
            500
        )


    # ======================================================
    # OUTPUT FILE PATHS
    # ======================================================

    cleaned_path = os.path.join(

        PIPELINE2_CLEANED_DIR,

        image_name +
        "_cleaned.png"
    )


    valley_path = os.path.join(

        PIPELINE2_VALLEY_DIR,

        image_name +
        "_valleys.png"
    )


    skeleton_path = os.path.join(

        PIPELINE2_SKELETON_DIR,

        image_name +
        "_skeleton.png"
    )


    direction_path = os.path.join(

        PIPELINE2_DIRECTION_DIR,

        image_name +
        "_flow_direction.png"
    )


    reconstruction_path = os.path.join(

        PIPELINE2_RECONSTRUCTION_DIR,

        image_name +
        "_flow_reconstruction.png"
    )


    # ======================================================
    # PIPELINE 2 JSON
    # ======================================================

    pipeline2_json_path = os.path.join(

        PIPELINE2_JSON_DIR,

        image_name +
        "_pipeline2_improved.json"
    )


    pipeline2_data = {}


    if os.path.exists(
        pipeline2_json_path
    ):

        try:

            with open(

                pipeline2_json_path,

                "r",

                encoding="utf-8"

            ) as f:

                pipeline2_data = json.load(
                    f
                )


        except Exception as exc:

            print(
                "Could not read Pipeline 2 JSON:",
                exc
            )


    # ======================================================
    # SUMMARY DATA
    # ======================================================

    summary_data = pipeline2_data.get(

        "summary",

        {}

    )


    if not isinstance(
        summary_data,
        dict
    ):

        summary_data = {}


    # ======================================================
    # ACCEPTED VALLEYS
    # ======================================================

    accepted_valleys = summary_data.get(
        "accepted_valleys"
    )


    if accepted_valleys is None:

        accepted_valleys = 0


    try:

        accepted_valleys = int(
            accepted_valleys
        )

    except Exception:

        accepted_valleys = 0


    # ======================================================
    # RECONSTRUCTED FLOW PATHS
    # ======================================================

    reconstructed_paths = summary_data.get(

        "reconstructed_valleys"

    )


    if reconstructed_paths is None:

        flow_results = pipeline2_data.get(

            "flow_results",

            []

        )


        if isinstance(
            flow_results,
            list
        ):

            reconstructed_paths = len(
                flow_results
            )

        else:

            reconstructed_paths = 0


    try:

        reconstructed_paths = int(
            reconstructed_paths
        )

    except Exception:

        reconstructed_paths = 0


    # ======================================================
    # ACCEPTED FLOW PATHS
    # ======================================================
    # Pipeline 2 stores accepted_path_count per reconstructed valley.
    # The frontend needs the overall total, so sum those per-valley
    # counts instead of using reconstructed_valleys.
    flow_results = pipeline2_data.get(
        "flow_results",
        []
    )

    if isinstance(flow_results, list):
        accepted_paths = sum(
            int(result.get("accepted_path_count", 0))
            for result in flow_results
            if isinstance(result, dict)
        )
    else:
        accepted_paths = 0

    # ======================================================
    # SKIPPED VALLEYS
    # ======================================================

    skipped_valleys = summary_data.get(

        "skipped_valleys",

        0

    )


    try:

        skipped_valleys = int(
            skipped_valleys
        )

    except Exception:

        skipped_valleys = 0


    # ======================================================
    # SKELETON PIXELS
    # ======================================================

    skeleton_pixels = summary_data.get(

        "skeleton_pixels",

        0

    )


    try:

        skeleton_pixels = int(
            skeleton_pixels
        )

    except Exception:

        skeleton_pixels = 0


    # ======================================================
    # SKELETON COMPONENTS
    # ======================================================

    skeleton_components = summary_data.get(

        "skeleton_components",

        0

    )


    try:

        skeleton_components = int(
            skeleton_components
        )

    except Exception:

        skeleton_components = 0


    # ======================================================
    # BEST PATH
    # ======================================================

    (
        best_path_length,
        best_path_confidence

    ) = find_best_path_information(

        pipeline2_data

    )


    # ======================================================
    # FALLBACK BEST PATH
    # ======================================================

    if best_path_length is None:

        best_path_length = summary_data.get(

            "best_path_length"

        )


    if best_path_confidence is None:

        best_path_confidence = summary_data.get(

            "best_path_confidence"

        )


    # ======================================================
    # FORMAT BEST PATH LENGTH
    # ======================================================

    if best_path_length is not None:

        try:

            best_path_length = float(
                best_path_length
            )

        except Exception:

            best_path_length = None


    # ======================================================
    # FORMAT CONFIDENCE
    # ======================================================

    if best_path_confidence is not None:

        try:

            best_path_confidence = float(
                best_path_confidence
            )

        except Exception:

            best_path_confidence = None


    # ======================================================
    # EVALUATION METRICS
    # ======================================================

    metrics_data = {}


    metrics_json_path = os.path.join(

        EVALUATION_DIR,

        image_name +
        "_metrics.json"
    )


    if os.path.exists(
        metrics_json_path
    ):

        try:

            with open(

                metrics_json_path,

                "r",

                encoding="utf-8"

            ) as f:

                metrics_data = json.load(
                    f
                )


        except Exception as exc:

            print(
                "Could not read metrics JSON:",
                exc
            )


    metrics = metrics_data.get(

        "metrics",

        metrics_data

    )


    # ======================================================
    # RELATIVE OUTPUT PATH
    # ======================================================

    def relative_output(path):

        return os.path.relpath(

            path,

            OUTPUT_DIR

        ).replace(
            "\\",
            "/"
        )


    # ======================================================
    # CHECK ACTUAL OUTPUTS
    # ======================================================

    def output_url(path):

        if path is None:
            return None

        if not os.path.exists(path):
            return None

        return (
            "/api/output/"
            +
            relative_output(path)
        )


    # ======================================================
    # BUILD API RESPONSE
    # ======================================================

    response = {

        "success":
            True,


        # ==================================================
        # IMAGE
        # ==================================================

        "image": {

            "name":
                original_filename,

            "url":
                f"/api/upload/{original_filename}"

        },


        # ==================================================
        # PREDICTION MODE
        # ==================================================

        "prediction_mode":
            prediction_mode,


        "prediction_path":
            prediction_path,


        # ==================================================
        # PANEL IMAGES
        # ==================================================

        "images": {

            # PANEL 1
            "original":
                f"/api/upload/{original_filename}",


            # PANEL 2
            "prediction":
                output_url(
                    prediction_path
                ),


            # PANEL 3
            "cleaned":
                output_url(
                    cleaned_path
                ),


            # PANEL 4
            "valley":
                output_url(
                    valley_path
                ),


            # PANEL 5
            "skeleton":
                output_url(
                    skeleton_path
                ),


            # PANEL 6
            "direction":
                output_url(
                    direction_path
                ),


            # PANEL 7
            "reconstruction":
                output_url(
                    reconstruction_path
                )

        },


        # ==================================================
        # PANEL 8 / 9 / 10 DATA
        # ==================================================

        "summary": {

            "valleys_detected":
                accepted_valleys,

            "flow_paths":
                reconstructed_paths,

            "accepted_paths":
                accepted_paths,

            "skeleton_pixels":
                skeleton_pixels,

            "skeleton_components":
                skeleton_components,

            "skipped_valleys":
                skipped_valleys,

            "best_path_length":
                best_path_length,

            "best_path_confidence":
                best_path_confidence,

            "processing_time":
                processing_time,

            "analysis_status":
                "Completed"

        },


        # ==================================================
        # PIPELINE STATUS
        # ==================================================

        "pipeline": {

            "model_used":
                "unet_model.pth",

            "sam2_mask":
                "Available",

            "prediction":
                "Generated",

            "cleaned_mask":
                "Generated",


            "valley_detection":

                (
                    f"{accepted_valleys} detected"

                    if accepted_valleys > 0

                    else
                    "No valleys detected"
                ),


            "skeleton":

                (
                    f"Generated ({skeleton_pixels} pixels)"

                    if skeleton_pixels > 0

                    else
                    "Generated"
                ),


            "flow_direction":

                (
                    "Available"

                    if reconstructed_paths > 0

                    else
                    "Not Available"
                ),


            "flow_reconstruction":

                (
                    f"{reconstructed_paths} "
                    f"{'path' if reconstructed_paths == 1 else 'paths'} "
                    f"reconstructed"

                    if reconstructed_paths > 0

                    else
                    "Not Available"
                )

        },


        # ==================================================
        # EVALUATION METRICS
        # ==================================================

        "metrics":
            metrics

    }


    # ======================================================
    # DEBUG OUTPUT
    # ======================================================

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        "Image:",
        original_filename
    )

    print(
        "Prediction mode:",
        prediction_mode
    )

    print(
        "Prediction:",
        prediction_path
    )

    print(
        "Accepted valleys:",
        accepted_valleys
    )

    print(
        "Reconstructed paths:",
        reconstructed_paths
    )

    print(
        "Skeleton pixels:",
        skeleton_pixels
    )

    print(
        "Skeleton components:",
        skeleton_components
    )

    print(
        "Skipped valleys:",
        skipped_valleys
    )

    print(
        "Best path length:",
        best_path_length
    )

    print(
        "Best path confidence:",
        best_path_confidence
    )

    print(
        "Processing time:",
        round(
            processing_time,
            2
        ),
        "seconds"
    )

    print("=" * 70)


    return jsonify(
        response
    )


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MARS WATER DETECTION - FLASK API")
    print("=" * 60)

    print()

    print(
        "Pipeline 3:",
        PIPELINE3_PATH
    )

    print(
        "Upload directory:",
        UPLOAD_DIR
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Prediction directory:",
        PREDICTION_DIR
    )

    print()

    print(
        "API:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False
    )
