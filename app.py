
# ============================================================
# PIXELPROOF
# app.py — COMPLETE FLASK APPLICATION
#
# Pipeline:
# 1. Image upload and validation
# 2. Fourier analysis
# 3. Pixel correlation analysis
# 4. Spatial structure analysis
# 5. Relative depth analysis
# 6. Extraction of all 35 mathematical features
# 7. Eight individual mathematical model predictions
# 8. Mathematical soft-ensemble prediction
# 9. ResNet50 CNN prediction
# 10. Final hybrid prediction
# ============================================================

import os
import uuid
import logging
from pathlib import Path

import numpy as np

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

from config import (
    MAX_CONTENT_LENGTH,
    RESULT_FOLDER,
    APP_HOST,
    APP_PORT,
    DEBUG
)

import config


# ============================================================
# 2. VISUAL ANALYSIS MODULES
# ============================================================

from src.fourier import analyze_fourier

from src.correlation import analyze_correlation

from src.stereogram import analyze_spatial_structure

from src.depth_estimation import (
    analyze_depth,
    get_depth_model
)


# ============================================================
# 3. MATHEMATICAL FEATURE EXTRACTION
# ============================================================

from src.feature_extraction import (
    extract_all_features
)


# ============================================================
# 4. INDIVIDUAL MATHEMATICAL MODELS
# ============================================================

from src.individual_predictions import (
    predict_individual_models,
    get_individual_prediction_summary
)


# ============================================================
# 5. HYBRID PREDICTION
# ============================================================

from src.hybrid_prediction import (
    predict_hybrid
)


# ============================================================
# 6. FLASK INITIALIZATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PixelProof")


# ============================================================
# 7. DIRECTORY CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = Path(
    getattr(
        config,
        "UPLOAD_FOLDER",
        BASE_DIR / "uploads"
    )
)

RESULT_FOLDER = Path(RESULT_FOLDER)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ============================================================
# 8. JSON SERIALIZATION
# ============================================================

def make_json_serializable(value):

    """
    Converts NumPy and Python objects into JSON-compatible values.
    """

    if isinstance(value, dict):

        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return make_json_serializable(value.item())

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, float):

        if not np.isfinite(value):
            return None

        return float(value)

    # Check bool before int because bool is a subclass of int.
    if isinstance(value, bool):

        return value

    if isinstance(value, int):

        return int(value)

    if isinstance(value, str) or value is None:

        return value

    return str(value)


# ============================================================
# 9. FILE VALIDATION
# ============================================================

def allowed_file(filename):

    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# 10. IMAGE LOADING AND VALIDATION
# ============================================================

def load_uploaded_image(file_storage):

    """
    Validates the uploaded image and returns a PIL RGB image.
    """

    try:

        image = Image.open(file_storage)

        image.verify()

        file_storage.seek(0)

        image = Image.open(file_storage)

        image = image.convert("RGB")

        if image.width < 3 or image.height < 3:

            raise ValueError(
                "Image dimensions must be at least 3 × 3 pixels."
            )

        return image

    except UnidentifiedImageError:

        raise ValueError(
            "The uploaded file is not a valid image."
        )

    except OSError:

        raise ValueError(
            "Unable to read the uploaded image."
        )


# ============================================================
# 11. SAVE UPLOADED IMAGE
# ============================================================

def save_uploaded_image(image, analysis_id):

    """
    Saves the validated image for feature extraction.
    """

    image_path = UPLOAD_FOLDER / f"{analysis_id}.png"

    image.save(
        image_path,
        format="PNG"
    )

    return image_path


# ============================================================
# 12. CREATE RESULT PATHS
# ============================================================

def create_result_paths(analysis_id):

    """
    Creates output paths for all four visual analyses.
    """

    paths = {

        "fourier": RESULT_FOLDER / f"{analysis_id}_fourier.png",

        "correlation": RESULT_FOLDER / f"{analysis_id}_correlation.png",

        "spatial": RESULT_FOLDER / f"{analysis_id}_spatial.png",

        "depth": RESULT_FOLDER / f"{analysis_id}_depth.png"

    }

    return paths


# ============================================================
# 13. VISUAL ANALYSIS PIPELINE
# ============================================================

def perform_visual_analysis(image, analysis_id):

    """
    Performs the four existing image-analysis operations.
    """

    visual_image = image.copy()

    visual_image.thumbnail(
        (1024, 1024)
    )

    image_array = np.asarray(
        visual_image,
        dtype=np.uint8
    )

    paths = create_result_paths(analysis_id)

    # Fourier analysis

    logger.info("Running Fourier analysis...")

    fourier_details = analyze_fourier(
        image_array,
        str(paths["fourier"])
    )

    # Correlation analysis

    logger.info("Running correlation analysis...")

    correlation_details = analyze_correlation(
        image_array,
        str(paths["correlation"])
    )

    # Spatial analysis

    logger.info("Running spatial analysis...")

    spatial_details = analyze_spatial_structure(
        image_array,
        str(paths["spatial"])
    )

    # Depth analysis

    logger.info("Running depth analysis...")

    depth_details = analyze_depth(
        visual_image,
        str(paths["depth"])
    )

    # Prepare visualization response

    visual_results = {

        "fourier": {

            "image": url_for(
                "static",
                filename=f"results/{paths['fourier'].name}"
            ),

            "details": fourier_details

        },

        "correlation": {

            "image": url_for(
                "static",
                filename=f"results/{paths['correlation'].name}"
            ),

            "details": correlation_details

        },

        "spatial": {

            "image": url_for(
                "static",
                filename=f"results/{paths['spatial'].name}"
            ),

            "details": spatial_details

        },

        "depth": {

            "image": url_for(
                "static",
                filename=f"results/{paths['depth'].name}"
            ),

            "details": depth_details

        }

    }

    return visual_results


# ============================================================
# 14. EXTRACT ALL 35 MATHEMATICAL FEATURES
# ============================================================

def extract_image_features(image, image_path):

    """
    Extracts the original 35-feature vector:

    Fourier features       : 5
    Correlation features   : 5
    Spatial features       : 16
    Depth features         : 9

    Total                  : 35
    """

    logger.info("Loading depth model for feature extraction...")

    depth_model = get_depth_model()

    gray_image = np.asarray(
        image.convert("L"),
        dtype=np.uint8
    )

    logger.info("Extracting 35 mathematical features...")

    features = extract_all_features(
        gray_image,
        str(image_path),
        depth_model
    )

    if not isinstance(features, dict):

        raise TypeError(
            "Feature extraction must return a dictionary."
        )

    if len(features) != 35:

        raise ValueError(
            f"Expected 35 features, but received {len(features)}."
        )

    feature_values = np.asarray(
        list(features.values()),
        dtype=np.float64
    )

    if not np.all(np.isfinite(feature_values)):

        raise ValueError(
            "Feature extraction returned NaN or infinite values."
        )

    logger.info(
        "Successfully extracted %d features.",
        len(features)
    )

    return features


# ============================================================
# 15. INDIVIDUAL MATHEMATICAL MODEL PREDICTIONS
# ============================================================

def perform_individual_predictions(features):

    """
    Runs all eight individual mathematical models.

    Returns:
        individual_results:
            Dictionary containing exactly the eight model records.

        individual_summary:
            Aggregate statistics kept separately.
    """

    logger.info(
        "Running eight individual mathematical models..."
    )

    individual_results = predict_individual_models(
        features
    )

    individual_summary = get_individual_prediction_summary(
        individual_results
    )

    return individual_results, individual_summary


# ============================================================
# 16. MATHEMATICAL ENSEMBLE, CNN AND HYBRID PREDICTION
# ============================================================

def perform_hybrid_prediction(image, features):

    """
    Runs:

    1. Mathematical soft ensemble
    2. ResNet50 CNN
    3. Final hybrid prediction

    The hybrid module already calls the mathematical ensemble
    and CNN internally. Therefore, they are not called separately
    here a second time.
    """

    logger.info(
        "Running mathematical ensemble and ResNet50 CNN..."
    )

    hybrid_result = predict_hybrid(
        image,
        features
    )

    mathematical_result = hybrid_result.get(
        "mathematical_result",
        {}
    )

    cnn_result = hybrid_result.get(
        "cnn_result",
        {}
    )

    if not mathematical_result:

        raise RuntimeError(
            "Mathematical ensemble result is missing."
        )

    if not cnn_result:

        raise RuntimeError(
            "CNN prediction result is missing."
        )

    return (
        mathematical_result,
        cnn_result,
        hybrid_result
    )


# ============================================================
# 17. HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return render_template("index.html")


# ============================================================
# 18. ABOUT PAGE
# ============================================================

@app.route("/about", methods=["GET"])
def about():

    return render_template("about.html")


# ============================================================
# 19. MAIN IMAGE ANALYSIS ROUTE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # ----------------------------------------------------
        # A. Check uploaded file
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error": "No image was uploaded."

            }), 400

        uploaded_file = request.files["image"]

        if uploaded_file.filename == "":

            return jsonify({

                "success": False,

                "error": "Please select an image."

            }), 400

        if not allowed_file(uploaded_file.filename):

            return jsonify({

                "success": False,

                "error": "Only PNG, JPG and JPEG images are supported."

            }), 400

        original_filename = secure_filename(
            uploaded_file.filename
        )

        # ----------------------------------------------------
        # B. Validate and load image
        # ----------------------------------------------------

        logger.info(
            "Starting PixelProof analysis for %s",
            original_filename
        )

        image = load_uploaded_image(
            uploaded_file
        )

        original_width = image.width
        original_height = image.height

        # ----------------------------------------------------
        # C. Generate analysis ID
        # ----------------------------------------------------

        analysis_id = str(
            uuid.uuid4()
        )

        # ----------------------------------------------------
        # D. Save image
        # ----------------------------------------------------

        image_path = save_uploaded_image(
            image,
            analysis_id
        )

        # ----------------------------------------------------
        # E. Four visual analyses
        # ----------------------------------------------------

        logger.info(
            "[%s] Starting visual analysis.",
            analysis_id
        )

        visual_results = perform_visual_analysis(
            image,
            analysis_id
        )

        # ----------------------------------------------------
        # F. Extract 35 features
        # ----------------------------------------------------

        logger.info(
            "[%s] Starting feature extraction.",
            analysis_id
        )

        features = extract_image_features(
            image,
            image_path
        )

        # ----------------------------------------------------
        # G. Eight individual mathematical models
        # ----------------------------------------------------

        logger.info(
            "[%s] Starting individual model predictions.",
            analysis_id
        )

        individual_results, individual_summary = (
            perform_individual_predictions(features)
        )

        # ----------------------------------------------------
        # H. Mathematical ensemble + CNN + hybrid
        # ----------------------------------------------------

        logger.info(
            "[%s] Starting hybrid prediction.",
            analysis_id
        )

        (
            mathematical_result,
            cnn_result,
            hybrid_result
        ) = perform_hybrid_prediction(
            image,
            features
        )

        # ----------------------------------------------------
        # I. Normalize final hybrid result
        # ----------------------------------------------------

        final_prediction = hybrid_result.get(
            "final_prediction",
            hybrid_result.get("prediction")
        )

        if final_prediction is None:

            raise ValueError(
                "Hybrid prediction did not return a final verdict."
            )

        hybrid_result["prediction"] = final_prediction

        # ----------------------------------------------------
        # J. Complete response
        # ----------------------------------------------------

        response = {

            "success": True,

            "analysis_id": analysis_id,

            "filename": original_filename,

            "image": {

                "width": original_width,

                "height": original_height

            },

            # Four visual analyses
            "mathematical_analysis": visual_results,

            # Exactly eight individual mathematical model results
            "individual_models": individual_results,

            # Aggregate summary is separate
            "individual_summary": individual_summary,

            # Combined mathematical ensemble
            "mathematical_prediction": mathematical_result,

            # ResNet50 CNN prediction
            "cnn_prediction": cnn_result,

            # Final hybrid prediction
            "hybrid_prediction": hybrid_result,

            # Direct final-result fields
            "final_prediction": final_prediction,

            "hybrid_real_evidence_percentage": hybrid_result.get(
                "hybrid_real_evidence_percentage"
            ),

            "hybrid_fake_evidence_percentage": hybrid_result.get(
                "hybrid_fake_evidence_percentage"
            ),

            "model_agreement": hybrid_result.get(
                "model_agreement"
            ),

            # Pipeline status
            "prediction_status": "completed",

            # Feature extraction information
            "feature_extraction": {

                "total_features": len(features),

                "feature_count_expected": 35,

                "status": "completed"

            }

        }

        response = make_json_serializable(
            response
        )

        logger.info(
            "[%s] PixelProof analysis completed. Verdict: %s",
            analysis_id,
            final_prediction
        )

        return jsonify(response), 200

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except ValueError as error:

        logger.warning(
            "Validation or processing error: %s",
            str(error)
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400

    except FileNotFoundError as error:

        logger.exception(
            "Required model or file not found."
        )

        return jsonify({

            "success": False,

            "error": (
                "A required model or file is missing. "
                "Check the models directory and config.py."
            ),

            "details": str(error)

        }), 500

    except Exception as error:

        logger.exception(
            "Unexpected PixelProof analysis error."
        )

        return jsonify({

            "success": False,

            "error": "Image analysis failed.",

            "details": str(error)

        }), 500


# ============================================================
# 20. FILE SIZE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": "Image exceeds the maximum upload size of 10 MB."

    }), 413


# ============================================================
# 21. APPLICATION HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,

        "application": "PixelProof",

        "status": "running"

    }), 200


# ============================================================
# 22. APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting PixelProof Flask application..."
    )

    logger.info(
        "Host: %s",
        APP_HOST
    )

    logger.info(
        "Port: %s",
        APP_PORT
    )

    app.run(

        host=APP_HOST,

        port=APP_PORT,

        debug=DEBUG

    )
