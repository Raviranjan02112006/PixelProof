
import os


# ============================================================
# 1. PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

APP_HOST = "127.0.0.1"
APP_PORT = 5000
DEBUG = True

MAX_CONTENT_LENGTH = 10 * 1024 * 1024


# ============================================================
# 3. DIRECTORY PATHS
# ============================================================

SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "models")

INDIVIDUAL_MODEL_DIR = os.path.join(
    MODEL_DIR, "individual"
)

MATHEMATICAL_MODEL_DIR = os.path.join(
    MODEL_DIR, "mathematical"
)

CNN_MODEL_DIR = os.path.join(
    MODEL_DIR, "cnn"
)

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

RESULT_FOLDER = os.path.join(
    STATIC_DIR, "results"
)

REPORT_FOLDER = os.path.join(BASE_DIR, "reports")


# ============================================================
# 4. INDIVIDUAL MATHEMATICAL MODEL PATHS
# ============================================================

INDIVIDUAL_MODELS = {

    "Logistic Regression": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "logistic_regression.joblib"
    ),

    "Random Forest": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "random_forest.joblib"
    ),

    "SVM": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "svm.joblib"
    ),

    "Gradient Boosting": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "gradient_boosting.joblib"
    ),

    "MDMD-35": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "mdmd_35.joblib"
    ),

    "ZS-35": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "zs_35.joblib"
    ),

    "EDS-35": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "eds_35.joblib"
    ),

    "CFS-35": os.path.join(
        INDIVIDUAL_MODEL_DIR,
        "cfs_35.joblib"
    )
}


# ============================================================
# 5. MDMD-35 INFERENCE ARTIFACTS
# ============================================================

MDMD_ARTIFACTS_PATH = os.path.join(
    INDIVIDUAL_MODEL_DIR,
    "mdmd_35_artifacts.joblib"
)


# ============================================================
# 6. MATHEMATICAL ENSEMBLE MODEL
# ============================================================

MATHEMATICAL_ENSEMBLE_PATH = os.path.join(
    MATHEMATICAL_MODEL_DIR,
    "pixelproof_final_soft_ensemble.joblib"
)


# ============================================================
# 7. CNN MODEL
# ============================================================

CNN_MODEL_PATH = os.path.join(
    CNN_MODEL_DIR,
    "pixelproof_resnet50_best.keras"
)


# ============================================================
# 8. MATHEMATICAL FEATURE CONFIGURATION
# ============================================================

EXPECTED_FEATURE_COUNT = 35

FEATURE_NAMES = [

    # Fourier features (5)
    "fft_mean",
    "fft_std",
    "low_frequency_ratio",
    "high_frequency_ratio",
    "spectral_entropy",

    # Correlation features (5)
    "horizontal_correlation",
    "vertical_correlation",
    "diagonal_correlation",
    "local_variance_mean",
    "local_variance_std",

    # Spatial features (16)
    "gradient_mean",
    "gradient_std",
    "gradient_max",
    "edge_density",
    "intensity_mean",
    "intensity_std",
    "intensity_skewness",
    "intensity_kurtosis",

    "gradient_direction_0",
    "gradient_direction_1",
    "gradient_direction_2",
    "gradient_direction_3",
    "gradient_direction_4",
    "gradient_direction_5",
    "gradient_direction_6",
    "gradient_direction_7",

    # Depth features (9)
    "depth_mean",
    "depth_std",
    "depth_min",
    "depth_max",
    "depth_median",
    "depth_q25",
    "depth_q75",
    "depth_gradient_mean",
    "depth_gradient_std"
]


# ============================================================
# 9. CNN CONFIGURATION
# ============================================================

CNN_IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "REAL",
    "FAKE"
]

CNN_CLASS_NAMES = CLASS_NAMES


# ============================================================
# 10. PREDICTION CONFIGURATION
# ============================================================

CLASS_REAL = "REAL"
CLASS_FAKE = "FAKE"

DEFAULT_PREDICTION_THRESHOLD = 0.5


# ============================================================
# 11. RESULT IMAGE CONFIGURATION
# ============================================================

RESULT_IMAGE_FORMAT = "png"
RESULT_IMAGE_DPI = 150


# ============================================================
# 12. CREATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [

    SRC_DIR,
    MODEL_DIR,
    INDIVIDUAL_MODEL_DIR,
    MATHEMATICAL_MODEL_DIR,
    CNN_MODEL_DIR,
    TEMPLATE_DIR,
    STATIC_DIR,
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    RESULT_FOLDER,
    REPORT_FOLDER
]


def create_required_directories():

    for directory in REQUIRED_DIRECTORIES:

        os.makedirs(
            directory,
            exist_ok=True
        )


create_required_directories()
