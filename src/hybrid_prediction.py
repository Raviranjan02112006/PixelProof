
# ============================================================
# PIXELPROOF
# STEP 12 — HYBRID PREDICTION
#
# Mathematical Ensemble + ResNet50 CNN
# ============================================================

import numpy as np

from src.mathematical_prediction import (
    predict_mathematical
)

from src.cnn_prediction import (
    predict_cnn
)


# ============================================================
# 1. HYBRID CONFIGURATION
# ============================================================

MATHEMATICAL_WEIGHT = 0.50
CNN_WEIGHT = 0.50

HYBRID_THRESHOLD = 0.50


# ============================================================
# 2. VALIDATE HYBRID CONFIGURATION
# ============================================================

def validate_configuration():

    if not np.isclose(
        MATHEMATICAL_WEIGHT + CNN_WEIGHT,
        1.0
    ):
        raise ValueError(
            "Mathematical and CNN weights must sum to 1."
        )

    if not 0.0 <= HYBRID_THRESHOLD <= 1.0:
        raise ValueError(
            "Hybrid threshold must be between 0 and 1."
        )


# ============================================================
# 3. CALCULATE HYBRID EVIDENCE
# ============================================================

def calculate_hybrid_evidence(
    mathematical_fake,
    cnn_fake
):

    hybrid_fake = (
        MATHEMATICAL_WEIGHT * mathematical_fake
        +
        CNN_WEIGHT * cnn_fake
    )

    hybrid_real = 1.0 - hybrid_fake

    return hybrid_real, hybrid_fake


# ============================================================
# 4. FINAL HYBRID PREDICTION
# ============================================================

def predict_hybrid(image, features):

    validate_configuration()

    # ----------------------------------------
    # MATHEMATICAL ENSEMBLE PREDICTION
    # ----------------------------------------

    mathematical_result = predict_mathematical(
        features
    )

    # ----------------------------------------
    # CNN PREDICTION
    # ----------------------------------------

    cnn_result = predict_cnn(
        image
    )

    # ----------------------------------------
    # EXTRACT FAKE EVIDENCE
    # ----------------------------------------

    mathematical_fake = (
        mathematical_result[
            "fake_support_percentage"
        ] / 100.0
    )

    cnn_fake = (
        cnn_result[
            "fake_evidence_percentage"
        ] / 100.0
    )

    # ----------------------------------------
    # COMBINE BOTH APPROACHES
    # ----------------------------------------

    hybrid_real, hybrid_fake = (
        calculate_hybrid_evidence(
            mathematical_fake,
            cnn_fake
        )
    )

    # ----------------------------------------
    # FINAL VERDICT
    # ----------------------------------------

    prediction = (
        "FAKE"
        if hybrid_fake >= HYBRID_THRESHOLD
        else "REAL"
    )

    # ----------------------------------------
    # AGREEMENT CHECK
    # ----------------------------------------

    mathematical_prediction = (
        mathematical_result["prediction"]
    )

    cnn_prediction = (
        cnn_result["prediction"]
    )

    models_agree = (
        mathematical_prediction == cnn_prediction
    )

    if models_agree:

        agreement_status = "AGREEMENT"

    else:

        agreement_status = "DISAGREEMENT"

    # ----------------------------------------
    # RETURN COMPLETE RESULT
    # ----------------------------------------

    return {

        "project": "PixelProof",

        "final_prediction": prediction,

        "hybrid_fake_evidence_percentage": round(
            hybrid_fake * 100,
            2
        ),

        "hybrid_real_evidence_percentage": round(
            hybrid_real * 100,
            2
        ),

        "hybrid_threshold": HYBRID_THRESHOLD,

        "hybrid_weights": {

            "mathematical_weight_percentage": (
                MATHEMATICAL_WEIGHT * 100
            ),

            "cnn_weight_percentage": (
                CNN_WEIGHT * 100
            )
        },

        "model_agreement": {

            "status": agreement_status,

            "mathematical_prediction": (
                mathematical_prediction
            ),

            "cnn_prediction": cnn_prediction,

            "models_agree": models_agree
        },

        "mathematical_result": mathematical_result,

        "cnn_result": cnn_result,

        "evidence_note": (
            "Hybrid evidence is a weighted combination "
            "of the mathematical ensemble support and CNN "
            "sigmoid score. It is not a calibrated probability."
        )
    }


# ============================================================
# 5. DISPLAY HYBRID RESULT
# ============================================================

def display_hybrid_result(result):

    print("\n")
    print("=" * 65)
    print("              PIXELPROOF — FINAL ANALYSIS")
    print("=" * 65)

    print("\nMATHEMATICAL ENSEMBLE")
    print("-" * 65)

    math_result = result["mathematical_result"]

    print(
        "REAL Support:",
        math_result["real_support_percentage"],
        "%"
    )

    print(
        "FAKE Support:",
        math_result["fake_support_percentage"],
        "%"
    )

    print(
        "Prediction:",
        math_result["prediction"]
    )

    print("\nRESNET50 CNN")
    print("-" * 65)

    cnn_result = result["cnn_result"]

    print(
        "REAL Evidence:",
        cnn_result["real_evidence_percentage"],
        "%"
    )

    print(
        "FAKE Evidence:",
        cnn_result["fake_evidence_percentage"],
        "%"
    )

    print(
        "Prediction:",
        cnn_result["prediction"]
    )

    print("\nMODEL AGREEMENT")
    print("-" * 65)

    agreement = result["model_agreement"]

    print(
        "Status:",
        agreement["status"]
    )

    print("\nHYBRID RESULT")
    print("-" * 65)

    print(
        "REAL Evidence:",
        result["hybrid_real_evidence_percentage"],
        "%"
    )

    print(
        "FAKE Evidence:",
        result["hybrid_fake_evidence_percentage"],
        "%"
    )

    print(
        "Final Prediction:",
        result["final_prediction"]
    )

    print("=" * 65)


# ============================================================
# 6. DIRECT USAGE
# ============================================================

if __name__ == "__main__":

    print("PixelProof Hybrid Prediction Module")
    print("------------------------------------")

    validate_configuration()

    print("Mathematical weight:", MATHEMATICAL_WEIGHT)
    print("CNN weight:", CNN_WEIGHT)

    print("Hybrid threshold:", HYBRID_THRESHOLD)

    print("\nConfiguration check completed.")

    print(
        "\nTo predict an image, call:\n"
        "predict_hybrid(image, features)"
    )
