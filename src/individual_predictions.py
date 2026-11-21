
# ============================================================
# PIXELPROOF
# src/individual_predictions.py
#
# INDIVIDUAL PREDICTION ENGINE — ALL 8 MODELS
#
# Machine Learning:
#   1. Logistic Regression
#   2. Random Forest
#   3. SVM
#   4. Gradient Boosting
#
# Original Statistical Techniques:
#   5. MDMD-35
#   6. ZS-35
#   7. EDS-35
#   8. CFS-35
#
# Custom techniques reproduce the original Colab formulas.
# ============================================================

import os
import logging
import joblib
import numpy as np
import pandas as pd

from functools import lru_cache

from config import (
    INDIVIDUAL_MODELS,
    MATHEMATICAL_ENSEMBLE_PATH,
    MDMD_ARTIFACTS_PATH,
    FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    DEFAULT_PREDICTION_THRESHOLD
)


logger = logging.getLogger(__name__)

EPSILON = 1e-10


# ============================================================
# 1. LOAD INDIVIDUAL MODELS
# ============================================================

@lru_cache(maxsize=1)
def load_individual_models():

    loaded_models = {}

    print("\nLoading eight individual mathematical models...")

    for model_name, model_path in INDIVIDUAL_MODELS.items():

        print(f"Loading {model_name}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found for {model_name}: {model_path}"
            )

        model = joblib.load(model_path)

        if model is None:
            raise ValueError(
                f"Empty model file: {model_name}"
            )

        loaded_models[model_name] = model

    print("\nAll eight individual models loaded.")

    return loaded_models


# ============================================================
# 2. LOAD MATHEMATICAL ENSEMBLE METADATA
# ============================================================

@lru_cache(maxsize=1)
def load_ensemble_metadata():

    if not os.path.exists(MATHEMATICAL_ENSEMBLE_PATH):
        raise FileNotFoundError(
            "Mathematical ensemble file not found: "
            + MATHEMATICAL_ENSEMBLE_PATH
        )

    metadata = joblib.load(MATHEMATICAL_ENSEMBLE_PATH)

    required_keys = [
        "models",
        "weights",
        "model_thresholds",
        "mathematical_score_scales",
        "ensemble_threshold",
        "feature_names",
        "model_names"
    ]

    for key in required_keys:

        if key not in metadata:
            raise KeyError(
                f"Missing ensemble metadata key: {key}"
            )

    if len(metadata["feature_names"]) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Ensemble metadata feature count does not match configuration."
        )

    if list(metadata["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError(
            "Ensemble feature order does not match config.FEATURE_NAMES."
        )

    return metadata


# ============================================================
# 3. LOAD ORIGINAL MDMD-35 ARTIFACTS
# ============================================================

@lru_cache(maxsize=1)
def load_mdmd_artifacts():

    if not os.path.exists(MDMD_ARTIFACTS_PATH):
        raise FileNotFoundError(
            "MDMD-35 supporting artifact not found: "
            + MDMD_ARTIFACTS_PATH
        )

    artifacts = joblib.load(MDMD_ARTIFACTS_PATH)

    required_keys = [
        "domain_scalers",
        "mdmd_references",
        "domain_weights",
        "feature_domains",
        "all_features",
        "epsilon"
    ]

    for key in required_keys:

        if key not in artifacts:
            raise KeyError(
                f"MDMD-35 artifact is missing: {key}"
            )

    if len(artifacts["all_features"]) != 35:
        raise ValueError(
            "MDMD-35 must contain exactly 35 features."
        )

    if list(artifacts["all_features"]) != list(FEATURE_NAMES):
        raise ValueError(
            "MDMD-35 feature order differs from config.FEATURE_NAMES."
        )

    weights = artifacts["domain_weights"]

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(
            "MDMD-35 domain weights do not sum to 1."
        )

    for domain_name, features in artifacts["feature_domains"].items():

        if domain_name not in artifacts["domain_scalers"]:
            raise KeyError(
                f"Missing scaler for MDMD domain: {domain_name}"
            )

        if domain_name not in artifacts["mdmd_references"]:
            raise KeyError(
                f"Missing references for MDMD domain: {domain_name}"
            )

        for class_name in ["REAL", "FAKE"]:

            reference = artifacts["mdmd_references"][
                domain_name
            ][class_name]

            for key in [
                "mean",
                "inverse_covariance"
            ]:

                if key not in reference:
                    raise KeyError(
                        f"Missing MDMD reference value: "
                        f"{domain_name}/{class_name}/{key}"
                    )

            if len(reference["mean"]) != len(features):
                raise ValueError(
                    f"Invalid MDMD mean dimensions: {domain_name}/{class_name}"
                )

            if np.asarray(
                reference["inverse_covariance"]
            ).shape != (len(features), len(features)):

                raise ValueError(
                    f"Invalid MDMD covariance dimensions: "
                    f"{domain_name}/{class_name}"
                )

    print("Original MDMD-35 artifacts loaded successfully.")

    return artifacts


# ============================================================
# 4. PREPARE INPUT FEATURES
# ============================================================

def prepare_features(features):

    feature_names = list(FEATURE_NAMES)

    if isinstance(features, pd.DataFrame):

        if len(features) != 1:
            raise ValueError(
                "Individual prediction expects exactly one image."
            )

        if list(features.columns) != feature_names:
            raise ValueError(
                "Input DataFrame does not match the required "
                "35-feature order."
            )

        X = features.copy()

    elif isinstance(features, dict):

        missing = [
            name for name in feature_names
            if name not in features
        ]

        if missing:
            raise ValueError(
                f"Missing extracted features: {missing}"
            )

        X = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names
        )

    else:

        values = np.asarray(features, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.shape != (1, EXPECTED_FEATURE_COUNT):
            raise ValueError(
                f"Expected one image with 35 features. "
                f"Received shape {values.shape}."
            )

        X = pd.DataFrame(
            values,
            columns=feature_names
        )

    X = X.astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Extracted features contain NaN or infinite values."
        )

    return X


# ============================================================
# 5. MACHINE LEARNING MODEL SCORING
# ============================================================

def get_ml_fake_score(model, X):

    if not hasattr(model, "predict_proba"):

        raise TypeError(
            "ML model does not support predict_proba."
        )

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 in classes:
        fake_index = classes.index(1)

    elif "FAKE" in classes:
        fake_index = classes.index("FAKE")

    else:
        raise ValueError(
            "FAKE class not found in model classes."
        )

    score = float(probabilities[0][fake_index])

    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# 6. ORIGINAL MDMD-35 IMPLEMENTATION
# ============================================================

def prepare_mdmd_scaled_features(X, artifacts):

    feature_domains = artifacts["feature_domains"]

    all_features = artifacts["all_features"]

    X_scaled = pd.DataFrame(index=X.index)

    for domain_name, domain_features in feature_domains.items():

        scaler = artifacts["domain_scalers"][domain_name]

        domain_values = X[domain_features]

        scaled_values = scaler.transform(domain_values)

        X_scaled[domain_features] = scaled_values

    X_scaled = X_scaled[all_features]

    return X_scaled


def calculate_mdmd_mahalanobis_distance(
    X_scaled,
    domain_name,
    class_name,
    artifacts
):

    features = artifacts["feature_domains"][domain_name]

    reference = artifacts["mdmd_references"][
        domain_name
    ][class_name]

    X_values = X_scaled[features].to_numpy(dtype=np.float64)

    mean_vector = np.asarray(
        reference["mean"],
        dtype=np.float64
    )

    inverse_covariance = np.asarray(
        reference["inverse_covariance"],
        dtype=np.float64
    )

    difference = X_values - mean_vector

    distances = np.einsum(
        "ij,jk,ik->i",
        difference,
        inverse_covariance,
        difference
    )

    return np.maximum(distances, 0.0)


def calculate_mdmd_domain_evidence(X_scaled, artifacts):

    evidence = pd.DataFrame(index=X_scaled.index)

    epsilon = artifacts["epsilon"]

    for domain_name in artifacts["feature_domains"]:

        real_distance = calculate_mdmd_mahalanobis_distance(
            X_scaled,
            domain_name,
            "REAL",
            artifacts
        )

        fake_distance = calculate_mdmd_mahalanobis_distance(
            X_scaled,
            domain_name,
            "FAKE",
            artifacts
        )

        evidence[domain_name] = (
            (real_distance - fake_distance)
            /
            (real_distance + fake_distance + epsilon)
        )

    return evidence


def calculate_mdmd_raw_score(X, artifacts=None):

    if artifacts is None:
        artifacts = load_mdmd_artifacts()

    X = prepare_features(X)

    X_scaled = prepare_mdmd_scaled_features(
        X,
        artifacts
    )

    evidence_df = calculate_mdmd_domain_evidence(
        X_scaled,
        artifacts
    )

    # Original Colab weighted multidomain score
    score = 0.0

    for domain_name, weight in artifacts["domain_weights"].items():

        score += (
            float(weight)
            *
            float(evidence_df[domain_name].iloc[0])
        )

    return float(score)


# ============================================================
# 7. ORIGINAL ZS-35 IMPLEMENTATION
# ============================================================

def calculate_zs35_raw_score(model, X):

    feature_names = model["feature_names"]

    if list(X.columns) != list(feature_names):
        raise ValueError(
            "ZS-35 feature order does not match saved model."
        )

    values = X.to_numpy(dtype=float)

    if values.shape[1] != 35:
        raise ValueError(
            "ZS-35 requires exactly 35 features."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "ZS-35 input contains NaN or infinite values."
        )

    real_mean = np.asarray(
        model["real_mean"], dtype=float
    )

    fake_mean = np.asarray(
        model["fake_mean"], dtype=float
    )

    real_std_safe = np.asarray(
        model["real_std_safe"], dtype=float
    )

    fake_std_safe = np.asarray(
        model["fake_std_safe"], dtype=float
    )

    real_z = np.abs(
        (values - real_mean) / real_std_safe
    )

    fake_z = np.abs(
        (values - fake_mean) / fake_std_safe
    )

    real_scores = np.mean(real_z, axis=1)

    fake_scores = np.mean(fake_z, axis=1)

    # Original Colab evidence
    fake_evidence = real_scores - fake_scores

    return float(fake_evidence[0])


# ============================================================
# 8. ORIGINAL EDS-35 IMPLEMENTATION
# ============================================================

def calculate_eds35_raw_score(model, X):

    feature_names = model["feature_names"]

    if list(X.columns) != list(feature_names):
        raise ValueError(
            "EDS-35 feature order does not match saved model."
        )

    values = X.to_numpy(dtype=float)

    if values.shape[1] != 35:
        raise ValueError(
            "EDS-35 requires exactly 35 features."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "EDS-35 input contains NaN or infinite values."
        )

    scaler = model["scaler"]

    scaled_values = scaler.transform(X)

    real_centroid = np.asarray(
        model["real_centroid"],
        dtype=float
    )

    fake_centroid = np.asarray(
        model["fake_centroid"],
        dtype=float
    )

    real_distances = np.linalg.norm(
        scaled_values - real_centroid,
        axis=1
    )

    fake_distances = np.linalg.norm(
        scaled_values - fake_centroid,
        axis=1
    )

    # Original Colab evidence
    fake_evidence = real_distances - fake_distances

    return float(fake_evidence[0])


# ============================================================
# 9. ORIGINAL CFS-35 IMPLEMENTATION
# ============================================================

def safe_pearson_correlation(A, B, epsilon=1e-12):

    A_centered = A - np.mean(
        A,
        axis=1,
        keepdims=True
    )

    B_centered = B - np.mean(B)

    numerator = np.sum(
        A_centered * B_centered.reshape(1, -1),
        axis=1
    )

    A_norm = np.sqrt(
        np.sum(A_centered ** 2, axis=1)
    )

    B_norm = np.sqrt(
        np.sum(B_centered ** 2)
    )

    denominator = A_norm * B_norm

    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > epsilon
    )

    return correlations


def calculate_cfs35_raw_score(model, X):

    feature_names = model["feature_names"]

    if list(X.columns) != list(feature_names):
        raise ValueError(
            "CFS-35 feature order does not match saved model."
        )

    values = X.to_numpy(dtype=float)

    if values.shape[1] != 35:
        raise ValueError(
            "CFS-35 requires exactly 35 features."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "CFS-35 input contains NaN or infinite values."
        )

    scaler = model["scaler"]

    scaled_values = scaler.transform(X)

    real_prototype = np.asarray(
        model["real_prototype"],
        dtype=float
    )

    fake_prototype = np.asarray(
        model["fake_prototype"],
        dtype=float
    )

    real_correlations = safe_pearson_correlation(
        scaled_values,
        real_prototype
    )

    fake_correlations = safe_pearson_correlation(
        scaled_values,
        fake_prototype
    )

    # Original Colab evidence
    fake_evidence = (
        fake_correlations - real_correlations
    )

    return float(fake_evidence[0])


# ============================================================
# 10. CONVERT RAW CUSTOM SCORES TO FAKE EVIDENCE
# ============================================================

def convert_to_fake_evidence(
    model_name,
    raw_score,
    metadata
):

    thresholds = metadata["model_thresholds"]

    scales = metadata["mathematical_score_scales"]

    if model_name not in thresholds:
        raise KeyError(
            f"Missing threshold for {model_name}."
        )

    if model_name not in scales:
        raise KeyError(
            f"Missing score scale for {model_name}."
        )

    threshold = float(thresholds[model_name])

    scale = float(scales[model_name])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Invalid mathematical score scale for {model_name}."
        )

    probability = 0.5 * (
        1.0
        +
        np.tanh(
            (float(raw_score) - threshold) / scale
        )
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


# ============================================================
# 11. CUSTOM MODEL RAW SCORING
# ============================================================

def get_custom_raw_score(model, X, name):

    if name == "MDMD-35":

        return calculate_mdmd_raw_score(
            X,
            load_mdmd_artifacts()
        )

    if name == "ZS-35":

        return calculate_zs35_raw_score(
            model,
            X
        )

    if name == "EDS-35":

        return calculate_eds35_raw_score(
            model,
            X
        )

    if name == "CFS-35":

        return calculate_cfs35_raw_score(
            model,
            X
        )

    raise ValueError(
        f"Unknown custom model: {name}"
    )


# ============================================================
# 12. INDIVIDUAL MODEL PREDICTIONS
# ============================================================

def predict_individual_models(features):

    X = prepare_features(features)

    models = load_individual_models()

    metadata = load_ensemble_metadata()

    results = {}

    ml_model_names = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Gradient Boosting"
    ]

    custom_model_names = [
        "MDMD-35",
        "ZS-35",
        "EDS-35",
        "CFS-35"
    ]

    print("\nStarting individual model predictions.")

    for model_name, model in models.items():

        print(f"Predicting with {model_name}...")

        try:

            if model_name in ml_model_names:

                fake_probability = get_ml_fake_score(
                    model,
                    X
                )

                raw_score = None

            elif model_name in custom_model_names:

                raw_score = get_custom_raw_score(
                    model,
                    X,
                    model_name
                )

                fake_probability = convert_to_fake_evidence(
                    model_name,
                    raw_score,
                    metadata
                )

            else:

                raise ValueError(
                    f"Unsupported individual model: {model_name}"
                )

            real_probability = 1.0 - fake_probability

            prediction = (
                "FAKE"
                if fake_probability >= DEFAULT_PREDICTION_THRESHOLD
                else "REAL"
            )

            results[model_name] = {
                "model_name": model_name,
                "raw_score": raw_score,
                "fake_probability": float(fake_probability),
                "real_probability": float(real_probability),
                "fake_percentage": round(
                    fake_probability * 100, 2
                ),
                "real_percentage": round(
                    real_probability * 100, 2
                ),
                "prediction": prediction,
                "status": "success"
            }

            print(
                f"{model_name}: "
                f"FAKE={fake_probability * 100:.2f}% | "
                f"REAL={real_probability * 100:.2f}% | "
                f"{prediction}"
            )

        except Exception as error:

            logger.exception(
                "Individual prediction failed for %s",
                model_name
            )

            results[model_name] = {
                "model_name": model_name,
                "raw_score": None,
                "fake_probability": None,
                "real_probability": None,
                "fake_percentage": None,
                "real_percentage": None,
                "prediction": "ERROR",
                "status": "error",
                "error": str(error)
            }

    successful_results = [
        result
        for result in results.values()
        if result["status"] == "success"
    ]

    if not successful_results:

        raise RuntimeError(
            "All eight individual model predictions failed."
        )

    return results


# ============================================================
# 13. INDIVIDUAL PREDICTION SUMMARY
# ============================================================

def get_individual_prediction_summary(results):

    successful_results = [
        result
        for result in results.values()
        if result["status"] == "success"
    ]

    if not successful_results:

        return {
            "results": results,
            "total_models": len(results),
            "successful_models": 0,
            "failed_models": len(results),
            "average_fake_probability": None,
            "average_real_probability": None,
            "overall_prediction": "ERROR"
        }

    fake_values = [
        result["fake_probability"]
        for result in successful_results
    ]

    real_values = [
        result["real_probability"]
        for result in successful_results
    ]

    average_fake = float(np.mean(fake_values))

    average_real = float(np.mean(real_values))

    overall_prediction = (
        "FAKE"
        if average_fake >= DEFAULT_PREDICTION_THRESHOLD
        else "REAL"
    )

    return {
        "results": results,
        "total_models": len(results),
        "successful_models": len(successful_results),
        "failed_models": (
            len(results) - len(successful_results)
        ),
        "average_fake_probability": average_fake,
        "average_real_probability": average_real,
        "average_fake_percentage": round(
            average_fake * 100, 2
        ),
        "average_real_percentage": round(
            average_real * 100, 2
        ),
        "overall_prediction": overall_prediction
    }


# ============================================================
# 14. DIRECT MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 65)
    print("PIXELPROOF — INDIVIDUAL MODEL FILE CHECK")
    print("=" * 65)

    try:

        loaded_models = load_individual_models()

        print("\nLoaded individual model files:\n")

        for name, model in loaded_models.items():

            print(
                f"{name:25} | {type(model).__name__}"
            )

        print("\nFeature count:", len(FEATURE_NAMES))

        mdmd_artifacts = load_mdmd_artifacts()

        print(
            "MDMD domains:",
            list(mdmd_artifacts["feature_domains"].keys())
        )

        print(
            "MDMD feature count:",
            len(mdmd_artifacts["all_features"])
        )

        print(
            "MDMD domain weights:",
            mdmd_artifacts["domain_weights"]
        )

        metadata = load_ensemble_metadata()

        print("\nMathematical ensemble metadata loaded.")

        print(
            "Ensemble feature count:",
            len(metadata["feature_names"])
        )

        print("\nAll model files and metadata loaded successfully.")

    except Exception:

        logger.exception(
            "PixelProof individual model verification failed."
        )

        raise
