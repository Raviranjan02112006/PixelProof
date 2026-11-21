
# ============================================================
# PIXELPROOF
# STEP 9 — MATHEMATICAL ENSEMBLE PREDICTION
#
# Original Step 5.3:
# Validation-Based Weights + Weighted Soft Voting
#
# Original custom scoring:
# MDMD-35, ZS-35, EDS-35, CFS-35
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean

from config import (
    MATHEMATICAL_ENSEMBLE_PATH,
    INDIVIDUAL_MODELS,
    MDMD_ARTIFACTS_PATH
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Gradient Boosting",
    "MDMD-35",
    "ZS-35",
    "EDS-35",
    "CFS-35"
]

ML_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Gradient Boosting"
]

NON_ML_MODELS = [
    "MDMD-35",
    "ZS-35",
    "EDS-35",
    "CFS-35"
]

EPSILON = 1e-10

_ENSEMBLE = None
_MDMD_ARTIFACTS = None
_CUSTOM_ARTIFACTS = {}


# ============================================================
# 2. LOAD MATHEMATICAL ENSEMBLE
# ============================================================

def load_mathematical_ensemble():

    global _ENSEMBLE

    if _ENSEMBLE is not None:
        return _ENSEMBLE

    model_path = MATHEMATICAL_ENSEMBLE_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Mathematical ensemble not found: {model_path}"
        )

    print("Loading PixelProof mathematical ensemble...")

    bundle = joblib.load(model_path)

    required_keys = [
        "models",
        "weights",
        "model_thresholds",
        "mathematical_score_scales",
        "ensemble_threshold",
        "feature_names",
        "model_names"
    ]

    missing = [
        key for key in required_keys
        if key not in bundle
    ]

    if missing:
        raise ValueError(
            f"Invalid ensemble bundle. Missing keys: {missing}"
        )

    if set(bundle["model_names"]) != set(MODEL_NAMES):
        raise ValueError(
            "Saved model names do not match the expected eight models."
        )

    if set(bundle["models"].keys()) != set(MODEL_NAMES):
        raise ValueError(
            "Saved model objects are incomplete."
        )

    if set(bundle["weights"].keys()) != set(MODEL_NAMES):
        raise ValueError(
            "Saved model weights are incomplete."
        )

    total_weight = sum(
        float(bundle["weights"][name])
        for name in MODEL_NAMES
    )

    if not np.isclose(total_weight, 1.0, atol=1e-5):
        raise ValueError(
            f"Model weights must sum to 1. Current sum: {total_weight}"
        )

    if len(bundle["feature_names"]) != 35:
        raise ValueError(
            "Expected exactly 35 mathematical features."
        )

    if len(set(bundle["feature_names"])) != 35:
        raise ValueError(
            "Duplicate feature names found in ensemble bundle."
        )

    for name in NON_ML_MODELS:

        if name not in bundle["mathematical_score_scales"]:
            raise ValueError(
                f"Missing score scale for {name}."
            )

        if name not in bundle["model_thresholds"]:
            raise ValueError(
                f"Missing model threshold for {name}."
            )

        scale = float(
            bundle["mathematical_score_scales"][name]
        )

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid score scale for {name}: {scale}"
            )

    threshold = float(bundle["ensemble_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Invalid ensemble threshold."
        )

    _ENSEMBLE = bundle

    print("Mathematical ensemble loaded successfully.")

    return _ENSEMBLE


# ============================================================
# 3. PREPARE 35-FEATURE INPUT
# ============================================================

def prepare_feature_vector(features):

    bundle = load_mathematical_ensemble()

    feature_names = list(bundle["feature_names"])

    if isinstance(features, pd.DataFrame):

        if len(features) != 1:
            raise ValueError(
                "Exactly one image must be predicted at a time."
            )

        missing = [
            name for name in feature_names
            if name not in features.columns
        ]

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        X = features[feature_names].copy()

    elif isinstance(features, dict):

        missing = [
            name for name in feature_names
            if name not in features
        ]

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        X = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names
        )

    else:

        values = np.asarray(
            features,
            dtype=np.float64
        )

        if values.ndim == 1:
            values = values.reshape(1, -1)

        expected_shape = (1, len(feature_names))

        if values.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {values.shape}."
            )

        X = pd.DataFrame(
            values,
            columns=feature_names
        )

    X = X.astype(np.float64)

    if X.shape != (1, 35):
        raise ValueError(
            f"Expected one row with 35 features, got {X.shape}."
        )

    if not np.all(np.isfinite(X.to_numpy())):
        raise ValueError(
            "Feature vector contains NaN or infinite values."
        )

    return X


# ============================================================
# 4. LOAD ORIGINAL CUSTOM MODEL ARTIFACTS
# ============================================================

def load_custom_artifact(model_name):

    global _CUSTOM_ARTIFACTS

    if model_name in _CUSTOM_ARTIFACTS:
        return _CUSTOM_ARTIFACTS[model_name]

    # First try the individual model file from config.py.
    model_path = INDIVIDUAL_MODELS.get(model_name)

    if model_path and os.path.isfile(model_path):

        artifact = joblib.load(model_path)

        _CUSTOM_ARTIFACTS[model_name] = artifact

        print(f"Loaded original artifact: {model_name}")

        return artifact

    # Fallback to the corresponding object in the ensemble bundle.
    bundle = load_mathematical_ensemble()

    artifact = bundle["models"].get(model_name)

    if artifact is None:
        raise FileNotFoundError(
            f"Could not locate the saved artifact for {model_name}. "
            "Check INDIVIDUAL_MODELS in config.py."
        )

    _CUSTOM_ARTIFACTS[model_name] = artifact

    return artifact


# ============================================================
# 5. LOAD MDMD-35 DOMAIN ARTIFACTS
# ============================================================

def load_mdmd_artifacts():

    global _MDMD_ARTIFACTS

    if _MDMD_ARTIFACTS is not None:
        return _MDMD_ARTIFACTS

    if not os.path.isfile(MDMD_ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"MDMD-35 artifacts not found: {MDMD_ARTIFACTS_PATH}"
        )

    print("Loading original MDMD-35 domain artifacts...")

    artifacts = joblib.load(MDMD_ARTIFACTS_PATH)

    required_keys = [
        "domain_scalers",
        "mdmd_references",
        "domain_weights",
        "feature_domains",
        "all_features"
    ]

    missing = [
        key for key in required_keys
        if key not in artifacts
    ]

    if missing:
        raise ValueError(
            f"Invalid MDMD-35 artifacts. Missing keys: {missing}"
        )

    domains = artifacts["feature_domains"]

    for domain_name, domain_features in domains.items():

        if domain_name not in artifacts["domain_scalers"]:
            raise ValueError(
                f"Missing MDMD scaler for {domain_name}."
            )

        if domain_name not in artifacts["mdmd_references"]:
            raise ValueError(
                f"Missing MDMD references for {domain_name}."
            )

        for class_name in ["REAL", "FAKE"]:

            reference = artifacts["mdmd_references"][
                domain_name
            ].get(class_name)

            if reference is None:
                raise ValueError(
                    f"Missing {class_name} reference for {domain_name}."
                )

            for key in [
                "mean",
                "inverse_covariance"
            ]:

                if key not in reference:
                    raise ValueError(
                        f"Missing {key} in {domain_name}/{class_name}."
                    )

            expected_size = len(domain_features)

            if len(reference["mean"]) != expected_size:
                raise ValueError(
                    f"Invalid mean vector size for "
                    f"{domain_name}/{class_name}."
                )

            if np.asarray(
                reference["inverse_covariance"]
            ).shape != (expected_size, expected_size):

                raise ValueError(
                    f"Invalid inverse covariance matrix for "
                    f"{domain_name}/{class_name}."
                )

    weight_sum = sum(
        float(value)
        for value in artifacts["domain_weights"].values()
    )

    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(
            f"MDMD domain weights must sum to 1. Got {weight_sum}."
        )

    _MDMD_ARTIFACTS = artifacts

    print("MDMD-35 artifacts loaded successfully.")

    return _MDMD_ARTIFACTS


# ============================================================
# 6. ML MODEL FAKE PROBABILITY
# ============================================================

def get_ml_fake_score(model, X, model_name):

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model_name} does not support predict_proba()."
        )

    # Preserve the feature-name format used during training.
    if hasattr(model, "feature_names_in_"):
        model_input = X
    else:
        model_input = X.to_numpy(dtype=np.float64)

    probabilities = np.asarray(
        model.predict_proba(model_input),
        dtype=np.float64
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"{model_name}: invalid probability output."
        )

    if probabilities.shape[0] != 1:
        raise ValueError(
            f"{model_name}: expected one prediction."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"{model_name}: FAKE class label 1 not found."
        )

    fake_index = classes.index(1)

    score = float(
        probabilities[0, fake_index]
    )

    if not np.isfinite(score):
        raise ValueError(
            f"{model_name}: invalid FAKE probability."
        )

    return score


# ============================================================
# 7. ORIGINAL MDMD-35 SCORING
# ============================================================

def calculate_mdmd_raw_score(X):

    artifacts = load_mdmd_artifacts()

    feature_domains = artifacts["feature_domains"]
    domain_scalers = artifacts["domain_scalers"]
    references = artifacts["mdmd_references"]
    domain_weights = artifacts["domain_weights"]

    domain_evidence = {}

    for domain_name, domain_features in feature_domains.items():

        # Apply the original training-fitted scaler.
        scaler = domain_scalers[domain_name]

        domain_values = X[domain_features].to_numpy(
            dtype=np.float64
        )

        scaled_values = scaler.transform(domain_values)

        # Original REAL reference.
        real_reference = references[
            domain_name
        ]["REAL"]

        # Original FAKE reference.
        fake_reference = references[
            domain_name
        ]["FAKE"]

        real_difference = (
            scaled_values -
            np.asarray(real_reference["mean"], dtype=np.float64)
        )

        fake_difference = (
            scaled_values -
            np.asarray(fake_reference["mean"], dtype=np.float64)
        )

        real_inverse_covariance = np.asarray(
            real_reference["inverse_covariance"],
            dtype=np.float64
        )

        fake_inverse_covariance = np.asarray(
            fake_reference["inverse_covariance"],
            dtype=np.float64
        )

        # Original Mahalanobis distance formula.
        real_distance = np.einsum(
            "ij,jk,ik->i",
            real_difference,
            real_inverse_covariance,
            real_difference
        )

        fake_distance = np.einsum(
            "ij,jk,ik->i",
            fake_difference,
            fake_inverse_covariance,
            fake_difference
        )

        real_distance = np.maximum(
            real_distance,
            0.0
        )

        fake_distance = np.maximum(
            fake_distance,
            0.0
        )

        # Original domain evidence:
        # (REAL distance - FAKE distance) /
        # (REAL distance + FAKE distance + epsilon)
        evidence = (
            (real_distance - fake_distance) /
            (
                real_distance +
                fake_distance +
                float(artifacts.get("epsilon", EPSILON))
            )
        )

        domain_evidence[domain_name] = float(evidence[0])

    # Original weighted MDMD score.
    raw_score = 0.0

    for domain_name, weight in domain_weights.items():

        raw_score += (
            float(weight) *
            domain_evidence[domain_name]
        )

    return float(raw_score)


# ============================================================
# 8. ORIGINAL ZS-35 SCORING
# ============================================================

def calculate_zs_raw_score(X):

    artifact = load_custom_artifact("ZS-35")

    required_keys = [
        "feature_names",
        "real_mean",
        "fake_mean",
        "real_std_safe",
        "fake_std_safe"
    ]

    for key in required_keys:
        if key not in artifact:
            raise ValueError(
                f"ZS-35 artifact is missing '{key}'."
            )

    feature_names = list(artifact["feature_names"])

    if list(X.columns) != feature_names:
        raise ValueError(
            "ZS-35 feature order does not match the input feature order."
        )

    values = X[feature_names].to_numpy(
        dtype=np.float64
    )

    real_mean = np.asarray(
        artifact["real_mean"],
        dtype=np.float64
    )

    fake_mean = np.asarray(
        artifact["fake_mean"],
        dtype=np.float64
    )

    real_std_safe = np.asarray(
        artifact["real_std_safe"],
        dtype=np.float64
    )

    fake_std_safe = np.asarray(
        artifact["fake_std_safe"],
        dtype=np.float64
    )

    # Original ZS-35 formula.
    real_z = np.abs(
        (values - real_mean) / real_std_safe
    )

    fake_z = np.abs(
        (values - fake_mean) / fake_std_safe
    )

    real_score = float(
        np.mean(real_z, axis=1)[0]
    )

    fake_score = float(
        np.mean(fake_z, axis=1)[0]
    )

    # Positive evidence indicates FAKE.
    raw_score = real_score - fake_score

    return float(raw_score)


# ============================================================
# 9. ORIGINAL EDS-35 SCORING
# ============================================================

def calculate_eds_raw_score(X):

    artifact = load_custom_artifact("EDS-35")

    required_keys = [
        "feature_names",
        "scaler",
        "real_centroid",
        "fake_centroid"
    ]

    for key in required_keys:
        if key not in artifact:
            raise ValueError(
                f"EDS-35 artifact is missing '{key}'."
            )

    feature_names = list(artifact["feature_names"])

    if list(X.columns) != feature_names:
        raise ValueError(
            "EDS-35 feature order does not match the input feature order."
        )

    values = X[feature_names].to_numpy(
        dtype=np.float64
    )

    # Original training-fitted StandardScaler.
    scaled_values = artifact["scaler"].transform(values)

    real_centroid = np.asarray(
        artifact["real_centroid"],
        dtype=np.float64
    )

    fake_centroid = np.asarray(
        artifact["fake_centroid"],
        dtype=np.float64
    )

    # Original Euclidean distance to class centroids.
    real_distances = np.linalg.norm(
        scaled_values - real_centroid,
        axis=1
    )

    fake_distances = np.linalg.norm(
        scaled_values - fake_centroid,
        axis=1
    )

    # Positive evidence indicates FAKE.
    raw_score = (
        real_distances[0] -
        fake_distances[0]
    )

    return float(raw_score)


# ============================================================
# 10. ORIGINAL CFS-35 SCORING
# ============================================================

def safe_pearson_correlation(A, B):

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A_centered = (
        A -
        np.mean(A, axis=1, keepdims=True)
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
        where=denominator > EPSILON
    )

    return correlations


def calculate_cfs_raw_score(X):

    artifact = load_custom_artifact("CFS-35")

    required_keys = [
        "feature_names",
        "scaler",
        "real_prototype",
        "fake_prototype"
    ]

    for key in required_keys:
        if key not in artifact:
            raise ValueError(
                f"CFS-35 artifact is missing '{key}'."
            )

    feature_names = list(artifact["feature_names"])

    if list(X.columns) != feature_names:
        raise ValueError(
            "CFS-35 feature order does not match the input feature order."
        )

    values = X[feature_names].to_numpy(
        dtype=np.float64
    )

    # Original training-fitted StandardScaler.
    scaled_values = artifact["scaler"].transform(values)

    real_prototype = np.asarray(
        artifact["real_prototype"],
        dtype=np.float64
    )

    fake_prototype = np.asarray(
        artifact["fake_prototype"],
        dtype=np.float64
    )

    # Original row-wise Pearson correlation.
    real_correlation = safe_pearson_correlation(
        scaled_values,
        real_prototype
    )

    fake_correlation = safe_pearson_correlation(
        scaled_values,
        fake_prototype
    )

    # Positive evidence indicates FAKE.
    raw_score = (
        fake_correlation[0] -
        real_correlation[0]
    )

    return float(raw_score)


# ============================================================
# 11. CUSTOM MATHEMATICAL MODEL DISPATCHER
# ============================================================

def get_non_ml_raw_score(model, X, model_name):

    if model_name == "MDMD-35":

        return calculate_mdmd_raw_score(X)

    elif model_name == "ZS-35":

        return calculate_zs_raw_score(X)

    elif model_name == "EDS-35":

        return calculate_eds_raw_score(X)

    elif model_name == "CFS-35":

        return calculate_cfs_raw_score(X)

    raise ValueError(
        f"Unknown custom mathematical model: {model_name}"
    )


# ============================================================
# 12. CONVERT RAW SCORES INTO FAKE EVIDENCE
# ============================================================

def convert_to_evidence(model_name, raw_score, bundle):

    score = float(raw_score)

    if not np.isfinite(score):
        raise ValueError(
            f"{model_name}: raw score is not finite."
        )

    if model_name in NON_ML_MODELS:

        threshold = float(
            bundle["model_thresholds"][model_name]
        )

        scale = float(
            bundle["mathematical_score_scales"][model_name]
        )

        if scale <= 0:
            raise ValueError(
                f"Invalid score scale for {model_name}."
            )

        # Original training-time score conversion.
        evidence = 0.5 * (
            1.0 + np.tanh(
                (score - threshold) / scale
            )
        )

    else:

        # ML model output is already FAKE probability.
        evidence = score

    return float(
        np.clip(evidence, 0.0, 1.0)
    )


# ============================================================
# 13. CALCULATE INDIVIDUAL MODEL RESULTS
# ============================================================

def calculate_individual_results(features):

    bundle = load_mathematical_ensemble()

    X = prepare_feature_vector(features)

    results = {}

    for name in MODEL_NAMES:

        model = bundle["models"][name]

        if name in ML_MODELS:

            raw_score = get_ml_fake_score(
                model,
                X,
                name
            )

        else:

            raw_score = get_non_ml_raw_score(
                model,
                X,
                name
            )

        fake_evidence = convert_to_evidence(
            name,
            raw_score,
            bundle
        )

        real_evidence = 1.0 - fake_evidence

        results[name] = {

            "raw_score": round(
                raw_score,
                6
            ),

            "fake_evidence_percentage": round(
                fake_evidence * 100,
                2
            ),

            "real_evidence_percentage": round(
                real_evidence * 100,
                2
            ),

            "weight_percentage": round(
                float(bundle["weights"][name]) * 100,
                4
            ),

            "prediction": (
                "FAKE"
                if fake_evidence >= 0.5
                else "REAL"
            )
        }

    return results


# ============================================================
# 14. ORIGINAL WEIGHTED SOFT-VOTING FORMULA
# ============================================================

def weighted_soft_vote(individual_results):

    bundle = load_mathematical_ensemble()

    fake_support = 0.0

    for name in MODEL_NAMES:

        weight = float(
            bundle["weights"][name]
        )

        fake_evidence = (
            individual_results[name][
                "fake_evidence_percentage"
            ] / 100.0
        )

        fake_support += (
            weight * fake_evidence
        )

    real_support = 1.0 - fake_support

    return real_support, fake_support


# ============================================================
# 15. FINAL MATHEMATICAL PREDICTION
# ============================================================

def predict_mathematical(features):

    bundle = load_mathematical_ensemble()

    # Individual model results.
    individual_results = calculate_individual_results(
        features
    )

    # Combined weighted support.
    real_support, fake_support = weighted_soft_vote(
        individual_results
    )

    # Frozen validation threshold.
    threshold = float(
        bundle["ensemble_threshold"]
    )

    prediction = (
        "FAKE"
        if fake_support >= threshold
        else "REAL"
    )

    return {

        "model": "PixelProof Mathematical Ensemble",

        "ensemble_type": (
            "Performance-weighted soft voting"
        ),

        "individual_models": individual_results,

        "real_support_percentage": round(
            real_support * 100,
            2
        ),

        "fake_support_percentage": round(
            fake_support * 100,
            2
        ),

        "ensemble_threshold": round(
            threshold,
            6
        ),

        "prediction": prediction,

        "evidence_note": bundle.get(
            "evidence_note",
            "Weighted evidence support, not calibrated probability."
        )
    }


# ============================================================
# 16. BUNDLE INFORMATION TEST
# ============================================================

if __name__ == "__main__":

    bundle = load_mathematical_ensemble()

    print("\nPIXELPROOF MATHEMATICAL ENSEMBLE")
    print("=" * 55)

    print("Models:", len(bundle["model_names"]))
    print("Features:", len(bundle["feature_names"]))

    print(
        "Ensemble threshold:",
        bundle["ensemble_threshold"]
    )

    print(
        "Total model weight:",
        sum(bundle["weights"].values())
    )

    print("\nModel weights:")

    for name in MODEL_NAMES:

        print(
            f"{name:24s}: "
            f"{bundle['weights'][name] * 100:.4f}%"
        )

    print("\nChecking custom mathematical artifacts...")

    load_mdmd_artifacts()

    for name in ["ZS-35", "EDS-35", "CFS-35"]:

        artifact = load_custom_artifact(name)

        print(
            f"{name}: loaded "
            f"({len(artifact['feature_names'])} features)"
        )

    print("\nEnsemble and custom artifact loading test completed.")
