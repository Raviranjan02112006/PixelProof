
import numpy as np
import cv2

from PIL import Image

from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis


# ============================================================
# 1. FOURIER FEATURE EXTRACTION
# Total: 5 features
# ============================================================

def extract_fourier_features(gray):

    image = gray.astype(np.float32)

    fft = np.fft.fft2(image)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    magnitude_log = np.log1p(magnitude)

    h, w = gray.shape

    y, x = np.ogrid[:h, :w]

    center_y = h // 2
    center_x = w // 2

    radius = np.sqrt(
        (x - center_x)**2 +
        (y - center_y)**2
    )

    max_radius = np.sqrt(
        center_x**2 + center_y**2
    )

    normalized_radius = radius / max_radius

    low_mask = normalized_radius <= 0.25

    high_mask = normalized_radius >= 0.50

    energy = magnitude**2

    total_energy = np.sum(energy) + 1e-10

    low_energy_ratio = (
        np.sum(energy[low_mask]) / total_energy
    )

    high_energy_ratio = (
        np.sum(energy[high_mask]) / total_energy
    )

    probabilities = magnitude.ravel()

    probabilities = probabilities / (
        probabilities.sum() + 1e-10
    )

    spectral_entropy = -np.sum(
        probabilities *
        np.log2(probabilities + 1e-10)
    )

    return {

        "fft_mean": float(np.mean(magnitude_log)),

        "fft_std": float(np.std(magnitude_log)),

        "low_frequency_ratio": float(low_energy_ratio),

        "high_frequency_ratio": float(high_energy_ratio),

        "spectral_entropy": float(spectral_entropy)

    }


# ============================================================
# 2. CORRELATION FEATURE EXTRACTION
# Total: 5 features
# ============================================================

def calculate_correlation(a, b):

    a = a.astype(np.float64).ravel()

    b = b.astype(np.float64).ravel()

    if (
        len(a) < 2 or
        np.std(a) < 1e-10 or
        np.std(b) < 1e-10
    ):

        return 0.0

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def extract_correlation_features(gray):

    image = gray.astype(np.float32)

    horizontal = calculate_correlation(
        image[:, :-1],
        image[:, 1:]
    )

    vertical = calculate_correlation(
        image[:-1, :],
        image[1:, :]
    )

    diagonal = calculate_correlation(
        image[:-1, :-1],
        image[1:, 1:]
    )

    local_mean = uniform_filter(
        image,
        size=3
    )

    local_variance = uniform_filter(
        image**2,
        size=3
    ) - local_mean**2

    return {

        "horizontal_correlation": float(horizontal),

        "vertical_correlation": float(vertical),

        "diagonal_correlation": float(diagonal),

        "local_variance_mean": float(
            np.mean(local_variance)
        ),

        "local_variance_std": float(
            np.std(local_variance)
        )

    }


# ============================================================
# 3. SPATIAL FEATURE EXTRACTION
# Total: 16 features
# ============================================================

def extract_spatial_features(gray):

    image = gray.astype(np.float32)

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx**2 + gy**2
    )

    angle = np.arctan2(
        gy,
        gx
    )

    edge_threshold = (
        np.mean(magnitude) +
        np.std(magnitude)
    )

    edge_density = np.mean(
        magnitude > edge_threshold
    )

    direction_histogram, _ = np.histogram(
        angle,
        bins=8,
        range=(-np.pi, np.pi),
        density=True
    )

    features = {

        "gradient_mean": float(
            np.mean(magnitude)
        ),

        "gradient_std": float(
            np.std(magnitude)
        ),

        "gradient_max": float(
            np.max(magnitude)
        ),

        "edge_density": float(
            edge_density
        ),

        "intensity_mean": float(
            np.mean(image)
        ),

        "intensity_std": float(
            np.std(image)
        ),

        "intensity_skewness": float(
            skew(image.ravel())
        ),

        "intensity_kurtosis": float(
            kurtosis(image.ravel())
        )

    }

    for i, value in enumerate(direction_histogram):

        features[f"gradient_direction_{i}"] = float(value)

    return features


# ============================================================
# 4. DEPTH FEATURE EXTRACTION
# Total: 9 features
# ============================================================

def extract_depth_features(image_path, depth_model):

    image = Image.open(
        image_path
    ).convert("RGB")

    result = depth_model(image)

    depth = np.asarray(
        result["depth"]
    ).astype(np.float32)

    if depth.size == 0:

        raise ValueError(
            "Empty depth map returned."
        )

    minimum = np.min(depth)

    maximum = np.max(depth)

    normalized = (
        (depth - minimum) /
        (maximum - minimum + 1e-8)
    )

    gy, gx = np.gradient(normalized)

    gradient = np.sqrt(
        gx**2 + gy**2
    )

    features = {

        "depth_mean": float(
            np.mean(normalized)
        ),

        "depth_std": float(
            np.std(normalized)
        ),

        "depth_min": float(
            np.min(normalized)
        ),

        "depth_max": float(
            np.max(normalized)
        ),

        "depth_median": float(
            np.median(normalized)
        ),

        "depth_q25": float(
            np.percentile(normalized, 25)
        ),

        "depth_q75": float(
            np.percentile(normalized, 75)
        ),

        "depth_gradient_mean": float(
            np.mean(gradient)
        ),

        "depth_gradient_std": float(
            np.std(gradient)
        )

    }

    return features


# ============================================================
# 5. COMBINED MATHEMATICAL FEATURE EXTRACTION
# Total: 35 features
# ============================================================

def extract_all_features(gray, image_path, depth_model):

    fourier_features = extract_fourier_features(gray)

    correlation_features = extract_correlation_features(gray)

    spatial_features = extract_spatial_features(gray)

    depth_features = extract_depth_features(
        image_path,
        depth_model
    )

    all_features = {}

    all_features.update(fourier_features)

    all_features.update(correlation_features)

    all_features.update(spatial_features)

    all_features.update(depth_features)

    if len(all_features) != 35:

        raise ValueError(
            f"Expected 35 features, got {len(all_features)}."
        )

    feature_values = np.asarray(
        list(all_features.values()),
        dtype=np.float64
    )

    if not np.all(np.isfinite(feature_values)):

        raise ValueError(
            "Feature extraction produced NaN or infinite values."
        )

    return all_features


# ============================================================
# 6. FEATURE VECTOR GENERATION
# ============================================================

def create_feature_vector(features, feature_names):

    missing_features = [
        name for name in feature_names
        if name not in features
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    feature_vector = np.asarray(
        [[features[name] for name in feature_names]],
        dtype=np.float64
    )

    if feature_vector.shape[1] != 35:

        raise ValueError(
            f"Expected 35 columns, got {feature_vector.shape[1]}."
        )

    return feature_vector
