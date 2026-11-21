
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# SPATIAL STRUCTURE ANALYSIS AND VISUALIZATION
# ============================================================

def analyze_spatial_structure(image, output_path):

    if image is None or image.size == 0:
        raise ValueError(
            "Invalid image supplied for spatial analysis."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with three channels."
        )

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # 1. HORIZONTAL AND VERTICAL GRADIENTS
    # --------------------------------------------------------

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # --------------------------------------------------------
    # 2. GRADIENT MAGNITUDE
    # --------------------------------------------------------

    magnitude = np.sqrt(
        gradient_x ** 2 +
        gradient_y ** 2
    )

    magnitude = np.log1p(
        magnitude
    )

    # --------------------------------------------------------
    # 3. NORMALIZATION FOR VISUALIZATION
    # --------------------------------------------------------

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 4. VISUALIZATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    axes[0].imshow(
        gray,
        cmap="gray"
    )

    axes[0].set_title(
        "Original Grayscale Image"
    )

    axes[0].axis("off")

    axes[1].imshow(
        magnitude,
        cmap="viridis"
    )

    axes[1].set_title(
        "Spatial Structure / Edge Map"
    )

    axes[1].axis("off")

    plt.tight_layout()

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    plt.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 5. RETURN RESULTS
    # --------------------------------------------------------

    return {

        "description": (
            "Spatial gradient and edge structure calculated. "
            "This is not a true depth estimate."
        ),

        "mean_edge_strength": round(
            float(np.mean(magnitude)),
            4
        )

    }
