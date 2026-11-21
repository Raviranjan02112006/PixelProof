
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter


# ============================================================
# PIXEL CORRELATION ANALYSIS AND VISUALIZATION
# ============================================================

def analyze_correlation(image, output_path):

    if image is None or image.size == 0:
        raise ValueError(
            "Invalid image supplied for correlation analysis."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with three channels."
        )

    gray = np.mean(
        image,
        axis=2
    ).astype(np.float64)

    if gray.shape[0] < 3 or gray.shape[1] < 4:
        raise ValueError(
            "Image is too small for correlation analysis."
        )

    # --------------------------------------------------------
    # 1. GLOBAL HORIZONTAL CORRELATION
    # --------------------------------------------------------

    x = gray[:, :-1]

    y = gray[:, 1:]

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        global_corr = 0.0

    else:
        global_corr = np.corrcoef(
            x.ravel(),
            y.ravel()
        )[0, 1]

    if not np.isfinite(global_corr):
        global_corr = 0.0

    # --------------------------------------------------------
    # 2. LOCAL CORRELATION
    # --------------------------------------------------------

    window = min(
        15,
        gray.shape[0],
        gray.shape[1] - 1
    )

    if window % 2 == 0:
        window -= 1

    window = max(window, 3)

    mean_x = uniform_filter(
        x,
        size=window
    )

    mean_y = uniform_filter(
        y,
        size=window
    )

    mean_x2 = uniform_filter(
        x * x,
        size=window
    )

    mean_y2 = uniform_filter(
        y * y,
        size=window
    )

    mean_xy = uniform_filter(
        x * y,
        size=window
    )

    covariance = (
        mean_xy - mean_x * mean_y
    )

    variance_x = np.maximum(
        mean_x2 - mean_x ** 2,
        0
    )

    variance_y = np.maximum(
        mean_y2 - mean_y ** 2,
        0
    )

    denominator = np.sqrt(
        variance_x * variance_y
    )

    local_corr = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-8
    )

    local_corr = np.clip(
        local_corr,
        -1,
        1
    )

    # --------------------------------------------------------
    # 3. VISUALIZATION
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
        "Original Image"
    )

    axes[0].axis("off")

    plot = axes[1].imshow(
        local_corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    axes[1].set_title(
        "Local Pixel Correlation"
    )

    axes[1].axis("off")

    fig.colorbar(
        plot,
        ax=axes[1],
        fraction=0.046
    )

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
    # 4. RETURN RESULTS
    # --------------------------------------------------------

    return {

        "description": (
            "Horizontal neighbouring-pixel correlation calculated."
        ),

        "horizontal_correlation": round(
            float(global_corr),
            4
        )

    }
