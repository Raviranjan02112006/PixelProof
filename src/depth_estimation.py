
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from transformers import pipeline


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

_MODEL = None


# ============================================================
# LOAD DEPTH MODEL ONCE
# ============================================================

def get_depth_model():

    global _MODEL

    if _MODEL is None:

        print("Loading Depth Anything V2 model...")
        print("The first download may take some time.")

        _MODEL = pipeline(
            task="depth-estimation",
            model=MODEL_NAME,
            framework="pt",
            device=-1
        )

        print(
            "Depth estimation model loaded successfully."
        )

    return _MODEL


# ============================================================
# DEPTH ESTIMATION AND VISUALIZATION
# ============================================================

def analyze_depth(image, output_path):
    """
    Estimate relative depth from one RGB PIL image
    and save a colour depth-map visualization.
    """

    if image is None:
        raise ValueError(
            "No image supplied for depth estimation."
        )

    # Load the model (only once)
    depth_model = get_depth_model()

    # Run pretrained model on uploaded image
    result = depth_model(image)

    # The model returns a PIL depth image
    depth_image = result["depth"]

    depth_array = np.asarray(
        depth_image
    ).astype(np.float32)

    if depth_array.size == 0:
        raise ValueError(
            "The depth model returned an empty result."
        )

    if not np.all(np.isfinite(depth_array)):
        raise ValueError(
            "Depth model returned invalid values."
        )

    # --------------------------------------------------------
    # NORMALIZE DEPTH VALUES
    # --------------------------------------------------------

    minimum = float(
        np.min(depth_array)
    )

    maximum = float(
        np.max(depth_array)
    )

    if maximum - minimum < 1e-8:

        normalized = np.zeros_like(
            depth_array
        )

    else:

        normalized = (
            (depth_array - minimum) /
            (maximum - minimum)
        )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5)
    )

    axes[0].imshow(image)

    axes[0].set_title(
        "Uploaded Image"
    )

    axes[0].axis("off")

    depth_plot = axes[1].imshow(
        normalized,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Estimated Relative Depth"
    )

    axes[1].axis("off")

    fig.colorbar(
        depth_plot,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Relative depth"
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
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "description": (
            "Depth Anything V2 generated a single-image "
            "relative depth map."
        ),

        "model": "Depth Anything V2 Small",

        "depth_range": (
            "Relative depth, not physical distance"
        ),

        "minimum_value": round(
            minimum,
            4
        ),

        "maximum_value": round(
            maximum,
            4
        )

    }


# ============================================================
# ACCESS DEPTH MODEL FOR MATHEMATICAL FEATURE EXTRACTION
# ============================================================

def get_depth_map(image):

    depth_model = get_depth_model()

    result = depth_model(image)

    depth = np.asarray(
        result["depth"]
    ).astype(np.float32)

    if depth.size == 0:
        raise ValueError(
            "Empty depth map returned."
        )

    if not np.all(np.isfinite(depth)):
        raise ValueError(
            "Depth map contains invalid values."
        )

    return depth
