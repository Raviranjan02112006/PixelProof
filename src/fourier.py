
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# FOURIER ANALYSIS AND VISUALIZATION
# ============================================================

def analyze_fourier(image, output_path):

    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied for Fourier analysis.")

    # Convert RGB image to grayscale
    gray = np.mean(image, axis=2)

    # Apply 2D Fourier Transform
    fft_result = np.fft.fft2(gray)

    # Move low frequencies to the centre
    fft_shift = np.fft.fftshift(fft_result)

    # Log magnitude spectrum
    magnitude = 20 * np.log(
        np.abs(fft_shift) + 1
    )

    # Create visualization
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
        cmap="inferno"
    )

    axes[1].set_title(
        "Fourier Frequency Spectrum"
    )

    axes[1].axis("off")

    plt.tight_layout()

    output_directory = os.path.dirname(output_path)

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

    return {

        "description": "2D Fourier frequency spectrum generated.",

        "mean_frequency_magnitude": round(
            float(np.mean(magnitude)),
            4
        )

    }
