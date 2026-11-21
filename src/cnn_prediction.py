
# ============================================================
# PIXELPROOF
# STEP 11 — CNN PREDICTION
#
# ResNet50 Deep Learning Model
# ============================================================

import os
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError

from config import (
    CNN_MODEL_PATH,
    CNN_IMAGE_SIZE,
    CLASS_NAMES
)


# ============================================================
# 1. REGISTER RESNET50 PREPROCESSING FUNCTION
# ============================================================

RESNET50_PREPROCESS_INPUT = (
    tf.keras.applications.resnet50.preprocess_input
)


# ============================================================
# 2. LOAD CNN MODEL
# ============================================================

_CNN_MODEL = None


def load_cnn_model():

    global _CNN_MODEL

    if _CNN_MODEL is not None:
        return _CNN_MODEL

    if not os.path.isfile(CNN_MODEL_PATH):
        raise FileNotFoundError(
            f"CNN model not found: {CNN_MODEL_PATH}"
        )

    print("Loading PixelProof ResNet50 CNN model...")

    _CNN_MODEL = tf.keras.models.load_model(
        CNN_MODEL_PATH,
        custom_objects={
            "preprocess_input": RESNET50_PREPROCESS_INPUT
        },
        compile=False
    )

    print("ResNet50 CNN model loaded successfully.")

    return _CNN_MODEL


# ============================================================
# 3. PREPROCESS IMAGE
# ============================================================

def preprocess_cnn_image(image):

    if isinstance(image, (str, os.PathLike)):

        if not os.path.isfile(image):
            raise FileNotFoundError(
                f"Image not found: {image}"
            )

        try:
            image = Image.open(image).convert("RGB")

        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(
                "Unable to open the supplied image."
            ) from error

    elif isinstance(image, Image.Image):

        image = image.convert("RGB")

    else:

        raise TypeError(
            "Image must be a PIL Image or image file path."
        )

    width, height = image.size

    if width < 3 or height < 3:
        raise ValueError(
            "Image dimensions must be at least 3 × 3."
        )

    target_width, target_height = CNN_IMAGE_SIZE

    image = image.resize(
        (target_width, target_height),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Preprocessing is already included in the trained model.
    # Do not apply preprocess_input again here.

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image_batch


# ============================================================
# 4. GET CNN OUTPUT
# ============================================================

def get_cnn_fake_score(model, image_batch):

    output = model.predict(
        image_batch,
        verbose=0
    )

    output = np.asarray(
        output,
        dtype=np.float64
    )

    if output.size != 1:
        raise ValueError(
            "Expected a single sigmoid output from the CNN. "
            f"Received output shape: {output.shape}"
        )

    fake_score = float(
        output.reshape(-1)[0]
    )

    if not np.isfinite(fake_score):
        raise ValueError(
            "CNN returned an invalid prediction score."
        )

    if not 0.0 <= fake_score <= 1.0:
        raise ValueError(
            f"CNN output is outside the expected sigmoid range: "
            f"{fake_score}"
        )

    return fake_score


# ============================================================
# 5. CNN PREDICTION
# ============================================================

def predict_cnn(image):

    model = load_cnn_model()

    image_batch = preprocess_cnn_image(
        image
    )

    fake_score = get_cnn_fake_score(
        model,
        image_batch
    )

    real_score = 1.0 - fake_score

    prediction = (
        "FAKE"
        if fake_score >= 0.5
        else "REAL"
    )

    return {

        "model": "PixelProof ResNet50 CNN",

        "fake_evidence_percentage": round(
            fake_score * 100,
            2
        ),

        "real_evidence_percentage": round(
            real_score * 100,
            2
        ),

        "prediction": prediction,

        "threshold": 0.5,

        "evidence_note": (
            "CNN sigmoid output. Treat as model score, "
            "not a calibrated probability unless calibration "
            "has been performed."
        )
    }


# ============================================================
# 6. DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\nPIXELPROOF — CNN MODEL CHECK")
    print("=" * 55)

    model = load_cnn_model()

    print("\nModel input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    print("\nExpected image size:", CNN_IMAGE_SIZE)
    print("Configured classes:", CLASS_NAMES)

    print("\nCNN model loading test completed.")

    print(
        "\nTo test an actual image, call:\n"
        "predict_cnn('path/to/image.jpg')"
    )
