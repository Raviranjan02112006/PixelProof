
import numpy as np

from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. IMAGE VALIDATION
# ============================================================

def validate_image(image):

    if not isinstance(image, Image.Image):

        raise TypeError(
            "Input must be a PIL Image."
        )

    if image.width < 3 or image.height < 3:

        raise ValueError(
            "Image must be at least 3 × 3 pixels."
        )

    return True


# ============================================================
# 2. LOAD IMAGE FROM FILE PATH
# ============================================================

def load_image(image_path):

    try:

        image = Image.open(image_path)

        image.verify()

        image = Image.open(image_path)

        image = image.convert("RGB")

        validate_image(image)

        return image

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        raise ValueError(
            f"Unable to load image: {error}"
        )


# ============================================================
# 3. CONVERT IMAGE TO RGB
# ============================================================

def convert_to_rgb(image):

    validate_image(image)

    return image.convert("RGB")


# ============================================================
# 4. MATHEMATICAL IMAGE PREPROCESSING
# ============================================================

def prepare_mathematical_image(image, max_size=1024):

    """
    Prepare an image for mathematical feature extraction.

    Returns:
        PIL Image
        NumPy RGB array
        NumPy grayscale array

    IMPORTANT:
    The exact resizing and grayscale procedure must be
    verified against the original mathematical training
    feature extraction pipeline before model inference.
    """

    image = convert_to_rgb(image)

    image = image.copy()

    image.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    rgb_array = np.asarray(
        image,
        dtype=np.uint8
    )

    grayscale_array = np.asarray(
        image.convert("L"),
        dtype=np.uint8
    )

    return {

        "image": image,

        "rgb": rgb_array,

        "grayscale": grayscale_array,

        "width": image.width,

        "height": image.height

    }


# ============================================================
# 5. CNN IMAGE PREPROCESSING
# ============================================================

def prepare_cnn_image(image, image_size=(224, 224)):

    """
    Prepare a single image for the PixelProof ResNet50 model.

    The saved Keras model contains its own ResNet50
    preprocess_input layer.

    Therefore, this function returns RGB pixel values
    in the 0–255 range without additional normalization.
    """

    image = convert_to_rgb(image)

    resized_image = image.resize(
        image_size,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 6. GENERAL IMAGE INFORMATION
# ============================================================

def get_image_information(image):

    validate_image(image)

    image = convert_to_rgb(image)

    return {

        "width": image.width,

        "height": image.height,

        "mode": image.mode,

        "format": image.format,

        "channels": 3,

        "total_pixels": image.width * image.height

    }
