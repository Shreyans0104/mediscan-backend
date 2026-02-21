import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image(img, target_size=(380, 380)):
    """
    Preprocess image for EfficientNet B0 model
    EXACT SAME pipeline as training
    """

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, target_size)

    # Convert to float32
    img = img.astype("float32")

    # EfficientNet normalization
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img