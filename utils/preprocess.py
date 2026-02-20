import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image(img, target_size=(380, 380)):
    """
    Preprocess image for EfficientNet models (B0 & B4).
    """

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, target_size)

    # Convert to float
    img = img.astype("float32")

    # EfficientNet preprocessing
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
