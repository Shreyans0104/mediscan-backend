import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Force TensorFlow to use ONLY the CPU and minimize memory overhead
tf.config.set_visible_devices([], 'GPU')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

B0_PATH = os.path.join(MODEL_DIR, "fracture_b0.keras")
B4_PATH = os.path.join(MODEL_DIR, "fracture_b4.keras")

# Google Drive IDs
B0_ID = "1-XTjfnayM6lh2c1cYqSIqk0oYdqy7XTy"
B4_ID = "1Tn7mbDg9AsplfDSnBWVuaIXGVU7JZSI5"

def download_models():
    if not os.path.exists(B0_PATH):
        print("⬇ Downloading B0 model...")
        gdown.download(f"https://drive.google.com/uc?id={B0_ID}", B0_PATH, quiet=False)

    if not os.path.exists(B4_PATH):
        print("⬇ Downloading B4 model...")
        gdown.download(f"https://drive.google.com/uc?id={B4_ID}", B4_PATH, quiet=False)

# 2. Use global variables for lazy loading
model_b0 = None
model_b4 = None

def get_model(model_type="b0"):
    """Loads model ONLY when requested to save RAM during startup."""
    global model_b0, model_b4
    
    download_models()
    
    if model_type == "b0":
        if model_b0 is None:
            print("🧠 Loading B0 model...")
            # compile=False saves memory by not loading optimizer state
            model_b0 = load_model(B0_PATH, compile=False)
        return model_b0
    else:
        if model_b4 is None:
            print("🧠 Loading B4 model...")
            model_b4 = load_model(B4_PATH, compile=False)
        return model_b4

# Example of how you would use it in your FastAPI route:
# @app.post("/predict")
# def predict(data):
#     model = get_model("b0") # Model loads here, not at startup
#     return model.predict(data)