import os
import cv2
import gdown
import numpy as np
import tensorflow as tf
import keras

from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File

from utils.preprocess import preprocess_image

app = FastAPI()

# -------------------------------------------------
# ⚙️ Reduce logs & force CPU (important for Render)
# -------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")

# -------------------------------------------------
# 📦 MODEL SETTINGS (B0 ONLY)
# -------------------------------------------------

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

B0_PATH = os.path.join(MODEL_DIR, "fracture_b0.keras")
B0_ID = "1-XTjfnayM6lh2c1cYqSIqk0oYdqy7XTy"

# -------------------------------------------------
# ⬇ Download model from Google Drive
# -------------------------------------------------

def download_models():
    if not os.path.exists(B0_PATH):
        print("⬇ Downloading B0 model...")
        gdown.download(
            f"https://drive.google.com/uc?id={B0_ID}",
            B0_PATH,
            quiet=False
        )

# -------------------------------------------------
# 🧠 Load model once (cached)
# -------------------------------------------------

model_b0 = None

def get_model():
    global model_b0

    download_models()

    if model_b0 is None:
        print("🧠 Loading B0 model...")
        model_b0 = load_model(B0_PATH, compile=False)

    return model_b0

# -------------------------------------------------
# 🏠 HEALTH CHECK
# -------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "Service Live",
        "model": "EfficientNet-B0"
    }

# -------------------------------------------------
# 🏷️ MULTICLASS LABELS (EDIT IF NEEDED)
# -------------------------------------------------

CLASS_NAMES = [
    "Hairline Fracture",
    "Comminuted Fracture",
    "Displaced Fracture",
    "Fracture Dislocation"
]

# -------------------------------------------------
# 🔮 PREDICTION ENDPOINT
# -------------------------------------------------

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    model = get_model()

    # Read uploaded image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # ✅ Correct preprocessing for EfficientNet
    img = preprocess_image(img)

    # Predict
    prediction = model.predict(img)[0]

    # Multiclass handling
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    result = CLASS_NAMES[class_index]

    return {
        "prediction": result,
        "confidence": round(confidence, 4)
    }