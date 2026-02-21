import gdown
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# Reduce TensorFlow logs & force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

B0_PATH = os.path.join(MODEL_DIR, "fracture_b0.keras")
B0_ID = "1-XTjfnayM6lh2c1cYqSIqk0oYdqy7XTy"

# ---------------- DOWNLOAD MODEL ----------------

def download_models():
    if not os.path.exists(B0_PATH):
        print("⬇ Downloading B0 model...")
        gdown.download(
            f"https://drive.google.com/uc?id={B0_ID}",
            B0_PATH,
            quiet=False
        )

# ---------------- LOAD MODEL ----------------

model_b0 = None

def get_model():
    global model_b0
    download_models()
    if model_b0 is None:
        print("🧠 Loading B0 model...")
        model_b0 = load_model(B0_PATH, compile=False)
    return model_b0

# ---------------- PREPROCESS ----------------

def preprocess_image(img, target_size=(380, 380)):

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, target_size)

    # Convert to float
    img = img.astype("float32")

    # EfficientNet preprocessing
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {
        "status": "Service Live",
        "supported_model": "EfficientNetB0"
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    model = get_model()

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # ⭐ USE CORRECT PREPROCESSING
    img = preprocess_image(img)

    prediction = model.predict(img)

    score = float(prediction.flatten()[0])

    result = "Fracture Detected" if score > 0.5 else "No Fracture"

    return {
        "prediction": result,
        "confidence": round(score, 4)
    }