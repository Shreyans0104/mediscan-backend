import gdown
import os
import cv2
import numpy as np
import tensorflow as tf
# Important: Keras 3 compatibility for EfficientNet
import keras 
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

# Force CPU only and reduce logging to save a bit of memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

B0_PATH = os.path.join(MODEL_DIR, "fracture_b0.keras")
B0_ID = "1-XTjfnayM6lh2c1cYqSIqk0oYdqy7XTy"

def download_models():
    if not os.path.exists(B0_PATH):
        print("⬇ Downloading B0 model...")
        gdown.download(f"https://drive.google.com/uc?id={B0_ID}", B0_PATH, quiet=False)

# Global variable for the single supported model
model_b0 = None

def get_model():
    global model_b0
    download_models()
    if model_b0 is None:
        print("🧠 Loading B0 model...")
        # compile=False is critical to save RAM (skips optimizer loading)
        model_b0 = load_model(B0_PATH, compile=False)
    return model_b0

@app.get("/")
def home():
    return {"status": "Service Live", "supported_model": "EfficientNetB0"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1. Load B0 model (B4 is disabled to save RAM)
    model = get_model() 
    
    # 2. Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Preprocess
    # Ensure resize matches the input shape of your B0 model (usually 224 or 380)
    img = cv2.resize(img, (224, 224)) 
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)

    # 4. Predict
    prediction = model.predict(img)
    score = float(prediction[0][0])
    
    # Adjust threshold as per your training
    result = "Fracture Detected" if score > 0.5 else "No Fracture"

    return {
        "prediction": result,
        "confidence": round(score, 4)
    }