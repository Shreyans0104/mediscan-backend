import gdown
import os
from tensorflow.keras.models import load_model

# -----------------------------------
# 🔹 Create models directory
# -----------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

B0_PATH = os.path.join(MODEL_DIR, "fracture_b0.keras")
B4_PATH = os.path.join(MODEL_DIR, "fracture_b4.keras")

# -----------------------------------
# 🔹 Google Drive IDs (YOUR FILES)
# -----------------------------------
B0_ID = "1-XTjfnayM6lh2c1cYqSIqk0oYdqy7XTy"
B4_ID = "1Tn7mbDg9AsplfDSnBWVuaIXGVU7JZSI5"

# -----------------------------------
# 🔹 Download if not present
# -----------------------------------
if not os.path.exists(B0_PATH):
    print("⬇ Downloading B0 model...")
    gdown.download(
        f"https://drive.google.com/uc?id={B0_ID}",
        B0_PATH,
        quiet=False
    )

if not os.path.exists(B4_PATH):
    print("⬇ Downloading B4 model...")
    gdown.download(
        f"https://drive.google.com/uc?id={B4_ID}",
        B4_PATH,
        quiet=False
    )

# -----------------------------------
# 🔹 Load models
# -----------------------------------
print("🧠 Loading models...")
model_b0 = load_model(B0_PATH)
model_b4 = load_model(B4_PATH)
print("✅ Models loaded successfully!")