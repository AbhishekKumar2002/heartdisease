from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from joblib import load
import os
import requests
import uvicorn

app = FastAPI()

# Allow frontend access (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your actual frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model setup
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749283968550.bin"
MODEL_PATH = "heart_model.joblib"

# Download the model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from S3...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded.")
        else:
            raise Exception(f"Failed to download model: {response.status_code}")

download_model()

# Load model
model = load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is live."}

@app.post("/predict")
async def predict(
    op: float = Form(...),     # Oldpeak
    mhra: int = Form(...),     # Max Heart Rate
    eia: int = Form(...),      # Exercise Induced Angina (0/1)
    nmv: int = Form(...),      # No. of Major Vessels (0-4)
    tcp: int = Form(...),      # Chest Pain Type (0-3)
    age: int = Form(...),      # Age
    thal: int = Form(...)      # Thalassemia (1-3)
):
    try:
        input_array = np.array([[op, mhra, eia, nmv, tcp, age, thal]])
        prediction = model.predict(input_array)
        return {"result": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
