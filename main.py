from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os

app = FastAPI(title="PurrtectMe Scam Detector")

# Replace with your actual Hugging Face model path from Cell 6 of the training script
MODEL_ID = "tshai1/purrtectme-scam-detector-v1"

# A simple shared-secret key so random people on the internet can't call your model for free.
# Set this in Render's environment variables (Step 4 below) — don't hardcode a real value here.
API_KEY = os.environ.get("API_KEY", "change-me")

# Keep memory and CPU usage as low as possible on Render's free tier
torch.set_num_threads(1)

classifier = None  # loaded on startup, see below


@app.on_event("startup")
def load_model():
    global classifier
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        torch_dtype=torch.float32,
        model_kwargs={"low_cpu_mem_usage": True},
    )


class ClassifyRequest(BaseModel):
    text: str
    key: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "PurrtectMe scam detector is running"}


@app.post("/classify")
def classify(request: ClassifyRequest):
    if request.key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    result = classifier(request.text[:2000])  # cap length to match training
    top = result[0]

    return {
        "label": top["label"],        # "scam" or "legitimate"
        "score": round(top["score"], 4),
    }
