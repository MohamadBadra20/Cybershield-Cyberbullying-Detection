# Run with: uvicorn app:app --reload
# Test at:  http://127.0.0.1:8000/docs

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── PATHS ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROBERTA_DIR = BASE_DIR / "roberta_best"

# ── INIT APP ─────────────────────────────────────────────────
app = FastAPI(
    title="Cyberbullying Type Detector",
    description="Detects the type of cyberbullying in social media text",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODELS ───────────────────────────────────────────────
# Note: roberta_best/model.safetensors is not included in this repo (476 MB,
# over GitHub's file size limit). See README for download instructions before
# running this app.
le              = joblib.load(BASE_DIR / "label_encoder.pkl")
classical_model = joblib.load(BASE_DIR / "tuned_lr_model.pkl")

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
dl_model  = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR).to(DEVICE)
dl_model.eval()

# ── SCHEMAS ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: str = "roberta"   # "roberta" or "classical"

class PredictResponse(BaseModel):
    text:       str
    prediction: str
    confidence: float
    all_scores: dict
    model_used: str

# ── HELPER ────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ── ENDPOINTS ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Cyberbullying Type Detector API is running!"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text

    if req.model == "roberta":
        enc = tokenizer(text, return_tensors='pt',
                        max_length=128, truncation=True, padding='max_length').to(DEVICE)
        with torch.no_grad():
            logits = dl_model(**enc).logits.cpu().numpy()[0]

        probs      = softmax(logits)
        pred_idx   = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        all_scores = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)}

    else:
        scores     = classical_model.decision_function([text])[0]
        probs      = softmax(scores)
        pred_idx   = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        all_scores = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return PredictResponse(
        text=text,
        prediction=pred_label,
        confidence=confidence,
        all_scores=all_scores,
        model_used=req.model
    )


@app.get("/classes")
def get_classes():
    return {"classes": list(le.classes_)}


@app.get("/health")
def health():
    return {"status": "ok"}
