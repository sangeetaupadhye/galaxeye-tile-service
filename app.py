"""
Part 2 - the thin working slice.

ONE real endpoint: POST /classify
    -> accepts an uploaded tile image
    -> runs it through the CNN trained from scratch in Colab
    -> stores the result in SQLite
    -> returns the prediction to the caller

Everything else (querying/listing past results, auth, batch upload,
retraining triggers, etc.) is intentionally NOT built here - see README.md
for what's stubbed and why.

Run: uvicorn app:app --reload
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from cnn_model import TileClassifier
import db

MODEL_VERSION = "scratch-cnn-v1"  # bump this if you retrain/change the architecture

app = FastAPI(title="Tile Classification Service (thin slice)")

# Loaded once at startup, not per-request.
_classifier: TileClassifier | None = None


@app.on_event("startup")
def load_model():
    global _classifier
    _classifier = TileClassifier()
    print("[app] CNN loaded, ready to serve")


@app.post("/classify")
async def classify_tile(file: UploadFile = File(...)):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    result = _classifier.classify(image)

    row_id = db.insert_result(
        filename=file.filename or "unknown",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        all_class_probabilities=result["all_class_probabilities"],
        model_version=MODEL_VERSION,
    )

    return {
        "id": row_id,
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_class_probabilities": result["all_class_probabilities"],
        "model_version": MODEL_VERSION,
    }