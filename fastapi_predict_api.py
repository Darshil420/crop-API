from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import os
import traceback
import tempfile
import sys

app = FastAPI(title="Prediction API")

# === change this to your actual model path (raw string recommended) ===
MODEL_PATH = r"C:\Users\DEVEL\OneDrive\Desktop\crop\model.h5"

# === CLASS NAMES (must match model output order) ===
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Load model at startup with clear logging
model = None
load_error = None
try:
    print(f"Attempting to load model from: {MODEL_PATH}", file=sys.stderr)
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.", file=sys.stderr)
except Exception as e:
    load_error = "".join(traceback.format_exception(None, e, e.__traceback__))
    model = None
    print("Model load failed. See load_error for details.", file=sys.stderr)
    print(load_error, file=sys.stderr)


def _get_input_shape():
    """Return (height, width, channels) expected by the model, or None."""
    if model is None:
        return None
    shape = model.input_shape
    # typical shapes: (None, H, W, C) or (None, C, H, W) or (H, W, C)
    if not shape:
        return None
    # Normalize to tuple
    if len(shape) == 4:
        # shape like (None, a, b, c) or (None, c, a, b)
        _, a, b, c = shape
        # Heuristic: if 'a' is 1 or 3 it's channel-first (C,H,W)
        if a in (1, 3):
            return (b, c, a)
        else:
            return (a, b, c)
    elif len(shape) == 3:
        return shape
    else:
        return None


def _load_and_preprocess(image_path: str):
    """Open image, resize to model input, normalize, and return batched array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    target = _get_input_shape()
    if target is None:
        raise RuntimeError("Unable to determine model input shape")

    img = Image.open(image_path).convert("RGB")
    # target is (H, W, C)
    img = img.resize((target[1], target[0]))
    arr = np.array(img).astype("float32") / 255.0

    # If arr is grayscale or shape mismatched handle conversions
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    if arr.shape[-1] != target[2]:
        if target[2] == 1:
            img = img.convert("L")
            arr = np.expand_dims(np.array(img).astype("float32") / 255.0, -1)
        elif target[2] == 3:
            img = img.convert("RGB")
            arr = np.array(img).astype("float32") / 255.0
        else:
            # repeat or truncate channels
            if arr.shape[-1] < target[2]:
                arr = np.tile(arr, (1, 1, target[2] // arr.shape[-1] + 1))[:, :, :target[2]]
            else:
                arr = arr[:, :, :target[2]]

    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr


class PredictPath(BaseModel):
    image_path: str


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "load_error": load_error,
    }


@app.post("/predict_from_path")
async def predict_from_path(payload: PredictPath):
    """Predict by giving a server-local image path (image must be accessible by server)."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    try:
        arr = _load_and_preprocess(payload.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")

    preds = model.predict(arr)
    # Normalize prediction to probability list
    if isinstance(preds, list):
        preds = np.array(preds)
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0].tolist()
    else:
        probs = preds.flatten().tolist()

    pred_index = int(np.argmax(probs))
    pred_prob = float(np.max(probs))
    pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"class_{pred_index}"

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_index,
        "predicted_prob": pred_prob,
        "probabilities": probs,
    }


@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    """Predict by uploading an image file (temporary file used, cross-platform)."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    tmp_path = None
    try:
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        arr = _load_and_preprocess(tmp_path)
        preds = model.predict(arr)
        preds = np.array(preds)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0].tolist()
        else:
            probs = preds.flatten().tolist()

        pred_index = int(np.argmax(probs))
        pred_prob = float(np.max(probs))
        pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"class_{pred_index}"

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_index,
            "predicted_prob": pred_prob,
            "probabilities": probs,
        }

    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        print("Error in /predict_upload:", tb, file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Exception during predict_upload: {str(e)}\n\nTraceback:\n{tb}")

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # run with: python fastapi_predict_api.py
    uvicorn.run("fastapi_predict_api:app", host="0.0.0.0", port=8000, reload=False)
