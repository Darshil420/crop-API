# fastapi_predict_api.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Query
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import os
import traceback
import tempfile
import sys
import requests  # only used if MODEL_URL is provided
from typing import Optional

app = FastAPI(title="Prediction API")

# Use env var MODEL_PATH if provided; otherwise default to local ./model.h5
MODEL_PATH = os.environ.get("MODEL_PATH", "./model.h5")
# Optional: set MODEL_URL to download the model at startup (direct file URL)
MODEL_URL = os.environ.get("MODEL_URL", None)

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

# ===== Utility: try to download model if MODEL_URL is provided and file missing =====
def _maybe_download_model(path: str, url: Optional[str]) -> Optional[str]:
    if os.path.exists(path):
        return None
    if not url:
        return f"Model file not found at: {path} and no MODEL_URL provided"
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return None
    except Exception as e:
        return f"Failed to download model from MODEL_URL: {url} : {str(e)}"

# ===== Load model at startup with clear logging =====
model = None
load_error: Optional[str] = None

precheck_err = _maybe_download_model(MODEL_PATH, MODEL_URL)
if precheck_err:
    load_error = precheck_err
    print("Model precheck failed:", load_error, file=sys.stderr)
else:
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
    """
    Return (height, width, channels) expected by the model, or None if unknown.
    Handles common Keras shapes like (None, H, W, C) or (None, C, H, W) or (H, W, C).
    """
    if model is None:
        return None
    shape = getattr(model, "input_shape", None)
    if not shape:
        return None

    shape = tuple(shape)
    if len(shape) == 4 and shape[0] is None:
        _, a, b, c = shape
        if a in (1, 3):
            return (b, c, a)
        else:
            return (a, b, c)
    elif len(shape) == 4:
        _, a, b, c = shape
        if a in (1, 3):
            return (b, c, a)
        else:
            return (a, b, c)
    elif len(shape) == 3:
        return shape
    else:
        return None


def _load_and_preprocess(image_path: str):
    """Open image, resize to model input, normalize, and return batched array (1,H,W,C)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    target = _get_input_shape()
    if target is None:
        raise RuntimeError("Unable to determine model input shape")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((int(target[1]), int(target[0])))
    arr = np.array(img).astype("float32") / 255.0

    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    # Adjust channels if necessary
    if arr.shape[-1] != target[2]:
        if target[2] == 1:
            img = img.convert("L")
            arr = np.expand_dims(np.array(img).astype("float32") / 255.0, -1)
        elif target[2] == 3:
            img = img.convert("RGB")
            arr = np.array(img).astype("float32") / 255.0
        else:
            if arr.shape[-1] < target[2]:
                reps = (1, 1, (target[2] // arr.shape[-1]) + 1)
                arr = np.tile(arr, reps)[:, :, :target[2]]
            else:
                arr = arr[:, :, :target[2]]

    arr = np.expand_dims(arr, axis=0)
    return arr


class PredictPath(BaseModel):
    image_path: str


@app.get("/")
async def root():
    return {
        "message": "Prediction API is running",
        "health_url": "/health",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "load_error": load_error,
        "model_path": MODEL_PATH,
        "model_url_env": bool(MODEL_URL)
    }


def _process_predictions(preds: np.ndarray):
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    else:
        probs = preds.flatten()
    probs_list = probs.tolist()
    pred_index = int(np.argmax(probs_list))
    pred_prob = float(np.max(probs_list))
    pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"class_{pred_index}"
    return {
        "predicted_class": pred_class,
        "predicted_index": pred_index,
        "predicted_prob": pred_prob,
        "probabilities": probs_list,
    }


@app.post("/predict_from_path")
async def predict_from_path(payload: PredictPath):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    try:
        arr = _load_and_preprocess(payload.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")

    try:
        preds = model.predict(arr)
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"Error during model.predict(): {str(e)}\n\nTraceback:\n{tb}")

    return _process_predictions(preds)


@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
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
        return _process_predictions(preds)

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


# ------------------ NEW unified endpoint /predict ------------------
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    """
    Unified endpoint:
      - If multipart/form-data with `file` is sent, it will use that file.
      - Otherwise expects JSON body: {"image_path": "absolute/or/relative/path.jpg"}
    """
    # If a file was uploaded via multipart/form-data
    if file is not None:
        return await predict_upload(file)

    # Try to parse JSON body for image_path
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="No file uploaded and JSON body missing or invalid. Provide multipart 'file' or JSON {'image_path': '...'}.")
    image_path = body.get("image_path") if isinstance(body, dict) else None
    if not image_path:
        raise HTTPException(status_code=400, detail="JSON must include 'image_path' when not uploading a file.")
    return await predict_from_path(PredictPath(image_path=image_path))
# ------------------ end /predict ------------------


# ------------------ NEW convenience GET endpoint for browser testing ------------------
@app.get("/predict")
async def predict_get(image_path: Optional[str] = Query(None, description="Server-local image path (absolute or relative)")):
    """
    Convenience GET endpoint for quick browser checks.
    Usage: /predict?image_path=C:\path\to\image.jpg
    NOTE: This uses server's filesystem; path must be accessible to the server.
    """
    if image_path is None:
        return {
            "message": "Use GET /predict?image_path=<path> to predict from a server-local image path, or POST /predict to upload a file or send JSON {'image_path': '...'}."
        }

    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    try:
        # Reuse the existing predict_from_path logic by calling it
        return await predict_from_path(PredictPath(image_path=image_path))
    except HTTPException:
        # propagate known HTTPExceptions
        raise
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=f"Error handling GET /predict: {str(e)}\n\nTraceback:\n{tb}")
# ------------------ end GET convenience endpoint ------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_predict_api:app", host="0.0.0.0", port=port, reload=False)
