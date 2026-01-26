# fastapi_predict_api.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uvicorn
import os
import traceback
import tempfile
import sys
import requests
from typing import Optional

# Initialize App
app = FastAPI(title="Prediction API")

# ==========================================
# 1. FIXED CORS SECTION (Hardcoded for Safety)
# ==========================================
origins = [
    "http://127.0.0.1:5500",                  # Local VS Code Live Server
    "http://localhost:5500",                  # Localhost alternative
    "https://cropfront-dxnh.onrender.com"     # <--- YOUR DEPLOYED FRONTEND
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# ==========================================

# Use env var MODEL_PATH if provided; otherwise default to local ./model.h5
MODEL_PATH = os.environ.get("MODEL_PATH", "./model.h5")
MODEL_URL = os.environ.get("MODEL_URL", None)

# === CLASS NAMES ===
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

# ===== Utility: Download model if missing =====
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

# ===== Load model at startup =====
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
        print("Model load failed.", file=sys.stderr)

def _get_input_shape():
    if model is None: return None
    shape = getattr(model, "input_shape", None)
    if not shape: return None
    shape = tuple(shape)
    # Handle various Keras shape formats
    if len(shape) == 4:
        _, a, b, c = shape
        return (b, c, a) if a in (1, 3) else (a, b, c)
    elif len(shape) == 3:
        return shape
    return None

def _load_and_preprocess(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    target = _get_input_shape()
    if target is None:
        raise RuntimeError("Unable to determine model input shape")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((int(target[1]), int(target[0])))
    arr = np.array(img).astype("float32") / 255.0

    if arr.ndim == 2: arr = np.expand_dims(arr, -1)
    
    if arr.shape[-1] != target[2]:
        if target[2] == 1:
            img = img.convert("L")
            arr = np.expand_dims(np.array(img).astype("float32") / 255.0, -1)
        elif target[2] == 3:
            img = img.convert("RGB")
            arr = np.array(img).astype("float32") / 255.0
        else:
            arr = arr[:, :, :target[2]]

    arr = np.expand_dims(arr, axis=0)
    return arr

def _process_predictions(preds: np.ndarray):
    preds = np.array(preds)
    probs = preds[0] if preds.ndim == 2 and preds.shape[0] == 1 else preds.flatten()
    probs_list = probs.tolist()
    pred_index = int(np.argmax(probs_list))
    pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"class_{pred_index}"
    return {
        "predicted_class": pred_class,
        "predicted_prob": float(np.max(probs_list)),
        "probabilities": probs_list,
    }

class PredictPath(BaseModel):
    image_path: str

@app.get("/")
async def root():
    return {"message": "Prediction API is running", "health_url": "/health"}

@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "load_error": load_error
    }

@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    allowed_exts = [".jpg", ".jpeg"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only .jpg or .jpeg files allowed.")

    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        arr = _load_and_preprocess(tmp_path)
        preds = model.predict(arr)
        return _process_predictions(preds)

    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        print(f"Error: {tb}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# Unified Endpoint
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    # 1. Handle File Upload (This is what your frontend uses)
    if file is not None:
        return await predict_upload(file)

    # 2. Handle JSON Body (Fallback)
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="No file uploaded and JSON invalid.")
    
    # Note: Ensure you have the `predict_from_path` function defined if you need this JSON feature. 
    # Since your frontend uploads a file, the code above (predict_upload) will handle it.
    raise HTTPException(status_code=400, detail="JSON path prediction not fully implemented in this snippet.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_predict_api:app", host="0.0.0.0", port=port, reload=False)