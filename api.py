"""Optional FastAPI wrapper around the segmentation model."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
import torch

from checkpoint import DEFAULT_WEIGHTS, load_model
from predict import predict_image


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL: torch.nn.Module | None = None
CHECKPOINT: dict | None = None
DEVICE: torch.device | None = None


class Prediction(BaseModel):
    width: int
    height: int
    threshold: float
    mask_ratio: float
    mask_png_base64: str
    overlay_png_base64: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, CHECKPOINT, DEVICE
    weights = Path(os.getenv("MODEL_WEIGHTS", str(DEFAULT_WEIGHTS)))
    MODEL, CHECKPOINT, DEVICE = load_model(weights)
    yield
    MODEL = CHECKPOINT = DEVICE = None


app = FastAPI(
    title="Dark Spot and Melasma Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL is not None else "unavailable"}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), threshold: float = 0.5) -> Prediction:
    if MODEL is None or CHECKPOINT is None or DEVICE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not 0 < threshold < 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    mask, overlay = predict_image(
        image,
        MODEL,
        DEVICE,
        image_size=int(CHECKPOINT.get("img_size", 640)),
        threshold=threshold,
    )

    def encode_png(value: Image.Image) -> str:
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    mask_array = np.asarray(mask) > 0
    return Prediction(
        width=image.width,
        height=image.height,
        threshold=threshold,
        mask_ratio=float(mask_array.mean()),
        mask_png_base64=encode_png(mask),
        overlay_png_base64=encode_png(overlay),
    )

