"""FastAPI application for speech-emotion inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.predict import DEFAULT_CHECKPOINT, predict

APP_TITLE = "Speech Emotion Recognition API"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a"})
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

app = FastAPI(
    title=APP_TITLE,
    description="Predict an emotion from a short speech recording.",
    version="1.0.0",
)

_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    checkpoint: str


class PredictionResponse(BaseModel):
    emotion: str
    confidence_scores: dict[str, float]
    processing_time_ms: float


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and whether the configured model is available."""
    return HealthResponse(
        status="ok",
        model_available=CHECKPOINT_PATH.is_file(),
        checkpoint=str(CHECKPOINT_PATH),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(
    file: Annotated[UploadFile, File(description="A short WAV, FLAC, MP3, OGG, or M4A recording")],
) -> PredictionResponse:
    """Predict the emotion expressed in an uploaded audio recording."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if not CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint is unavailable: {CHECKPOINT_PATH}",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file must be smaller than 25 MB")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        result = predict(temporary_path, CHECKPOINT_PATH)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/predict-realtime", response_model=PredictionResponse)
async def predict_realtime(
    file: Annotated[UploadFile, File(description="A short audio chunk")],
) -> PredictionResponse:
    """Predict a short audio chunk using the same validated inference path."""
    return await predict_audio(file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
