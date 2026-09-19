"""API contract tests."""

from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import api.main as api_module

client = TestClient(api_module.app)


def wav_bytes() -> bytes:
    """Create a short valid WAV payload entirely in memory."""
    samples = np.zeros(1600, dtype=np.float32)
    samples[0] = 0.25
    buffer = __import__("io").BytesIO()
    sf.write(buffer, samples, 16_000, format="WAV")
    return buffer.getvalue()


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_available" in body
    assert "checkpoint" in body


def test_predict_rejects_non_wav_files() -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_rejects_empty_files() -> None:
    response = client.post(
        "/predict",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded audio is empty"


def test_predict_returns_503_when_checkpoint_is_missing(monkeypatch) -> None:
    missing_checkpoint = Path("/tmp/speech-emotion-checkpoint-does-not-exist.pt")
    monkeypatch.setattr(api_module, "CHECKPOINT_PATH", missing_checkpoint)

    response = client.post(
        "/predict",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_predict_returns_valid_prediction_contract(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "cnn_lstm.pt"
    checkpoint.write_bytes(b"test checkpoint marker")
    monkeypatch.setattr(api_module, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(
        api_module,
        "predict",
        lambda input_path, checkpoint_path: {
            "emotion": "happy",
            "confidence_scores": {
                "happy": 0.91,
                "sad": 0.04,
                "angry": 0.03,
                "neutral": 0.02,
            },
            "processing_time_ms": 12.5,
        },
    )

    response = client.post(
        "/predict",
        files={"file": ("sample.wav", wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] == "happy"
    assert body["confidence_scores"]["happy"] == 0.91
    assert body["processing_time_ms"] == 12.5
