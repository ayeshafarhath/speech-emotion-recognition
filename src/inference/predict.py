"""Run emotion prediction from a trained CNN-BiLSTM-Attention checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.features.audio import extract_features_from_file
from src.models.cnn_lstm import CNNBiLSTMAttention

DEFAULT_CHECKPOINT = Path("models/cnn_lstm.pt")


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint onto the selected device with basic metadata validation."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "class_names", "input_features"}
    missing = required.difference(checkpoint)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"checkpoint is missing required metadata: {missing_names}")
    if not isinstance(checkpoint["class_names"], list) or len(checkpoint["class_names"]) < 2:
        raise ValueError("checkpoint class_names must contain at least two classes")
    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> CNNBiLSTMAttention:
    """Reconstruct the model architecture recorded by the training pipeline."""
    model = CNNBiLSTMAttention(
        input_features=int(checkpoint["input_features"]),
        num_classes=len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def predict(
    input_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Predict one audio file and return class probabilities and latency."""
    started = time.perf_counter()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(checkpoint_path, selected_device)
    model = build_model(checkpoint, selected_device)

    sample_rate = int(checkpoint.get("sample_rate", 16_000))
    duration_seconds = float(checkpoint.get("duration_seconds", 5.0))
    features = extract_features_from_file(
        input_path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    expected_features = int(checkpoint["input_features"])
    if features.shape != (expected_features,):
        raise ValueError(
            f"feature dimension mismatch: checkpoint expects {expected_features}, "
            f"but extraction produced {features.shape[0]}"
        )

    feature_tensor = torch.from_numpy(features).to(selected_device).view(1, 1, -1)
    with torch.inference_mode():
        probabilities = model.predict_proba(feature_tensor)[0].detach().cpu().tolist()

    class_names = [str(name) for name in checkpoint["class_names"]]
    confidence_scores = {
        name: round(float(score), 6) for name, score in zip(class_names, probabilities)
    }
    emotion = max(confidence_scores, key=confidence_scores.get)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "emotion": emotion,
        "confidence_scores": confidence_scores,
        "processing_time_ms": round(elapsed_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an emotion from a speech recording")
    parser.add_argument("--input", required=True, type=Path, help="path to a supported audio file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="path to a CNN-BiLSTM-Attention checkpoint",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    print(json.dumps(predict(args.input, args.checkpoint, device=args.device), indent=2))


if __name__ == "__main__":
    main()
