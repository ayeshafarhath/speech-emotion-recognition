"""Train a CNN-BiLSTM baseline on a folder-labelled audio dataset.

The project intentionally keeps the machine-learning stack minimal and reproducible.
It supports: dataset collection, deterministic splitting, model training, baseline
comparison, and evaluation summary generation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.audio import SUPPORTED_AUDIO_EXTENSIONS, extract_features_from_file
from src.models.cnn_lstm import CNNBiLSTMAttention


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    output: str = "models/cnn_lstm.pt"
    baseline_output: str = "models/logistic_regression.joblib"
    results_dir: str = "results"
    sample_rate: int = 16_000
    duration_seconds: float = 5.0
    test_size: float = 0.2
    validation_size: float = 0.2
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 6
    seed: int = 42
    min_speakers: int = 3


def seed_everything(seed: int) -> None:
    """Make data splits and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_speaker_id(path: Path, label_dir: Path) -> str | None:
    """Infer a speaker identifier when the dataset layout supports it.

    The function is intentionally conservative. It only returns a candidate when the
    parent directory is not obviously a class label or a split folder.
    """
    candidate_names = [
        path.parent.name,
        path.parent.parent.name,
    ]
    for candidate in candidate_names:
        if not candidate or candidate in {".", "..", label_dir.name, "train", "validation", "val", "test"}:
            continue
        return candidate
    return None


def collect_records(data_dir: Path) -> list[dict[str, Any]]:
    """Collect audio files and optional metadata from ``data_dir/<label>`` folders."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {data_dir}")

    records: list[dict[str, Any]] = []
    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                records.append(
                    {
                        "path": path,
                        "label": label_dir.name.lower(),
                        "speaker_id": infer_speaker_id(path, label_dir),
                    }
                )

    if not records:
        raise ValueError(f"no supported audio files found under {data_dir}")
    labels = {record["label"] for record in records}
    if len(labels) < 2:
        raise ValueError("dataset must contain at least two emotion directories")

    return records


def build_arrays(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Extract features and encode labels in deterministic alphabetical order."""
    records = collect_records(Path(config.data_dir))
    class_names = sorted({record["label"] for record in records})
    class_to_index = {label: index for index, label in enumerate(class_names)}

    vectors: list[np.ndarray] = []
    encoded: list[int] = []
    metadata: list[dict[str, Any]] = []

    for record in records:
        path = record["path"]
        label = record["label"]
        vectors.append(
            extract_features_from_file(
                path,
                sample_rate=config.sample_rate,
                duration_seconds=config.duration_seconds,
            )
        )
        encoded.append(class_to_index[label])
        metadata.append({
            "path": str(path),
            "label": label,
            "speaker_id": record["speaker_id"],
        })

    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(encoded, dtype=np.int64),
        class_names,
        metadata,
    )


def split_records(records: list[dict[str, Any]], config: TrainingConfig):
    """Create train/validation/test splits.

    If speaker identifiers are available, this uses a speaker-aware split. Otherwise
    it falls back to deterministic stratified splitting and documents the limitation
    through the returned metadata.
    """
    labels = np.asarray([record["label"] for record in records])
    speaker_ids = [record["speaker_id"] for record in records]
    speaker_available = all(speaker_id is not None for speaker_id in speaker_ids) and len(set(speaker_ids)) > 1

    if speaker_available:
        speaker_to_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            speaker_id = record["speaker_id"]
            if speaker_id is None:
                continue
            speaker_to_records.setdefault(speaker_id, []).append(record)

        speaker_ids_unique = sorted(speaker_to_records)
        if len(speaker_ids_unique) >= config.min_speakers:
            speaker_majority_label = {
                speaker: Counter(record["label"] for record in items).most_common(1)[0][0]
                for speaker, items in speaker_to_records.items()
            }
            train_speakers, remaining_speakers = train_test_split(
                speaker_ids_unique,
                test_size=config.test_size,
                random_state=config.seed,
                stratify=[speaker_majority_label[speaker] for speaker in speaker_ids_unique],
            )
            val_speakers, test_speakers = train_test_split(
                remaining_speakers,
                test_size=config.test_size / (1.0 - config.test_size),
                random_state=config.seed,
                stratify=[speaker_majority_label[speaker] for speaker in remaining_speakers],
            )

            train_records = [record for speaker in train_speakers for record in speaker_to_records[speaker]]
            validation_records = [record for speaker in val_speakers for record in speaker_to_records[speaker]]
            test_records = [record for speaker in test_speakers for record in speaker_to_records[speaker]]

            return {
                "train": train_records,
                "validation": validation_records,
                "test": test_records,
                "speaker_aware": True,
                "speaker_count": len(speaker_ids_unique),
                "warning": "speaker-aware split applied",
            }

    train_val_records, test_records = train_test_split(
        records,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    relative_validation_size = config.validation_size / (1.0 - config.test_size)
    train_records, validation_records = train_test_split(
        train_val_records,
        test_size=relative_validation_size,
        random_state=config.seed,
        stratify=[record["label"] for record in train_val_records],
    )

    return {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
        "speaker_aware": False,
        "speaker_count": len(set(speaker_ids)),
        "warning": "speaker-independent evaluation cannot be guaranteed without reliable speaker metadata",
    }


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Convert fixed vectors to one-step sequences accepted by the model."""
    tensors = TensorDataset(torch.from_numpy(features).unsqueeze(1), torch.from_numpy(labels))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def class_weights(labels: np.ndarray, classes: int) -> Tensor:
    """Return inverse-frequency weights for imbalanced training data."""
    counts = np.bincount(labels, minlength=classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("every class must be represented in the training partition")
    weights = counts.sum() / (classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    """Evaluate the model and return metrics with actual predictions."""
    model.eval()
    preds: list[int] = []
    true: list[int] = []

    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            logits = model(batch_features.to(device))
            probabilities = torch.softmax(logits, dim=-1)
            batch_predictions = torch.argmax(probabilities, dim=-1)
            preds.extend(batch_predictions.detach().cpu().tolist())
            true.extend(batch_labels.detach().cpu().tolist())

    labels = list(range(len(class_names)))
    precision = precision_score(true, preds, labels=labels, average=None, zero_division=0)
    recall = recall_score(true, preds, labels=labels, average=None, zero_division=0)
    f1 = f1_score(true, preds, labels=labels, average=None, zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(true, preds)),
        "precision": {class_names[index]: float(value) for index, value in enumerate(precision)},
        "recall": {class_names[index]: float(value) for index, value in enumerate(recall)},
        "f1": {class_names[index]: float(value) for index, value in enumerate(f1)},
        "macro_f1": float(f1_score(true, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, preds, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(true, preds, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, preds, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, preds, labels=labels).tolist(),
        "class_distribution": {
            class_names[index]: int((np.asarray(true) == index).sum()) for index in range(len(class_names))
        },
    }
    return metrics


def train_baseline_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Train a simple logistic-regression baseline on the same features."""
    model = LogisticRegression(max_iter=2000, multi_class="auto")
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision": {class_names[index]: float(value) for index, value in enumerate(precision_score(test_y, predictions, labels=list(range(len(class_names))), average=None, zero_division=0))},
        "recall": {class_names[index]: float(value) for index, value in enumerate(recall_score(test_y, predictions, labels=list(range(len(class_names))), average=None, zero_division=0))},
        "f1": {class_names[index]: float(value) for index, value in enumerate(f1_score(test_y, predictions, labels=list(range(len(class_names))), average=None, zero_division=0))},
        "macro_f1": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
    }
    return metrics


def save_metrics(results_dir: Path, filename: str, metrics: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Train the CNN-BiLSTM model and return metadata. The function is structured so
    the project can run meaningful evaluations when actual data is available."""
    seed_everything(config.seed)
    feature_matrix, labels, class_names, metadata = build_arrays(config)
    split_summary = split_records(metadata, config)
    train_records = split_summary["train"]
    validation_records = split_summary["validation"]
    test_records = split_summary["test"]

    train_paths = [record["path"] for record in train_records]
    validation_paths = [record["path"] for record in validation_records]
    test_paths = [record["path"] for record in test_records]

    train_features = np.asarray([
        extract_features_from_file(path, sample_rate=config.sample_rate, duration_seconds=config.duration_seconds)
        for path in train_paths
    ], dtype=np.float32)
    validation_features = np.asarray([
        extract_features_from_file(path, sample_rate=config.sample_rate, duration_seconds=config.duration_seconds)
        for path in validation_paths
    ], dtype=np.float32)
    test_features = np.asarray([
        extract_features_from_file(path, sample_rate=config.sample_rate, duration_seconds=config.duration_seconds)
        for path in test_paths
    ], dtype=np.float32)

    train_labels = np.asarray([class_names.index(record["label"]) for record in train_records], dtype=np.int64)
    validation_labels = np.asarray([class_names.index(record["label"]) for record in validation_records], dtype=np.int64)
    test_labels = np.asarray([class_names.index(record["label"]) for record in test_records], dtype=np.int64)

    if len(np.unique(train_labels)) < 2:
        raise ValueError("training split must contain at least two classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMAttention(train_features.shape[1], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, len(class_names)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    train_loader = make_loader(train_features, train_labels, config.batch_size, shuffle=True)
    validation_loader = make_loader(validation_features, validation_labels, config.batch_size, shuffle=False)
    test_loader = make_loader(test_features, test_labels, config.batch_size, shuffle=False)

    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        validation_loss = 0.0
        total_items = 0
        model.eval()
        with torch.inference_mode():
            for batch_features, batch_labels in validation_loader:
                logits = model(batch_features.to(device))
                loss = criterion(logits, batch_labels.to(device))
                validation_loss += float(loss) * batch_labels.size(0)
                total_items += batch_labels.size(0)
        validation_loss /= max(total_items, 1)
        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = evaluate_model(model, test_loader, device, class_names)
    baseline_metrics = train_baseline_classifier(train_features, train_labels, test_features, test_labels, class_names)

    save_metrics(results_dir, "metrics.json", {
        "model": "cnn_bilstm_attention",
        "split": split_summary,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "class_names": class_names,
        "config": asdict(config),
    })

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": feature_matrix.shape[1],
            "class_names": class_names,
            "sample_rate": config.sample_rate,
            "duration_seconds": config.duration_seconds,
            "config": asdict(config),
            "split_summary": split_summary,
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
        },
        output_path,
    )

    return {
        "output": str(output_path),
        "classes": class_names,
        "split_summary": split_summary,
        "results_dir": str(results_dir),
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", default="models/cnn_lstm.pt")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=str(args.data_dir),
        output=args.output,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(train_model(config), indent=2))


if __name__ == "__main__":
    main()













































