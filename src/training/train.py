"""Train the CNN-BiLSTM baseline on a folder-labelled audio dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.audio import SUPPORTED_AUDIO_EXTENSIONS, extract_features_from_file
from src.models.cnn_lstm import CNNBiLSTMAttention


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    output: str = "models/cnn_lstm.pt"
    sample_rate: int = 16_000
    duration_seconds: float = 5.0
    test_size: float = 0.2
    validation_size: float = 0.2
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 6
    seed: int = 42


def seed_everything(seed: int) -> None:
    """Make data splits and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_records(data_dir: Path) -> tuple[list[Path], list[str]]:
    """Collect audio files from ``data_dir/<label>/file`` directories."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {data_dir}")
    paths: list[Path] = []
    labels: list[str] = []
    for label_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                paths.append(path)
                labels.append(label_dir.name.lower())
    if len(set(labels)) < 2:
        raise ValueError("dataset must contain at least two emotion directories")
    return paths, labels


def build_arrays(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features and encode labels in deterministic alphabetical order."""
    paths, labels = collect_records(Path(config.data_dir))
    class_names = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(class_names)}
    vectors: list[np.ndarray] = []
    encoded: list[int] = []
    for path, label in zip(paths, labels):
        vectors.append(
            extract_features_from_file(
                path,
                sample_rate=config.sample_rate,
                duration_seconds=config.duration_seconds,
            )
        )
        encoded.append(class_to_index[label])
    return np.asarray(vectors, dtype=np.float32), np.asarray(encoded, dtype=np.int64), class_names


def split_data(features: np.ndarray, labels: np.ndarray, config: TrainingConfig):
    """Create stratified train, validation, and test partitions."""
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    relative_validation_size = config.validation_size / (1.0 - config.test_size)
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_val_x,
        train_val_y,
        test_size=relative_validation_size,
        random_state=config.seed,
        stratify=train_val_y,
    )
    return train_x, train_y, validation_x, validation_y, test_x, test_y


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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            total_loss += float(loss) * labels.size(0)
            total_items += labels.size(0)
    return total_loss / max(total_items, 1)


def train_model(config: TrainingConfig) -> dict:
    seed_everything(config.seed)
    features, labels, class_names = build_arrays(config)
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_data(features, labels, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMAttention(features.shape[1], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, len(class_names)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    train_loader = make_loader(train_x, train_y, config.batch_size, shuffle=True)
    validation_loader = make_loader(validation_x, validation_y, config.batch_size, shuffle=False)
    test_loader = make_loader(test_x, test_y, config.batch_size, shuffle=False)

    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        validation_loss = evaluate(model, validation_loader, criterion, device)
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
    test_loss = evaluate(model, test_loader, criterion, device)
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_features": features.shape[1],
        "class_names": class_names,
        "sample_rate": config.sample_rate,
        "duration_seconds": config.duration_seconds,
        "config": asdict(config),
        "test_loss": test_loss,
    }, output_path)
    return {"output": str(output_path), "classes": class_names, "test_loss": test_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", default="models/cnn_lstm.pt")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    config = TrainingConfig(
        data_dir=str(args.data_dir),
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(train_model(config), indent=2))


if __name__ == "__main__":
    main()
