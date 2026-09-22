# Speech Emotion Recognition

## Overview

This repository implements a speech emotion recognition baseline based on short audio recordings and handcrafted acoustic features. The project is intentionally framed as a reproducible ML engineering effort rather than a production-ready system.

The current implementation is organized around:

- audio validation and loading
- feature extraction from short waveform segments
- a CNN + BiLSTM + attention classifier
- deterministic train/validation/test splitting
- model checkpointing
- FastAPI inference
- Docker-based runtime startup
- lightweight automated tests

## Problem

Speech emotion recognition is a challenging multivariate audio classification problem. A model must cope with:

- class imbalance
- speaker variability
- audio quality differences
- inconsistent recording conditions
- label ambiguity

A repository can only be considered portfolio-grade if the dataset, split methodology, and evaluation pipeline are documented and reproducible.

## Solution

This project provides a minimal but structured pipeline for:

1. loading audio and validating it
2. extracting fixed-length acoustic features
3. training a baseline classifier
4. saving model checkpoints with metadata
5. serving predictions via FastAPI
6. running lightweight validation tests

This project should be viewed as a defensible experimental baseline, not as an already-validated commercial system.

## Architecture

```text
Audio file
  ↓
Validation
  ↓
Preprocessing
  ↓
Feature extraction
  ↓
Train / validation / test split
  ↓
CNN-BiLSTM-Attention model
  ↓
Checkpoint + metrics
  ↓
FastAPI prediction service
  ↓
Docker runtime
```

## Dataset

The repository does not contain a dataset. The expected workflow is to acquire a labeled audio dataset and organize it into class directories, as documented in `DATASET.md`.

Dataset provenance could not be fully verified from the repository. The project intentionally documents this limitation rather than guessing at a dataset source.

## Dataset Preparation

The repository expects a folder-based structure similar to:

```text
data/
  angry/
    speaker_001_001.wav
    speaker_001_002.wav
  happy/
    speaker_002_001.wav
  neutral/
    speaker_003_001.wav
  sad/
    speaker_004_001.wav
```

The exact directory layout should match the dataset used in the training run. The split strategy is documented in `src/training/train.py` and `DATASET.md`.

## Data Split

The project supports deterministic splitting with a fixed random seed.

A speaker-aware split is supported when speaker metadata can be inferred from the dataset structure. If speaker IDs are not available, the repository falls back to a deterministic stratified split and explicitly documents the limitation.

The project does not claim speaker-independent generalization unless the dataset and metadata support it.

## Feature Extraction

The extracted acoustic vector includes summary statistics for:

- MFCCs
- delta MFCCs
- chroma features
- mel spectrogram features
- spectral contrast
- zero-crossing rate
- RMS energy

All features are normalized to deterministic fixed-size outputs and validated for finite values.

## Models

### Primary model

The primary model is a CNN + BiLSTM + attention model implemented in `src/models/cnn_lstm.py`.

### Baseline philosophy

A simple baseline is supported in the training pipeline but should be used intentionally to answer the question: "Does the more complex model provide meaningful improvement over a simpler classifier?"

The repository does not claim that the advanced model is always better without running the comparison on the same split.

## Training

The training workflow is implemented in `src/training/train.py`.

It includes:

- deterministic seeding
- train/validation/test splitting
- feature extraction from labelled audio files
- class-weighted cross-entropy
- early stopping
- checkpoint saving with metadata

The training script can be used as follows:

```bash
python -m src.training.train --data-dir /path/to/data --output models/cnn_lstm.pt
```

## Evaluation

The repository contains a reproducible evaluation pipeline in the training code. It produces actual metrics only when a dataset and checkpoint are available.

The metrics intended for evaluation include:

- accuracy
- precision
- recall
- F1-score
- macro F1
- weighted F1
- confusion matrix
- per-class metrics
- class distribution

At this point, no verified metrics are included in the repository because the project does not ship a dataset or a trained checkpoint.

## Results

No verified model metrics are currently reported in this repository.

This repository does not claim real accuracy, F1, or benchmark results without a real dataset and a completed training run.

## Error Analysis

Error analysis is implemented as a valid evaluation step when the training pipeline is run on a real dataset. The workflow is designed to report:

- confused emotion pairs
- weak classes
- class imbalance effects
- audio-quality issues
- speaker variability effects

If the dataset is unavailable or the model has not been trained, this analysis remains pending.

## API

The FastAPI app is defined in `api/main.py`.

It provides:

- `GET /health`
- `POST /predict`
- `POST /predict-realtime`

The API validates file type, file size, empty payloads, and missing checkpoints.

Example usage:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/sample.wav"
```

## Docker

A Dockerfile and docker-compose configuration are included. These provide a runtime environment for the API and a sane CPU-based setup.

The project should be treated as a locally runnable ML service baseline, not a production deployment.

## Testing

The project includes automated tests for:

- audio loading
- feature extraction
- invalid audio handling
- API contract validation
- checkpoint unavailability behavior

Run tests with:

```bash
pytest tests/ -q
```

## Project Structure

```text
speech-emotion-recognition/
├── api/
│   └── main.py
├── src/
│   ├── features/
│   │   └── audio.py
│   ├── inference/
│   │   └── predict.py
│   ├── models/
│   │   ├── cnn_lstm.py
│   │   └── wav2vec_finetune.py
│   └── training/
│       └── train.py
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── DATASET.md
├── Dockerfile
├── README.md
├── requirements.txt
├── .flake8
├── .gitignore
├── LICENSE
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

System dependencies may also be required for audio processing:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

## Usage

Train a model from a labeled dataset:

```bash
python -m src.training.train --data-dir /path/to/data --output models/cnn_lstm.pt
```

Run the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Run Docker:

```bash
docker build -t speech-emotion-recognition .
docker run --rm -p 8000:8000 speech-emotion-recognition
```

## Limitations

The project has several important limitations that must be documented honestly:

- The dataset is not shipped in the repository.
- Dataset provenance could not be fully verified from the repo alone.
- Speaker independence cannot be guaranteed unless metadata is explicitly available and validated.
- The project currently provides a foundation for reproducible training and evaluation, not a fully validated benchmark.
- Model results must be generated by running the actual training pipeline on a real dataset.
- Audio quality and class imbalance can strongly affect performance.

## Future Improvements

- add a stricter speaker-aware data split when dataset metadata is available
- add a real baseline comparison
- add a more detailed experiment report
- improve evaluation artifact handling
- compare feature extraction strategies
- add stronger API validation around invalid audio and duration
- add richer CI smoke testing

## License

This project is licensed under the MIT License.

---

This repository is best understood as a reproducible experimental baseline for speech emotion recognition, not as a benchmark-validated commercial system.

The project is intentionally honest about what has and has not been verified.


































































