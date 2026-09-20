# Speech Emotion Recognition

A small experimental speech emotion recognition project using Python, PyTorch, and audio feature extraction.

This repository is designed as a practical baseline for speech classification using audio signals. It is intentionally presented as an experimental project rather than a production-ready or benchmarked system.

## Project Goal

The project explores whether short speech clips can be classified into emotion categories using:
- audio preprocessing
- feature extraction
- a neural classifier
- a simple inference pipeline

## Architecture

The project includes:
- feature extraction from audio files
- model definition for a CNN + BiLSTM style architecture
- training logic
- prediction/inference flow
- a small FastAPI application for serving predictions

## Tech Stack

- Python
- PyTorch
- Librosa
- NumPy
- scikit-learn
- FastAPI
- Docker
- pytest

## Repository Structure

```text
src/
  features/
  inference/
  models/
  training/
api/
tests/
requirements.txt
Dockerfile
docker-compose.yml
```

## Getting Started

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train the model:

```bash
python -m src.training.train --help
```

Run the API locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open:
- http://localhost:8000/docs

## Notes

- This is a research-style baseline project.
- Dataset availability and preprocessing assumptions should be checked before running training.
- No public accuracy benchmark is claimed in this repository.
- This project is intended to demonstrate ML engineering and audio processing workflow, not to claim deployment-ready performance.

## Limitations

- The dataset and model performance must be validated separately.
- Audio quality, class balance, and dataset split choices strongly affect results.
- Model outputs should be treated as experimental predictions rather than validated business-grade decisions.
- This project is best viewed as a learning and prototyping project.
