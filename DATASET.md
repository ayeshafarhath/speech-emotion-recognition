# Dataset documentation

## Status

Dataset provenance could not be fully verified from the repository alone. This project does not ship a dataset, and no dataset files are committed to the repository. The repository expects a local labeled dataset to be prepared by the user.

The project therefore documents the expected workflow and required structure without claiming a specific dataset has been verified in this repository.

## Expected dataset format

The training code expects a folder-structured dataset in which each class is a top-level directory and each file is an audio recording.

Example:

```text
data/
  angry/
    sample_001.wav
    sample_002.wav
  happy/
    sample_001.wav
  neutral/
    sample_001.wav
  sad/
    sample_001.wav
```

The repository reads all files under each label directory and accepts the following audio extensions:

- .wav
- .flac
- .mp3
- .ogg
- .m4a

## Known assumptions

The training pipeline currently uses the following assumptions:

- sample rate: 16,000 Hz
- clip duration: 5.0 seconds
- mono audio
- fixed-length waveform normalization
- deterministic file ordering for consistent processing

These assumptions are defined in the feature extraction and training configuration in `src/features/audio.py` and `src/training/train.py`.

## Speaker leakage

Speaker leakage is a serious issue in emotion recognition projects. If the same speaker appears in both training and test sets, report quality can be overestimated.

The repository supports speaker-aware splitting when speaker metadata can be inferred from the dataset layout. The training pipeline will attempt to infer a speaker identifier from a file path or directory structure when possible.

If the dataset does not contain speaker metadata or the structure does not support reliable inference, the project falls back to a deterministic stratified split and explicitly documents that speaker-independent evaluation cannot be guaranteed.

## Required metadata

For the most defensible evaluation, the dataset should include:

- speaker identifier
- file path
- label
- split assignment (train / validation / test)

When speaker metadata is absent, the project cannot honestly claim speaker-independent performance.

## Official source

This repository does not include an official dataset reference. The user must obtain the dataset from an official, legally usable source and verify license terms before training.

## License and usage restrictions

The repository does not assume or claim a license for any external dataset. Any dataset used must comply with its own licensing and terms of use.

## Dataset preparation workflow

1. Acquire a labeled emotion dataset from an official source.
2. Organize files by class folder.
3. Verify audio sampling rate and format.
4. Confirm per-class label counts.
5. Apply a deterministic split.
6. Confirm the split is speaker-aware if possible.
7. Run the training script.

## Missing information

The exact dataset used by this repository could not be verified from the codebase alone. The following facts are therefore not claimed:

- exact dataset name
- exact number of samples
- exact class distribution
- exact train/validation/test counts
- speaker count
- final benchmark numbers

The project remains intentionally honest about this limitation.

---

This document is not a claim that the dataset is available or validated. It is a clear statement of the repository’s expected dataset contract and the constraints under which the project should be used.
