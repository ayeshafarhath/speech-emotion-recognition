"""Tests for audio loading, feature extraction, and augmentation."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.features.audio import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    augment_audio,
    extract_features,
    extract_features_from_file,
    load_audio,
)


def write_sine_wave(path: Path, *, sample_rate: int = 16_000, seconds: float = 1.0) -> None:
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    signal = (0.4 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    sf.write(path, signal, sample_rate)


def test_load_audio_returns_normalized_fixed_length(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    write_sine_wave(path, seconds=0.5)

    signal = load_audio(path, sample_rate=DEFAULT_SAMPLE_RATE, duration_seconds=DEFAULT_DURATION_SECONDS)

    assert signal.shape == (int(DEFAULT_SAMPLE_RATE * DEFAULT_DURATION_SECONDS),)
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) > 0


def test_feature_vector_is_fixed_size_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    write_sine_wave(path)

    features = extract_features_from_file(path)
    expected_size = 2 * (40 + 40 + 12 + 128 + 7 + 1 + 1)

    assert features.shape == (expected_size,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_extract_features_rejects_stereo_signal() -> None:
    stereo = np.ones((1000, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="one-dimensional"):
        extract_features(stereo)


def test_augmentation_is_reproducible_with_seed() -> None:
    signal = np.sin(np.linspace(0, 2 * np.pi, 16_000, dtype=np.float32))

    first = augment_audio(signal, rng=np.random.default_rng(123))
    second = augment_audio(signal, rng=np.random.default_rng(123))

    np.testing.assert_allclose(first, second)
    assert first.shape == signal.shape
    assert np.isfinite(first).all()
