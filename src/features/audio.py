"""Audio loading, preprocessing, feature extraction, and augmentation utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION_SECONDS = 5.0
SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a"})


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> np.ndarray:
    """Load one audio file as a mono, fixed-length floating-point signal.

    Audio longer than ``duration_seconds`` is truncated; shorter audio is padded
    with zeros. This gives every downstream model the same temporal input size.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    signal, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        duration=duration_seconds,
    )
    target_length = int(round(sample_rate * duration_seconds))
    signal = signal[:target_length]
    if signal.size < target_length:
        signal = np.pad(signal, (0, target_length - signal.size))

    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f"Audio contains non-finite samples: {audio_path}")
    if not np.any(np.abs(signal) > 1e-8):
        raise ValueError(f"Audio contains no usable signal: {audio_path}")

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    return signal


def _summary_statistics(features: np.ndarray) -> np.ndarray:
    """Convert a feature-by-frame matrix into a fixed-length vector."""
    return np.concatenate((features.mean(axis=1), features.std(axis=1))).astype(np.float32)


def extract_features(
    signal: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = 40,
    n_mels: int = 128,
) -> np.ndarray:
    """Extract a fixed-size acoustic feature vector from an audio signal.

    The vector contains summary statistics for MFCCs, delta MFCCs, chroma,
    mel-spectrogram energy, spectral contrast, zero-crossing rate, and RMS
    energy. Feature extraction is deterministic for a given signal and config.
    """
    if sample_rate <= 0 or n_mfcc <= 0 or n_mels <= 0:
        raise ValueError("sample_rate, n_mfcc, and n_mels must be positive")

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional mono array")
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("signal must contain finite samples")
    if not np.any(np.abs(signal) > 1e-8):
        raise ValueError("signal contains no usable samples")

    stft_magnitude = np.abs(librosa.stft(signal, n_fft=1024, hop_length=256))
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(
        S=stft_magnitude,
        sr=sample_rate,
        n_chroma=12,
    )
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
    )
    spectral_contrast = librosa.feature.spectral_contrast(
        S=stft_magnitude,
        sr=sample_rate,
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)

    blocks = (
        mfcc,
        delta_mfcc,
        chroma,
        librosa.power_to_db(mel, ref=np.max),
        spectral_contrast,
        zero_crossing_rate,
        rms,
    )
    vector = np.concatenate([_summary_statistics(block) for block in blocks])
    if not np.isfinite(vector).all():
        raise ValueError("feature extraction produced non-finite values")
    return vector.astype(np.float32)


def extract_features_from_file(
    path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> np.ndarray:
    """Load an audio file and return its fixed-size feature vector."""
    signal = load_audio(
        path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
    return extract_features(signal, sample_rate=sample_rate)


def augment_audio(
    signal: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise_scale: float = 0.005,
    semitones: float = 0.0,
    rate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply optional noise, pitch, and time-stretch augmentation.

    Augmentation is explicit and deterministic when a seeded Generator is
    supplied. The output has the same length as the input.
    """
    if sample_rate <= 0 or noise_scale < 0 or rate <= 0:
        raise ValueError("sample_rate must be positive; noise_scale cannot be negative; rate must be positive")

    generator = rng or np.random.default_rng()
    augmented = np.asarray(signal, dtype=np.float32).copy()
    if augmented.ndim != 1 or not np.isfinite(augmented).all():
        raise ValueError("signal must be a finite one-dimensional array")

    if noise_scale:
        augmented += generator.normal(0.0, noise_scale, size=augmented.shape).astype(np.float32)
    if semitones:
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=semitones)
    if rate != 1.0:
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    if augmented.size < signal.size:
        augmented = np.pad(augmented, (0, signal.size - augmented.size))
    augmented = augmented[: signal.size]
    peak = float(np.max(np.abs(augmented)))
    if peak > 1.0:
        augmented = augmented / peak
    return augmented.astype(np.float32)
