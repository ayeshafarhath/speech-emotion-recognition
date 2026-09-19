"""CNN + bidirectional LSTM + attention baseline for audio features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalAttention(nn.Module):
    """Learn a weighted summary over the recurrent time dimension."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Return the weighted context vector and attention weights."""
        logits = self.score(sequence).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    """Baseline classifier for sequences of frame-level audio features.

    Input shape is ``(batch, time, features)``. The convolution captures local
    temporal patterns, the BiLSTM models longer context, and attention makes the
    final classification focus on informative frames.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        *,
        convolution_channels: int = 128,
        recurrent_hidden_size: int = 128,
        recurrent_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_features <= 0 or num_classes <= 1:
            raise ValueError("input_features must be positive and num_classes must exceed one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_features = input_features
        self.num_classes = num_classes
        self.convolution = nn.Sequential(
            nn.Conv1d(input_features, convolution_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(convolution_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(convolution_channels, convolution_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(convolution_channels),
            nn.GELU(),
        )
        self.recurrent = nn.LSTM(
            input_size=convolution_channels,
            hidden_size=recurrent_hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        recurrent_output_size = recurrent_hidden_size * 2
        self.attention = TemporalAttention(recurrent_output_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(recurrent_output_size),
            nn.Dropout(dropout),
            nn.Linear(recurrent_output_size, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return unnormalized class logits for a feature sequence."""
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, time, features)")
        if features.shape[-1] != self.input_features:
            raise ValueError(
                f"expected {self.input_features} features per frame, got {features.shape[-1]}"
            )
        encoded = self.convolution(features.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(encoded)
        context, _ = self.attention(sequence)
        return self.classifier(context)

    def predict_proba(self, features: Tensor) -> Tensor:
        """Return softmax probabilities without changing module training mode."""
        return torch.softmax(self(features), dim=-1)
