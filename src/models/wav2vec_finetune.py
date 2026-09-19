"""Optional Wav2Vec2 sequence-classification model factory.

The pretrained checkpoint is downloaded by Hugging Face Transformers at runtime;
model weights are intentionally not stored in the repository. Fine-tuning should
be run with a labeled dataset and an explicitly documented experiment config.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

DEFAULT_CHECKPOINT = "facebook/wav2vec2-base"


class Wav2Vec2EmotionClassifier(nn.Module):
    """Wav2Vec2 encoder with a configurable emotion-classification head."""

    def __init__(
        self,
        num_classes: int,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        freeze_feature_encoder: bool = True,
        id2label: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")

        labels = id2label or {index: f"class_{index}" for index in range(num_classes)}
        label2id = {label: index for index, label in labels.items()}
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            id2label=labels,
            label2id=label2id,
            problem_type="single_label_classification",
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )
        if freeze_feature_encoder and hasattr(self.model, "freeze_feature_encoder"):
            self.model.freeze_feature_encoder()

    def forward(
        self,
        input_values: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ):
        """Return the Transformers sequence-classification output."""
        return self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.inference_mode()
    def predict_proba(
        self,
        input_values: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return class probabilities for a batch of raw waveforms."""
        self.eval()
        outputs = self.forward(input_values, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=-1)


def load_processor(checkpoint: str = DEFAULT_CHECKPOINT) -> Wav2Vec2FeatureExtractor:
    """Load the matching raw-waveform processor from Hugging Face."""
    return Wav2Vec2FeatureExtractor.from_pretrained(
        checkpoint,
        sampling_rate=16_000,
        return_attention_mask=True,
    )


def build_model(
    labels: Sequence[str],
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    freeze_feature_encoder: bool = True,
) -> Wav2Vec2EmotionClassifier:
    """Build a classifier with stable integer-to-label mappings."""
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two unique emotion names")
    id2label = {index: label for index, label in enumerate(labels)}
    return Wav2Vec2EmotionClassifier(
        len(labels),
        checkpoint=checkpoint,
        freeze_feature_encoder=freeze_feature_encoder,
        id2label=id2label,
    )
