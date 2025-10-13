"""Shared configuration and constants for the Alzheimer MRI toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

DEFAULT_CLASSES: Tuple[str, ...] = (
    "Moderate Impairment",
    "Mild Impairment",
    "Very Mild Impairment",
    "No Impairment",
)

CLASS_DISPLAY: Dict[str, str] = {
    "Moderate Impairment": "Orta Alzheimer Bulgusu",
    "Mild Impairment": "Hafif Alzheimer Bulgusu",
    "Very Mild Impairment": "Çok Hafif Alzheimer Bulgusu",
    "No Impairment": "Sağlıklı",
}

SEVERITY_ORDER: Dict[str, int] = {
    "Moderate Impairment": 3,
    "Mild Impairment": 2,
    "Very Mild Impairment": 1,
    "No Impairment": 0,
}

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_IN_CHANNELS_RGB = 3
DEFAULT_IN_CHANNELS_GRAY = 1
DEFAULT_MODEL_VERSION = "SimpleCNN_v1"
DEFAULT_CHECKPOINT = Path("alzheimer_cnn_torch.pt")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

NORMALIZATION_RGB: Dict[str, Sequence[float]] = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
NORMALIZATION_GRAY: Dict[str, Sequence[float]] = {
    "mean": (0.5,),
    "std": (0.25,),
}


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Settings shared by training and inference pipelines."""

    classes: Tuple[str, ...] = DEFAULT_CLASSES
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    grayscale: bool = False
    model_version: str = DEFAULT_MODEL_VERSION

    @property
    def in_channels(self) -> int:
        return DEFAULT_IN_CHANNELS_GRAY if self.grayscale else DEFAULT_IN_CHANNELS_RGB

    @property
    def normalization(self) -> Dict[str, Sequence[float]]:
        return NORMALIZATION_GRAY if self.grayscale else NORMALIZATION_RGB


DEFAULT_RUNTIME_CONFIG = ModelRuntimeConfig()
