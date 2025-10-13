"""Utilities for loading the trained PyTorch model and running inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from .config import DEFAULT_RUNTIME_CONFIG, ModelRuntimeConfig
from .models import LegacySimpleCNN, SimpleCNN


@dataclass(frozen=True)
class PredictionResult:
    """Container for model outputs."""

    label: str
    confidence: float
    probabilities: Dict[str, float]


class ModelService:
    """Load the trained model once and expose simple prediction helpers."""

    def __init__(self, checkpoint_path: Path | str, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(chosen_device)

        self.runtime_config: ModelRuntimeConfig = DEFAULT_RUNTIME_CONFIG
        self.model_version: str = DEFAULT_RUNTIME_CONFIG.model_version
        self.arch: str = "simple_cnn"
        self.classes: Sequence[str] = DEFAULT_RUNTIME_CONFIG.classes
        self.model: nn.Module | None = None
        self.transform: transforms.Compose | None = None
        self.max_batch_size: int = 16

        self._load_checkpoint()

    # ------------------------------------------------------------------ Checkpoint & transforms
    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        runtime_config = ModelRuntimeConfig(
            classes=tuple(checkpoint.get("classes", DEFAULT_RUNTIME_CONFIG.classes)),
            image_size=tuple(checkpoint.get("img_size", DEFAULT_RUNTIME_CONFIG.image_size)),
            grayscale=bool(checkpoint.get("grayscale", DEFAULT_RUNTIME_CONFIG.grayscale)),
            model_version=str(checkpoint.get("model_version", DEFAULT_RUNTIME_CONFIG.model_version)),
        )
        self.runtime_config = runtime_config
        self.classes = runtime_config.classes
        self.model_version = runtime_config.model_version
        self.max_batch_size = int(checkpoint.get("max_batch_size", self.max_batch_size))

        self.arch = str(checkpoint.get("arch", "simple_cnn"))

        input_channels = int(
            checkpoint.get(
                "input_channels",
                checkpoint.get("in_channels", runtime_config.in_channels),
            )
        )
        state_dict = checkpoint["model_state"]

        if self.arch == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, len(self.classes)),
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            model = SimpleCNN(num_classes=len(self.classes), in_channels=input_channels)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                print(
                    "[INFO] Falling back to legacy SimpleCNN (no BatchNorm) to load checkpoint."
                )
                model = LegacySimpleCNN(num_classes=len(self.classes), in_channels=input_channels)
                model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model = model
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        normalization = self.runtime_config.normalization
        normalize = transforms.Normalize(
            mean=list(normalization["mean"]), std=list(normalization["std"])
        )
        height, width = self.runtime_config.image_size
        pipeline = [transforms.Resize((height, width))]
        if self.runtime_config.grayscale:
            pipeline.append(transforms.Grayscale(num_output_channels=1))
        pipeline.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(pipeline)

    # ------------------------------------------------------------------ Inference helpers
    def _ensure_ready(self) -> None:
        if self.model is None or self.transform is None:
            raise RuntimeError("Model is not initialised.")

    def _prepare_tensor(self, image_path: Path | str) -> torch.Tensor:
        self._ensure_ready()
        path = Path(image_path)
        with Image.open(path) as img:
            img = img.convert("L" if self.runtime_config.grayscale else "RGB")
            tensor = self.transform(img)  # type: ignore[arg-type]
        return tensor

    def _infer_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        assert self.model is not None  # for mypy/static hints
        with torch.no_grad():
            logits = self.model(batch.to(self.device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1).cpu()
        return probabilities

    def _build_prediction(self, probability_vector: torch.Tensor) -> PredictionResult:
        probs_dict = {
            cls: float(probability_vector[idx]) for idx, cls in enumerate(self.classes)
        }
        top_idx = int(probability_vector.argmax().item())
        top_label = self.classes[top_idx]
        top_confidence = probs_dict[top_label]
        return PredictionResult(label=top_label, confidence=top_confidence, probabilities=probs_dict)

    # ------------------------------------------------------------------ Public API
    def predict(self, image_path: Path | str) -> PredictionResult:
        tensor = self._prepare_tensor(image_path).unsqueeze(0)
        probabilities = self._infer_tensor_batch(tensor)[0]
        return self._build_prediction(probabilities)

    def predict_batch(
        self, image_paths: Iterable[Path | str], batch_size: int | None = None
    ) -> List[PredictionResult]:
        batch_limit = batch_size or self.max_batch_size
        batch_limit = max(1, batch_limit)

        tensors: List[torch.Tensor] = []
        results: List[PredictionResult] = []

        def flush() -> None:
            nonlocal tensors
            if not tensors:
                return
            stacked = torch.stack(tensors, dim=0)
            probs = self._infer_tensor_batch(stacked)
            results.extend(self._build_prediction(row) for row in probs)
            tensors = []

        for path in image_paths:
            try:
                tensors.append(self._prepare_tensor(path))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] Failed to run inference on {path}: {exc}")
                continue
            if len(tensors) >= batch_limit:
                flush()

        flush()
        return results
