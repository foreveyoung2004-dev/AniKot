from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class CamieTagger:
    """Локальный ONNX-инференс Camie Tagger v2.

    Модель загружается лениво, чтобы бот мог стартовать даже без скачанных весов.
    """

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.session = None
        self.metadata: dict[str, Any] | None = None
        self.idx_to_tag: dict[str, str] = {}
        self.tag_to_category: dict[str, str] = {}
        self.image_size = 512

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.metadata_path.exists()

    def _load(self) -> None:
        if self.session is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                "Camie Tagger v2 ещё не готов. На Bothost модель скачивается автоматически при первом запуске."
            )

        import onnxruntime as ort

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        dataset_info = self.metadata["dataset_info"]
        tag_mapping = dataset_info["tag_mapping"]
        self.idx_to_tag = tag_mapping["idx_to_tag"]
        self.tag_to_category = tag_mapping["tag_to_category"]
        self.image_size = int(self.metadata.get("model_info", {}).get("img_size", 512))

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

    def _preprocess(self, image_path: str) -> np.ndarray:
        size = self.image_size
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            ratio = width / max(height, 1)
            if ratio > 1:
                new_w = size
                new_h = max(1, int(size / ratio))
            else:
                new_h = size
                new_w = max(1, int(size * ratio))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (124, 116, 104))
            canvas.paste(image, ((size - new_w) // 2, (size - new_h) // 2))

            arr = np.asarray(canvas).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0).astype(np.float32)

    def predict(self, image_path: str, threshold: float = 0.5, top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
        self._load()
        assert self.session is not None
        tensor = self._preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]
        probs = 1.0 / (1.0 + np.exp(-logits))

        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        indices = np.where(probs[0] >= threshold)[0]
        for idx in indices:
            tag = self.idx_to_tag.get(str(int(idx)), f"unknown-{idx}")
            category = self.tag_to_category.get(tag, "general")
            grouped[category].append((tag, float(probs[0, idx])))

        for category in list(grouped):
            grouped[category] = sorted(grouped[category], key=lambda x: x[1], reverse=True)[:top_k]
        return dict(grouped)

    def best_sources(self, image_path: str, threshold: float, top_k: int = 5) -> list[tuple[str, float]]:
        grouped = self.predict(image_path, threshold=min(threshold, 0.35), top_k=max(top_k, 10))
        candidates = grouped.get("copyright", [])
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
