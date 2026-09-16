from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class PixAIResult:
    general: dict[str, float]
    characters: dict[str, float]
    ips: list[str]
    available: bool = True


class PixAITagger:
    """Lightweight direct ONNX integration for PixAI Tagger v0.9.

    This avoids torch/timm/dghs-imgutils on hosting. The public DeepGHS ONNX
    export uses 448x448 RGB input normalized to [-1, 1]. The implementation
    inspects the runtime input/output shapes so it tolerates NCHW/NHWC exports.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_path: str = "",
        tags_path: str = "",
        general_threshold: float = 0.30,
        character_threshold: float = 0.85,
    ):
        self.enabled = enabled
        self.model_path = Path(model_path) if model_path else Path("missing.onnx")
        self.tags_path = Path(tags_path) if tags_path else Path("missing.csv")
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.session = None
        self.tags: list[dict[str, Any]] = []
        self.input_size = 448

    @property
    def available(self) -> bool:
        return self.enabled and self.model_path.is_file() and self.tags_path.is_file()

    def _load(self) -> None:
        if self.session is not None:
            return
        if not self.available:
            raise FileNotFoundError("PixAI ONNX files are not ready yet")

        import onnxruntime as ort

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        with self.tags_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        def tag_id(row: dict[str, str]) -> int:
            try:
                return int(row.get("tag_id", "0"))
            except ValueError:
                return 0

        self.tags = sorted(rows, key=tag_id)

        shape = self.session.get_inputs()[0].shape
        ints = [x for x in shape if isinstance(x, int) and x > 3]
        if ints:
            self.input_size = int(ints[-1])

    def _preprocess(self, image_path: str) -> np.ndarray:
        assert self.session is not None
        size = self.input_size
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5

        input_shape = self.session.get_inputs()[0].shape
        # NCHW is the usual PixAI ONNX layout, but handle NHWC defensively.
        if len(input_shape) == 4 and input_shape[1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _predict_sync(self, image_path: str) -> PixAIResult:
        if not self.enabled or not self.available:
            return PixAIResult({}, {}, [], False)
        self._load()
        assert self.session is not None

        tensor = self._preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})

        scores = None
        tag_count = len(self.tags)
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim >= 2 and arr.shape[-1] == tag_count:
                scores = arr.reshape(-1, tag_count)[0]
                break
        if scores is None:
            raise RuntimeError("PixAI ONNX output does not match selected_tags.csv")

        scores = scores.astype(np.float32)
        if scores.min(initial=0.0) < 0.0 or scores.max(initial=1.0) > 1.0:
            scores = self._sigmoid(scores)

        general: dict[str, float] = {}
        characters: dict[str, float] = {}
        for row, score in zip(self.tags, scores, strict=False):
            try:
                category = int(row.get("category", "0"))
            except ValueError:
                category = 0
            name = (row.get("name") or "").strip()
            if not name:
                continue
            value = float(score)
            if category == 4:
                if value >= self.character_threshold:
                    characters[name] = value
            elif value >= self.general_threshold:
                general[name] = value

        general = dict(sorted(general.items(), key=lambda x: x[1], reverse=True)[:48])
        characters = dict(sorted(characters.items(), key=lambda x: x[1], reverse=True)[:24])
        # The public ONNX tag CSV contains general + character categories only.
        # Qwen receives the character/general hints and resolves the IP/source.
        return PixAIResult(general, characters, [], True)

    async def predict(self, image_path: str) -> PixAIResult:
        return await asyncio.to_thread(self._predict_sync, image_path)
