"""Chọn backend detector theo config / tham số."""
from __future__ import annotations

from typing import Optional

from src.core.config import settings
from src.detectors.base import RegionDetector


def get_detector(backend: Optional[str] = None) -> RegionDetector:
    backend = (backend or settings.detector_backend or "vllm").lower()
    if backend == "vllm":
        from src.detectors.vlm_vllm import VLLMDetector
        return VLLMDetector()
    if backend == "ollama":
        from src.detectors.vlm_ollama import OllamaDetector
        return OllamaDetector()
    raise ValueError(f"Backend không hỗ trợ: {backend!r} (chọn 'vllm' hoặc 'ollama')")
