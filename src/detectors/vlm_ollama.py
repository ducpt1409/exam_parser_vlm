"""Backend Ollama — gọi /api/chat với format=<json schema> (structured output).

Dễ dựng, nhưng model lớn có thể bị split sang CPU → chậm (xem PLAN §6.1).
"""
from __future__ import annotations

import json
from typing import Optional

import httpx
from PIL import Image

from src.core.config import settings
from src.core.logging import logger
from src.detectors.base import RegionDetector
from src.detectors.image_util import encode_jpeg_b64
from src.detectors.prompt import SYSTEM_PROMPT, build_user_prompt, gpage_to_regions
from src.detectors.schema import GPage
from src.schemas.region import Region


class OllamaDetector(RegionDetector):
    name = "ollama"

    def __init__(self):
        self._url = f"{settings.ollama_host}/api/chat"
        self._model = settings.ollama_model
        self._schema = GPage.model_json_schema()
        self._detect_answers = settings.detect_answers
        self._user_prompt = build_user_prompt(self._detect_answers)
        self.last_raw: Optional[str] = None

    def detect_page(self, image: Image.Image, page_index: int) -> list[Region]:
        page_w, page_h = image.size
        b64 = encode_jpeg_b64(image, settings.vlm_max_pixels)

        payload = {
            "model": self._model,
            "stream": False,
            "format": self._schema,
            "options": {
                "temperature": settings.vlm_temperature,
                "num_predict": settings.vlm_max_output_tokens,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt, "images": [b64]},
            ],
        }

        try:
            with httpx.Client(timeout=settings.vlm_timeout) as client:
                resp = client.post(self._url, json=payload)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ollama] trang {page_index}: lỗi gọi API — {e}")
            self.last_raw = None
            return []

        self.last_raw = content
        try:
            page = GPage.model_validate_json(content)
        except Exception:
            try:
                start, end = content.index("{"), content.rindex("}") + 1
                page = GPage.model_validate(json.loads(content[start:end]))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[ollama] trang {page_index}: parse JSON lỗi — {e}")
                return []

        regions = gpage_to_regions(page, page_index, page_w, page_h,
                                   drop_answers=not self._detect_answers)
        logger.info(f"[ollama] trang {page_index}: {len(regions)} vùng")
        return regions
