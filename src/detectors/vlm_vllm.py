"""Backend vLLM — gọi qua OpenAI-compatible API (/v1/chat/completions) + guided_json.

Chạy 100% GPU (xem PLAN §6.1). Yêu cầu `vllm serve ...` đang chạy ở VLM_BASE_URL.
"""
from __future__ import annotations

import json
from typing import Optional

from PIL import Image

from src.core.config import settings
from src.core.logging import logger
from src.detectors.base import RegionDetector
from src.detectors.image_util import data_url
from src.detectors.prompt import SYSTEM_PROMPT, USER_PROMPT, gpage_to_regions
from src.detectors.schema import GPage
from src.schemas.region import Region


class VLLMDetector(RegionDetector):
    name = "vllm"

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("Cần `pip install openai` để dùng backend vllm") from e

        self._client = OpenAI(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            timeout=settings.vlm_timeout,
        )
        self._model = settings.vlm_model
        self._schema = GPage.model_json_schema()
        # Lưu raw để debug (pipeline đọc ra ghi file)
        self.last_raw: Optional[str] = None

    def detect_page(self, image: Image.Image, page_index: int) -> list[Region]:
        page_w, page_h = image.size
        url = data_url(image, settings.vlm_max_pixels)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=settings.vlm_temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ],
                extra_body={"guided_json": self._schema},
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — fail-safe theo interface
            logger.error(f"[vllm] trang {page_index}: lỗi gọi API — {e}")
            self.last_raw = None
            return []

        self.last_raw = content
        try:
            page = GPage.model_validate_json(content)
        except Exception:
            # thử bóc JSON object đầu tiên nếu model lỡ kèm text
            try:
                start, end = content.index("{"), content.rindex("}") + 1
                page = GPage.model_validate(json.loads(content[start:end]))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[vllm] trang {page_index}: parse JSON lỗi — {e}")
                return []

        regions = gpage_to_regions(page, page_index, page_w, page_h)
        logger.info(f"[vllm] trang {page_index}: {len(regions)} vùng")
        return regions
