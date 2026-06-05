"""Interface RegionDetector — mọi backend (VLM, sau này YOLO) tuân theo.

Pipeline chỉ phụ thuộc interface này → đổi backend không sửa pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from src.schemas.region import Region


class RegionDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect_page(self, image: Image.Image, page_index: int) -> list[Region]:
        """Trả list[Region] (bbox PIXEL) cho 1 trang. KHÔNG raise — lỗi thì trả []."""
        raise NotImplementedError

    def detect(self, images: list[Image.Image]) -> list[list[Region]]:
        """Mặc định: chạy tuần tự từng trang."""
        return [self.detect_page(img, i) for i, img in enumerate(images)]
