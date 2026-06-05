"""Tiện ích ảnh cho VLM: resize theo max_pixels + encode base64.

Toạ độ VLM chuẩn hoá 0-1000 nên resize KHÔNG ảnh hưởng việc map về pixel trang gốc.
"""
from __future__ import annotations

import base64
import io
import math

from PIL import Image


def resize_for_vlm(img: Image.Image, max_pixels: int) -> Image.Image:
    """Thu nhỏ nếu tổng pixel vượt max_pixels (giữ tỉ lệ). Không phóng to."""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / float(w * h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def encode_jpeg_b64(img: Image.Image, max_pixels: int, quality: int = 90) -> str:
    """Trả base64 (không kèm prefix)."""
    img = resize_for_vlm(img, max_pixels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def data_url(img: Image.Image, max_pixels: int, quality: int = 90) -> str:
    """Trả data URL dùng cho OpenAI-compatible image_url."""
    b64 = encode_jpeg_b64(img, max_pixels, quality)
    return f"data:image/jpeg;base64,{b64}"
