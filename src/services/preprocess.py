"""Stage 1 — Load PDF/ảnh → list[PIL.Image] (render 300 DPI + deskew tùy chọn)."""
from __future__ import annotations

from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from src.core.config import settings
from src.core.logging import logger

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_input(input_path: str, dpi: int) -> list[Image.Image]:
    path = Path(input_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _render_pdf(path, dpi)
    if ext in SUPPORTED_IMAGE_EXTS:
        return [Image.open(input_path).convert("RGB")]
    raise ValueError(f"Định dạng không hỗ trợ: {ext} (PDF hoặc {sorted(SUPPORTED_IMAGE_EXTS)})")


def _render_pdf(pdf_path: Path, dpi: int) -> list[Image.Image]:
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        logger.debug(f"  render trang {i + 1}: {pix.width}x{pix.height}")
    doc.close()
    logger.info(f"Load {len(images)} trang từ PDF (DPI={dpi})")
    return images


def deskew(img: Image.Image, threshold_degrees: float) -> Image.Image:
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < threshold_degrees:
        return img
    h, w = arr.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    logger.debug(f"  deskew {angle:.2f}°")
    return Image.fromarray(rotated)


def preprocess(input_path: str, dpi: int | None = None, do_deskew: bool | None = None) -> list[Image.Image]:
    dpi = dpi if dpi is not None else settings.render_dpi
    do_deskew = settings.do_deskew if do_deskew is None else do_deskew
    logger.info(f"Preprocess: {input_path}")
    images = load_input(input_path, dpi)
    if do_deskew:
        images = [deskew(im, settings.deskew_threshold_degrees) for im in images]
    return images
