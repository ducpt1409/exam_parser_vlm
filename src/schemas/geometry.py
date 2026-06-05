"""Kiểu bbox + helper hình học. bbox = (x1, y1, x2, y2) pixel trên 1 trang."""
from __future__ import annotations

from typing import Iterable

BBox = tuple[float, float, float, float]


def union_bbox(boxes: Iterable[BBox]) -> BBox | None:
    """Hộp bao của nhiều bbox."""
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def clamp_bbox(b: BBox, width: int, height: int) -> BBox:
    """Ép bbox vào trong biên ảnh, đảm bảo x1<x2, y1<y2."""
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def pad_bbox(b: BBox, pad: float, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = b
    return clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def bbox_center(b: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_valid(b: BBox, min_size: float = 2.0) -> bool:
    x1, y1, x2, y2 = b
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / (bbox_area(a) + bbox_area(b) - inter)


def y_overlap_ratio(a: BBox, b: BBox) -> float:
    """Tỉ lệ chồng lấn theo trục y so với chiều cao nhỏ hơn."""
    iy1, iy2 = max(a[1], b[1]), min(a[3], b[3])
    inter = max(0.0, iy2 - iy1)
    h = min(a[3] - a[1], b[3] - b[1])
    return inter / h if h > 0 else 0.0
