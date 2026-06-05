"""Stage 7 — Cropper: crop câu/đáp án/group/header từ ảnh trang.

- Mỗi Part được box-snap (nếu bật) ngay trước khi crop → khít ink, bao trọn hình.
- Nhiều Part (vắt trang) → ghép dọc (vertical stitch).
- Vẽ overlay bbox màu lên ảnh gốc để soát mắt thường.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from src.core.logging import logger
from src.schemas.exam import CropImage
from src.schemas.geometry import BBox, clamp_bbox
from src.services.assembly import AssemblyResult, Part
from src.services.box_snap import PageInk

# Màu overlay
C_QUESTION = (30, 90, 220)    # 🔵 full câu
C_STEM = (150, 30, 200)       # 🟣 stem
C_ANSWER = (30, 160, 60)      # 🟢 đáp án
C_GROUP = (220, 30, 30)       # 🔴 group
C_HEADER = (230, 140, 0)      # 🟠 header
LINE_W = 3


def _snap_box(part: Part, page_inks: list[PageInk], snap: bool, pad: int) -> BBox:
    if snap and 0 <= part.page_index < len(page_inks):
        return page_inks[part.page_index].snap(part.bbox, pad)
    pi = page_inks[part.page_index]
    return clamp_bbox(part.bbox, pi.width, pi.height)


def _crop_one(img: Image.Image, b: BBox) -> Optional[Image.Image]:
    x1, y1, x2, y2 = [int(v) for v in b]
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img.crop((x1, y1, x2, y2))


def _stitch_vertical(crops: list[Image.Image], gap: int = 8) -> Image.Image:
    max_w = max(c.width for c in crops)
    total_h = sum(c.height for c in crops) + gap * (len(crops) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for c in crops:
        canvas.paste(c, (0, y))
        y += c.height + gap
    return canvas


def _make_crop(
    parts: list[Part], images: list[Image.Image], page_inks: list[PageInk],
    out_dir: Path, rel_name: str, snap: bool, pad: int,
    overlay_acc: dict[int, list[tuple[BBox, tuple]]], color: tuple,
) -> Optional[CropImage]:
    """Crop (snap) các part → lưu PNG → CropImage. Ghi nhận bbox vào overlay_acc."""
    snapped: list[tuple[int, BBox]] = []
    crops: list[Image.Image] = []
    for part in parts:
        if part.page_index >= len(images):
            continue
        b = _snap_box(part, page_inks, snap, pad)
        c = _crop_one(images[part.page_index], b)
        if c is None:
            continue
        crops.append(c)
        snapped.append((part.page_index, b))
        overlay_acc.setdefault(part.page_index, []).append((b, color))

    if not crops:
        return None
    out = crops[0] if len(crops) == 1 else _stitch_vertical(crops)

    path = out_dir / "crops" / rel_name
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(path), "PNG")

    return CropImage(
        path=f"crops/{rel_name}",
        bbox=snapped[0][1] if snapped else None,
        page_span=sorted({p for p, _ in snapped}),
        width=out.width, height=out.height, size_bytes=path.stat().st_size,
    )


def crop_all(
    result: AssemblyResult, images: list[Image.Image], page_inks: list[PageInk],
    out_dir: Path, snap: bool, pad: int,
) -> None:
    doc = result.document
    q_by_id = {q.id: q for q in doc.questions}
    overlay_acc: dict[int, list[tuple[BBox, tuple]]] = {}

    # Câu
    for q_id, lay in result.q_layouts.items():
        q = q_by_id.get(q_id)
        if q is None:
            continue
        n = lay.number
        q.full_image = _make_crop(lay.full_parts, images, page_inks, out_dir,
                                  f"q{n}_full.png", snap, pad, overlay_acc, C_QUESTION)
        q.stem_image = _make_crop(lay.stem_parts, images, page_inks, out_dir,
                                  f"q{n}_stem.png", snap, pad, overlay_acc, C_STEM)
        for ai, (label, parts) in enumerate(lay.answer_parts):
            ci = _make_crop(parts, images, page_inks, out_dir,
                            f"q{n}_{label}.png", snap, pad, overlay_acc, C_ANSWER)
            if ai < len(q.answers):
                q.answers[ai].image = ci

    # Groups
    for g in doc.groups:
        glay = result.group_layouts.get(g.id)
        if glay:
            g.image = _make_crop(glay.parts, images, page_inks, out_dir,
                                 f"{g.id}.png", snap, pad, overlay_acc, C_GROUP)

    # Header
    if result.header_layout and doc.header:
        doc.header.image = _make_crop(result.header_layout.parts, images, page_inks, out_dir,
                                      "header.png", snap, pad, overlay_acc, C_HEADER)

    _draw_overlay(images, overlay_acc, out_dir)

    n_crops = len(list((out_dir / "crops").glob("*.png"))) if (out_dir / "crops").exists() else 0
    logger.info(f"Cropper: {n_crops} ảnh tại {out_dir / 'crops'}")


def _draw_overlay(images: list[Image.Image], acc: dict[int, list[tuple[BBox, tuple]]], out_dir: Path) -> None:
    od = out_dir / "overlay"
    od.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)
        for b, color in acc.get(i, []):
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_W)
        canvas.save(str(od / f"page_{i:02d}.png"), "PNG")
    logger.info(f"Overlay: {len(images)} trang tại {od}")
