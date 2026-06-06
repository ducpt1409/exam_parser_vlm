"""Orchestrator — chạy toàn bộ stage [1]→[7] cho 1 file đề.

PDF nhiều trang hoặc 1 ảnh → output/{exam_id}/{exam.json, crops/, overlay/, vlm_raw/, summary.txt}.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.core.logging import logger
from src.detectors.base import RegionDetector
from src.detectors.factory import get_detector
from src.services import assembly as asm
from src.services.box_snap import PageInk
from src.services.cropper import crop_all
from src.services.crosspage import merge_cross_page
from src.services.preprocess import preprocess


def _write_summary(out_dir: Path, result: asm.AssemblyResult) -> None:
    doc = result.document
    n_review = sum(1 for q in doc.questions if q.needs_review)
    n_cross = sum(1 for q in doc.questions if len(q.page_indices) > 1)
    lines = [
        f"exam_id      : {doc.exam_id}",
        f"source       : {doc.source_file}",
        f"backend      : {doc.detector_backend}",
        f"pages        : {doc.n_pages}",
        f"questions    : {len(doc.questions)}",
        f"sections     : {len(doc.sections)}",
        f"groups       : {len(doc.groups)}",
        f"header       : {'có' if doc.header else 'không'}",
        f"needs_review : {n_review}",
        f"cross-page   : {n_cross}",
        "",
        "Câu (number | type | #answers | pages | review):",
    ]
    for q in doc.questions:
        flag = "⚠" if q.needs_review else " "
        lines.append(
            f"  {flag} Câu {q.number:>3} | {q.type.value:<12} | "
            f"{len(q.answers)} đáp án | trang {q.page_indices} "
            f"| sec={q.section_id} grp={q.group_id}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_exam(
    input_path: str,
    out_root: Optional[str] = None,
    backend: Optional[str] = None,
    snap: Optional[bool] = None,
    detector: Optional[RegionDetector] = None,
) -> asm.AssemblyResult:
    backend = backend or settings.detector_backend
    snap = settings.snap_enabled if snap is None else snap
    out_root = out_root or settings.output_dir

    src = Path(input_path)
    exam_id = uuid.uuid4().hex[:8]
    out_dir = Path(out_root) / exam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Parse {src.name} → {out_dir} (backend={backend}, snap={snap}) ===")

    # [1] Preprocess
    images = preprocess(str(src))
    page_sizes = [im.size for im in images]

    # [2] Region detection (VLM)
    det = detector or get_detector(backend)
    raw_dir = out_dir / "vlm_raw"
    regions_per_page: list[list] = []
    for i, img in enumerate(images):
        regions = det.detect_page(img, i)
        regions_per_page.append(regions)
        if settings.save_vlm_raw and getattr(det, "last_raw", None):
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"page_{i:02d}.json").write_text(det.last_raw, encoding="utf-8")

    # [3] Box-snap (precompute ink masks; áp dụng trong cropper)
    page_inks = [PageInk(im) for im in images]

    # [4] Assembly → DOM
    result = asm.build_document(regions_per_page, page_sizes, exam_id, src.name, backend)

    # [5] Cross-page merge (nối ảnh câu vắt trang)
    merge_cross_page(result, page_sizes, regions_per_page)

    # [7] Crop + overlay (snap áp dụng tại đây)
    crop_all(result, images, page_inks, out_dir, snap, settings.snap_pad)

    # Stats + ghi file
    doc = result.document
    doc.stats = {
        "n_questions": len(doc.questions),
        "n_sections": len(doc.sections),
        "n_groups": len(doc.groups),
        "n_cross_page": sum(1 for q in doc.questions if len(q.page_indices) > 1),
        "n_needs_review": sum(1 for q in doc.questions if q.needs_review),
    }
    (out_dir / "exam.json").write_text(
        doc.model_dump_json(indent=2), encoding="utf-8"
    )
    _write_summary(out_dir, result)
    logger.info(f"=== Xong: {len(doc.questions)} câu → {out_dir / 'exam.json'} ===")
    return result
