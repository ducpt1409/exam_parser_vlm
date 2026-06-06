"""Stage 5 — Cross-page merge: gộp câu/đáp án vắt trang rồi nối ảnh (PLAN §5.3).

Ba ca:
1. VLM đánh cùng số câu ở cuối trang N và đầu trang N+1 → gộp 2 mảnh thành 1 câu.
2. Cờ continues_to_next/continues_from_prev → thêm mảnh đầu trang N+1 vào câu cuối trang N.
3. Heuristic (khi 4B không set cờ): đầu trang N+1 có NỘI DUNG phía trên câu hỏi đầu tiên, không phải
   header/section/group → đó là phần ĐUÔI của câu cuối trang N → nối vào.
Ảnh nhiều mảnh sẽ được cropper ghép DỌC (vertical stitch) như paddle.
"""
from __future__ import annotations

from src.core.logging import logger
from src.schemas.geometry import BBox
from src.schemas.region import Region, RegionClass
from src.services.assembly import AssemblyResult, Part

_BLOCKER_CLASSES = (
    RegionClass.EXAM_HEADER,
    RegionClass.SECTION_HEADER,
    RegionClass.GROUP_INSTRUCTION,
    RegionClass.PASSAGE,
)


def _last_on_page(result: AssemblyResult, page: int):
    cands = [l for l in result.q_layouts.values() if l.page_index == page]
    return max(cands, key=lambda l: l.y_top) if cands else None


def _first_on_page(result: AssemblyResult, page: int):
    cands = [l for l in result.q_layouts.values() if l.page_index == page]
    return min(cands, key=lambda l: l.y_top) if cands else None


def _has_blocker_above(regions: list[Region], y: float) -> bool:
    """Có header/section/group/passage nằm phía trên y không (→ không phải đuôi câu)."""
    return any(r.cls in _BLOCKER_CLASSES and r.bbox[1] < y for r in regions)


def merge_cross_page(
    result: AssemblyResult,
    page_sizes: list[tuple[int, int]],
    regions_per_page: list[list[Region]] | None = None,
) -> None:
    doc = result.document
    q_by_id = {q.id: q for q in doc.questions}
    n_pages = doc.n_pages
    merged = 0

    for p in range(n_pages - 1):
        last = _last_on_page(result, p)
        first = _first_on_page(result, p + 1)
        if last is None or first is None:
            continue

        # Ca 1: cùng số câu → gộp first vào last, bỏ first
        if first.number == last.number and first.q_id != last.q_id:
            last.full_parts += first.full_parts
            last.stem_parts += first.stem_parts
            last.answer_parts += first.answer_parts
            last.continues_to_next = first.continues_to_next
            lq, fq = q_by_id.get(last.q_id), q_by_id.get(first.q_id)
            if lq and fq:
                lq.answers += fq.answers
                lq.page_indices = sorted(set(lq.page_indices + fq.page_indices))
                lq.needs_review = True
                lq.notes.append(f"merge mảnh vắt trang {p}->{p+1}")
                doc.questions = [q for q in doc.questions if q.id != fq.id]
            result.q_layouts.pop(first.q_id, None)
            merged += 1
            continue

        # Ca 2 + 3: đuôi câu cuối trang N nằm ở đầu trang N+1 (phía trên câu đầu N+1)
        page_h = page_sizes[p + 1][1] if p + 1 < len(page_sizes) else 0
        flag_signal = last.continues_to_next or first.continues_from_prev
        heuristic_signal = False
        if regions_per_page is not None and page_h:
            regs = regions_per_page[p + 1]
            # đầu trang sau có khoảng trống lớn phía trên câu đầu + không có header/section chen vào
            if first.y_top > 0.12 * page_h and not _has_blocker_above(regs, first.y_top):
                heuristic_signal = True

        if flag_signal or heuristic_signal:
            xl, xr = first.col_bounds
            lead_box: BBox = (xl, 0.0, xr, max(1.0, first.y_top))
            last.full_parts.append(Part(p + 1, lead_box))
            last.stem_parts.append(Part(p + 1, lead_box))
            lq = q_by_id.get(last.q_id)
            if lq:
                lq.page_indices = sorted(set(lq.page_indices + [p + 1]))
                lq.needs_review = True
                why = "cờ VLM" if flag_signal else "heuristic khoảng trống đầu trang"
                lq.notes.append(f"đuôi câu vắt sang trang {p+1} ({why})")
            merged += 1

    logger.info(f"Cross-page: nối {merged} câu vắt trang" if merged else "Cross-page: không có câu vắt trang")
