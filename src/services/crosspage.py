"""Stage 5 — Cross-page merge: gộp câu vắt trang (PLAN §5.3).

Hai ca:
1. VLM đánh cùng số câu ở cuối trang N và đầu trang N+1 → gộp 2 mảnh thành 1 câu.
2. Câu cuối trang N có cờ continues_to_next (hoặc câu đầu trang N+1 có continues_from_prev),
   nhưng đầu trang N+1 là CÂU MỚI → phần nội dung phía trên câu mới đó là phần đuôi của câu N
   → thêm 1 mảnh (leading region) trang N+1 vào câu N.
"""
from __future__ import annotations

from src.core.logging import logger
from src.schemas.geometry import BBox
from src.services.assembly import AssemblyResult, Part


def _last_on_page(result: AssemblyResult, page: int):
    cands = [l for l in result.q_layouts.values() if l.page_index == page]
    return max(cands, key=lambda l: l.y_top) if cands else None


def _first_on_page(result: AssemblyResult, page: int):
    cands = [l for l in result.q_layouts.values() if l.page_index == page]
    return min(cands, key=lambda l: l.y_top) if cands else None


def merge_cross_page(result: AssemblyResult, page_sizes: list[tuple[int, int]]) -> None:
    """Sửa in-place result (q_layouts + document.questions)."""
    doc = result.document
    q_by_id = {q.id: q for q in doc.questions}
    n_pages = doc.n_pages
    merged_count = 0

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
            merged_count += 1
            continue

        # Ca 2: cờ continuation → thêm leading region trang p+1 vào last
        if last.continues_to_next or first.continues_from_prev:
            xl, xr = first.col_bounds
            lead_box: BBox = (xl, 0.0, xr, max(1.0, first.y_top))
            last.full_parts.append(Part(p + 1, lead_box))
            last.stem_parts.append(Part(p + 1, lead_box))
            lq = q_by_id.get(last.q_id)
            if lq:
                lq.page_indices = sorted(set(lq.page_indices + [p + 1]))
                lq.needs_review = True
                lq.notes.append(f"đuôi câu vắt sang trang {p+1}")
            merged_count += 1

    if merged_count:
        logger.info(f"Cross-page: gộp/nối {merged_count} câu vắt trang")
    else:
        logger.info("Cross-page: không có câu vắt trang")
