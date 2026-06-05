"""Stage 4 — Assembly: list[Region] (đã snap) → ExamDocument (DOM) + Layout để crop.

Logic (PLAN §5.2):
- Sort câu theo cột rồi y → band dọc [đỉnh câu N, đỉnh câu N+1) trong cùng cột.
- answer_option gắn vào câu chứa nó (cùng cột + trong band).
- full box câu = [mép cột, đỉnh câu, đáy band] (box-snap sẽ tighten + bao trọn hình).
- section_header → chia section; group_instruction/passage → group (gán câu theo covers/vị trí).
KHÔNG xử lý vắt trang ở đây — để crosspage.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.logging import logger
from src.schemas.exam import (
    Answer,
    ExamDocument,
    ExamHeader,
    Group,
    Question,
    QuestionType,
    Section,
)
from src.schemas.geometry import BBox, bbox_center
from src.schemas.region import Region, RegionClass


# ============================================================
# Layout structures (truyền sang cropper)
# ============================================================
@dataclass
class Part:
    page_index: int
    bbox: BBox


@dataclass
class QLayout:
    q_id: str
    number: int
    page_index: int
    y_top: float
    band_bottom: float
    col_bounds: tuple[float, float]
    full_parts: list[Part] = field(default_factory=list)
    stem_parts: list[Part] = field(default_factory=list)
    answer_parts: list[tuple[str, list[Part]]] = field(default_factory=list)
    continues_from_prev: bool = False
    continues_to_next: bool = False


@dataclass
class GroupLayout:
    g_id: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class HeaderLayout:
    parts: list[Part] = field(default_factory=list)


@dataclass
class AssemblyResult:
    document: ExamDocument
    q_layouts: dict[str, QLayout]
    group_layouts: dict[str, GroupLayout]
    header_layout: Optional[HeaderLayout]


# ============================================================
# Helpers
# ============================================================
def _detect_columns(qregions: list[Region], page_w: int) -> Optional[float]:
    """Trả x-split nếu trang 2 cột, None nếu 1 cột."""
    if len(qregions) < 2:
        return None
    left = [r for r in qregions if bbox_center(r.bbox)[0] < 0.5 * page_w]
    right = [r for r in qregions if bbox_center(r.bbox)[0] >= 0.5 * page_w]
    if not left or not right:
        return None
    max_left_x2 = max(r.bbox[2] for r in left)
    min_right_x1 = min(r.bbox[0] for r in right)
    if max_left_x2 < min_right_x1:           # có khe trắng giữa 2 cột
        return (max_left_x2 + min_right_x1) / 2.0
    return None


def _col_index(b: BBox, split: Optional[float]) -> int:
    if split is None:
        return 0
    return 0 if bbox_center(b)[0] < split else 1


def _col_bounds(col_idx: int, split: Optional[float], page_w: int) -> tuple[float, float]:
    if split is None:
        return (0.0, float(page_w))
    return (0.0, split) if col_idx == 0 else (split, float(page_w))


def _infer_type(qtype: Optional[str], n_answers: int) -> tuple[QuestionType, bool]:
    """Trả (type, needs_review)."""
    t = QuestionType.from_str(qtype)
    if t != QuestionType.UNKNOWN:
        # MCQ nhưng thiếu/đủ đáp án
        if t in (QuestionType.MCQ_SINGLE, QuestionType.MCQ_MULTI) and n_answers not in (4,):
            return t, True
        return t, False
    if n_answers >= 3:
        return QuestionType.MCQ_SINGLE, (n_answers != 4)
    if n_answers in (1, 2):
        return QuestionType.UNKNOWN, True
    return QuestionType.UNKNOWN, False  # 0 đáp án — có thể tự luận


# ============================================================
# Main
# ============================================================
def build_document(
    regions_per_page: list[list[Region]],
    page_sizes: list[tuple[int, int]],
    exam_id: str,
    source_file: str,
    backend: str = "vllm",
) -> AssemblyResult:
    n_pages = len(regions_per_page)
    doc = ExamDocument(
        exam_id=exam_id, source_file=source_file, n_pages=n_pages, detector_backend=backend,
    )
    q_layouts: dict[str, QLayout] = {}
    group_layouts: dict[str, GroupLayout] = {}
    header_layout: Optional[HeaderLayout] = None

    # --- gom câu theo trang, tính band + cột ---
    # qinfo[page] = list of dict
    questions: list[Question] = []
    # lưu tạm thông tin định vị câu để gán đáp án + section/group
    q_records: list[dict] = []   # {q_id, number, page, y_top, band_bottom, col, region}

    for page_idx, regions in enumerate(regions_per_page):
        page_w, page_h = page_sizes[page_idx]
        qregs = [r for r in regions if r.cls == RegionClass.QUESTION]
        if not qregs:
            continue
        split = _detect_columns(qregs, page_w)

        # content bottom theo cột (loại footer)
        non_footer = [r for r in regions if r.cls not in (RegionClass.FOOTER,)]
        def col_content_bottom(col_idx: int) -> float:
            ys = [r.bbox[3] for r in non_footer if _col_index(r.bbox, split) == col_idx]
            return max(ys) if ys else float(page_h)

        # sort câu trong từng cột theo y
        for col_idx in (0, 1):
            col_qs = [r for r in qregs if _col_index(r.bbox, split) == col_idx]
            if not col_qs:
                continue
            col_qs.sort(key=lambda r: r.bbox[1])
            cb = col_content_bottom(col_idx)
            bounds = _col_bounds(col_idx, split, page_w)
            for i, r in enumerate(col_qs):
                y_top = r.bbox[1]
                band_bottom = col_qs[i + 1].bbox[1] if i + 1 < len(col_qs) else cb
                q_records.append({
                    "page": page_idx, "y_top": y_top, "band_bottom": band_bottom,
                    "col": col_idx, "split": split, "bounds": bounds, "region": r,
                })

    # --- đánh số + id (giữ số VLM, fallback tăng dần) ---
    q_records.sort(key=lambda d: (d["page"], d["bounds"][0], d["y_top"]))
    used_numbers: set[int] = set()
    running = 0
    for rec in q_records:
        num = rec["region"].number
        if num is None or num in used_numbers:
            running += 1
            num = running if num is None else num
            while num in used_numbers:
                num = (running := running + 1)
        used_numbers.add(num)
        running = max(running, num)
        rec["q_id"] = f"q{num}"
        rec["number"] = num

    # --- gán đáp án vào câu ---
    answers_by_q: dict[str, list[Region]] = {rec["q_id"]: [] for rec in q_records}
    for page_idx, regions in enumerate(regions_per_page):
        for r in regions:
            if r.cls != RegionClass.ANSWER_OPTION:
                continue
            cx, cy = bbox_center(r.bbox)
            cand = None
            for rec in q_records:
                if rec["page"] != page_idx:
                    continue
                xl, xr = rec["bounds"]
                if not (xl <= cx <= xr):
                    continue
                if rec["y_top"] <= cy < rec["band_bottom"]:
                    cand = rec
                    break
            if cand is None:
                # fallback: câu gần nhất phía trên cùng trang
                above = [rec for rec in q_records if rec["page"] == page_idx and rec["y_top"] <= cy]
                if above:
                    cand = max(above, key=lambda d: d["y_top"])
            if cand is not None:
                answers_by_q[cand["q_id"]].append(r)

    # --- dựng Question + QLayout ---
    for rec in q_records:
        q_id = rec["q_id"]
        page_idx = rec["page"]
        xl, xr = rec["bounds"]
        ans_regs = sorted(answers_by_q[q_id], key=lambda r: (r.bbox[1], r.bbox[0]))

        # full box = cột × band
        full_box: BBox = (xl, rec["y_top"], xr, rec["band_bottom"])
        # stem = từ đỉnh câu tới đáp án đầu (nếu có)
        if ans_regs:
            first_ans_top = min(a.bbox[1] for a in ans_regs)
            stem_box: BBox = (xl, rec["y_top"], xr, max(rec["y_top"] + 1, first_ans_top))
        else:
            stem_box = full_box

        qtype, review = _infer_type(rec["region"].qtype, len(ans_regs))
        answers = [Answer(label=(a.label or "?")) for a in ans_regs]

        q = Question(
            id=q_id, number=rec["number"], type=qtype,
            answers=answers, page_indices=[page_idx],
            confidence=float(rec["region"].score),
            needs_review=review,
        )
        if rec["region"].continues_from_prev or rec["region"].continues_to_next:
            q.needs_review = True
            q.notes.append("vlm: câu có thể vắt trang")
        questions.append(q)

        q_layouts[q_id] = QLayout(
            q_id=q_id, number=rec["number"], page_index=page_idx,
            y_top=rec["y_top"], band_bottom=rec["band_bottom"], col_bounds=(xl, xr),
            full_parts=[Part(page_idx, full_box)],
            stem_parts=[Part(page_idx, stem_box)],
            answer_parts=[((a.label or "?"), [Part(page_idx, a.bbox)]) for a in ans_regs],
            continues_from_prev=rec["region"].continues_from_prev,
            continues_to_next=rec["region"].continues_to_next,
        )

    questions.sort(key=lambda q: q.number)
    doc.questions = questions

    # --- sections ---
    section_regs: list[Region] = []
    for regions in regions_per_page:
        section_regs += [r for r in regions if r.cls == RegionClass.SECTION_HEADER]
    section_regs.sort(key=lambda r: r.global_pos())
    if section_regs:
        sections: list[Section] = []
        for i, sr in enumerate(section_regs):
            sid = f"s{i + 1}"
            sections.append(Section(id=sid, title=(sr.title or "").strip(),
                                     page_indices=[sr.page_index]))
        # gán câu vào section theo vị trí global
        def gpos_q(q: Question) -> tuple[int, float]:
            lay = q_layouts[q.id]
            return (lay.page_index, lay.y_top)
        for q in questions:
            qp = gpos_q(q)
            chosen = None
            for i, sr in enumerate(section_regs):
                if sr.global_pos() <= qp:
                    chosen = i
                else:
                    break
            if chosen is not None:
                sid = f"s{chosen + 1}"
                q.section_id = sid
                sections[chosen].question_ids.append(q.id)
                if q.page_indices[0] not in sections[chosen].page_indices:
                    sections[chosen].page_indices.append(q.page_indices[0])
        doc.sections = sections

    # --- groups (instruction/passage) ---
    group_regs: list[Region] = []
    for regions in regions_per_page:
        group_regs += [r for r in regions
                       if r.cls in (RegionClass.GROUP_INSTRUCTION, RegionClass.PASSAGE)]
    group_regs.sort(key=lambda r: r.global_pos())
    if group_regs:
        groups: list[Group] = []
        for i, gr in enumerate(group_regs):
            gid = f"g{i + 1}"
            kind = "passage" if gr.cls == RegionClass.PASSAGE else "instruction"
            groups.append(Group(id=gid, kind=kind, page_indices=[gr.page_index]))
            group_layouts[gid] = GroupLayout(g_id=gid, parts=[Part(gr.page_index, gr.bbox)])
        # gán câu: ưu tiên covers, fallback vị trí (giữa group này và group/section kế)
        for i, gr in enumerate(group_regs):
            gid = f"g{i + 1}"
            next_pos = group_regs[i + 1].global_pos() if i + 1 < len(group_regs) else (10**9, 0.0)
            for q in questions:
                lay = q_layouts[q.id]
                qp = (lay.page_index, lay.y_top)
                in_group = False
                if gr.covers:
                    in_group = gr.covers[0] <= q.number <= gr.covers[1]
                else:
                    in_group = gr.global_pos() <= qp < next_pos
                if in_group:
                    q.group_id = gid
                    groups[i].question_ids.append(q.id)
                    if lay.page_index not in groups[i].page_indices:
                        groups[i].page_indices.append(lay.page_index)
        doc.groups = groups

    # --- header ---
    for regions in regions_per_page:
        hdr = [r for r in regions if r.cls == RegionClass.EXAM_HEADER]
        if hdr:
            h = hdr[0]
            header_layout = HeaderLayout(parts=[Part(h.page_index, h.bbox)])
            doc.header = ExamHeader(raw_text=h.text or "")
            break

    # --- answer_key (chỉ ghi nhận có tồn tại; parse để sau) ---
    has_key = any(r.cls == RegionClass.ANSWER_KEY for regions in regions_per_page for r in regions)

    logger.info(
        f"Assembly: {len(questions)} câu, {len(doc.sections)} section, "
        f"{len(doc.groups)} group, header={'có' if doc.header else 'không'}, "
        f"answer_key={'có' if has_key else 'không'}"
    )
    return AssemblyResult(doc, q_layouts, group_layouts, header_layout)
