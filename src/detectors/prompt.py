"""Prompt grounding + chuyển GPage (toạ độ 0-1000) → list[Region] (pixel)."""
from __future__ import annotations

from src.detectors.schema import GPage, GRegion
from src.schemas.geometry import clamp_bbox
from src.schemas.region import Region, RegionClass

SYSTEM_PROMPT = (
    "Bạn là hệ thống phân tích BỐ CỤC đề thi. Bạn CHỈ khoanh vùng và phân loại, "
    "TUYỆT ĐỐI KHÔNG giải bài, KHÔNG chép lại nội dung. "
    "Toạ độ luôn theo thang 0-1000 so với chiều rộng và cao của ảnh."
)

USER_PROMPT = """Phân tích ảnh MỘT TRANG đề thi. Liệt kê TẤT CẢ các vùng từ trên xuống dưới, trái sang phải.

Mỗi vùng trả 1 object với:
- "cls": một trong:
  • "exam_header"        : khối tiêu đề đầu đề (Sở GD, trường, môn, thời gian, mã đề, số câu)
  • "section_header"     : dòng "PHẦN I", "PHẦN II", "Phần A"...
  • "group_instruction"  : câu dẫn chung cho nhiều câu ("Đọc đoạn sau trả lời câu 1-5", "Mark the letter...")
  • "passage"            : đoạn văn/ngữ liệu dùng chung cho nhiều câu
  • "question"           : khối MỘT câu hỏi (phần đề bài, KÈM hình/đồ thị/công thức của nó nếu có)
  • "answer_option"      : MỘT ô đáp án (A. / B. / C. / D. hoặc a) b) c) d))
  • "answer_key"         : bảng đáp án cuối đề (1.C 2.A ...)
  • "footer"             : số trang / chân trang (sẽ bị bỏ qua)
- "box": [x1,y1,x2,y2] thang 0-1000, ôm KHÍT vùng đó.
- Nếu cls="question": thêm "number" (số câu, ví dụ "Câu 5" → 5), "qtype"
  (mcq_single|mcq_multi|true_false|fill_blank|essay), và 2 cờ "continues_from_prev"/"continues_to_next"
  (true nếu câu bị cắt ở mép trên/dưới trang).
- Nếu cls="answer_option": thêm "label" ("A"/"B"/"C"/"D"...).
- Nếu cls="section_header": thêm "title".
- Nếu cls="group_instruction" hoặc "section_header" và đọc được dải câu: thêm "covers_start","covers_end".

QUY TẮC:
- Với câu hỏi có hình/đồ thị: box của "question" phải BAO TRỌN cả hình đó.
- Mỗi đáp án A/B/C/D là MỘT "answer_option" riêng, kể cả khi nằm cùng hàng.
- KHÔNG bỏ sót câu nào. KHÔNG gộp 2 câu vào 1.
Trả về JSON đúng schema {"regions": [...]}."""


def _meta_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def gregion_to_region(g: GRegion, page_index: int, page_w: int, page_h: int) -> Region | None:
    """Map 1 GRegion (0-1000) → Region (pixel)."""
    if not g.box or len(g.box) != 4:
        return None
    x1, y1, x2, y2 = g.box
    sx, sy = page_w / 1000.0, page_h / 1000.0
    bbox = clamp_bbox((x1 * sx, y1 * sy, x2 * sx, y2 * sy), page_w, page_h)

    cls = RegionClass.from_str(g.cls)
    covers = None
    cs, ce = _meta_int(g.covers_start), _meta_int(g.covers_end)
    if cs is not None and ce is not None:
        covers = (cs, ce)

    return Region(
        page_index=page_index,
        cls=cls,
        bbox=bbox,
        score=0.9,
        number=_meta_int(g.number),
        label=(g.label.strip() if g.label else None),
        qtype=g.qtype,
        title=g.title,
        covers=covers,
        continues_from_prev=bool(g.continues_from_prev),
        continues_to_next=bool(g.continues_to_next),
        source="vlm",
    )


def gpage_to_regions(page: GPage, page_index: int, page_w: int, page_h: int) -> list[Region]:
    out: list[Region] = []
    for g in page.regions:
        r = gregion_to_region(g, page_index, page_w, page_h)
        if r is not None:
            out.append(r)
    return out
