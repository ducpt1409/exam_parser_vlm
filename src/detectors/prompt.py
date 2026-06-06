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

# Phần chung
_COMMON_HEAD = """Phân tích ảnh MỘT TRANG đề thi. Liệt kê TẤT CẢ các vùng từ trên xuống dưới, trái sang phải.

Mỗi vùng trả 1 object với:
- "cls": một trong:
  • "exam_header"        : khối tiêu đề đầu đề (Sở GD, trường, môn, thời gian, mã đề, số câu)
  • "section_header"     : dòng "PHẦN I", "PHẦN II", "Phần A"...
  • "group_instruction"  : câu dẫn chung cho nhiều câu ("Mark the letter...", "Đọc đoạn sau trả lời câu 1-5")
  • "passage"            : đoạn văn/ngữ liệu dùng chung cho nhiều câu
  • "question"           : khối MỘT câu hỏi
  • "answer_key"         : bảng đáp án cuối đề (1.C 2.A ...)
  • "footer"             : số trang / chân trang (sẽ bị bỏ qua)
- "box": [x1,y1,x2,y2] thang 0-1000, ôm KHÍT vùng đó."""

# Biến thể KHÔNG tách đáp án (mặc định) — gom cả phương án vào trong "question"
_QUESTION_RULES_NO_ANSWERS = """
QUAN TRỌNG về "question":
- Box của MỘT câu hỏi = từ dòng "Question N"/"Câu N" cho tới HẾT các phương án A/B/C/D của CHÍNH câu đó
  (gộp cả đề bài + các lựa chọn vào MỘT vùng duy nhất). KHÔNG tách riêng từng đáp án.
- Thêm "number" (số câu, ví dụ "Question 3" → 3) và "qtype"
  (mcq_single|mcq_multi|true_false|fill_blank|essay).
- MỖI dòng bắt đầu bằng "Question N"/"Câu N" là MỘT vùng RIÊNG. TUYỆT ĐỐI KHÔNG gộp 2 câu liền nhau
  (vd Question 1 và Question 2) thành 1 vùng. Số vùng "question" = đúng số câu nhìn thấy.
- Box DỪNG ngay sau phương án cuối cùng. KHÔNG kéo xuống ô/khung trả lời TRỐNG hay khoảng trắng phía dưới,
  KHÔNG lấn sang câu kế tiếp.
- Nếu câu bị cắt ở mép trên/dưới trang: đặt cờ "continues_from_prev"/"continues_to_next" = true."""

# Biến thể CÓ tách đáp án
_QUESTION_RULES_WITH_ANSWERS = """
Thêm class:
  • "answer_option"      : MỘT ô đáp án A/B/C/D (hoặc a/b/c/d), thêm "label" ("A"/"B"/"C"/"D").
QUAN TRỌNG về "question":
- Box "question" = phần ĐỀ BÀI (stem), KHÔNG gồm các phương án (phương án để riêng ở answer_option).
- Thêm "number" và "qtype". MỖI "Question N"/"Câu N" là MỘT vùng RIÊNG, cấm gộp 2 câu.
- Mỗi phương án A/B/C/D là MỘT "answer_option" riêng, kể cả khi nằm cùng hàng.
- Cờ "continues_from_prev"/"continues_to_next" nếu câu vắt trang."""

_COMMON_TAIL = """
- "section_header": thêm "title". Nếu group/section đọc được dải câu: thêm "covers_start","covers_end".
- KHÔNG bỏ sót câu nào.

LƯU Ý QUAN TRỌNG:
- BỎ QUA HOÀN TOÀN phần LỜI GIẢI / HƯỚNG DẪN GIẢI CHI TIẾT / ĐÁP ÁN CHI TIẾT (thường ở cuối đề) —
  KHÔNG tạo vùng cho chúng. CHỈ lấy đề bài (câu hỏi + phương án).
- Box phải BAO TRỌN công thức toán: phân số nhiều tầng (cả TỬ SỐ phía trên lẫn MẪU SỐ phía dưới),
  dấu tích phân, căn, ma trận — KHÔNG được cắt lẹm phần trên/dưới của công thức.
- Câu hỏi có HÌNH/ĐỒ THỊ: box phải bao cả hình.
- Nếu một câu (hoặc các phương án của nó) bị cắt ở MÉP DƯỚI trang (còn tiếp trang sau):
  đặt "continues_to_next"=true cho câu đó. Nếu PHẦN ĐẦU trang là phần tiếp của câu trang trước:
  đặt "continues_from_prev"=true cho câu đầu trang.
Trả về JSON đúng schema {"regions": [...]}."""


def build_user_prompt(detect_answers: bool) -> str:
    rules = _QUESTION_RULES_WITH_ANSWERS if detect_answers else _QUESTION_RULES_NO_ANSWERS
    return _COMMON_HEAD + rules + _COMMON_TAIL


def _meta_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def gregion_to_region(g: GRegion, page_index: int, page_w: int, page_h: int) -> Region | None:
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


def gpage_to_regions(
    page: GPage, page_index: int, page_w: int, page_h: int,
    drop_answers: bool = False,
) -> list[Region]:
    out: list[Region] = []
    for g in page.regions:
        r = gregion_to_region(g, page_index, page_w, page_h)
        if r is None:
            continue
        if drop_answers and r.cls == RegionClass.ANSWER_OPTION:
            continue
        out.append(r)
    return out
