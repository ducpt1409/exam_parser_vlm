"""Exam Document Object Model (DOM) — output JSON cuối (xem PLAN_GROUNDING.md §2).

Tinh gọn theo yêu cầu: chỉ câu/đáp án + nhóm (section/group) + header. KHÔNG tách figure/table/
formula riêng — hình/công thức nằm trong ảnh crop của câu.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.geometry import BBox


class QuestionType(str, Enum):
    MCQ_SINGLE = "mcq_single"          # trắc nghiệm 1 đáp án A/B/C/D
    MCQ_MULTI = "mcq_multi"            # chọn nhiều
    TRUE_FALSE = "true_false"          # đúng/sai a)b)c)d)
    FILL_BLANK = "fill_blank"          # điền số/từ
    SHORT_ANSWER = "short_answer"      # tự luận ngắn
    ESSAY = "essay"                    # tự luận dài
    READING = "reading"               # đọc hiểu (thuộc group)
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, s: Optional[str]) -> "QuestionType":
        s = (s or "").strip().lower()
        for m in cls:
            if m.value == s:
                return m
        alias = {
            "mcq": cls.MCQ_SINGLE, "single": cls.MCQ_SINGLE,
            "multi": cls.MCQ_MULTI, "multiple": cls.MCQ_MULTI,
            "truefalse": cls.TRUE_FALSE, "dung_sai": cls.TRUE_FALSE,
            "fill": cls.FILL_BLANK, "blank": cls.FILL_BLANK, "numeric": cls.FILL_BLANK,
            "tu_luan": cls.ESSAY, "essay_long": cls.ESSAY,
        }
        return alias.get(s, cls.UNKNOWN)


class CropImage(BaseModel):
    """1 ảnh đã crop. page_span >1 phần tử nếu vắt trang (ảnh ghép dọc)."""
    path: str = ""                     # đường dẫn local (Phase 3)
    key: str = ""                      # MinIO object key (Phase sau)
    bbox: Optional[BBox] = None        # bbox trên trang ĐẦU (tham khảo)
    page_span: list[int] = Field(default_factory=list)
    width: int = 0
    height: int = 0
    size_bytes: int = 0


class Answer(BaseModel):
    label: str                         # "A".."D" / "a".."d"
    image: Optional[CropImage] = None
    is_correct: Optional[bool] = None


class Question(BaseModel):
    id: str                            # "q1"
    number: int
    type: QuestionType = QuestionType.UNKNOWN
    section_id: Optional[str] = None
    group_id: Optional[str] = None

    full_image: Optional[CropImage] = None     # TRỌN câu (stem + đáp án + hình) — ảnh FE dùng chính
    stem_image: Optional[CropImage] = None     # chỉ phần đề bài
    answers: list[Answer] = Field(default_factory=list)

    page_indices: list[int] = Field(default_factory=list)
    confidence: float = 1.0
    needs_review: bool = False
    notes: list[str] = Field(default_factory=list)


class Section(BaseModel):
    id: str                            # "s1"
    title: str = ""
    page_indices: list[int] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)


class Group(BaseModel):
    id: str                            # "g1"
    kind: str = "instruction"          # instruction | passage
    image: Optional[CropImage] = None  # ảnh đoạn dẫn / passage
    page_indices: list[int] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)


class ExamHeader(BaseModel):
    image: Optional[CropImage] = None
    so_gd: Optional[str] = None
    truong: Optional[str] = None
    mon: Optional[str] = None
    thoi_gian_phut: Optional[int] = None
    ma_de: Optional[str] = None
    so_cau: Optional[int] = None
    raw_text: str = ""


class ExamDocument(BaseModel):
    exam_id: str
    source_file: str
    n_pages: int
    model_version: str = "vlm-grounding-0.1"
    schema_version: str = "1.0"
    detector_backend: str = "vllm"

    header: Optional[ExamHeader] = None
    sections: list[Section] = Field(default_factory=list)
    groups: list[Group] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    answer_key: dict[str, str] = Field(default_factory=dict)

    preview_pdf_path: str = ""
    stats: dict = Field(default_factory=dict)
