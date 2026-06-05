"""Region — đầu ra chuẩn hoá của MỌI detector (VLM hôm nay, YOLO sau này).

Pipeline phía sau chỉ làm việc với list[Region], KHÔNG biết backend nào sinh ra.
bbox đã ở toạ độ PIXEL của trang (đã map từ 0-1000 của VLM về pixel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.schemas.geometry import BBox


class RegionClass(str, Enum):
    EXAM_HEADER = "exam_header"
    SECTION_HEADER = "section_header"
    GROUP_INSTRUCTION = "group_instruction"
    PASSAGE = "passage"
    QUESTION = "question"
    ANSWER_OPTION = "answer_option"
    ANSWER_KEY = "answer_key"
    FOOTER = "footer"
    OTHER = "other"

    @classmethod
    def from_str(cls, s: str) -> "RegionClass":
        s = (s or "").strip().lower()
        for m in cls:
            if m.value == s:
                return m
        # vài alias hay gặp
        alias = {
            "header": cls.EXAM_HEADER,
            "metadata": cls.EXAM_HEADER,
            "section": cls.SECTION_HEADER,
            "instruction": cls.GROUP_INSTRUCTION,
            "group": cls.GROUP_INSTRUCTION,
            "answer": cls.ANSWER_OPTION,
            "option": cls.ANSWER_OPTION,
            "answerkey": cls.ANSWER_KEY,
            "answer_keys": cls.ANSWER_KEY,
            "page_number": cls.FOOTER,
        }
        return alias.get(s, cls.OTHER)


@dataclass
class Region:
    page_index: int
    cls: RegionClass
    bbox: BBox                                  # pixel
    score: float = 0.9
    # meta tùy class:
    number: Optional[int] = None                # QUESTION: số câu
    label: Optional[str] = None                 # ANSWER_OPTION: "A".."D" / "a".."d"
    qtype: Optional[str] = None                 # QUESTION: loại câu (VLM đoán)
    title: Optional[str] = None                 # SECTION_HEADER: "PHẦN I"
    covers: Optional[tuple[int, int]] = None    # GROUP/SECTION: dải số câu [start,end]
    continues_from_prev: bool = False           # QUESTION: nối từ trang trước
    continues_to_next: bool = False             # QUESTION: còn tiếp ở trang sau
    text: str = ""                              # text thô (nếu VLM trả, tham khảo)
    source: str = "vlm"

    def global_pos(self) -> tuple[int, float]:
        return (self.page_index, self.bbox[1])
