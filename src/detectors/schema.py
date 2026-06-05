"""Schema JSON thô mà VLM phải trả (dùng cho structured output / guided_json).

Toạ độ box chuẩn hoá thang 0-1000 theo (rộng, cao) ảnh trang.
Giữ schema PHẲNG + ít field để model nhỏ (8B) bám đúng dễ hơn.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GRegion(BaseModel):
    cls: str = Field(description="Loại vùng: exam_header|section_header|group_instruction|"
                                 "passage|question|answer_option|answer_key|footer")
    box: list[int] = Field(description="[x1,y1,x2,y2] thang 0-1000", min_length=4, max_length=4)
    number: Optional[int] = Field(default=None, description="Số câu nếu cls=question")
    label: Optional[str] = Field(default=None, description="Nhãn đáp án A/B/C/D nếu cls=answer_option")
    qtype: Optional[str] = Field(default=None, description="Loại câu: mcq_single|mcq_multi|"
                                                           "true_false|fill_blank|essay")
    title: Optional[str] = Field(default=None, description="Tiêu đề section nếu cls=section_header")
    covers_start: Optional[int] = Field(default=None, description="Câu bắt đầu group/section")
    covers_end: Optional[int] = Field(default=None, description="Câu kết thúc group/section")
    continues_from_prev: bool = Field(default=False, description="Câu nối từ trang trước")
    continues_to_next: bool = Field(default=False, description="Câu còn tiếp ở trang sau")


class GPage(BaseModel):
    regions: list[GRegion] = Field(default_factory=list)
