"""Parse JSON từ VLM một cách KHOAN DUNG.

4B đôi khi trả JSON méo (thiếu value, dư phẩy) hoặc bị cắt. Thay vì fail cả trang,
ta cứu được bao nhiêu vùng hợp lệ thì cứu.
"""
from __future__ import annotations

from src.core.logging import logger
from src.detectors.schema import GPage, GRegion


def _iter_objects(s: str):
    """Yield mọi object `{...}` cân bằng ngoặc (kể cả lồng nhau, kể cả khi phần ngoài bị cắt)."""
    stack: list[int] = []
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                start = stack.pop()
                yield s[start:i + 1]


def parse_gpage(content: str, page_index: int = -1) -> GPage:
    if not content:
        return GPage(regions=[])

    text = content.strip()
    # bỏ rào markdown ```json ... ```
    if text.startswith("```"):
        text = text.strip("`").lstrip()
        if text[:4].lower() == "json":
            text = text[4:].lstrip()

    # 1) parse nguyên khối
    try:
        return GPage.model_validate_json(text)
    except Exception:
        pass

    # 2) cắt từ '{' đầu tới '}' cuối
    try:
        s = text.index("{")
        e = text.rindex("}") + 1
        return GPage.model_validate_json(text[s:e])
    except Exception:
        pass

    # 3) cứu từng region object (bỏ object ngoài chứa "regions")
    regions: list[GRegion] = []
    for obj in _iter_objects(text):
        if '"regions"' in obj:
            continue
        try:
            regions.append(GRegion.model_validate_json(obj))
        except Exception:
            continue
    if regions:
        logger.warning(f"[parse] trang {page_index}: JSON méo/cắt — cứu được {len(regions)} vùng")
    else:
        logger.error(f"[parse] trang {page_index}: không cứu được vùng nào từ JSON")
    return GPage(regions=regions)
