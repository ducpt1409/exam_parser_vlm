"""Stage 3 — CV Box-Snap: ép bbox thô (từ VLM) khít vào ink thật trên trang.

Nguyên tắc: trong bbox cho trước, tìm HỘP BAO của toàn bộ ink (mép ngoài), bỏ khoảng trắng
rìa, cộng pad nhỏ. Vì chỉ lấy bao ngoài nên khoảng trắng GIỮA (vd ruột đồ thị) được giữ
nguyên → câu có hình vẫn bao trọn hình (xem PLAN §5.1). Snap chỉ THU HẸP về nội dung,
KHÔNG mở rộng quá bbox đầu vào → không nuốt sang vùng câu kế.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.schemas.geometry import BBox, clamp_bbox


class PageInk:
    """Tiền xử lý 1 trang: mask ink (1=có mực) để snap nhiều bbox nhanh."""

    def __init__(self, image: Image.Image):
        self.width, self.height = image.size
        arr = np.array(image.convert("L"))
        # Otsu inverted: chữ/nét → 255, nền → 0
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.ink = (th > 0).astype(np.uint8)   # HxW, {0,1}

    def snap(self, bbox: BBox, pad: int = 8, expand_x: int = 0, expand_y: int = 0) -> BBox:
        """Snap bbox về hộp bao ink. Mở rộng vùng TÌM (expand) trước khi snap để cứu nội dung
        bị box VLM cắt (vd tử số phân số, đỉnh tích phân). Không có ink → trả bbox clamp."""
        x1, y1, x2, y2 = clamp_bbox(bbox, self.width, self.height)
        if expand_x or expand_y:
            x1, y1, x2, y2 = clamp_bbox(
                (x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y),
                self.width, self.height,
            )
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            return (x1, y1, x2, y2)

        sub = self.ink[iy1:iy2, ix1:ix2]
        if sub.size == 0 or sub.sum() == 0:
            return (x1, y1, x2, y2)

        row_ink = sub.sum(axis=1)              # mỗi hàng: số pixel ink
        col_ink = sub.sum(axis=0)              # mỗi cột: số pixel ink

        # Ngưỡng nhỏ để bỏ nhiễu lốm đốm (không bỏ nét thật)
        w = ix2 - ix1
        h = iy2 - iy1
        row_thr = max(1.0, 0.004 * w)
        col_thr = max(1.0, 0.004 * h)

        rows = np.where(row_ink > row_thr)[0]
        cols = np.where(col_ink > col_thr)[0]
        if len(rows) == 0 or len(cols) == 0:
            # fallback: dùng ngưỡng >0 (giữ mọi ink)
            rows = np.where(row_ink > 0)[0]
            cols = np.where(col_ink > 0)[0]
        if len(rows) == 0 or len(cols) == 0:
            return (x1, y1, x2, y2)

        ny1 = iy1 + int(rows[0])
        ny2 = iy1 + int(rows[-1]) + 1
        nx1 = ix1 + int(cols[0])
        nx2 = ix1 + int(cols[-1]) + 1

        snapped = (nx1 - pad, ny1 - pad, nx2 + pad, ny2 + pad)
        return clamp_bbox(snapped, self.width, self.height)

    def ink_ratio(self, bbox: BBox) -> float:
        """Tỉ lệ pixel có ink trong bbox (0..1). Dùng để biết vùng có nội dung thật không."""
        x1, y1, x2, y2 = clamp_bbox(bbox, self.width, self.height)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        if ix2 - ix1 < 1 or iy2 - iy1 < 1:
            return 0.0
        sub = self.ink[iy1:iy2, ix1:ix2]
        return float(sub.sum()) / sub.size if sub.size else 0.0
