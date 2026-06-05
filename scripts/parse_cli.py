#!/usr/bin/env python
"""CLI — parse 1 file đề (PDF nhiều trang hoặc 1 ảnh) → crops + overlay + exam.json.

Ví dụ:
    python scripts/parse_cli.py input/de_thi.pdf
    python scripts/parse_cli.py input/trang1.jpg --backend ollama
    python scripts/parse_cli.py input/de.pdf --no-snap --debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# cho phép chạy trực tiếp từ thư mục project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Exam VLM grounding parser (Phase 3)")
    ap.add_argument("input", help="File PDF hoặc ảnh (png/jpg/jpeg/bmp/tif/webp)")
    ap.add_argument("--backend", choices=["vllm", "ollama"], default=None,
                    help="Ép backend detector (mặc định lấy từ .env)")
    ap.add_argument("--out", default=None, help="Thư mục output gốc (mặc định ./output)")
    ap.add_argument("--no-snap", action="store_true", help="Tắt CV box-snap (để so sánh)")
    ap.add_argument("--debug", action="store_true", help="Log DEBUG")
    args = ap.parse_args()

    if args.debug:
        import os
        os.environ["LOG_LEVEL"] = "DEBUG"

    # import sau khi set env để logger lấy đúng level
    from src.core.logging import logger
    from src.services.pipeline import parse_exam

    src = Path(args.input)
    if not src.exists():
        logger.error(f"Không tìm thấy file: {src}")
        return 1

    try:
        result = parse_exam(
            str(src),
            out_root=args.out,
            backend=args.backend,
            snap=(False if args.no_snap else None),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pipeline lỗi: {e}")
        return 2

    doc = result.document
    print("\n" + "=" * 60)
    print(f"  {doc.source_file}: {len(doc.questions)} câu, "
          f"{len(doc.sections)} section, {len(doc.groups)} group")
    print(f"  Output: {Path(args.out or './output') / doc.exam_id}")
    print(f"  → mở overlay/page_*.png để kiểm khung, crops/ để xem ảnh cắt")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
