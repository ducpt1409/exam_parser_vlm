# Changelog — exam_parser_vlm

## [Phase 3] - 2026-06-05 - Lõi pipeline VLM grounding (chạy được, kiểm crop)

### Mục tiêu
Dựng pipeline top-down (grounding-first) chạy local bằng CLI để **kiểm độ chính xác crop**,
tương đương "Phase 3" của project paddle. Input: PDF nhiều trang HOẶC 1 ảnh (png/jpg/jpeg/...).
Chưa làm service/MinIO/API.

### Kiến trúc (detector-agnostic)
```
Preprocess → RegionDetector(VLM) → Assembly(DOM) → Cross-page → Box-snap+Crop+Overlay
```

### Đã thêm
- **Cấu hình & hạ tầng**: `requirements.txt`, `.env.example`, `SETUP.md` (conda `exam_parser_vlm`,
  vLLM Qwen3-VL-8B-AWQ + cờ `--gpu-memory-utilization 0.55` cho GPU chia sẻ 13GB Windows), `README.md`.
- **`src/core/`**: `config.py` (pydantic-settings), `logging.py` (loguru).
- **`src/schemas/`**:
  - `geometry.py` — BBox + helper (union/clamp/pad/iou/y_overlap).
  - `region.py` — `Region` + `RegionClass` (đầu ra chuẩn của mọi detector, bbox pixel).
  - `exam.py` — DOM: `ExamDocument/Section/Group/Question/Answer/ExamHeader/CropImage` + `QuestionType`.
- **`src/detectors/`** (2 backend cùng interface):
  - `base.py` — `RegionDetector` ABC.
  - `schema.py` — `GPage/GRegion` (JSON thô VLM, box 0-1000) cho structured output.
  - `prompt.py` — prompt grounding tiếng Việt + map GPage→Region (0-1000 → pixel).
  - `image_util.py` — resize theo `VLM_MAX_PIXELS` + encode base64/data-url.
  - `vlm_vllm.py` — gọi vLLM qua OpenAI API + `guided_json` (KHUYẾN NGHỊ).
  - `vlm_ollama.py` — gọi Ollama `/api/chat` + `format` schema.
  - `factory.py` — chọn backend theo `DETECTOR_BACKEND`.
- **`src/services/`**:
  - `preprocess.py` — PDF/ảnh → list[PIL.Image], deskew.
  - `box_snap.py` — `PageInk`: mask ink + snap bbox về hộp bao ink (bao trọn hình, không nuốt câu kế).
  - `assembly.py` — regions → DOM: cột/band, gán đáp án, section, group (covers/vị trí).
  - `crosspage.py` — gộp câu vắt trang (cùng số câu / cờ continuation).
  - `cropper.py` — snap + crop full/stem/đáp án/group/header, ghép dọc vắt trang, vẽ overlay.
  - `pipeline.py` — orchestrator → `output/{exam_id}/{exam.json, crops/, overlay/, vlm_raw/, summary.txt}`.
- **`scripts/parse_cli.py`** — CLI: `python scripts/parse_cli.py <input> [--backend] [--no-snap] [--debug]`.

### Thiết kế đáng chú ý
- **OCR KHÔNG quyết định khung**: khung sinh từ VLM grounding + CV-snap → tránh nhóm lỗi của paddle.
- **Hình trong câu**: KHÔNG detect figure riêng; full box câu = band, box-snap lấy bao ngoài ink
  → đồ thị/công thức tự được bao trọn.
- **Nhóm (section/group) + vắt trang**: hạng nhất trong DOM.
- **Detector-agnostic**: đổi `vllm`↔`ollama` (và sau này YOLO) không sửa pipeline.

### Đã verify
- `py_compile` toàn bộ src/ + scripts/ → OK (chưa chạy runtime: cần cài deps trên WSL theo SETUP.md).

### Còn lại (Phase sau)
- OCR parse metadata header + bảng đáp án (`is_correct`).
- MinIO upload + FastAPI `/v1/exams` + `/v1/images`→base64.
- Tinh chỉnh assembly sau khi xem overlay thực tế (cột phức tạp, group đa trang).
- Thu nhãn → train YOLO (backend production).

### Cần làm (người dùng, trên WSL)
1. Cài theo `SETUP.md` (conda `exam_parser_vlm` + torch cu128 + vLLM + `pip install -r requirements.txt`).
2. `vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ --gpu-memory-utilization 0.55 ...` (terminal riêng).
3. `python scripts/parse_cli.py input/de_mau_azota_toan_THPT.pdf` → mở `overlay/` kiểm khung.
