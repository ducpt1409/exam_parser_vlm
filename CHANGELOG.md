# Changelog — exam_parser_vlm

## [Phase 3.3] - 2026-06-06 - Parser JSON khoan dung + cross-page đo ink

### Vấn đề
- `Expecting value ... (char 426)` (toan8 trang 2): JSON 4B méo/cắt → parser fail cứng → mất cả trang.
- Đề azota báo "không có câu vắt trang" dù có: heuristic cũ chỉ theo ngưỡng vị trí, không đáng tin.

### Giải pháp
- **`detectors/parse.py`** (MỚI) — `parse_gpage()` khoan dung: thử nguyên khối → cắt `{...}` →
  **cứu từng region object hợp lệ** (quét cân bằng ngoặc) khi JSON tổng méo/bị cắt. Không mất cả trang.
  Hai backend dùng chung, bỏ parse thủ công.
- **Cross-page đo mật độ ink** (`box_snap.PageInk.ink_ratio`): câu vắt trang được nhận khi đầu trang
  sau có NỘI DUNG THẬT phía trên câu đầu (ink_ratio > 0.4%) + không có header/section chen — thay vì
  chỉ dựa ngưỡng vị trí. `crosspage.merge_cross_page` nhận thêm `page_inks`.

---

## [Phase 3.2] - 2026-06-06 - Bật lại đáp án A/B/C/D + cứu công thức bị cắt + nối ảnh vắt trang

### Làm rõ yêu cầu
- "đáp án" = phương án A/B/C/D (KHÔNG phải lời giải). CẦN: câu hỏi + full câu hỏi + đáp án.
  KHÔNG detect phần lời giải/hướng dẫn giải.

### Vấn đề (4B-FP8, đề toán Azota)
- Tắt đáp án (3.1) làm mất hết phương án trong crop.
- Câu/đáp án có phân số/tích phân bị **cắt lẹm mép trên & chiều cao**.
- Câu 45 (có đồ thị): full crop không bao hình + bỏ qua đáp án.
- Đáp án vắt trang chưa được nối ảnh.

### Giải pháp
- **`DETECT_ANSWERS` mặc định = true** → detect lại A/B/C/D. Prompt thêm: BỎ QUA lời giải;
  box bao TRỌN công thức (cả tử/mẫu phân số, tích phân, căn); bao cả hình; set cờ vắt trang.
- **Full câu = băng full-height [đỉnh câu → đỉnh câu kế] + snap ink** (không cap theo box VLM khi
  answers on) → tự bao hình (Câu 45), phân số, đủ đáp án; không cắt lẹm.
- **Box-snap có `expand`** (`box_snap.py`): mở rộng vùng tìm ink trước khi snap → cứu tử số phân số/
  đỉnh tích phân bị box VLM cắt. CHỈ áp cho crop ĐÁP ÁN (full là băng nên không expand → tránh nuốt câu kế).
  Config: `snap_expand_y_ratio`, `snap_expand_x_ratio`.
- **Cross-page nối ảnh** (`crosspage.py`): thêm Ca 3 — heuristic phát hiện đuôi câu ở đầu trang sau
  (khoảng trống lớn trên câu đầu + không có header/section chen) → nối mảnh, cropper ghép dọc như paddle.
  Nhận thêm `regions_per_page`.

### Cần làm (người dùng)
- Sửa `.env`: `DETECT_ANSWERS=true` (bản cũ đang false). KHÔNG cần serve lại vLLM.
- Chạy lại 3 đề, soi `overlay/` (full câu bao đủ hình/công thức/đáp án; câu vắt trang được nối).

---

## [Phase 3.1] - 2026-06-06 - Bỏ detect đáp án (mặc định) + prompt cứng + cap box theo VLM

### Vấn đề (chạy thật đề Tiếng Anh, 4B-FP8)
- Khoanh cả từng đáp án (chưa cần) → nhiễu + bắt 4B liệt kê ~250 vùng/đề → quá tải.
- 2 câu sát nhau bị gộp làm 1; cắt thiếu/thừa; cắt cả ô trắng dưới đáp án. Đề dày càng sai.

### Nguyên nhân
- Một phần **model 4B** đuối khi nhiều mục tiêu/trang (sót marker → gộp câu).
- Một phần **prompt/code**: ta bắt detect cả answer_option; full box kéo tới đỉnh câu kế (nuốt ô trắng).

### Giải pháp
- **`DETECT_ANSWERS` (mặc định false)**: chỉ khoanh VÙNG CÂU HỎI trọn vẹn (gồm phương án bên trong),
  KHÔNG tách đáp án → giảm ~5× số mục tiêu → 4B chính xác hơn, ít gộp câu, JSON ngắn (ít truncate).
  - `core/config.py`: thêm `detect_answers`.
  - `detectors/prompt.py`: tách `build_user_prompt(detect_answers)` — 2 biến thể; biến thể no-answers
    dặn "mỗi Question N là 1 vùng riêng, cấm gộp; box dừng ở phương án cuối, không lấn ô trống/câu kế".
  - `vlm_vllm.py`/`vlm_ollama.py`: dùng prompt động + `drop_answers` khi map region.
- **`assembly.py`**: khi không tách đáp án, full box dùng MÉP DƯỚI box VLM (cap trong band, +8px)
  thay vì kéo tới đỉnh câu kế → bớt nuốt ô trắng / lấn câu sau.

### Còn lại (do model — cần test 8B để tách bạch)
- Gộp 2 câu / sót câu trên trang dày: chủ yếu giới hạn 4B. Lên 8B/32B (khi có VRAM) sẽ giảm.

---

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
