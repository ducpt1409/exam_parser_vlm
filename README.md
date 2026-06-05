# exam_parser_vlm

Bóc tách **vùng câu hỏi / đáp án** từ đề thi (PDF nhiều trang hoặc 1 ảnh) bằng **VLM grounding +
CV box-snap**, gom **section/group**, xử lý **câu vắt trang** — crop ảnh để hiển thị như Azota.

- Hướng tiếp cận & lý do: [`PLAN_GROUNDING.md`](PLAN_GROUNDING.md)
- Cài đặt chi tiết: [`SETUP.md`](SETUP.md)

## Tóm tắt kiến trúc
```
PDF/ảnh → Preprocess → RegionDetector(VLM) → CV box-snap → Assembly(DOM) → Cross-page → Crop+Overlay
```
- **Top-down**: VLM khoanh vùng theo ngữ nghĩa; OCR KHÔNG quyết định khung.
- **Detector-agnostic**: đổi backend `vllm`↔`ollama` (và sau này YOLO) không sửa pipeline.

## Chạy nhanh
```bash
conda activate exam_parser_vlm
cp .env.example .env
# (terminal riêng) vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ --gpu-memory-utilization 0.55 ...
python scripts/parse_cli.py input/de_thi.pdf
# xem output/{exam_id}/overlay/*.png và crops/
```

## Phạm vi hiện tại
Tới **Phase 3** (pipeline + crop local, CLI) để kiểm độ chính xác. Service/MinIO/API làm sau.

## Cấu trúc
```
src/
  core/        config, logging
  schemas/     geometry, region (detector output), exam (DOM)
  detectors/   base, prompt+schema, vLLM backend, Ollama backend, factory
  services/    preprocess, box_snap, assembly, crosspage, cropper, pipeline
scripts/parse_cli.py
input/  output/
```
