# SETUP — exam_parser_vlm

Hướng dẫn cài đặt đầy đủ: môi trường, thư viện, model VLM (vLLM), và cách chạy thử.
Phần cứng giả định: **RTX 5090 32GB**, WSL2 Ubuntu. Conda env: **`exam_parser_vlm`**.

> Bối cảnh VRAM: ~13GB đang bị 1 model phía Windows chiếm → WSL còn ~19GB. Vì vậy POC dùng
> **Qwen3-VL-8B-AWQ** (vừa GPU, chạy 100% GPU). Xem `PLAN_GROUNDING.md §6.1`.

---

## 1. Conda env + PyTorch (CUDA 12.8 cho Blackwell sm_120)

```bash
conda create -n exam_parser_vlm python=3.11 -y
conda activate exam_parser_vlm

# PyTorch CUDA 12.8 — BẮT BUỘC cho RTX 5090 (Blackwell sm_120).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Kiểm tra GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Thư viện pipeline

```bash
cd clode_azota
pip install -r requirements.txt
```

## 3. vLLM + model Qwen3-VL-8B-AWQ

vLLM serve chạy **tiến trình riêng** (1 terminal), pipeline gọi qua HTTP. Cài vLLM:

```bash
# vLLM bản hỗ trợ CUDA 12.8 / Qwen3-VL. Nếu pip thường lỗi, dùng wheel cu128:
pip install -U vllm
# (nếu cần) pip install -U "transformers>=4.49" accelerate
```

Tải model tự động ở lần serve đầu (từ HuggingFace). Nếu mạng chậm, set HF cache:
```bash
export HF_HOME=~/.cache/huggingface
```

### 3.1 Khởi động vLLM (TERMINAL RIÊNG, để chạy nền)

```bash
conda activate exam_parser_vlm
vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ \
  --gpu-memory-utilization 0.55 \      # chừa ~13GB của Windows; 0.55*32 ≈ 17.6GB
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --port 8000
```

Giải thích cờ:
- `--gpu-memory-utilization 0.55`: **quan trọng** — vLLM mặc định chiếm 90% GPU sẽ đụng 13GB của
  Windows → OOM. Hạ xuống ~0.55 để nằm gọn trong 19GB trống. Nếu vẫn OOM, hạ tiếp 0.5/0.45.
- `--limit-mm-per-prompt image=1`: mỗi prompt 1 ảnh (ta gửi 1 trang/lượt) → tiết kiệm KV cache.
- `--max-model-len 8192`: đủ cho 1 trang + JSON output.

Kiểm tra server sống:
```bash
curl http://localhost:8000/v1/models
```

### 3.2 (Khi đã giải phóng được 13GB) nâng lên 32B
```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct-AWQ \
  --gpu-memory-utilization 0.9 --max-model-len 8192 --limit-mm-per-prompt image=1 --port 8000
# rồi đổi VLM_MODEL trong .env → Qwen/Qwen3-VL-32B-Instruct-AWQ  (KHÔNG cần sửa code)
```

## 4. (Tùy chọn) Backend Ollama thay cho vLLM

Nếu chưa muốn dựng vLLM, dùng Ollama (chấp nhận chậm hơn nếu model lớn):
```bash
ollama pull qwen3-vl:8b          # hoặc qwen3-vl:32b-instruct (sẽ split CPU nếu thiếu VRAM)
# .env: DETECTOR_BACKEND=ollama ; OLLAMA_MODEL=qwen3-vl:8b
```

## 5. Cấu hình project

```bash
cp .env.example .env
# Sửa .env nếu cần (mặc định đã trỏ vLLM 8B ở localhost:8000)
```

## 6. Chạy thử (Phase 3 — kiểm crop)

```bash
conda activate exam_parser_vlm

# PDF nhiều trang
python scripts/parse_cli.py input/de_thi.pdf

# 1 file ảnh
python scripts/parse_cli.py input/trang1.jpg

# Tùy chọn
python scripts/parse_cli.py input/de.pdf --backend ollama   # ép backend
python scripts/parse_cli.py input/de.pdf --no-snap          # tắt box-snap để so sánh
python scripts/parse_cli.py input/de.pdf --debug            # log chi tiết
```

Output tại `output/{exam_id}/`:
```
exam.json        # cấu trúc DOM (câu/đáp án/section/group/header) + đường dẫn ảnh
crops/           # q{N}_full.png, q{N}_{A..}.png, q{N}_stem.png, g{k}_*.png, header.png
overlay/         # page_xx.png — bbox màu chồng lên ảnh gốc để soát mắt thường
vlm_raw/         # JSON thô VLM trả về mỗi trang (debug)
summary.txt      # tóm tắt số câu/section/group/cross-page
```

**Cách kiểm độ chính xác**: mở `overlay/page_*.png` xem khung có khít + đủ câu/đáp án + bao trọn
hình không; mở `crops/` xem ảnh cắt ra. So với project paddle cũ.

## 7. Checklist nhanh
```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
curl http://localhost:8000/v1/models                          # vLLM sống (nếu dùng vllm)
python scripts/parse_cli.py input/<đề>.pdf                     # ra output/
```

## 8. Sự cố thường gặp
| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| vLLM OOM khi khởi động | util quá cao, đụng 13GB Windows | hạ `--gpu-memory-utilization` 0.5/0.45 |
| `CUDA error: no kernel image` | torch không phải cu128 | cài lại torch theta `--index-url .../cu128` |
| Kết nối `localhost:8000` fail | chưa `vllm serve` | mở terminal riêng chạy §3.1 |
| VLM trả JSON sai schema | model nhỏ/độ phân giải thấp | tăng `VLM_MAX_PIXELS`, hoặc đổi model lớn hơn |
| Box lệch nhiều | độ phân giải gửi VLM thấp | tăng `VLM_MAX_PIXELS`; bật `SNAP_ENABLED=true` |
