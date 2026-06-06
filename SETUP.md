# SETUP — exam_parser_vlm

Hướng dẫn cài đặt đầy đủ: môi trường, thư viện, model VLM (vLLM), và cách chạy thử.
Phần cứng giả định: **RTX 5090 32GB**, WSL2 Ubuntu. Conda env: **`exam_parser_vlm`**.

> Bối cảnh VRAM: ~13GB đang bị 1 model phía Windows chiếm → WSL còn ~19GB. Vì vậy POC dùng
> **Qwen3-VL-8B-AWQ** (vừa GPU, chạy 100% GPU). Xem `PLAN_GROUNDING.md §6.1`.

---

## 1. Conda env

```bash
conda create -n exam_parser_vlm python=3.11 -y
conda activate exam_parser_vlm
```

> KHÔNG cài `torch` thủ công ở đây. Pipeline parse KHÔNG dùng torch; chỉ vLLM dùng — và để
> **vLLM tự kéo torch đúng CUDA** (xem §3) để tránh lệch `libcudart`.

## 2. Thư viện pipeline

```bash
cd clode_azota
pip install -r requirements.txt
```

## 3. vLLM + model Qwen3-VL-8B-AWQ

vLLM serve chạy **tiến trình riêng** (1 terminal), pipeline gọi qua HTTP.

> ⚠️ QUAN TRỌNG — tránh lỗi `libcudart.so.13: cannot open shared object file`:
> torch và vLLM phải **cùng một phiên bản CUDA**. Cách chắc nhất là để **`uv` tự dò** theo driver,
> KHÔNG cài torch cu128 thủ công trước rồi mới `pip install vllm` (sẽ lệch CUDA).

```bash
# xem driver hỗ trợ CUDA tối đa (góc trên phải)
nvidia-smi

# nếu đã lỡ cài torch/vllm trước đó, gỡ sạch cho khỏi lệch
pip uninstall -y vllm torch torchvision xformers

# cài vLLM + torch khớp CUDA tự động
pip install uv
uv pip install vllm --torch-backend=auto

# kiểm tra torch thấy GPU 5090
python -c "import torch; print(torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

- Nếu dòng trên in `True` + `NVIDIA GeForce RTX 5090` → OK.
- Nếu in `False`: driver quá cũ cho CUDA mặc định của vLLM → ép cu128:
  `uv pip install vllm --torch-backend=cu128`
- Không muốn dùng `uv`: thay 2 lệnh cài bằng `pip install vllm` (vLLM tự kéo torch khớp với nó).

Model tải tự động lần serve đầu (HuggingFace). Mạng chậm thì set: `export HF_HOME=~/.cache/huggingface`

### 3.1 Khởi động vLLM (TERMINAL RIÊNG)

> **Cấu hình ĐÃ CHẠY ĐƯỢC trên 5090 với ~17.8GB free (Windows chiếm ~13.5GB).**
> Dùng **4B-FP8** vì 8B-FP8 (~10GB) + activation vision profiling vượt 17.8GB free → KV cache âm.
> Lệnh **1 dòng** (KHÔNG dán kèm comment, sẽ vỡ nối dòng):

```bash
conda activate exam_parser_vlm
pkill -f "vllm serve"; sleep 3                 # dọn tiến trình cũ còn giữ VRAM
export VLLM_USE_FLASHINFER_SAMPLER=0           # Blackwell: tắt FlashInfer sampler (báo nhầm "requires sm75")
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --gpu-memory-utilization 0.55 --max-model-len 16384 --enforce-eager --max-num-seqs 1 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":2304000}' --port 8001
```

Cập nhật `.env` cho khớp:
```
VLM_BASE_URL=http://localhost:8001/v1
VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct-FP8
VLM_MAX_PIXELS=2304000
```

Giải thích các cờ (đều cần để vượt qua chuỗi lỗi đã gặp):
- `VLLM_USE_FLASHINFER_SAMPLER=0` — **bắt buộc trên Blackwell sm_120**: FlashInfer báo nhầm
  `requires sm75` → tắt sampler này, dùng sampler native.
- `--enforce-eager` — **bắt buộc khi VRAM hẹp**: bỏ CUDA graph (ngốn nhiều GB) → dành cho KV cache.
  Thiếu cờ này → lỗi `No available memory for the cache blocks` (KV cache âm).
- `--gpu-memory-utilization 0.55` — 0.55×32≈17.6GB, vừa khít 17.8GB free. ĐỪNG đẩy 0.6 (=19.6GB > free → OOM).
- `--mm-processor-kwargs '{"max_pixels":2304000}'` — chặn độ phân giải ảnh để activation vừa VRAM;
  4B còn dư VRAM nên để ~2.3M (cao) giúp model nhỏ nhìn rõ marker → bù độ chính xác grounding.
- `--max-num-seqs 1` — xử lý 1 trang/lượt, không cần batch.
- `--max-model-len 16384` — phải đủ cho vision tokens + prompt + **output JSON của trang nhiều câu**
  (đề Tiếng Anh ~250 region). Thấp quá (vd 4096) → JSON bị cắt → lỗi `Expecting ',' delimiter`.
- `--limit-mm-per-prompt '{"image":1}'` — mỗi prompt 1 ảnh (bản vLLM mới yêu cầu JSON).

Kiểm tra server sống (đúng port 8001):
```bash
curl http://localhost:8001/v1/models
```

> ⚠️ Cảnh báo `SM 12.x requires CUDA >= 12.9` trong log: torch hiện là CUDA 12.8, hơi thiếu cho
> Blackwell. Không chặn chạy, nhưng nếu sau này gặp lỗi kernel lạ thì nâng torch/vLLM lên build CUDA 12.9+.

### 3.2 Nâng cấp model khi có thêm VRAM (grounding tốt hơn — KHÔNG cần sửa code)

4B chỉ là model bootstrap cho VRAM hiện tại. Khi **giải phóng được 13GB phía Windows** (tắt model
kia), có ~30GB → lên model lớn để grounding chính xác hơn:

```bash
# 8B-FP8 (cần ~free 16GB+)
vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --gpu-memory-utilization 0.6 --max-model-len 4096 --enforce-eager --max-num-seqs 1 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":2304000}' --port 8001

# 32B-FP8 (cần ~free 35GB+ → phải giải phóng hết Windows; có thể bỏ --enforce-eager nếu dư VRAM)
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 --gpu-memory-utilization 0.9 --max-model-len 8192 --limit-mm-per-prompt '{"image":1}' --port 8001
```
Rồi chỉ đổi `VLM_MODEL` trong `.env` (KHÔNG sửa code). Nếu repo `-FP8` không có thì thử `-Instruct`
(BF16, nặng hơn) hoặc kiểm tra tên qua lệnh list ở §3.

> So sánh nhanh theo VRAM: 4B-FP8 ~5GB · 8B-FP8 ~10GB · 32B-FP8 ~33GB (cộng activation + KV).

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
curl http://localhost:8001/v1/models                          # vLLM sống (port 8001)
python scripts/parse_cli.py input/<đề>.pdf                     # ra output/
```

## 8. Sự cố thường gặp (đã gặp & cách xử lý — theo đúng thứ tự lúc dựng)
| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `libcudart.so.13: cannot open shared object file` | torch (cu128) lệch CUDA với vLLM (cu13) | gỡ hết rồi `uv pip install vllm --torch-backend=auto` (§3) |
| `--max-model-len: command not found` | dán lệnh kèm comment `# ...` sau `\` → vỡ nối dòng | dùng lệnh 1 dòng ở §3.1 |
| `--limit-mm-per-prompt: Value image=1 cannot be converted` | bản vLLM mới cần JSON | `--limit-mm-per-prompt '{"image":1}'` |
| `Address already in use` | tiến trình vLLM cũ còn sống | `pkill -f "vllm serve"` hoặc `fuser -k 8001/tcp`, hoặc đổi `--port` |
| `RepositoryNotFound / 401` model | tên repo sai (bản `-AWQ` không có) | dùng `-FP8`; list repo bằng lệnh ở §3 |
| `FlashInfer requires GPUs with sm75 or higher` | FlashInfer chưa nhận Blackwell sm_120 (báo nhầm) | `export VLLM_USE_FLASHINFER_SAMPLER=0`; hoặc `pip uninstall -y flashinfer-python` |
| `No available memory for the cache blocks` (KV cache âm) | VRAM hẹp + CUDA graph/vision activation ngốn hết | `--enforce-eager` + `--mm-processor-kwargs '{"max_pixels":...}'` + giảm model về 4B (§3.1) |
| `Expecting ',' delimiter` khi parse JSON (đề nhiều câu) | output JSON vượt context → bị cắt cụt | tăng `--max-model-len` (vd 16384); xem `vlm_raw/*.json` thấy JSON đứt cuối |
| `SM 12.x requires CUDA >= 12.9` (warning) | torch CUDA 12.8 hơi thiếu cho Blackwell | thường bỏ qua được; lỗi kernel lạ thì nâng torch/vLLM build CUDA 12.9+ |
| `CUDA error: no kernel image` | torch không hỗ trợ sm_120 | cài torch CUDA mới qua vLLM (§3), không pin cu cũ |
| Kết nối `localhost:8000` fail | chưa `vllm serve` | mở terminal riêng chạy §3.1 |
| VLM trả JSON sai schema | model nhỏ/độ phân giải thấp | tăng `VLM_MAX_PIXELS`, hoặc đổi model lớn hơn |
| Box lệch nhiều | độ phân giải gửi VLM thấp | tăng `VLM_MAX_PIXELS`; bật `SNAP_ENABLED=true` |
