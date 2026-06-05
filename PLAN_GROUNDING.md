# Kế hoạch Production: Exam Layout Parser (grounding-first, Azota-style)

**Ngày**: 2026-06-05 (cập nhật) · **Phần cứng**: RTX 5090 32GB (WSL2) · **Mục tiêu cuối**: ~1000 đề/ngày
**Bài toán**: upload PDF/ảnh đề thi → tự động **khoanh chính xác câu hỏi & đáp án** (kèm gom **nhóm
câu hỏi**, bóc **header/metadata**) → crop ảnh lưu MinIO → FE lấy ra dưới dạng base64.

> Thay hướng `exam_parser_paddle` (OCR→regex→snake walker: chậm vì Paddle chạy CPU trên Blackwell,
> sai vì khung sinh từ OCR). Bài học chi tiết: `../exam_parser_paddle/CHANGELOG.md`.

---

## 0. Phạm vi (đã chốt với người dùng) — ĐỌC TRƯỚC

**LÀM** (detect & khoanh):
- ✅ **Câu hỏi** (stem) và **đáp án** (A/B/C/D, đúng/sai, ...) → đơn vị crop chính.
- ✅ **Nhóm câu hỏi**: `section` (Phần I/II — đề Toán) và `group` (instruction/passage — đề Tiếng Anh).
  Nhóm **có thể trải nhiều trang** → phải gom đúng.
- ✅ **Header/metadata** đầu đề (ảnh người dùng gửi: Sở GD / trường / môn / thời gian / số câu).
- ✅ **Vắt trang**: câu/nhóm dài bị cắt qua ≥2 trang → gom lại, ghép ảnh.
- ✅ (optional) **Bảng đáp án cuối đề** → nếu cần chấm đúng/sai.

**KHÔNG LÀM** (cố ý bỏ để đơn giản, theo yêu cầu):
- ❌ KHÔNG detect riêng `formula`, `table`, `figure`. Hình/đồ thị/công thức **nằm sẵn trong vùng
  crop** của câu → cứ khoanh trọn câu là có cả hình. Không phân loại "câu này có công thức/hình".
- ❌ KHÔNG cần OCR chính xác nội dung để phục vụ FE (FE dùng **ảnh crop**, không dùng text).
  OCR chỉ chạy phụ cho header/metadata + (tùy chọn) search.

> Hệ quả thiết kế: vùng crop của câu phải **bao trọn mọi nội dung** giữa câu này và câu kế (gồm
> hình/công thức) — đây chính là điều CV box-snap (§5.1) đảm bảo. "Không detect figure riêng"
> KHÔNG có nghĩa cắt mất hình; nghĩa là không tách hình thành node riêng.

---

## 1. Nguyên tắc kiến trúc (phần đúng — giữ nguyên)

1. **Top-down, không bottom-up**: detect *vùng* trước; KHÔNG dựng câu từ mảnh OCR.
2. **Khoanh vùng tách rời OCR**: khung quyết định bởi detector + CV, không bởi nội dung chữ.
3. **Một hợp đồng `RegionDetector` duy nhất, nhiều backend**: VLM (POC) và YOLO/RT-DETR (production)
   là 2 cài đặt cùng interface. Pipeline phía sau (snap, assembly, crop, serve) **detector-agnostic**
   → đổi backend không viết lại pipeline.

---

## 2. Exam Document Object Model (DOM) — tinh gọn

Cây tài liệu chỉ còn các nút thực sự cần cho FE:

```
ExamDocument
├── header        ExamHeader   ← khối metadata {so_gd, truong, mon, thoi_gian, ma_de, so_cau}
├── sections[]    Section      ← "PHẦN I / II" (đề Toán). Có thể vắt trang.
│                                {title, page_span[], question_ids[]}
├── groups[]      Group        ← cụm dùng chung (Tiếng Anh: instruction + passage). Vắt trang.
│                                {kind: instruction|passage, region, page_span[], question_ids[]}
└── questions[]   Question
        ├── number, type, confidence, page_span[], needs_review
        ├── section_id?, group_id?       ← thuộc nhóm nào (null nếu đứng lẻ)
        ├── full_region    ← khoanh TRỌN câu (stem + đáp án + hình nếu có) → ảnh FE dùng chính
        ├── stem_region    ← (optional) chỉ phần đề bài
        └── answers[]  Answer{label, region}   ← từng đáp án khoanh riêng
```

Không còn `figures[]`, `tables[]`, `formula`, `blanks[]` như bản trước — theo yêu cầu đơn giản hoá.
Hình/công thức nằm trong `full_region`/`answer.region`, không tách.

### 2.1 Region taxonomy (class detector phải nhận) — RÚT GỌN
| Class | Mô tả | Xử lý |
|---|---|---|
| `exam_header` | Khối tiêu đề/metadata đầu đề | OCR → parse metadata |
| `section_header` | "PHẦN I. ...", "PHẦN II. ..." | mốc chia section (gom câu) |
| `group_instruction` | "Đọc đoạn sau trả lời câu 1-5", "Mark the letter..." | tạo/gắn group |
| `passage` | Đoạn văn đọc hiểu dùng chung | crop riêng, thuộc group |
| `question` | Khối 1 câu hỏi (stem) | đơn vị chính |
| `answer_option` | 1 ô đáp án A/B/C/D (hoặc a/b/c/d) | crop riêng, gắn câu |
| `answer_key` | Bảng đáp án cuối đề (1.C 2.A...) | (optional) parse đáp án đúng |
| `footer`/`page_number` | Chân trang | bỏ khi crop (lọc nhiễu) |

→ **5 lớp lõi**: header, section_header, group_instruction/passage, question, answer_option.
KHÔNG có figure/table/formula. Ít lớp hơn ⇒ detector dễ học, prompt VLM gọn, ít lỗi.

---

## 3. Pipeline (detector-agnostic)

```
INPUT (PDF/JPG/PNG)
  │
  ▼ [1] PREPROCESS        render 300DPI + deskew → list[Page image]
  │
  ▼ [2] REGION DETECTION  RegionDetector.detect(page) → list[Region{class,bbox,score, +meta}]
  │                       meta: số câu, nhãn đáp án, dải câu của group, cờ vắt trang
  │                       backend = VLM (POC) | YOLO (production)                  ← §4
  │
  ▼ [3] CV BOX-SNAP       Ép mỗi bbox khít ink thật (projection profile, OpenCV).
  │                       Câu: snap "bao trọn" trong band → gồm cả hình/công thức.  ← §5.1
  │
  ▼ [4] ASSEMBLY (DOM)    Reading-order + gom nhóm:
  │                       - answer_option → gắn question chứa nó
  │                       - question → gắn group/section theo dải số & vị trí       ← §5.2
  │                       - section/group span nhiều trang
  │
  ▼ [5] CROSS-PAGE MERGE  Câu/nhóm vắt ≥2 trang → 1 node, page_span=[N..N+k], stitch dọc ← §5.3
  │
  ▼ [6] VALIDATE / QA     liên tục số câu, đủ đáp án, header tồn tại → confidence/needs_review ← §7
  │
  ▼ [7] CROP              Crop full_region + từng answer (+ passage/header) → WEBP. Overlay review.
  │
  ▼ [8] PERSIST           Upload MinIO (ảnh) + ghi DB (DOM JSON, key MinIO, status)  ← §6
  │
  ▼ FE đọc qua API → API lấy ảnh từ MinIO → trả base64                              ← §6
```

`[3]→[8]` không biết detector là VLM hay YOLO.

---

## 4. Hai backend (cùng interface) — POC dùng VLM, KHÔNG dùng YOLO

> **YOLO pretrained KHÔNG đủ cho POC.** YOLO/DocLayout pretrained chỉ biết lớp generic
> (text/title/figure/table) — không biết "đâu là 1 câu hỏi / đáp án / section / group / header".
> Muốn YOLO khoanh theo taxonomy §2.1 phải **fine-tune trên data có nhãn** — mà data đó chính là
> thứ POC sinh ra. Vì vậy:

### 4.A — VLM grounding (POC — triển khai NGAY, 0 data) ✅ dùng cho bây giờ
- **1 lượt/trang**: ảnh trang → JSON mọi region {class, box 0-1000, số câu, nhãn đáp án,
  dải câu group, cờ vắt trang}. Prompt liệt kê từ trên xuống, KHÔNG giải/chép nội dung.
- Vai trò: chạy ngay khi chưa có model; đồng thời **sinh nhãn** cho 4.B.
- Box thô → **CV snap (§5.1) bù pixel**.
- **2 backend cùng interface** (chọn qua env `DETECTOR_BACKEND`), xem §6.1:
  - `vllm` (KHUYẾN NGHỊ): chạy 100% GPU, kiểm soát độ phân giải + processor Qwen → grounding tốt hơn.
  - `ollama`: dễ dựng, nhưng nếu model không vừa VRAM sẽ **split sang CPU → chậm**.

### 4.B — Layout detector (PRODUCTION — sau POC, khi đã có data)
- Fine-tune RT-DETR / YOLOv11 trên taxonomy §2.1 (~300-500 trang, bootstrap từ 4.A + review).
- ~30-80ms/trang trên 5090 → tốc độ Azota. VLM lùi về fallback ca khó.

```
POC: 100% VLM ──chạy + review──▶ tích nhãn ──▶ train YOLO ──▶ YOLO chủ lực + VLM fallback
```

---

## 5. Thuật toán then chốt

### 5.1 CV Box-Snap (lõi độ chính xác, gồm cả hình)
- Binarize trang (Otsu inverted) → mask ink. Mỗi bbox: cắt khoảng trắng rìa theo row/col ink-profile,
  pad nhẹ (~8px), **chặn trong band** [đỉnh câu N, đỉnh câu N+1) → không nuốt câu kế.
- **Câu có hình**: vì band kéo từ đỉnh câu đến đỉnh câu kế và snap theo *bao ngoài toàn bộ ink trong
  band* (không trim khoảng trắng giữa), đồ thị/hình nằm giữa stem và đáp án **được bao trọn tự động**.
  → đáp đúng "câu có hình phải cắt cả hình" mà KHÔNG cần detect figure riêng.

### 5.2 Assembly: gom nhóm (Section & Group) — phần được nhấn mạnh
- **Reading order**: phát hiện 1 cột / 2 cột (phân bố x của các `question`) → sort theo cột rồi y.
- **answer_option → question**: gắn vào câu mà nó nằm trong band.
- **Section (Phần I/II)**: mỗi `section_header` mở một section; mọi câu sau nó (tới section kế)
  thuộc section đó. Đề Toán mới: Phần I trắc nghiệm, Phần II đúng/sai, Phần III điền đáp án...
- **Group (Tiếng Anh)**: `group_instruction`/`passage` đứng trên một dải câu → tạo Group; gán
  `question_ids` theo **dải số trong instruction** ("Questions 1-5", "câu 1 đến câu 5") nếu đọc được,
  fallback theo **vùng phủ vị trí** (các câu nằm dưới passage tới instruction/section kế).
- Câu đứng lẻ: `section_id=group_id=null`.

### 5.3 Cross-page merge — CÂU và NHÓM đều có thể vắt trang
- **Câu vắt trang**: cờ `continues_to_next`/`continues_from_prev` (VLM tự đánh giá khi nhìn mép
  trang) **hoặc** câu cuối trang N thiếu đáp án + đầu trang N+1 không có số câu mới → gộp 1 node,
  `page_span=[N..N+k]`, crop từng phần **stitch dọc**.
- **Nhóm vắt trang** (yêu cầu mới): passage/instruction ở trang N, các câu của nhóm rải sang trang
  N+1, N+2 → Group có `page_span` nhiều trang; `question_ids` gom xuyên trang theo dải số. Section
  tương tự (Phần II bắt đầu trang 3, kéo hết trang 4-5).
- Tín hiệu nhóm xuyên trang: dải số trong instruction/section + tính liên tục số câu giữa các trang.

### 5.4 Header & answer-key
- `exam_header`: ở đầu trang 1 → OCR → regex parse {sở, trường, môn, thời gian, mã đề, số câu}.
  Không nhầm thành câu hỏi (khác class).
- `answer_key` (optional): bảng "1.C 2.A..." cuối đề → map đáp án đúng (điền `is_correct`).

---

## 6. Service & cách FE lấy ảnh (theo mô tả người dùng)

```
                ┌──────────── FastAPI (stateless) ────────────┐
 FE ───POST────▶│ POST /v1/exams           (upload) → 202 +id │
                │ GET  /v1/exams/{id}      → DOM JSON          │
                │ GET  /v1/exams/{id}/status                  │
                │ GET  /v1/images?key=...  → base64 (đọc MinIO)│ ◀── FE gọi khi cần hiển thị ảnh
                │ PATCH /v1/regions/{...}  (human review)     │
                └───────────────┬─────────────────────────────┘
                        enqueue │ (Celery + Redis)
                   ┌────────────▼────────────┐
                   │ Worker (GPU): Pipeline §3│──ảnh crop──▶ MinIO (exams/{id}/q{n}_full.webp ...)
                   │ detector = VLM | YOLO    │──DOM JSON──▶ Postgres
                   └─────────────┬────────────┘
                                 └── VLM qua Ollama (POC) / vLLM (prod)
```

**Luồng ảnh (đúng yêu cầu)**:
1. Worker crop câu/đáp án → **upload MinIO**, lưu **object key** vào DOM (`questions[].full_region.key`).
2. DOM JSON trả cho FE chỉ chứa **key/metadata**, KHÔNG nhúng ảnh (JSON nhẹ).
3. Khi FE cần render 1 câu → gọi `GET /v1/images?key=exams/{id}/q5_full.webp` → API **đọc object
   từ MinIO → encode base64 → trả về** (`{ "mime": "image/webp", "data": "<base64>" }`).
   - Tùy chọn: cache base64 (Redis/CDN) cho ảnh hay xem; hoặc dùng presigned URL nếu sau này muốn
     FE tải thẳng. POC dùng base64-qua-API như mô tả.

**Vận hành**: status `queued|processing|done|failed` + retry; structured logs; health check
GPU/Ollama/MinIO/DB; lưu `model_version`+`schema_version` mỗi đề để tái xử lý.

---

## 6.1 Serving VLM: vLLM (khuyến nghị) vs Ollama + VRAM trên GPU chia sẻ

**Bối cảnh máy hiện tại**: RTX 5090 32GB, nhưng **~13GB đã bị 1 model phía Windows chiếm** →
WSL chỉ còn **~19GB** dùng được.

### Khác biệt cốt lõi
- **Ollama**: hỗ trợ *partial offload* — model không vừa VRAM thì **đẩy bớt layer xuống CPU** →
  vẫn chạy nhưng **chậm** (đây là cái chậm đang gặp với 32B).
- **vLLM**: *all-or-nothing trên GPU* — model **vừa** thì chạy 100% GPU (nhanh); **không vừa** thì
  **OOM, không khởi động** (không tự tụt xuống CPU).

### Hệ quả với ~19GB trống
| Cấu hình | Vừa GPU? | Kết quả |
|---|---|---|
| Ollama + 32B | Không → split CPU | chậm |
| vLLM + 32B (AWQ ~18-20GB weights + KV/vision) | Không | **OOM** |
| **vLLM + Qwen3-VL-8B-AWQ** (~9-12GB) | **Vừa** | **nhanh, 100% GPU** ✅ |

→ **Chốt cho POC: `vllm` + `Qwen3-VL-8B-Instruct-AWQ`**, full-GPU trong 19GB, đẩy `max_pixels` cao
để bù model nhỏ. (8B-full-GPU + ảnh phân giải cao thường **hơn** 32B-split-CPU cả tốc độ lẫn box.)

### Lệnh khởi động vLLM (chi tiết trong SETUP.md)
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ \
  --gpu-memory-utilization 0.55 \      # CHỪA 13GB của Windows — KHÔNG để vLLM chiếm hết
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --port 8000
```
- `--gpu-memory-utilization 0.55` ≈ 0.55×32GB ≈ 17.6GB → nằm gọn trong 19GB trống, không đụng Windows.
- Khi giải phóng được 13GB (tắt model Windows) → nâng util + đổi sang **32B-AWQ** mà không sửa code
  (chỉ đổi `VLM_MODEL` trong `.env`).
- Pipeline gọi vLLM qua **OpenAI-compatible API** (`/v1/chat/completions`) + `guided_json` (structured).

---

## 7. QA gates & confidence
Đẩy câu vào review nếu: số câu không liên tục / ≠ `so_cau` header; MCQ ≠ số đáp án kỳ vọng;
band chồng lấn bất thường (IoU); câu hoặc nhóm vắt trang; detector score thấp.
Qua hết gate = auto-accept. Còn lại ra review UI.

---

## 8. Human-in-the-loop (QA + sinh data train 4.B)
Web UI: overlay bbox trên ảnh; admin sửa box/class/số câu/gom nhóm. Mỗi sửa lưu
`(ảnh trang, bbox đúng, class)` → dataset train YOLO. Active-learning: ưu tiên trang detector phân vân.

---

## 9. JSON output cho FE (rút gọn, chỉ key MinIO + cấu trúc nhóm)

```jsonc
{
  "exam_id": "uuid", "n_pages": 5,
  "model_version": "vlm-grounding-0.1", "schema_version": "1.0",
  "header": { "key": "exams/uuid/header.webp", "page": 0,
              "so_gd": "Hải Dương", "truong": "THPT Đoàn Thượng",
              "mon": "Toán", "thoi_gian_phut": 90, "ma_de": null, "so_cau": 50 },
  "sections": [ { "id":"s1","title":"PHẦN I","page_span":[0,1],"question_ids":["q1","..."] },
                { "id":"s2","title":"PHẦN II","page_span":[2,3],"question_ids":["q26","..."] } ],
  "groups":   [ { "id":"g1","kind":"passage","key":"exams/uuid/g1_passage.webp",
                  "page_span":[3,4],"question_ids":["q31","q32","q33","q34","q35"] } ],
  "questions": [
    { "id":"q31","number":31,"type":"mcq_single","section_id":"s2","group_id":"g1",
      "page_span":[3,4],"confidence":0.92,"needs_review":true,        // vắt trang
      "full_region": { "key":"exams/uuid/q31_full.webp","bbox":[...],"page_span":[3,4] },
      "answers":[ {"label":"A","key":"exams/uuid/q31_A.webp","bbox":[...],"is_correct":null},
                  {"label":"B","key":"exams/uuid/q31_B.webp","bbox":[...],"is_correct":null} ] }
  ],
  "answer_key": { "q1":"C", "q2":"A" },          // optional
  "preview_pdf_key": "exams/uuid/preview.pdf",
  "stats": { "n_questions":50,"n_sections":3,"n_groups":1,"n_cross_page":2,"avg_confidence":0.91 }
}
```
→ FE đọc JSON này, với mỗi `key` gọi `/v1/images?key=...` để lấy base64 khi cần hiển thị.

---

## 10. Roadmap
| Phase | Nội dung | Output |
|---|---|---|
| **P1 — Lõi VLM (1-2 tuần)** | Preprocess + `RegionDetector(VLM)` + CV-snap + Assembly (gồm **gom section/group, vắt trang**) + Crop + overlay. CLI. Chạy 3 đề mẫu. | Khoanh đúng câu/đáp án, gom nhóm, header, vắt trang |
| **P2 — DOM + QA (1-2 tuần)** | section/group/passage/header parse hoàn chỉnh; answer_key (optional); validate + confidence gates | DOM + needs_review chuẩn |
| **P3 — Service (1-2 tuần)** | FastAPI + Celery/Redis + Postgres + MinIO; `/images`→base64; status/retry; preview | API như Azota |
| **P4 — Review UI + data (2-3 tuần)** | Web review, lưu correction → dataset | data train 4.B |
| **P5 — YOLO production (2-3 tuần)** | Train RT-DETR/YOLO; backend chính, VLM fallback; vLLM | ~30-80ms/trang, 1000 đề/ngày |
| **P6 — Hardening** | auth, rate-limit, monitoring, docker-compose, batch | production |

---

### 10.1 Phạm vi build lần này (tương đương "Phase 3" của project paddle)
Mục tiêu: **chạy local bằng CLI để kiểm độ chính xác crop** — CHƯA dựng service/MinIO/API.

**Làm trong lần này** (đủ để test crop):
- Input: **PDF nhiều trang** hoặc **1 file ảnh** (`.png/.jpg/.jpeg/.bmp/.tif/.webp`).
- Pipeline §3 stage [1]→[7]: Preprocess → `RegionDetector` (vLLM/Ollama) → CV box-snap →
  Assembly (câu/đáp án + gom section/group) → Cross-page merge → Crop + overlay.
- Output local: `output/{exam_id}/{exam.json, crops/*.png, overlay/*.png, summary.txt, vlm_raw/}`.
- Backend chọn qua `.env` (`DETECTOR_BACKEND=vllm|ollama`).

**Để SAU khi crop đạt** (full hệ thống): MinIO upload, FastAPI `/v1/exams` + `/v1/images`→base64,
Celery/Redis, Postgres, review UI, train YOLO.

> Conda env: **`exam_parser_vlm`**. Cài đặt: xem `SETUP.md`.

---

## 11. Vì sao bản này hợp yêu cầu mới
1. **Gom nhóm là hạng nhất**: section (Phần I/II) + group (instruction/passage), **vắt nhiều trang** (§5.2-5.3).
2. **Đơn giản hoá**: chỉ 5 lớp lõi, BỎ figure/table/formula — nhưng câu có hình vẫn cắt trọn hình (§5.1).
3. **Output = ảnh crop trong MinIO**; FE hỏi → API đổi sang base64 (§6, §9).
4. **POC = VLM** (YOLO chưa train không đủ); detector-agnostic nên lên production đổi backend không gãy.

---

## 12. Quyết định cần duyệt trước khi code P1
1. DOM §2 (tinh gọn, nhấn nhóm) + taxonomy 5 lớp §2.1 — OK chứ?
2. Serving §6 (MinIO + `/images`→base64) đúng ý FE chưa? Có cần thêm presigned URL không?
3. Bảng đáp án cuối đề: cần bóc `answer_key` ngay ở POC hay để sau?
4. Bắt đầu code P1 (backend VLM) khi bạn sẵn sàng?
```
