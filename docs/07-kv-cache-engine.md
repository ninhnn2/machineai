# 3. KV cache, attention, và chọn engine

## 3.1 KV cache: chi phí kép

KV cache tốn **hai** thứ, và người ta thường chỉ nhớ thứ nhất:

1. **RAM** — chiếm chỗ trong 8GB ít ỏi
2. **Bandwidth** — phải đọc lại **toàn bộ** cache ở mỗi token sinh ra

Điều (2) nghĩa là tok/s **giảm tuyến tính theo độ dài context**:

```
bytes/token = W_bytes + KV_bytes(ctx)
tok/s       = BW / bytes/token
```

Llama-3.1-8B Q4_K_M trên board này (dùng **66.8 GB/s đo thật**, xem
[MEASUREMENTS.md](09-so-do-phan-cung.md)):

| Context | KV (FP16) | bytes/token | Trần tok/s | với KV q8_0 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 4.8 GB | 13.9 | 13.9 |
| 4096 | 0.54 GB | 5.34 GB | 12.5 | 13.2 |
| 8192 | 1.07 GB | 5.87 GB | 11.4 | 12.6 |
| 16384 | 2.15 GB | 6.95 GB | 9.6 (và sát OOM) | 11.4 |

Đây là lý do chat dài chậm dần — không phải "model mệt", mà là roofline.

## 3.2 Các lever giảm KV

| Kỹ thuật | Giảm | Chi phí |
|---|---|---|
| **GQA** (đã có sẵn trong Llama-3/Qwen2.5) | 4-8× so với MHA | Có sẵn, không làm gì |
| **INT8 KV cache** | 2× RAM **và** 2× bandwidth | Chất lượng gần như không đổi |
| **INT4 KV cache** | 4× | Bắt đầu ảnh hưởng, cần đo |
| **Sliding window / StreamingLLM** | Chặn trần theo window | Mất context xa |
| **PagedAttention** | Giảm phân mảnh, không giảm KV thật | Cần vLLM/TRT-LLM |

Trên llama.cpp:
```bash
llama-server -m model.gguf -c 8192 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ngl 99 --flash-attn
```
`-ngl 99` (offload hết layer lên GPU) và `--flash-attn` là bắt buộc trên Jetson.

## 3.3 FlashAttention — vì sao nó quan trọng

Attention naive materialize ma trận `S = QK^T` kích thước `T×T` ra DRAM rồi đọc lại
cho softmax. Với T=4096: 4096² × 2 byte = 33 MB **ghi + đọc lại**, trong khi FLOPs không đổi.
→ AI thấp → memory-bound một cách vô nghĩa.

FlashAttention tile lại, giữ trong SRAM (shared memory), softmax online.
Không bao giờ ghi `S` ra DRAM. FLOPs giống hệt, bytes giảm hàng chục lần.

Đây là ví dụ giáo khoa: **cùng một thuật toán, chỉ đổi memory access pattern**.
Repo esp32 làm y hệt ở scale nhỏ — cache attention scores, tính RoPE 1 lần/token
(RESULTS.md:157).

## 3.4 Unified memory — đặc thù Jetson

CPU và GPU **dùng chung** physical RAM, không có PCIe.

Hệ quả:
- `cudaMemcpy(H2D)` trên Jetson thường là **copy vô nghĩa** — bạn copy RAM sang RAM,
  đốt băng thông 2 lần
- Dùng `cudaHostAlloc(..., cudaHostAllocMapped)` hoặc `cudaMallocManaged` để zero-copy
- Code viết cho desktop port thẳng sang Jetson hay chậm bất thường vì lý do này

Kiểm tra nhanh: profile bằng `nsys`, nếu thấy nhiều thời gian trong memcpy → đây là thủ phạm.

## 3.5 Chọn engine

| Engine | Ưu | Nhược | Dùng khi |
|---|---|---|---|
| **llama.cpp** (CUDA backend) | Dễ nhất, GGUF sẵn trên HF, server OpenAI-compatible | Không tối ưu bằng TRT-LLM | Mặc định. Bắt đầu ở đây |
| **TensorRT-LLM** | Nhanh nhất, in-flight batching, INT4 AWQ kernel | Build engine lâu, phụ thuộc phiên bản chặt | Production, cần throughput |
| **MLC-LLM** | TVM-based, portable tốt | Cộng đồng nhỏ hơn | Thử nghiệm |
| **TensorRT** (ONNX) | Chuẩn cho vision/ASR | Không dành cho LLM động | YOLO, ResNet, Whisper encoder |
| **ONNX Runtime + TRT EP** | Dễ tích hợp Python | Overhead cao hơn TRT thuần | Prototyping vision |

Với `jetson-containers` (dusty-nv) bạn có sẵn image cho hầu hết những cái trên — tiết kiệm
rất nhiều thời gian so với build tay trên aarch64.

## 3.6 Speculative decoding — phá trần roofline

Đây là kỹ thuật duy nhất thực sự **vượt** trần `BW/W_bytes` ở batch=1.

Ý tưởng: model nhỏ (draft) sinh K token nháp rẻ; model lớn verify cả K token trong **một**
lần forward. Vì một forward đọc weights đúng 1 lần dù xử lý K token → nếu chấp nhận được
trung bình `a` token, bạn nhận `a×` tốc độ với chi phí gần như không đổi.

**Đo thật trên board này chứng minh vì sao nó hoạt động:** GEMM `M×4096×4096` FP16
mất 0.512 ms ở M=1 và 0.544 ms ở M=64 — chênh 6%. Verify 8 token nháp cùng lúc gần
như không tốn thêm gì so với sinh 1 token.

Llama-3.2-1B draft + 8B target, `a ≈ 2-3` trên văn bản thường.
Đổi lại tốn thêm RAM cho draft model — trên 8GB thì phải cân nhắc (8B Q4 4.8GB +
1B Q4 0.7GB + KV = đã 6GB).

llama.cpp có `llama-speculative`. TRT-LLM hỗ trợ native (Medusa, EAGLE, draft-target).

## 3.7 Checklist hệ thống trên Jetson (làm trước mọi benchmark)

```bash
# 0. MÁY CÓ RẢNH KHÔNG? -- bước hay bị quên nhất
tegrastats --interval 1000   # GR3D_FREQ phải ~0%, CPU thấp, RAM đủ trống
ps -eo pcpu,rss,comm --sort=-pcpu | head

# 1. MAXN_SUPER = MODE 2 trên Orin Nano Super (KHÔNG phải 0 -- mode 0 là 15W!)
sudo nvpmodel -m 2
sudo jetson_clocks
nvpmodel -q                  # phải in "MAXN_SUPER"
sudo jetson_clocks --show    # xác nhận GPU=1020MHz và xem EMC freq thật

# 2. Khôi phục sau khi đo
sudo jetson_clocks --restore && sudo nvpmodel -m 0

# Tăng swap nếu build engine lớn (8GB dễ OOM khi build TRT)
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

`jtop` (`sudo pip3 install jetson-stats`) tiện hơn tegrastats nhưng không cài sẵn.

Ba thủ phạm thầm lặng, theo thứ tự mức độ tôi đã đo được trên chính board này:

| Thủ phạm | Mức sai lệch đo được | Cách phát hiện |
|---|---:|---|
| Tải nền chiếm GPU | **2.4×** | `tegrastats` → GR3D_FREQ > 0% khi tưởng rảnh |
| Sai power mode | **1.9×** | `nvpmodel -q` |
| Thermal throttle | ~1.3× (chưa đo) | `tegrastats` → tj@ tăng dần, freq tụt |

Thermal: benchmark chạy 30s thì đẹp, chạy 10 phút tụt. Luôn đo ở trạng thái ổn định
nhiệt và ghi lại nhiệt độ. Board này chạy nền đã ở 60-62°C.

> **Cạm bẫy khi tạm dừng tiến trình qua SSH:** đừng dùng `pkill -f <tên>`. Cờ `-f`
> khớp toàn bộ dòng lệnh, và dòng lệnh SSH của bạn cũng chứa chuỗi đó → bạn tự
> SIGSTOP chính phiên làm việc của mình. Dùng `pgrep -x <tên> | xargs kill -STOP`.

## 3.8 Bài tập

1. Đo tok/s ở ctx = 512 / 2048 / 8192 với cùng model. Vẽ đồ thị, so với công thức 3.1.
2. Bật/tắt `--cache-type-k q8_0`. Đo cả tok/s lẫn ppl. Kết luận có đáng đổi không.
3. Chạy benchmark 30 giây vs 10 phút liên tục, ghi tegrastats. Đo mức throttle thật của board.

→ Quay lại: [README.md](README.md)
