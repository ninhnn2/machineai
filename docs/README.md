# Học LLM và tối ưu cho nhiều kiến trúc phần cứng

Giáo trình dựa trên chính repo này. Điểm mạnh của nó: **cùng một model, cùng một
`model.bin`, chạy trên ba kiến trúc** — MCU (ESP32-S3), CPU (x86/ARM), GPU (Ampere).
Giữ nguyên bài toán, đổi phần cứng, xem cái gì chuyển giao được.

Mọi con số trong tài liệu đều **đo thật**, không lấy từ datasheet. Chỗ nào chưa đo
được đều nói rõ.

---

## Lộ trình

### Phần 0 — chưa biết gì thì bắt đầu ở đây

| # | File | Nội dung |
|---|---|---|
| 0 | [00-nhap-mon.md](00-nhap-mon.md) | **Không giả định kiến thức nền.** Token/logits/sampling, 4 phép toán duy nhất, phép chia `tok/s ≤ BW/size`, 4 phép tính nhẩm, bản đồ repo, **3 bài thực hành 60 phút không cần GPU hay board** |
| — | [12-thuat-ngu.md](12-thuat-ngu.md) | Tra nhanh ~50 thuật ngữ: mỗi mục 1–2 câu + *trong repo này là gì* |
| — | [`begin_0/`](begin_0/README.md) | **Lộ trình riêng cho kỹ sư embedded**: vector → weight → matrix multiplication → gradient → backprop → transformer → quantization → runtime (TensorRT/ONNX/TIDL) → VLA. 11 file, mỗi công thức đối chiếu với code C/Python thật trong repo, nhiều ví dụ **đo/verify bằng số thật** (cosine similarity trên embedding đã train, backprop tay khớp autograd tới 6 số thập phân, NumPy tự viết lại attention khớp PyTorch tới 1e-7) |

### Phần A — hiểu model

| # | File | Nội dung |
|---|---|---|
| 1 | [`../DEPLOY.md`](../DEPLOY.md) | Sơ đồ kiến trúc, bảng tensor, byte layout `model.bin`, deploy + test |
| 2 | [02-hieu-model.md](02-hieu-model.md) | Giải phẫu transformer qua code thật: RMSNorm, RoPE, SwiGLU, KV cache, **tham số nằm ở đâu**, PLE |

### Phần B — nền tảng tối ưu

| # | File | Nội dung |
|---|---|---|
| 3 | [03-roofline.md](03-roofline.md) | Memory-bound vs compute-bound, arithmetic intensity. **Đọc kỹ nhất** |
| 4 | [04-quantization.md](04-quantization.md) | Group-wise int4, GPTQ/AWQ/SmoothQuant, cách validate |

### Phần C — tối ưu theo kiến trúc

| # | File | Nội dung |
|---|---|---|
| 5 | [05-kien-truc-phan-cung.md](05-kien-truc-phan-cung.md) | **Cùng phép toán trên 3 kiến trúc.** Thang tối ưu 5 bậc, đo trên x86 và ARM |
| 6 | [`../firmware/jetson/JETSON.md`](../firmware/jetson/JETSON.md) | Thiết kế kernel CUDA, sai số song song, launch overhead |
| 7 | [07-kv-cache-engine.md](07-kv-cache-engine.md) | KV cache, FlashAttention, unified memory, chọn engine |

### Phần D — phương pháp

| # | File | Nội dung |
|---|---|---|
| 8 | [08-nhat-ky-toi-uu.md](08-nhat-ky-toi-uu.md) | Nhật ký ESP32 0.57 → 9.5 tok/s, 4 kỹ thuật, **phương pháp đối chứng** |
| 9 | [09-so-do-phan-cung.md](09-so-do-phan-cung.md) | 7 phát hiện khi đo Jetson thật (power mode, EMC, tải nền) |

### Phần E — lý thuyết sâu

| # | File | Nội dung |
|---|---|---|
| 10 | [10-ly-thuyet-nen.md](10-ly-thuyet-nen.md) | Dấu phẩy động (FP32/BF16/FP16/FP8), toán của lượng tử hoá, toán của attention, MHA/MQA/GQA, batching, speculative decoding, phân cấp bộ nhớ |
| 11 | [11-toi-uu-nvidia.md](11-toi-uu-nvidia.md) | **Chuyên sâu NVIDIA**: occupancy, coalescing, bank conflict, Tensor Core, **CUDA Graphs (đo 1.59×)**, stream, Nsight, TensorRT-LLM, đặc thù Jetson |

### Phần F — Jetson trong thực tế: framework, TensorRT, DeepStream

| # | File | Nội dung |
|---|---|---|
| 13 | [13-jetson-framework.md](13-jetson-framework.md) | **Tầng framework.** 6 tầng một lời gọi `generate()` đi qua, L4T/JetPack từ gốc, vì sao `pip install torch` hỏng + đường cài **offline**, **bảng 4 tầng chi phí** (framework nào chữa tầng nào), ngân sách RAM 8GB, 6 lab, 10 bẫy |
| 14 | [14-tensorrt-deepstream.md](14-tensorrt-deepstream.md) | **TensorRT** (6 bước builder, `trtexec`, INT8/calibration, nghiệm thu bằng `polygraphy`, plugin) và **DeepStream** (GStreamer + NVMM zero-copy, cây metadata, `interval`, đo FPS đúng cách, cảnh báo **Orin Nano không có NVENC**) + lộ trình 4 tuần |
| 15 | [15-kernel-den-camera.md](15-kernel-den-camera.md) | **Làm thật trên board 2026-08-05**: tự build kernel L4T + OOT modules, deploy có đường lùi, chứng minh bằng kernel gốc, **PREEMPT_RT mất 40% throughput mà độ trễ còn tệ hơn**, và đo đường camera tách được `nvvidconv` khỏi "rời NVMM" (**NVDEC rẻ hơn CPU 14.7×**) |

---

## Sample chạy được

| Sample | Kiến trúc | Dạy gì |
|---|---|---|
| [`samples/cpu/matvec_ladder.c`](../samples/cpu/matvec_ladder.c) | CPU x86 + ARM | Thang 5 bậc tối ưu, tự hiệu chuẩn số luồng, quét batch |
| [`samples/gpu/bench_roofline.cu`](../samples/gpu/bench_roofline.cu) | GPU | Bandwidth + quét arithmetic intensity dựng đường roofline |
| [`samples/gpu/bench_decode.cu`](../samples/gpu/bench_decode.cu) | GPU | Decode vs prefill: M=1 và M=64 cùng thời gian |
| [`samples/gpu/roofline.py`](../samples/gpu/roofline.py) | — | Máy tính trần tok/s, RAM, TTFT (không cần dependency) |
| [`samples/gpu/devprobe.cu`](../samples/gpu/devprobe.cu) | GPU | Thông số SM/bộ nhớ thật + bảng occupancy đo được |
| [`firmware/jetson/`](../firmware/jetson/) | GPU | Runtime CUDA đầy đủ + verify + generate + tokenizer C |
| [`firmware/host_verify/`](../firmware/host_verify/) | CPU | Golden logits + perplexity |
| [`firmware/esp32_llm/`](../firmware/esp32_llm/) | MCU | Runtime ESP32-S3 |

```bash
# CPU: thang tối ưu, chạy cả x86 lẫn ARM
make -C samples/cpu run
make -C samples/cpu scaling

# GPU: roofline thực nghiệm
nvcc -O3 -arch=sm_87 samples/gpu/bench_roofline.cu -o /tmp/rf && /tmp/rf 256
nvcc -O3 -arch=sm_87 samples/gpu/bench_decode.cu -o /tmp/dc -lcublas && /tmp/dc

# Không cần GPU
python3 samples/gpu/roofline.py --hw orin-nano-super --model llama-3.1-8b --bits 4.8
```

---

## Bốn con số đáng nhớ

Đo trên chính phần cứng của dự án này:

```
66.8 GB/s   băng thông DRAM Orin Nano Super (datasheet nói 102 -> chỉ đạt 65%)
152         machine balance FP16. Kernel nào AI < 152 là memory-bound
8           arithmetic intensity của LLM decode FP16 -> memory-bound 19x
3.71 us     launch overhead 1 kernel trên Orin (2.5 us trên RTX 4060)
```

Hệ quả trực tiếp: `tok/s ≤ 66.8 / model_GB`. Llama-8B Q4 (4.8GB) → trần **14 tok/s**.
Không kernel nào phá được trần này ở batch=1, trừ speculative decoding.

---

## Ba bài học lớn nhất

**1. Đo, đừng tra bảng.** Board Orin Nano Super chỉ đạt 65% bandwidth datasheet vì EMC
khoá ở 2133 MHz. Mọi trần tok/s tính từ datasheet đều sai 1.53×.

**2. Biết khi nào NGỪNG tối ưu.** [RESULTS.md:149-152](../RESULTS.md): tác giả đo head
chiếm 57.6 ms, tính ra sàn bandwidth 40 ms → SIMD chỉ mua thêm ~15% → **không làm**,
chuyển sang giảm bytes đọc. Đây là kỹ năng, không phải mẹo.

**3. Không có tối ưu phổ quát.** Cùng model, cùng code:

| | ESP32-S3 | CPU x86 | GPU Orin |
|---|---|---|---|
| Nút thắt | bandwidth PSRAM | compute → cache bandwidth | **launch overhead 50%** |
| SIMD tay | +15% (bị chặn bởi bandwidth) | **+49%** | không áp dụng |
| SIMD tay trên ARM | — | **0%** (compiler đã tự làm) | — |
| Song song | gần như luôn đáng | **hỏng nếu > 8 luồng** (CPU lai) | phải batch ≥ 128 |

---

## Chuỗi kiểm chứng — 5 tầng

Nguyên tắc: **mỗi tầng tách một loại lỗi, đừng gộp.** Gộp thì không biết lỗi ở đâu.

| Tầng | Câu hỏi | Lệnh | Ngưỡng |
|---|---|---|---|
| 0 | tokenizer C == Python? | `make -C firmware/jetson tok` | 18/18 khớp |
| 1 | port có đúng không? | `make -C firmware/jetson verify` | **argmax MATCH** |
| 2 | 4-bit hỏng bao nhiêu? | `./firmware/jetson/run.sh quantize` | báo cáo nats, ≥2 seed |
| 3 | nghẽn ở đâu? | `make -C firmware/jetson bench` | so với trần roofline |
| 4 | nó viết ra cái gì? | `make -C firmware/jetson chat` | đọc bằng mắt |

Điểm tinh tế: `golden.txt` là logits của model **đã dequantize**
([`export.py:12`](../src/export.py#L12)), nên tầng 1 chỉ đo lỗi *port*, tách hẳn khỏi
lỗi *lượng tử hoá* đo ở tầng 2. **Tách biến.**

---

## Giới hạn — nói thẳng để không mất thời gian

- **Model là TinyStories.** Nó VIẾT TIẾP truyện, không TRẢ LỜI câu hỏi. Giới hạn ở
  core 1.5M tham số, không phải ở runtime hay PLE.
- **Chưa có prefill theo batch.** Mọi target đều decode 1 token/lần, kể cả khi nạp
  prompt. Đó là chỗ tensor core mới có việc làm.
- **Chưa dùng tensor core.** Bản CUDA toàn CUDA core fp32.
- ~~Chưa có CUDA Graphs~~ → **đã làm, đo được 1.59× trên Orin** (1116 → 1774 tok/s).
  Xem [11-toi-uu-nvidia.md §11.5](11-toi-uu-nvidia.md).
- **Chưa chạy LLM thật (Llama/Qwen/Gemma) trên Jetson để đối chiếu** trần roofline.
- **Training vẫn bằng PyTorch.** Viết backward bằng C/C++ là khả thi (tham chiếu:
  llm.c của Karpathy) nhưng chưa làm.

Mỗi giới hạn là một bài tập có chủ đích, không phải thiếu sót giấu đi.
