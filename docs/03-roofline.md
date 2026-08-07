# 1. Roofline — nền tảng của mọi tối ưu trên Jetson

> Nếu chỉ đọc một file, đọc file này. 80% quyết định tối ưu sau này đều suy ra từ đây.

## 1.1 Hai loại giới hạn

Mọi kernel chỉ làm 2 việc: **đọc/ghi bytes** từ DRAM, và **tính toán** trên GPU.
Thời gian chạy là max của hai thứ đó (không phải tổng — chúng chồng lấp được):

```
t = max( bytes / BW ,  flops / PEAK )
```

- Nếu `bytes/BW` lớn hơn → **memory-bound**. Tối ưu compute (SIMD, tensor core, kernel fusion
  cho phần tính) **không giúp gì cả**.
- Nếu `flops/PEAK` lớn hơn → **compute-bound**. Giảm bytes đọc không giúp gì.

**Arithmetic Intensity (AI)** = `flops / bytes` (đơn vị FLOP/byte). Đây là đặc trưng của
*thuật toán*, không phụ thuộc phần cứng.

**Machine balance** = `PEAK / BW` (cũng FLOP/byte). Đặc trưng của *phần cứng*.

```
AI < balance  →  memory-bound
AI > balance  →  compute-bound
```

## 1.2 Con số của Orin Nano Super (8GB)

Hai cột: datasheet, và **số đo thật trên board của bạn** (chi tiết cách đo trong
[MEASUREMENTS.md](09-so-do-phan-cung.md)).

| Thông số | Datasheet | **Đo thật** | đạt |
|---|---:|---:|---:|
| GPU | Ampere, 1024 CUDA core, 32 Tensor Core, 1020 MHz | ✓ xác nhận | |
| RAM bandwidth | 102 GB/s | **66.8 GB/s** | 65% |
| FP32 CUDA core | 2.09 TFLOPS | **1.88 TFLOPS** | 90% |
| FP16 tensor core | 16.7 TFLOPS | **10.14 TFLOPS** | 61% |
| INT8 tensor core (dense) | 33.5 TOPS | **12.8 TOPS** (cuBLAS) | 38% |
| CPU | 6× Cortex-A78AE @ 1.7 GHz | ✓ | |
| DLA | **không có** trên Orin Nano (chỉ NX/AGX có) | | |

Bandwidth chỉ đạt 65% datasheet vì EMC của board khoá ở 2133 MHz thay vì 3199 MHz
(→ trần phần cứng 68.3 GB/s, và ta đã đạt 98% của trần đó). Xem
[MEASUREMENTS.md §Phát hiện 5](09-so-do-phan-cung.md).

**Machine balance — theo số đo thật:**

```
FP32 : 1.88e12 / 66.8e9  ≈  28 FLOP/byte   (đo trực tiếp bằng AI sweep)
FP16 : 10.14e12 / 66.8e9 ≈ 152 FLOP/byte
INT8 : 12.8e12 / 66.8e9  ≈ 192 OP/byte
```

Nhớ con số **152**. Bất cứ kernel FP16 nào có AI < 152 đều memory-bound trên board này.

> ⚠️ Bandwidth là **dùng chung** giữa CPU và GPU. CPU copy dữ liệu, camera DMA, display —
> tất cả ăn chung. Khác biệt lớn nhất so với RTX 4060 (272 GB/s VRAM riêng).
> Trên board này tôi đo được tải nền `forklift_demo` làm tụt bandwidth **2.4×** —
> xem [MEASUREMENTS.md §Phát hiện 3](09-so-do-phan-cung.md).

## 1.3 Áp vào LLM: prefill vs decode khác nhau hoàn toàn

Đây là điểm 90% người mới hiểu sai.

### Decode (sinh 1 token, batch=1)

Mỗi token phải đọc **toàn bộ** trọng số một lần, nhưng chỉ làm 1 phép nhân
vector–ma trận trên mỗi trọng số:

```
flops ≈ 2 × P          (P = số tham số)
bytes ≈ P × (bits/8)   (+ KV cache)

AI = 2P / (P × bits/8) = 16/bits
```

| Định dạng | AI (FLOP/byte) | So với balance FP16 = 152 |
|---|---:|---|
| FP16 | 8 | memory-bound **19×** (152/8) |
| INT8 | 16 | memory-bound 9.5× |
| INT4 | 32 | memory-bound 4.75× |

**Kết luận: decode ở batch=1 LUÔN memory-bound, không có ngoại lệ.**
Tensor core của bạn nhàn rỗi >95% thời gian. Cách duy nhất để nhanh hơn là **đọc ít bytes hơn**
(quantize thấp hơn, model nhỏ hơn, KV cache nhỏ hơn) hoặc **sinh nhiều token trên mỗi lần đọc**
(batching, speculative decoding).

Trần trên board của bạn, dùng **bandwidth đo thật 66.8 GB/s**:

```
tok/s ≤ 66.8 GB/s ÷ (kích thước model tính bằng bytes)
```

| Model | Q4_K_M | Trần tok/s (đo thật) | nếu board đạt 102 GB/s |
|---|---:|---:|---:|
| Llama-3.2-1B | 0.7 GB | **95** | 145 |
| Llama-3.2-3B | 1.9 GB | **35** | 54 |
| Qwen2.5-7B | 4.2 GB | **16** | 24 |
| Llama-3.1-8B | 4.8 GB | **14** | 21 |

Thực tế đạt 50–70% trần là tốt. Nếu bạn đo được 5 tok/s trên 8B Q4 → còn ~2.8× dư địa.
Nếu đo được 10 tok/s → đã gần sàn, đừng tốn thời gian tối ưu kernel nữa, hãy đổi model.

**Bằng chứng trực tiếp cho luận điểm "decode luôn memory-bound":** `bench_decode` đo GEMM
`1 × 4096 × 4096` FP16 và thu được **65.5 GB/s** — trùng với 66.8 GB/s đo bằng
`bench_roofline` (kernel hoàn toàn khác). Tensor core lúc đó chỉ chạy ở **0.6% năng lực**.
Chi tiết: [MEASUREMENTS.md §Phát hiện 7](09-so-do-phan-cung.md).

### Prefill (nạp prompt, T token cùng lúc)

Bây giờ mỗi trọng số đọc 1 lần nhưng dùng cho T token:

```
AI ≈ T × 16/bits
```

Với FP16 và T ≈ 19 token là đã vượt balance 152 → **compute-bound**. Prompt dài luôn
compute-bound.

Số đo thật xác nhận: GEMM `M×4096×4096` FP16 chuyển từ memory-bound sang compute-bound
trong khoảng M = 64 → 128 (GB/s bắt đầu tụt từ 63.6 xuống 57.4). Hơi cao hơn dự đoán 19
vì cuBLAS cần M đủ lớn mới lấp đầy được 8 SM.

Đây là lý do:
- Prefill nhanh (tok/s cao) — tận dụng tensor core
- Decode chậm — chờ DRAM
- Và tại sao TTFT (time-to-first-token) với LLM cần tối ưu khác hẳn tok/s

### Batch size tới hạn

Lý thuyết với FP16: `B* = balance / AI_decode = 152/8 ≈ 19`.

**Đo thật còn tốt hơn thế:** M=1 mất 0.512 ms, M=64 mất 0.544 ms — chênh **6%**.
Tức là 64 token gần như miễn phí. Đó là toàn bộ lý do continuous batching và
speculative decoding tồn tại. Bảng đầy đủ ở [MEASUREMENTS.md §Phát hiện 7](09-so-do-phan-cung.md).

## 1.4 KV cache — thứ bị bỏ quên trên 8GB RAM

```
KV_bytes = 2 × layers × kv_heads × head_dim × ctx × batch × (bits/8)
```

Llama-3.1-8B (32 layer, GQA 8 kv-head, head_dim 128), FP16, ctx 8192, batch 1:

```
2 × 32 × 8 × 128 × 8192 × 2 = 1.07 GB
```

Trên board 8GB (thực dùng được ~6.5GB sau OS): model Q4 4.7GB + KV 1.07GB = 5.8GB — sát nút.
Và KV cache đó **phải đọc lại mỗi token** → nó cộng thẳng vào bytes trong công thức decode,
làm tok/s tụt dần khi context dài ra.

Lever: **INT8 KV cache** (chia đôi cả RAM lẫn bandwidth), GQA/MQA, sliding window.

## 1.5 Quy trình làm việc đúng

```
0. Kiểm máy có RẢNH không                   ← tegrastats, GR3D_FREQ phải ~0%
1. sudo nvpmodel -m 2 && sudo jetson_clocks ← MODE 2, không phải 0! (đo được: 1.9x compute)
2. Đo BW và PEAK thật của board bạn         ← bench/bench_roofline.cu
3. Tính trần roofline cho workload          ← bench/roofline.py
4. Đo thực tế
5. So sánh: cách trần bao xa?
     > 3×  → có bug hệ thống (thermal, clock, copy thừa, sai backend, tải nền)
     2–3×  → tối ưu kernel/engine có ý nghĩa
     < 1.5× → đã chạm sàn, phải đổi model hoặc đổi thuật toán
```

Bước 0 và 5 là hai bước hay bị bỏ qua nhất. Trên chính board này, bỏ bước 0 làm sai
số đo **2.4×**, bỏ bước 1 làm mất **1.9×** compute. Cộng lại là sai gần 5 lần —
đủ để dẫn tới kết luận hoàn toàn ngược.

## 1.6 Bài tập

1. Chạy `bench_roofline` ở cả 4 nvpmodel (7W/15W/25W/MAXN_SUPER). Giải thích vì sao
   bandwidth gần như không đổi giữa 15W và MAXN_SUPER trong khi FP32 tăng 1.9×.
   *(Gợi ý: xem bảng EMC MAX_FREQ ở [MEASUREMENTS.md §Phát hiện 1](09-so-do-phan-cung.md))*
2. Trong `bench_roofline`, so `read scalar` với `read VECTOR float4`. Giải thích 41% chênh lệch.
3. Chạy `bench_decode` với N=K=2048 và N=K=8192. Điểm gãy M dịch theo hướng nào? Vì sao?
4. Dùng `bench/roofline.py` tính trần cho model bạn định chạy, rồi đo thật bằng
   `llama-bench`. Ghi lại tỉ lệ đạt được.

→ Tiếp: [02-quantization.md](04-quantization.md)
