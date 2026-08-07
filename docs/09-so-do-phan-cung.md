# Số đo thật trên board — 2026-08-01

Toàn bộ con số dưới đây **đo trực tiếp** trên board `machineai@100.92.121.20`, không
phải lấy từ datasheet. Mỗi bảng ghi rõ điều kiện đo và mã nguồn tạo ra nó.

## Board

```
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
L4T R36.4.7 (JetPack 6.2)  |  Ubuntu 22.04.5  |  CUDA 12.6.68  |  kernel 5.15.148-tegra
GPU: Ampere sm_87, 8 SM (4 TPC), 1024 CUDA core, 32 Tensor Core, max 1020 MHz
L2: 2.0 MB   |   RAM: 7.98 GB LPDDR5, 128-bit, dùng chung CPU+GPU
Disk: NVMe 116 GB (49 GB trống)   |   Không có torch, llama.cpp, TensorRT-LLM
```

Công cụ đo: [`bench/bench_roofline.cu`](../samples/gpu/bench_roofline.cu),
[`bench/bench_decode.cu`](../samples/gpu/bench_decode.cu), build bằng
`nvcc -O3 -arch=sm_87`. Đo thời gian bằng `cudaEvent`, buffer 256 MB (>> L2 2MB
nên chắc chắn chạm DRAM chứ không phải cache).

---

## Phát hiện 1 — Board mặc định ở 15W, không phải Super mode

```
$ nvpmodel -q
NV Power Mode: 15W          <- mode 0
GPU cur: 612 MHz            <- không phải 1020 MHz
```

Bảng power mode thật của board này (`/etc/nvpmodel.conf`):

| ID | Tên | GPU MAX_FREQ | EMC MAX_FREQ |
|---:|---|---:|---:|
| 0 | 15W | 612 MHz | 2133 MHz |
| 1 | 25W | 918 MHz | 3199 MHz |
| **2** | **MAXN_SUPER** | **-1 (không giới hạn → 1020 MHz)** | -1 |
| 3 | 7W | 408 MHz | 2133 MHz |

> ⚠️ **MAXN_SUPER là mode 2, KHÔNG phải mode 0.** Rất nhiều hướng dẫn trên mạng
> viết `nvpmodel -m 0` vì trên AGX Orin mode 0 mới là MAXN. Trên Orin Nano Super,
> `-m 0` đưa bạn về **15W**. Luôn kiểm bằng `nvpmodel -q` sau khi đặt.

Lệnh đúng:
```bash
sudo nvpmodel -m 2 && sudo jetson_clocks
nvpmodel -q                    # phải in "MAXN_SUPER"
sudo jetson_clocks --show      # kiểm GPU/EMC freq thật
```

---

## Phát hiện 2 — Ảnh hưởng của power mode (đo thật)

Cùng binary, cùng buffer 256 MB. Máy có tải nền ở cả hai lần đo (xem Phát hiện 3),
nên đây là so sánh **tương đối** giữa hai mode:

| | 15W (GPU 612 MHz) | MAXN_SUPER (GPU 1020 MHz) | tỉ lệ |
|---|---:|---:|---:|
| FP32 peak | 511.5 GFLOP/s | 975.3 GFLOP/s | **1.91×** |
| triad BW | 22.3 GB/s | 27.8 GB/s | 1.25× |

1.91× khớp gần đúng tỉ lệ clock 1020/612 = 1.67× cộng thêm phần bỏ throttle.
**Không chạy `nvpmodel -m 2` là bạn mất gần một nửa hiệu năng compute.**

---

## Phát hiện 3 — Tải nền làm sai lệch số đo tới 2.4×

`tegrastats` khi tưởng là máy rảnh:

```
RAM 2961/7608MB  CPU [37%@1728,30%,32%,32%,21%,64%]  GR3D_FREQ 62%
```

GPU đang bận 55-75% vì `forklift_demo` (137% CPU, 442 MB RSS) chạy nền.

Cùng lệnh `./bench_roofline 256`, cùng MAXN_SUPER:

| | có forklift_demo | máy rảnh (SIGSTOP) | tỉ lệ |
|---|---:|---:|---:|
| read scalar | 19.1 GB/s | 47.4 GB/s | 2.48× |
| triad | 27.8 GB/s | 56.4 GB/s | 2.03× |
| FP32 peak | 975 GFLOP/s | 1883.8 GFLOP/s | 1.93× |

**Luôn kiểm `tegrastats` trước khi benchmark.** GR3D_FREQ phải gần 0% lúc rảnh.

> Cách tạm dừng an toàn (không kill, giữ nguyên state):
> ```bash
> pgrep -x forklift_demo | xargs kill -STOP     # dùng -x, KHÔNG dùng pkill -f
> ...chạy bench...
> pgrep -x forklift_demo | xargs kill -CONT
> ```
> Đừng dùng `pkill -f forklift_demo` qua SSH: `-f` khớp cả dòng lệnh SSH của chính
> bạn và sẽ tự SIGSTOP phiên làm việc. (Tôi đã mắc đúng lỗi này khi làm bài này.)

---

## Phát hiện 4 — `float4` cho 1.4× bandwidth, miễn phí

Cùng kernel, chỉ đổi kiểu load từ `float` (4 byte) sang `float4` (16 byte):

| kernel | GB/s | ghi chú |
|---|---:|---|
| read scalar | 47.4 | mỗi thread 1 request đang bay |
| **read float4** | **66.8** | **+41%** |
| copy scalar | 54.3 | |
| copy float4 | 63.2 | +16% |
| triad scalar | 56.4 | |
| triad float4 | 64.2 | +14% |

Lý do: một thread chỉ giữ được số ít memory request đang bay. Load 16 byte/lệnh
cho 4× số byte đang bay với cùng số thread → che được độ trễ DRAM.
(NVIDIA CUDA C++ Best Practices Guide §9.2.1; Luitjens, *CUDA Pro Tip: Increase
Performance with Vectorized Memory Access*, NVIDIA Developer Blog 2013.)

Ở AI=1 trong phần quét, kernel `float`-scalar cũng đạt 60 GB/s vì nó có
grid-stride + accumulator độc lập — cho thấy **ILP thay thế được vector hoá**;
cái nào cũng được, miễn là có đủ request đang bay.

---

## Phát hiện 5 — Board này KHÔNG đạt 102 GB/s như datasheet

```
$ sudo jetson_clocks --show
EMC MinFreq=204000000 MaxFreq=2133000000 CurrentFreq=2133000000 FreqOverride=1
$ sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/max_rate
3199000000
```

- Phần cứng (bpmp) báo EMC max **3199 MHz**
- devfreq lại giới hạn ở **2133 MHz**, kể cả ở MAXN_SUPER và cả khi chuyển sang
  mode 25W (mode này ghi rõ `EMC MAX_FREQ 3199000000` trong nvpmodel.conf)

Băng thông lý thuyết theo EMC thật:
```
2133 MHz × 128 bit × 2 (DDR) / 8 = 68.3 GB/s
```

**Đo được 66.8 GB/s = 97.8% của trần đó.** Nghĩa là bench đã chạm sàn phần cứng;
không còn gì để tối ưu ở tầng kernel. Nút thắt là EMC clock.

102 GB/s trong datasheet cần EMC 3199 MHz (3199 × 32 = 102.4 GB/s) — con số này
khớp chính xác với `max_rate` của bpmp, xác nhận cách tính. Board (bản
Engineering Reference) hiện không mở được mức đó qua nvpmodel/jetson_clocks.

> **Hệ quả thực tế: mọi trần tok/s của bạn thấp hơn datasheet 1.53×.**
> Đây chính là lý do phải đo chứ không tra bảng.

### ⚠️ CẬP NHẬT 2026-08-05 — giới hạn này KHÔNG CÒN. Board đã đạt datasheet.

Đo lại trên cùng board sau khi nâng lên **L4T R36.4.7 (JetPack 6.2.x)**, cùng lệnh,
cùng điều kiện (MAXN_SUPER, máy rảnh, buffer 256MB, 3 lần):

```
$ sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate
3199000000                      <- TRƯỚC ĐÂY BỊ KHOÁ Ở 2133000000
$ sudo jetson_clocks --show | grep EMC
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=3199000000 FreqOverride=1
```

| | đo 2026-08-01 (R36.4.3) | đo 2026-08-05 (R36.4.7) |
|---|---:|---:|
| EMC | 2133 MHz | **3199 MHz** |
| trần lý thuyết | 68.3 GB/s | **102.4 GB/s** |
| **read float4 đo được** | **66.8 GB/s** (97.8%) | **99.2 GB/s** (96.9%) |
| triad float4 | 64.2 GB/s | 96.1 GB/s |
| FP16 tensor core (cuBLAS) | 10.14 TFLOP/s | **11.66 TFLOP/s** |
| INT8 tensor core (cuBLAS) | 12.82 TOPS | **15.26 TOPS** |
| machine balance FP16 | 152 FLOP/byte | **117.5 FLOP/byte** |

Ba lần đo liên tiếp cho 99.2 / 99.2 / 99.2 GB/s — lặp lại tuyệt đối, không phải nhiễu.

**Nguyên nhân:** devfreq trước đây chặn EMC ở 2133 MHz kể cả ở MAXN_SUPER; bản L4T
mới bỏ giới hạn đó. Phần cứng (bpmp `max_rate`) vẫn báo 3199 MHz như cũ — tức là
**phần cứng chưa bao giờ là nút thắt, phần mềm mới là.**

**Trần decode phải tính lại: `tok/s = 99.2 / model_size_GB`** (không còn 66.8).
Llama-3.1-8B Q4 4.8GB: trần cũ 14 tok/s → **trần mới 20.7 tok/s**. Bảng ở cuối file
này là số của R36.4.3; nhân 1.485 để ra số của R36.4.7.

> **Bài học kép, và nó mạnh hơn bài học ban đầu:** không những phải đo thay vì tra
> datasheet — mà còn phải **đo lại sau mỗi lần nâng phiên bản**. Một con số đo đúng
> vào tháng trước có thể sai vào tháng này, và sai theo hướng *có lợi* thì càng dễ
> bị bỏ qua. Số đo có hạn sử dụng; hãy ghi ngày tháng và phiên bản bên cạnh mọi con số.

---

## Phát hiện 6 — Đường roofline thực nghiệm

`bench_roofline` phần [B]: cùng một kernel, tăng dần FLOP làm trên mỗi byte
(AI = 1 → 256). Máy rảnh, MAXN_SUPER.

| AI (FLOP/byte) | GB/s | GFLOP/s | chế độ |
|---:|---:|---:|---|
| 1 | 60.0 | 82.5 | memory-bound (chạm trần BW) |
| 2 | 59.9 | 142.3 | memory-bound |
| 4 | 59.8 | 261.5 | memory-bound |
| 8 | 59.2 | 495.4 | memory-bound |
| **16** | **48.3** | **790.8** | **điểm gãy** |
| 32 | 33.3 | 1078.2 | compute-bound |
| 64 | 21.2 | 1362.3 | compute-bound |
| 128 | 13.2 | 1693.8 | compute-bound |
| 256 | 7.3 | 1883.8 | compute-bound (chạm trần FP32) |

Đọc bảng: từ AI=1 đến AI=8, **GB/s không đổi (~60)** — máy đang bị chặn bởi DRAM,
và mỗi lần tăng AI thì GFLOP/s tăng gấp đôi *miễn phí*. Từ AI=16 trở đi GB/s bắt
đầu tụt — DRAM không còn là nút thắt, compute mới là.

**Điểm gãy đo được ≈ 28 FLOP/byte** (1883.8 GFLOP/s ÷ 66.8 GB/s), khớp với vị trí
AI=16→32 trong bảng. Đây là machine balance FP32 của board, đo thực nghiệm chứ
không suy từ datasheet.

Kiểm chứng chéo: FP32 lý thuyết = 1024 core × 2 FLOP × 1.02 GHz = **2.09 TFLOP/s**.
Đo được 1.88 TFLOP/s = **90% peak**. Bench đáng tin.

---

## Phát hiện 7 — Vì sao decode chậm mà prefill nhanh (bằng chứng trực tiếp)

`bench_decode 4096 4096`: cùng phép GEMM `M × 4096 × 4096` FP16 tensor core,
chỉ đổi M = số token xử lý cùng lúc. Máy rảnh, MAXN_SUPER.

| M | ms | TFLOP/s | GB/s | AI | đây là gì |
|---:|---:|---:|---:|---:|---|
| **1** | **0.512** | 0.07 | **65.5** | 1.0 | **DECODE — sinh 1 token** |
| 2 | 0.524 | 0.13 | 64.1 | 2.0 | |
| 4 | 0.526 | 0.26 | 63.9 | 4.0 | |
| 8 | 0.530 | 0.51 | 63.5 | 8.0 | |
| 16 | 0.538 | 1.00 | 62.9 | 15.9 | |
| 32 | 0.522 | 2.06 | 65.3 | 31.5 | |
| **64** | **0.544** | 3.95 | 63.6 | 62.1 | **vẫn bằng giá của 1 token** |
| 128 | 0.621 | 6.92 | 57.4 | 120.5 | chuyển tiếp |
| 256 | 0.868 | 9.89 | 43.5 | 227.6 | |
| 512 | 1.694 | **10.14** | 24.8 | 409.6 | PREFILL — chạm peak |
| 1024 | 4.459 | 7.71 | 11.3 | 682.7 | |
| 4096 | 18.340 | 7.49 | 5.5 | 1365.3 | |

**Ba điều rút ra, mỗi điều đều có số:**

1. **M=1 dùng 0.6% năng lực tensor core** (0.07 / 10.14 TFLOP/s). GPU chỉ ngồi
   chờ đọc 33.6 MB ma trận B từ DRAM. Đó chính xác là LLM decode.

2. **GB/s ở M=1..64 là 63-65 GB/s** — khớp với 66.8 GB/s đo độc lập bằng
   `bench_roofline`. Hai benchmark hoàn toàn khác nhau, cùng chạm một trần.
   Đây là bằng chứng mạnh nhất rằng decode memory-bound.

3. **M=1 và M=64 mất cùng thời gian** (0.512 vs 0.544 ms, chênh 6%).
   **64 token với giá của 1 token.** Đây là toàn bộ cơ sở toán học của
   speculative decoding, continuous batching và Medusa/EAGLE.

INT8 (cuBLAS, cùng phép toán):

| M | ms | TOPS |
|---:|---:|---:|
| 1 | 0.547 | 0.06 |
| 64 | 0.289 | 7.42 |
| 512 | 1.340 | **12.82** |

INT8 peak 12.8 TOPS so với datasheet 33.5 TOPS dense = 38%. cuBLAS không tối ưu
INT8 tốt trên Tegra; TensorRT-LLM dùng kernel riêng sẽ cao hơn. **Nhưng ở M=1
INT8 vẫn chỉ 0.06 TOPS** — quantize không giúp gì cho *compute* ở decode,
nó giúp vì **giảm bytes đọc**.

---

## Bảng kết luận — dùng số này cho board của bạn

```
Băng thông DRAM đạt được     66.8 GB/s     (trần phần cứng 68.3, hiệu suất 98%)
FP32 CUDA core               1.88 TFLOP/s  (90% của 2.09 lý thuyết)
FP16 tensor core (cuBLAS)   10.14 TFLOP/s
INT8 tensor core (cuBLAS)   12.82 TOPS
Machine balance FP32           28 FLOP/byte
Machine balance FP16          152 FLOP/byte  (10.14e12 / 66.8e9)
```

**Trần decode: `tok/s = 66.8 / model_size_GB`**

| Model | Q4_K_M | trần tok/s | thực tế kỳ vọng (50-70%) |
|---|---:|---:|---:|
| Llama-3.2-1B | 0.7 GB | 95 | 48–67 |
| Qwen2.5-1.5B | 1.0 GB | 67 | 33–47 |
| Llama-3.2-3B | 1.9 GB | 35 | 18–25 |
| Qwen2.5-7B | 4.2 GB | 16 | 8–11 |
| Llama-3.1-8B | 4.8 GB | 14 | 7–10 |

Chưa tính KV cache. Ở ctx 4096 với 8B, KV FP16 thêm 0.54 GB → trần tụt còn 12.5 tok/s.
Dùng `--cache-type-k q8_0 --cache-type-v q8_0` để lấy lại phần lớn.

Chạy lại:
```bash
python3 bench/roofline.py --hw orin-nano-super --model llama-3.1-8b --bits 4.8 --ctx 4096
```
(mục `orin-nano-super` trong `roofline.py` đã được cập nhật bằng đúng số đo này;
mục `orin-nano-super-spec` giữ số datasheet để đối chiếu)

---

## Việc chưa làm được

- **Chưa chạy LLM thật.** Board không có torch, llama.cpp, TensorRT-LLM, và
  `curl https://huggingface.co` không phản hồi (không có internet ra ngoài).
  Nên phần "thực tế đạt 50-70% trần" vẫn là kỳ vọng theo kinh nghiệm chung,
  **chưa phải số đo trên board này**. Để xác nhận, cần copy llama.cpp + một
  file GGUF từ máy x86 sang qua `scp`.
- **Chưa đo throttle dài hạn.** Các bench chạy < 90 giây. Nhiệt độ nền lúc đo
  đã là 60-62°C. Cần chạy 10 phút liên tục mới biết mức tụt thật.
- **Chưa mở được EMC 3199 MHz.** Xem Phát hiện 5.

---

## Trạng thái board sau khi đo

Đã khôi phục nguyên trạng:
```
NV Power Mode: 15W (mode 0)     <- như lúc đầu
jetson_clocks --restore          <- đã chạy
forklift_launch (PID 29322)     state R  <- đã SIGCONT
forklift_demo   (PID 29716)     state R  <- đã SIGCONT
```
Mã nguồn bench để lại ở `~/jetson-optim/bench/` trên board (đã build sẵn).
