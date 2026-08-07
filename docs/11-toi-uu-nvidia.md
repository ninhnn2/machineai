# 11. Tối ưu trên target NVIDIA

Mọi thông số trong tài liệu này **truy vấn trực tiếp từ phần cứng** bằng
[`samples/gpu/devprobe.cu`](../samples/gpu/devprobe.cu), không lấy từ datasheet.
Chạy nó trên board của bạn trước khi tin bất cứ con số nào ở đây.

```bash
nvcc -O3 -arch=sm_87 samples/gpu/devprobe.cu -o /tmp/devprobe && /tmp/devprobe
```

---

## 11.1 Thông số thật của Jetson Orin Nano Super

```
Orin  (compute capability 8.7 = sm_87, Ampere GA10B)
```

| | Giá trị | Ý nghĩa khi viết kernel |
|---|---:|---|
| SM | 8 | grid phải có ≥ 8 block mới dùng hết chip |
| warp size | 32 | đơn vị lập lịch, mọi thứ nên chia hết cho 32 |
| **max thread/SM** | **1536** | **48 warp/SM** — KHÔNG phải 2048 như GA100/consumer |
| max block/SM | 16 | block quá nhỏ (<96 thread) không lấp nổi SM |
| thanh ghi/SM | 65536 | 1536 thread → trung bình 42 reg/thread ở 100% occupancy |
| shared/SM | 164 KB | opt-in tối đa 163 KB cho một block |
| shared/block mặc định | 48 KB | muốn hơn phải `cudaFuncSetAttribute` |
| L2 | 2 MB | model 1.87 MB của repo **nằm gọn trong L2** |
| bus | 128 bit | |
| **integrated** | **CÓ** | iGPU: RAM dùng chung CPU, memcpy H2D là lãng phí |
| async engine | 2 | copy chồng lấn compute được |

**Đỉnh lý thuyết suy ra:** 8 SM × 128 FP32 lane × 2 × 1.02 GHz = **2.09 TFLOP/s FP32**.
Tensor Core: 32 cái (4/SM), FP16 ≈ 16.7 TFLOP/s, INT8 ≈ 33.4 TOPS.

**Đo thật** ([`samples/gpu/bench_decode.cu`](../samples/gpu/bench_decode.cu)):
FP32 1.88 TFLOP/s (90% đỉnh), FP16 cuBLAS 10.14 TFLOP/s (61%), INT8 12.8 TOPS (38%).

> Chênh lệch FP16/INT8 không phải phần cứng yếu mà là cuBLAS chưa tối ưu tốt cho
> Tegra ở kích thước này. TensorRT-LLM dùng kernel riêng sẽ cao hơn.

---

## 11.2 Mô hình thực thi — occupancy là gì và không phải là gì

GPU giấu độ trễ bộ nhớ bằng cách **chuyển sang warp khác** khi một warp phải chờ.
Occupancy = số warp đang cư trú trên SM / số warp tối đa.

`devprobe` đo occupancy thật của một kernel 10 thanh ghi trên Orin:

| blockDim | block/SM | warp/SM | occupancy |
|---:|---:|---:|---:|
| 32 | 16 | 16 | 33% |
| 64 | 16 | 32 | 67% |
| **128** | 12 | 48 | **100%** |
| **256** | 6 | 48 | **100%** |
| **512** | 3 | 48 | **100%** |
| 1024 | 1 | 32 | **67%** |

**Hai bài học cụ thể cho Orin:**

1. **`blockDim = 1024` chỉ đạt 67%.** Vì 1536 không chia hết cho 1024 → chỉ 1 block
   cư trú. Trên GPU có 2048 thread/SM thì 1024 lại đạt 100%. **Con số tối ưu phụ
   thuộc chip, phải đo.**
2. **`blockDim = 32` chỉ đạt 33%** vì trần 16 block/SM. Block quá nhỏ lãng phí SM.

Vùng an toàn trên Orin: **128–512 thread/block**.

### Ba giới hạn occupancy — cái nào chặn bạn?

```
warp/SM  ≤  min( 48,
                 16 · (blockDim/32),            ← trần block/SM
                 65536 / (reg_per_thread · 32), ← trần thanh ghi
                 164KB / shared_per_block · blockDim/32 )  ← trần shared
```

Kiểm bằng `nvcc -Xptxas -v` (in reg + shared mỗi kernel) hoặc
`cudaFuncGetAttributes`. Nếu thanh ghi là nút thắt, dùng `__launch_bounds__(N)` để
ép compiler tiết kiệm thanh ghi.

> **Occupancy cao ≠ nhanh.** Nó là *khả năng che độ trễ*. Kernel memory-bound cần
> occupancy cao để có đủ request đang bay. Kernel compute-bound dùng nhiều thanh ghi
> có thể nhanh nhất ở occupancy 25% — vì mỗi thread làm nhiều việc hơn (ILP thay cho
> TLP). Đây là kết quả kinh điển của Volkov, *Better Performance at Lower Occupancy*
> (GTC 2010).

---

## 11.3 Bộ nhớ — coalescing, sector, bank

### Coalescing

Khi 32 lane của một warp đọc global memory, phần cứng gom thành các **giao dịch 32 byte**
(sector). Chỉ tiêu duy nhất: *tổng số sector cần chạm*.

| Pattern (32 lane × 4 byte) | Sector chạm | Hiệu suất |
|---|---:|---|
| liền mạch, căn 128 B | 4 | 100% |
| liền mạch, lệch căn | 5 | 80% |
| bước nhảy 2 | 8 | 50% |
| bước nhảy 32 | 32 | **12.5%** |
| ngẫu nhiên trong 1 sector | 1 | 100% (broadcast) |

Trong [`k_matvec_q4`](../firmware/jetson/llm_cuda.cuh), 32 lane đọc `row[j>>1]` với
`j = lane, lane+32, …` → các lane liền nhau chạm byte liền nhau. **Coalesced.**

### Vector load — vì sao `float4` cho +41%

Đo được ở [`samples/gpu/bench_roofline.cu`](../samples/gpu/bench_roofline.cu):

| kernel | GB/s |
|---|---:|
| read scalar (`float`) | 47.4 |
| **read vector (`float4`)** | **66.8** |

Không phải vì đọc nhiều byte hơn — mà vì **số request đang bay**. Mỗi thread chỉ giữ
được số ít lệnh load chưa hoàn thành; `float4` mang 16 B/lệnh thay vì 4 B, nên cùng số
thread có 4× byte đang bay để che độ trễ DRAM (~400–600 chu kỳ).

ILP thay thế được: dùng nhiều accumulator độc lập cũng đạt hiệu quả tương tự (phần
quét AI trong cùng bench đạt 60 GB/s với load scalar nhưng 4 accumulator).

### Shared memory và bank conflict

Shared memory chia **32 bank × 4 byte**. Hai lane trong cùng warp chạm **địa chỉ khác
nhau trong cùng bank** → serialise.

```cuda
__shared__ float tile[32][32];
tile[threadIdx.y][threadIdx.x];   // OK: lane x liền nhau -> 32 bank khác nhau
tile[threadIdx.x][threadIdx.y];   // 32-way conflict: cột cùng bank -> chậm 32x
__shared__ float tile[32][33];    // padding 1 -> lệch bank -> hết conflict
```

Mẹo `+1` đó là kỹ thuật chuẩn khi transpose trong shared memory.

### Bộ nhớ hợp nhất trên Jetson — đừng copy vô ích

`devprobe` báo `integrated = CÓ`. Nghĩa là CPU và GPU **dùng chung DRAM vật lý**.

```c
// SAI trên Jetson: copy RAM -> RAM, đốt băng thông 2 lần
cudaMalloc(&d, n); cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);

// ĐÚNG: zero-copy, GPU đọc thẳng bộ nhớ host
cudaHostAlloc(&h, n, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d, h, 0);

// Hoặc: managed, driver tự lo
cudaMallocManaged(&p, n);
```

Code viết cho desktop port sang Jetson thường chậm bất thường vì lý do này. Kiểm bằng
`nsys` — nếu thấy nhiều thời gian trong memcpy, đó là thủ phạm.

---

## 11.4 Tensor Core — khi nào chúng thực sự chạy

Tensor Core thực hiện `D = A·B + C` trên **tile cố định**, không phải phép nhân vô hướng.
Ampere (sm_87) hỗ trợ hình dạng `m16n8k16` (FP16/BF16), `m16n8k32` (INT8),
`m16n8k64` (INT4).

Điều kiện để chúng được dùng:

1. **Phép toán phải là GEMM đủ lớn.** `M=1` (decode) chỉ dùng 1/16 hàng của tile →
   lãng phí 94% năng lực.
2. **Kiểu dữ liệu được hỗ trợ**: FP16, BF16, TF32, INT8, INT4. **Orin KHÔNG có FP8.**
3. **Layout và căn chỉnh** phải đúng (thường 16 byte).
4. Gọi qua cuBLAS/cuDNN/CUTLASS, hoặc `wmma::` / inline PTX `mma.sync`.

**Đo thật trên Orin** (`bench_decode.cu`, GEMM `M×4096×4096` FP16):

| M | TFLOP/s | % đỉnh |
|---:|---:|---:|
| 1 (decode) | 0.07 | **0.6%** |
| 64 | 3.95 | 39% |
| 512 (prefill) | 10.14 | 100% |

> **Kết luận quan trọng: ở decode batch=1, Tensor Core gần như nhàn rỗi hoàn toàn,
> và không kỹ thuật nào thay đổi được điều đó.** Chúng chỉ có việc khi prefill hoặc
> batch lớn. Đây là lý do tối ưu decode = giảm bytes, không phải tăng FLOP/s.

### DP4A — đường INT8 trên CUDA core

Ngoài Tensor Core, Ampere còn có `__dp4a(int a, int b, int c)`: tích vô hướng 4 cặp
INT8 → INT32 trong 1 lệnh, chạy trên **CUDA core** thường. Hữu ích cho kernel matvec
mảnh (decode) nơi Tensor Core không lấp đầy được.

Đây là bài tập 7 trong [JETSON.md](../firmware/jetson/JETSON.md): thay warp reduction
FP32 bằng `__dp4a`. Lưu ý trước khi làm: ở decode ta **memory-bound**, nên đừng kỳ
vọng nhiều — và hiểu vì sao mới là mục tiêu bài tập.

---

## 11.5 CUDA Graphs — lever lớn nhất cho model nhỏ

### Vấn đề

Mỗi `kernel<<<>>>` tốn công việc phía driver/CPU: kiểm tham số, cấp phát slot, ghi
command buffer, thông báo GPU. `bench_cuda` đo bằng **kernel rỗng**:

| Board | overhead/launch |
|---|---:|
| RTX 4060 Laptop (x86 driver) | ~2.5 µs |
| **Orin Nano Super (Cortex-A78)** | **3.56 µs** |

Model của repo phóng **117 kernel/token**. `117 × 3.56 µs = 0.416 ms` — trong khi cả
bước chỉ mất 0.896 ms. **~46% thời gian là overhead thuần.**

### Giải pháp

CUDA Graph ghi lại toàn bộ đồ thị phụ thuộc **một lần**, sau đó mỗi token chỉ cần
`cudaGraphLaunch`. Driver đã biết trước cấu trúc nên bỏ được phần lớn công việc/lần.

### Kết quả đo thật trên Orin

```
                   ms/token        tok/s   so eager
eager (117 launch)   0.8956       1116.5     1.00x
graph (1 launch)     0.5636       1774.5     1.59x
  tiết kiệm 0.332 ms/token; overhead ước tính 0.416 ms
  -> graph thu hồi 80% phần overhead dự đoán
```

**1.59× chỉ bằng cách gom launch**, không đụng một dòng nào bên trong kernel.

### Ba điều kiện — và bug thật đã gặp

**1. Tham số kernel bị "nướng chín" lúc capture.** Thứ đổi mỗi token (`token`, `pos`)
phải nằm trong **bộ nhớ mà kernel đọc**, không phải trong argument. Nên các kernel đã
được sửa để nhận `const int* pos_dev` thay vì `int pos`:

```cuda
__global__ void k_rope_freqs(..., const int *__restrict__ pos_dev) {
  int pos = *pos_dev;     // đọc lúc CHẠY, không phải lúc capture
```

**2. Không cấp phát/đồng bộ trong lúc capture.** Chạy nháp một lần trước để driver
cấp phát lazily xong xuôi.

**3. Capture trên stream không phải stream mặc định** (`cudaStreamNonBlocking`).

> ### Bug đã bị tầng verify bắt được — đáng đọc kỹ
>
> Lần đầu tôi đẩy `(token,pos)` xuống device bằng `cudaMemcpyAsync` từ **pinned host
> memory**. Verify FAIL ngay: `argmax DIFFER (580 vs 438)`.
>
> Nguyên nhân: `cudaMemcpyAsync` từ pinned memory là **bất đồng bộ thật**. Host chạy
> tiếp và ghi đè ô staging bằng token của bước SAU **trước khi** copy của bước TRƯỚC
> kịp thực thi. Model đọc nhầm token. **Không có lỗi CUDA nào cả** — chỉ ra số sai.
>
> Cách sửa: dùng **kernel** thay vì memcpy.
> ```cuda
> __global__ void k_set_step(int *tok, int *pos, int t, int p) {
>   if (threadIdx.x == 0) { *tok = t; *pos = p; }
> }
> ```
> Tham số của kernel được **chụp lại ngay lúc launch** (driver chép vào command
> buffer), nên không có cửa sổ race. Rẻ hơn 2 memcpy, và đúng.
>
> `k_set_step` nằm **ngoài** graph (vì tham số node trong graph bị cố định), nên
> tổng là 2 launch/token thay vì 117.

### Khi nào CUDA Graphs không giúp

- Model lớn (Llama-8B): mỗi kernel chạy hàng trăm µs, overhead 3.5 µs là nhiễu
- Grid/block đổi theo bước
- Có nhánh điều kiện phía host giữa các kernel

Quy tắc: **graph đáng làm khi `n_kernels × launch_overhead` chiếm > 20% thời gian bước.**
`bench_cuda` in ra đúng tỉ lệ đó.

---

## 11.6 Stream và chồng lấn

`devprobe` báo `async engine = 2` → Orin copy và tính đồng thời được.

```cuda
cudaStream_t s[2];
for (int i = 0; i < n; i++) {
  int k = i & 1;
  cudaMemcpyAsync(d[k], h[k], sz, cudaMemcpyHostToDevice, s[k]);  // double buffer
  kernel<<<g, b, 0, s[k]>>>(d[k]);
  cudaMemcpyAsync(h2[k], d[k], sz, cudaMemcpyDeviceToHost, s[k]);
}
```

Lưu ý trên Jetson: vì bộ nhớ hợp nhất, phần lớn trường hợp **không nên copy** — chồng
lấn copy/compute là bài toán của dGPU. Stream trên Jetson hữu ích để chạy **nhiều
kernel độc lập** đồng thời (ví dụ nhiều camera stream), không phải để giấu PCIe.

---

## 11.7 Profiling — dùng đúng công cụ cho đúng câu hỏi

| Câu hỏi | Công cụ | Lệnh |
|---|---|---|
| Thời gian đi đâu ở mức ứng dụng? | **Nsight Systems** | `nsys profile -t cuda,nvtx ./app` |
| Kernel này vì sao chậm? | **Nsight Compute** | `ncu --set full -k k_matvec_q4 ./app` |
| Occupancy/thanh ghi lúc biên dịch | nvcc | `nvcc -Xptxas -v` |
| Toàn hệ thống Jetson | tegrastats / jtop | `tegrastats --interval 1000` |

**Chỉ số quan trọng trong `ncu`, theo thứ tự:**

1. `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM bận bao nhiêu %
2. `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` — DRAM bận bao nhiêu %
3. `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — sector đọc (kiểm coalescing)
4. `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy đạt được

Cách đọc: **(1) cao → compute-bound. (2) cao → memory-bound. Cả hai thấp → latency-bound
hoặc launch-bound.** Trường hợp thứ ba chính là model của repo này.

> Trên Jetson `ncu` cần quyền: `sudo ncu ...` hoặc set
> `/proc/sys/kernel/perf_event_paranoid`.

---

## 11.8 Chọn engine — cái gì làm gì

| Engine | Nó thực sự làm gì | Dùng khi |
|---|---|---|
| **llama.cpp** (CUDA backend) | kernel viết tay cho GGUF, mmap model | mặc định, bắt đầu ở đây |
| **TensorRT** | phân tích graph ONNX → **fusion**, chọn kernel tốt nhất bằng **autotuning** trên chính máy bạn, cấp phát bộ nhớ tối ưu | vision, ASR, model tĩnh |
| **TensorRT-LLM** | TensorRT + kernel LLM chuyên dụng: paged KV, in-flight batching, INT4 AWQ, FlashAttention | production LLM |
| **MLC-LLM** | TVM: sinh kernel bằng auto-scheduling | thử nghiệm, portable |
| **ONNX Runtime + TRT EP** | wrapper, fallback về CUDA EP khi TRT không nuốt được layer | prototyping |

**Ba thứ TensorRT làm mà bạn khó tự làm:**

1. **Layer fusion** — gộp `Conv+Bias+ReLU` thành 1 kernel, bỏ hẳn round-trip qua DRAM
   cho tensor trung gian. Đây chính là phiên bản tự động của việc gom kernel mà ta làm
   thủ công bằng CUDA Graphs.
2. **Kernel autotuning** — thử nhiều thuật toán/tile size cho từng layer **trên chính
   phần cứng của bạn**, chọn cái nhanh nhất. Đó là lý do engine không portable giữa
   các GPU và phải build lại.
3. **Chọn precision theo layer** — giữ layer nhạy cảm ở FP16, hạ phần còn lại xuống INT8.

**Cái giá:** build engine lâu (phút đến giờ), phụ thuộc phiên bản TensorRT + driver
rất chặt, và engine gắn với đúng compute capability.

---

## 11.9 Đặc thù Jetson — checklist trước mọi phép đo

```bash
# 1. Power mode: MAXN_SUPER là MODE 2 trên Orin Nano Super (KHÔNG phải 0)
sudo nvpmodel -m 2 && sudo jetson_clocks
nvpmodel -q                                   # phải in MAXN_SUPER

# 2. Máy có rảnh không
tegrastats --interval 1000                    # GR3D_FREQ phải ~0%

# 3. EMC thật (cudaDeviceProp.memoryClockRate KHÔNG đáng tin trên Tegra)
sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate

# 4. Khôi phục sau khi đo
sudo jetson_clocks --restore && sudo nvpmodel -m 0
```

Ba thủ phạm đã đo được trên chính board này
([09-so-do-phan-cung.md](09-so-do-phan-cung.md)):

| Thủ phạm | Sai lệch | Phát hiện bằng |
|---|---:|---|
| Tải nền chiếm GPU | **2.4×** | `tegrastats` → GR3D_FREQ > 0% |
| Sai power mode | **1.9×** | `nvpmodel -q` |
| Tin datasheet bandwidth | **1.53×** | đo bằng `bench_roofline.cu` |

**DLA:** Orin Nano **không có** DLA. Orin NX/AGX có 1–2. Đừng mất thời gian nếu bạn
dùng Nano.

**Thermal:** board này chạy nền đã 60–62 °C. Benchmark 30 giây không thấy throttle;
phải chạy ≥ 10 phút mới biết mức tụt thật.

---

## 11.10 Lộ trình tối ưu trên NVIDIA — theo thứ tự lợi ích

Áp dụng đúng thứ tự này; đừng nhảy cóc.

| # | Bước | Lợi ích điển hình | Công sức |
|---:|---|---|---|
| 0 | `nvpmodel -m 2` + máy rảnh | **2–5×** | phút |
| 1 | Chọn quantization đúng (Q4_K_M/AWQ) | **2–4×** | giờ |
| 2 | Dùng đúng engine, `-ngl 99` | 1.5–3× | giờ |
| 3 | INT8 KV cache + FlashAttention | 1.1–1.4× | giờ |
| 4 | **CUDA Graphs** nếu launch-bound | **1.6×** (đo được) | ngày |
| 5 | Speculative decoding | 1.5–2.5× | ngày |
| 6 | Kernel tay (SIMD/DP4A/tensor core) | 1.0–1.3× | tuần |

**Bước 6 gần cuối bảng là có chủ ý.** Đó là bước tốn công nhất và lợi ít nhất, trừ khi
bạn đã chứng minh được mình compute-bound. Trên board này, `bench_cuda` cho thấy mọi
stage đều **không** compute-bound — nên bước 6 sẽ lãng phí thời gian.

Đó cũng chính là kết luận của tác giả bản ESP32 ở [RESULTS.md:149-152](../RESULTS.md):
tính ra sàn bandwidth trước, thấy SIMD chỉ mua thêm ~15%, nên **không làm**.

---

## 11.11 Bài tập

1. Chạy `devprobe` trên board bạn. `blockDim` nào cho 100% occupancy? Khác Orin không?
2. Sửa `mv()` trong `llm_cuda.cuh` từ 8 warp/block thành 2, 4, 16. Đo `bench_cuda`.
   Kết quả có khớp bảng occupancy không?
3. Bật `nvcc -Xptxas -v`, xem `k_matvec_q4` dùng bao nhiêu thanh ghi. Thêm
   `__launch_bounds__(256)` — thanh ghi giảm không? Tốc độ đổi ra sao?
4. Chạy `ncu --set full -k k_matvec_q4 ./bench_cuda`. So `sm__throughput` với
   `gpu__dram_throughput`. Kernel này bị chặn bởi cái gì?
5. Cố tình phá coalescing: đổi `k_matvec_q4` để lane `i` đọc cột `i*32`. Đo lại và
   so số sector trong `ncu`.
6. Đưa `k_set_step` **vào trong** graph (bỏ ra ngoài). Chạy `verify`. Giải thích vì
   sao sai.

→ Quay lại [README.md](README.md)
