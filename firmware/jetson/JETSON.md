# Cổng Jetson — chạy cùng model trên Orin Nano Super

Target thứ ba của repo, bên cạnh `esp32_llm/` (Xtensa LX7) và `host_verify/` (CPU x86).
Đọc **cùng `model.bin`**, kiểm bằng **cùng `golden.txt`**. Chỉ phần số học chuyển lên GPU.

```
src/export.py ──> firmware/model/model.bin ──┬──> esp32_llm/     (2 core LX7, scalar C)
                  firmware/model/golden.txt  ├──> host_verify/   (CPU, scalar C)
                                             └──> jetson/        (Ampere GPU, CUDA)  ← file này
```

`llm_load()` trong [`../common/llm.h`](../common/llm.h) được **dùng lại nguyên vẹn** để
parse header và bind tensor. Không viết lại parser, không đổi định dạng. Đó là điểm
mấu chốt: một artifact, ba runtime, một golden.

> **Mới dùng lần đầu?** Đọc [`../../DEPLOY.md`](../../DEPLOY.md) trước — có sơ đồ kiến
> trúc, bảng tensor, byte layout của `model.bin`, và hướng dẫn deploy/test từng bước
> cho cả laptop lẫn Jetson kèm số tham chiếu. File này đi sâu vào **thiết kế kernel**.

---

## 1. Vì sao port sang Jetson khi đã có llama.cpp

Không phải để nhanh hơn llama.cpp. Là để **học**, và repo này là chỗ hiếm để học vì bạn
sở hữu toàn bộ chuỗi.

Ba thứ chỉ hiểu được khi tự viết:

1. **Decode batch=1 là memory-bound** — bạn sẽ thấy tận mắt tensor core nhàn rỗi trong
   khi kernel chờ DRAM. Bench in ra GB/s đạt được của từng stage.
2. **Kernel-launch overhead** — model này nhỏ, mỗi kernel chạy vài µs, và bạn sẽ thấy
   overhead launch có thể lấn át cả tính toán. Đây là bài học không gặp khi chạy model 8B.
3. **Số học song song không tất định** — mục 4 dưới. Đây là thứ làm hỏng nhiều bản port
   và không tài liệu nào cảnh báo đủ.

---

## 2. Chạy — mọi thứ trong container, không cài gì lên host

### 2.1 Pipeline PyTorch (trên máy x86 có GPU)

```bash
cd <repo>
./firmware/jetson/run.sh build      # dựng image từ pytorch/pytorch chính thức
./firmware/jetson/run.sh prepare    # tải TinyStories 300MB, train BPE, ra token bins
./firmware/jetson/run.sh train      # train arm `ple`
./firmware/jetson/run.sh quantize   # đo degradation 4-bit
./firmware/jetson/run.sh export     # -> firmware/model/model.bin + golden.txt
```

Container mount repo qua bind-mount và chạy bằng UID của bạn, nên artifact rơi thẳng
vào `runs/`, `data/`, `firmware/model/` như chạy native. Host không có torch, không có
venv, không đụng driver — container dùng driver host qua `nvidia` runtime, đó là cách
nó vốn hoạt động.

Biến môi trường: `VOCAB=4096 STEPS=2000 TAG=jetson ./firmware/jetson/run.sh train`.

> Cấu hình deploy 28.9M trong RESULTS.md là `--vocab 32768 --d-model 96 --n-layers 6
> --ple-dim 128`. Nó train lâu hơn nhiều. Bắt đầu bằng vocab 4096 để chạy hết vòng
> pipeline trong ~30 phút, rồi mới scale lên.

### 2.2 Runtime CUDA (trên Jetson)

Board đã có CUDA 12.6 nên `nvcc` chạy trực tiếp, không cần container:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cd firmware/jetson
make                       # ARCH=sm_87 mặc định cho Orin
make verify                # tầng 1: đúng chưa?
make bench                 # tốc độ + chẩn đoán roofline từng stage
make generate              # xem model VIẾT RA cái gì
make host_verify           # bản C scalar trên cùng board, để so
```

`generate` cần [`vocab.h`](vocab.h) (token id → raw UTF-8 bytes), sinh bởi:

```bash
python src/gen_assets.py --vocab 4096 --out firmware/jetson/vocab.h \
                         --prompt "Once upon a time"
```

Nó in luôn `PROMPT_IDS` để dán vào `generate_cuda.cu` nếu muốn đổi prompt.
Tham số: `./generate_cuda <model.bin> <n_tokens> <temperature> <top_k> [seed]`.
`temperature 0` = greedy, tất định.

Muốn container trên Jetson thì dùng `nvcr.io/nvidia/l4t-cuda:12.6.11-devel` hoặc
`dustynv/l4t-pytorch`. Cần `--runtime nvidia`.

Dev trên RTX 4060 trước khi đụng board: `make ARCH=sm_89` (hoặc `sm_86` nếu CUDA < 11.8).

---

## 3. Chuỗi kiểm chứng — giống hệt bản ESP32

Repo tách 3 loại lỗi, mỗi loại một công cụ. Bản CUDA giữ nguyên cấu trúc đó.

| Tầng | Câu hỏi | Lệnh | Ngưỡng |
|---|---|---|---|
| 1 | Port CUDA có đúng không? | `make verify` | **argmax khớp** (xem mục 4) |
| 2 | 4-bit làm hỏng bao nhiêu? | `run.sh quantize` | báo cáo nats, ≥2 seed |
| 3 | Nhanh chậm ra sao, nghẽn đâu? | `make bench` | so với trần roofline |

`verify_cuda` in **ba** so sánh trong một lần chạy:

```
PyTorch  vs  scalar C   : port C có đúng không
PyTorch  vs  CUDA       : port CUDA có đúng không
scalar C vs  CUDA       : tách port khỏi lượng tử hoá
```

Cả ba đều ~1e-6..1e-5 và **argmax phải khớp**. Ba số này tách được đúng ba nguyên nhân;
nếu chỉ có một số, bạn không biết lỗi ở đâu. Đừng kỳ vọng số nào nhỏ hơn số nào — xem mục 4.

---

## 4. Sai số — thứ làm hỏng nhiều bản port GPU

Bản C scalar và bản CUDA **cộng theo thứ tự khác nhau**, nên không thể trùng bit.

Lý do: **cộng số thực không có tính kết hợp.** `(a+b)+c ≠ a+(b+c)` trong dấu phẩy động.

- [`llm.h:117-145`](../common/llm.h#L117-L145) cộng dot product **tuần tự trái sang phải**.
- [`llm_cuda.cuh` `k_matvec_q4`](llm_cuda.cuh) cộng bằng **warp reduction dạng cây**
  (`__shfl_down_sync`, 5 tầng).

> **ĐÍNH CHÍNH — dự đoán ban đầu của tài liệu này sai.** Tôi viết rằng CUDA sẽ lệch
> PyTorch **nhiều hơn** bản C. Số đo thật cho ngược lại (mục 10):
> ```
> PyTorch vs scalar C : 7.63e-06
> PyTorch vs CUDA     : 4.05e-06   <- CUDA GẦN HƠN
> ```
> Vì cộng dạng cây tích luỹ sai số làm tròn `O(log n)`, còn cộng tuần tự là `O(n)`.
> **Tree reduction chính xác HƠN naive summation** (pairwise summation, kết quả cổ điển
> trong numerical analysis). Song song không chỉ nhanh hơn — ở đây nó còn đúng hơn.
>
> Bài học thật không phải "GPU kém chính xác" mà là: **thứ tự cộng khác nhau thì kết
> quả khác nhau, và bạn không đoán được chiều nào tốt hơn. Phải đo.**

Thêm hai nguồn nữa:

| Nguồn | Ảnh hưởng | Kiểm soát |
|---|---|---|
| Thứ tự cộng (tree vs tuần tự) | ~1e-4 trên logit | không tránh được, đây là bản chất song song |
| `fmaf()` gộp nhân-cộng, không làm tròn giữa chừng | thường **chính xác hơn**, nhưng khác | `-fmad=false` để tắt (chậm hơn) |
| `__expf()` / `__half2float` intrinsic nhanh | ~1e-6 | dùng `expf()` nếu cần khớp chặt |

**Vậy kiểm cái gì?** Không phải `max abs diff`. Phải kiểm **argmax**:

```
Điều PHẢI đúng: argmax(logits) khớp nhau.
Vì greedy decoding chỉ dùng argmax -- nếu argmax khớp, text sinh ra giống hệt.
```

`verify_cuda` in rõ `argmax MATCH` hay `DIFFER`. Nếu DIFFER, khi đó mới là bug thật
(hoặc hai logit đầu bảng quá sát nhau — cũng cần kiểm bằng tay).

> **Bài học chuyển giao:** khi validate TensorRT engine hay bất kỳ kernel GPU nào, đừng
> đặt ngưỡng theo `1e-5` như CPU. Đặt theo **hành vi**: argmax, top-k overlap, hoặc
> perplexity. Rất nhiều người kết luận "TensorRT sai" khi thực ra chỉ là thứ tự cộng.

---

## 5. Thiết kế kernel — và vì sao chọn như vậy

### 5.1 `k_matvec_q4` — một WARP cho một hàng output

```cuda
int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // hàng nào
int lane = threadIdx.x & 31;                                // cột nào trong hàng
for (int j = lane; j < cols; j += 32) { ...unpack nibble... }
for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
```

Vì sao warp chứ không phải block: `cols = D` rất nhỏ (96 ở cấu hình deploy của
RESULTS.md, 128 ở model mẫu mục 10). Một block 256 thread cho 128 cột thì quá nửa số
thread ngồi chơi. Một warp (32 lane) × 4 cột/lane thì vừa vặn, và `__shfl_down_sync`
reduce trong warp **không cần shared memory, không cần `__syncthreads()`**.

Block 256 thread = 8 warp = 8 hàng output song song. Head có `V` hàng (4096 hoặc 32768)
→ 512 hoặc 4096 block. Đủ lấp đầy 8 SM của Orin Nano nhiều lần.

> Đây cũng là lý do model nhỏ bị launch-bound (mục 6): kernel *đủ song song* nhưng
> *quá ít việc*, nên thời gian khởi động lấn át thời gian chạy.

**Cách đọc địa chỉ int4** giống hệt `deq_row` ở [`llm.h:86`](../common/llm.h#L86):
`row_bytes = ceil(cols/2)`, nibble `j` nằm ở byte `j>>1`, nửa thấp nếu `j` chẵn.
Không repack, không đệm — **byte trên GPU chính là byte trong `model.bin`**.

### 5.2 `k_attention` — một BLOCK cho một head

Softmax 2 pass với trừ max, y hệt [`llm.h:327`](../common/llm.h#L327). Không có bước
trừ max thì `expf` tràn fp32.

FlashAttention chính là phiên bản **1 pass online softmax** của kernel này. Ở ctx 512
thì dạng naive đủ nhanh và dễ đọc hơn nhiều — đó là lý do giữ nguyên. Khi bạn muốn
học FlashAttention, hãy bắt đầu bằng cách sửa đúng kernel này.

### 5.3 `k_rmsnorm` — cố ý KHÔNG dùng `__restrict__`

`llm.h` gọi rmsnorm in-place (`rmsnorm(s->h, w, D, s->h)`). Bản CUDA cũng vậy. Kernel
an toàn in-place vì mọi lệnh đọc `x` xảy ra trước `__syncthreads()` đứng trước mọi lệnh
ghi `out`. Nhưng nếu khai `__restrict__` (hứa không alias), compiler được phép sắp xếp
lại và **kết quả sai âm thầm**.

> `nvcc` bắt được đúng lỗi này khi build lần đầu:
> `warning: passing argument 4 to 'restrict'-qualified parameter aliases with argument 1`.
> **Đọc warning của compiler.**

---

## 6. Đọc kết quả `make bench`

Bench in ms/token, MB đọc, và **GB/s đạt được** cho từng stage, kèm chẩn đoán:

```
stage         ms/token        %    MB đọc  GB/s đạt   chẩn đoán
head             0.xxx    xx.x%     x.xxx      xx.x   CHẠM SÀN bandwidth - đừng tối ưu nữa
attention        0.xxx    xx.x%     x.xxx      xx.x   XA sàn -> compute/launch-bound
```

Đây chính là lập luận ở [`RESULTS.md:149-152`](../../RESULTS.md), tự động hoá:

> *"The head is now PSRAM-bandwidth-bound, not compute-bound... Literal S3 vector-SIMD
> would cut that 17ms but not the 40ms bandwidth floor."*

**Quy tắc:** tìm stage chiếm % lớn nhất, rồi xem cột GB/s.
- Gần peak của board (66.8 GB/s đo được trên Orin Nano Super) → **chạm sàn, dừng lại**.
  Muốn nhanh hơn phải giảm bytes: quantize sâu hơn, head nhỏ hơn, vocab nhỏ hơn.
- Xa peak → nút thắt là compute hoặc **kernel-launch overhead**.

### Launch overhead — bench tự đo, không hardcode

`bench_cuda` chạy một kernel rỗng 2000 lần để đo **launch overhead thật của board này**,
rồi so từng stage với nó. Cần làm vậy vì overhead khác nhau rất nhiều theo CPU điều khiển:

| Board | overhead đo được (kernel rỗng) |
|---|---:|
| RTX 4060 Laptop (x86) | ~2.5 µs |
| **Orin Nano Super (Cortex-A78)** | **3.71 µs** |

Ngưỡng cứng kiểu "dưới 6 µs là launch-bound" sai trên ít nhất một trong hai máy. **Hiệu
chuẩn, đừng hardcode** — đúng tinh thần của repo.

Model này chạy 117 kernel/token. Số đo thật (mục 11): 50% thời gian mỗi token là
launch overhead thuần. **Model càng nhỏ, overhead càng chiếm ưu thế** — bài học riêng
của embedded AI, không gặp khi chạy model 8B.

Cách khắc phục: **CUDA Graphs** — ghi lại chuỗi 117 kernel một lần rồi replay bằng một
lệnh. Bài tập 4.

---

## 7. So sánh ba target

| | ESP32-S3 | Jetson Orin Nano Super | Chênh |
|---|---:|---:|---:|
| Bộ nhớ nhanh | 512 KB SRAM | 8 GB LPDDR5 | 16,000× |
| Bandwidth | 60.7 MB/s (PSRAM) | 66.8 GB/s (đo thật) | 1,100× |
| Compute | 2× LX7 scalar @240MHz | 1024 CUDA core + 32 tensor core | ~10,000× |
| Model 28.9M @4-bit | 14.9 MB — **vừa khít 16MB flash** | 14.9 MB — 0.2% RAM | |
| Nút thắt | bandwidth PSRAM (head) | **launch overhead** (model quá nhỏ) | |

Điểm thú vị: **cùng một model, nút thắt hoàn toàn khác nhau.** Trên ESP32 nó là
bandwidth. Trên Jetson model quá nhỏ nên overhead thắng. Đây là lý do không có "mẹo tối
ưu" phổ quát — phải profile trên chính phần cứng đích.

---

## 8. Bài tập — theo thứ tự khó dần

**1. Chạy hết chuỗi.** `run.sh build/prepare/train/quantize/export`, rồi `make verify`
trên Jetson. Ghi lại cả ba số của `verify_cuda`. Argmax có khớp không?

**2. Phá RoPE.** Sửa `k_rope_apply` từ split-half sang interleaved (dùng cặp `2i, 2i+1`
thay vì `i, i+Dh/2`). Chạy lại `verify`. Argmax lệch chứ? Đây là bug port phổ biến nhất,
và đây là cách bạn bắt được nó.

**3. Đo launch overhead.** Chạy `make bench`, cộng ms của 5 stage. Chia cho số kernel
(~86). So với ~5-10 µs/launch. Bao nhiêu phần trăm thời gian là overhead thuần?

**4. CUDA Graphs.** Bọc `llm_cuda_forward` bằng `cudaStreamBeginCapture` /
`cudaGraphInstantiate`, replay mỗi token. Đo lại. *Dự đoán trước khi đo:* tăng bao nhiêu?

**5. So với llama.cpp.** Convert model sang GGUF (cần viết script — kiến trúc PLE không
có sẵn trong llama.cpp, nên bài này là bài khó nhất và có thể bất khả thi). Nếu không
được, hãy viết ra **tại sao** — hiểu vì sao một kiến trúc mới khó đưa vào engine có sẵn
cũng là bài học.

**6. Bỏ hàng vocab thừa.** `esp32_llm.ino:153` cắt head còn `VOCAB_N` hàng thật. Làm
tương tự cho `k_matvec_q4` của head. Tiết kiệm bao nhiêu %?

**7. int8 activation.** `llm.h:161-198` có sẵn đường int8. Viết `k_matvec_q4_i8` dùng
`__dp4a` (intrinsic dot-product int8 của Ampere). So tốc độ và ppl. *Gợi ý: ở batch=1
memory-bound, đừng kỳ vọng nhiều — và hiểu vì sao mới là mục tiêu.*

---

## 9. Giới hạn của bản port này

Nói thẳng để bạn không mất thời gian:

- **Chưa có prefill theo batch.** Mỗi token của prompt vẫn chạy một forward riêng, đúng
  như bản ESP32. Prefill thật phải là GEMM `T×D×D`, và đó là chỗ tensor core mới có việc.
- **Chưa dùng tensor core.** Toàn bộ là CUDA core fp32. Muốn chạm tensor core cần
  batch/prefill hoặc `__dp4a`/WMMA (bài tập 7).
- **Chưa có CUDA Graphs** (bài tập 4) — và đó gần như chắc chắn là lever lớn nhất.
- **Chưa tối ưu attention.** Naive 2-pass, ctx 512. Không phải FlashAttention.
- **fp32 activation.** Chưa thử fp16/bf16 cho scratch.

Mỗi giới hạn là một bài tập có chủ đích, không phải thiếu sót giấu đi.

---

## 10. Số đo thật — chạy đủ chuỗi, 2026-08-01

Model tự train trong repo này (không phải cấu hình deploy 28.9M của RESULTS.md):
`V=4096 D=128 L=6 H=4 F=415 P=64`, 2000 steps, 32.8M token, **model.bin 1.87 MB**.

### Train (container, RTX 4060)

| arm | core | table | val loss | ppl |
|---|---:|---:|---:|---:|
| baseline | 1,498,496 | 0 | 2.2496 | 9.48 |
| **ple** | 1,499,328 | 1,572,864 | **2.2107** | **9.12** |

PLE hơn **+0.039 nats / 3.8% ppl** ở core khớp nhau trong 0.06%.
[`RESULTS.md`](../../RESULTS.md) báo +0.025 nats ở vocab 4096 — cùng chiều, cùng bậc.

### Lượng tử hoá 4-bit

| arm | fp32 | 4-bit | degradation |
|---|---:|---:|---:|
| baseline | 2.2769 | 2.3291 | +0.0522 |
| ple | 2.2364 | 2.2779 | **+0.0416** |

**Edge giữ 126%** — bảng lookup chịu 4-bit *tốt hơn* core dày đặc. RESULTS.md báo
124-128%; ta rơi đúng giữa, trên model train độc lập. Tái hiện thành công.

### Tầng 1 — golden logits (chạy trên chính Orin Nano Super)

```
PyTorch    vs scalar C   | max|d| 7.629e-06 | argmax MATCH (580)
PyTorch    vs CUDA       | max|d| 4.053e-06 | argmax MATCH (580)
scalar C   vs CUDA       | max|d| 7.391e-06 | argmax MATCH (580)
```

Port đúng. Và CUDA **gần PyTorch hơn** bản C — xem đính chính ở mục 4.

### Tầng 3 — tốc độ và chẩn đoán

Orin Nano Super, MAXN_SUPER, forklift_demo tạm dừng, 200 token:

```
throughput: 1141 tok/s  (0.876 ms/token)   -- chỉ 2% trần bandwidth
launch overhead đo được: 3.71 us/kernel

stage         ms/token       % MB đọc GB/s đạt  us/kern
input+PLE        0.047    5.3%     0.026       0.5     6.66
attention        0.329   37.6%     0.203       0.6     7.83
ffn              0.254   29.0%     0.495       1.9     7.05
ple gate         0.153   17.4%     0.051       0.3     5.09
head             0.049    5.6%     0.270       5.5    24.65

117 kernel/token, 7.49 us/kernel trung bình
=> 50% thời gian mỗi token là LAUNCH OVERHEAD THUẦN (0.434 / 0.876 ms)
```

**So RTX 4060 (cùng binary, cùng model.bin):** 2866 tok/s, 0.349 ms/token,
2.98 µs/kernel. Nhanh hơn Jetson 2.5× — gần đúng tỉ lệ launch overhead, **không phải**
tỉ lệ bandwidth hay compute. Bằng chứng thêm cho chẩn đoán launch-bound.

### Text sinh ra trên Orin Nano Super

`./generate_cuda ../model/model.bin 160 0.8 40 2026`, prompt "Once upon a time":

> Once upon a time, there was a little girl named Lily. One day, Lily's mommy told
> Lily to be careful not to get hurt. They went to the shop and saw a picture of a
> little bird. Lily was happy to see the beautiful bird and make friends again.
> Lily and her mommy went into the market to find a big pot. Lily saw a small fish
> and she asked her mommy if they could open it. Her mommy said yes, so she found a
> big, heavy bucket. Lily was so happy and thanked her mommy. When they got home,
> Lily's mommy helped her mix the yummy fish together. They mixed and baked
> sandwiches, and cookies.

**1053 tok/s** (0.949 ms/token). Mạch lạc ở mức câu và đoạn; logic dài hơi thì trôi
("mix the yummy fish... baked sandwiches, and cookies") — đúng như kỳ vọng của model
3.6M tham số train 2000 steps trên TinyStories. Không phải giới hạn của PLE mà của
core 1.5M, đúng như [`README.md`](../../README.md) nói.

Greedy (`temperature 0`) cho 1299 tok/s vì bỏ được bước top-k selection trên host.

> **Kết quả đáng chú ý: cùng seed, Jetson (sm_87) và RTX 4060 (sm_86) sinh ra text
> GIỐNG HỆT TỪNG CHỮ.** Dù mục 4 nói float không kết hợp, ở đây hai GPU cùng họ
> Ampere chạy cùng kernel với cùng thứ tự reduction → cùng bit. Sai số chỉ xuất hiện
> khi *đổi thuật toán cộng* (scalar C vs warp reduction), không phải khi đổi GPU.
>
> Đừng khái quát quá: đổi `blockDim`, đổi số warp, hay dùng cuBLAS thay kernel tự viết
> đều làm đổi thứ tự cộng và có thể làm text rẽ nhánh sau vài chục token.

### Luận điểm PLE, đo trên GPU

```
bảng PLE 0.81 MB nhưng chỉ đọc 192 B/token (0.0237%)
```

Đúng luận điểm của repo, ở phần cứng hoàn toàn khác. Trên ESP32 con số tương ứng là
25M tham số đọc ~450 B/token.

### Ba dự đoán, hai sai

| Dự đoán ban đầu | Thực tế | |
|---|---|---|
| CUDA lệch PyTorch nhiều hơn bản C | **Ngược lại**, CUDA gần hơn (tree reduction chính xác hơn) | ✗ |
| ~86 kernel/token, overhead 5-10 µs | 117 kernel, overhead 3.71 µs | ✗ |
| Model này launch-bound | Đúng, nhưng **50%** chứ không ~100% | ~ |

Giữ nguyên cả ba trong tài liệu thay vì sửa lặng lẽ — cùng tinh thần
[`README.md:83-88`](../../README.md#L83-L88) của repo gốc ("I left the messy history in
the repo on purpose").

---

## 11. Đọc thêm trong repo

| Muốn hiểu | Đọc |
|---|---|
| Model là gì, từng op vì sao | [`../../src/model.py`](../../src/model.py) + `jetson-optim/04-hieu-model.md` |
| Bản C tham chiếu, đối chiếu dòng-dòng với `llm_cuda.cuh` | [`../common/llm.h`](../common/llm.h) |
| Kế toán bộ nhớ 3 tầng | [`../../src/budget.py`](../../src/budget.py) |
| Lượng tử hoá group-wise int4 | [`../../src/quantize.py`](../../src/quantize.py) |
| Định dạng `model.bin` + golden | [`../../src/export.py`](../../src/export.py) |
| Nhật ký tối ưu ESP32 0.57→9.5 tok/s | [`../../RESULTS.md`](../../RESULTS.md) |
| Roofline + số đo thật của board | `../../../jetson-optim/` |
