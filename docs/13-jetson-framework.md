# 13. Framework trên Jetson Orin Nano Super — từ gốc đến tối ưu

[`11-toi-uu-nvidia.md`](11-toi-uu-nvidia.md) nói về **kernel bạn tự viết**.
[`09-so-do-phan-cung.md`](09-so-do-phan-cung.md) nói về **phần cứng đo được**.
File này lấp khoảng giữa: **tầng framework** — PyTorch, llama.cpp, TensorRT, ONNX
Runtime — thứ mà 95% công việc thực tế trên Jetson nằm ở đó.

Nguyên tắc xuyên suốt: **không dùng framework như hộp đen.** Mỗi phần dưới đây trả lời
"nó thực sự làm gì", "chi phí nằm ở tầng nào", "đo bằng cách nào".

Board tham chiếu là board thật trong [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md):
Orin Nano Engineering Reference Dev Kit Super, L4T R36.4.7 (JetPack 6.2), CUDA 12.6.68,
sm_87, 8 GB LPDDR5 dùng chung CPU+GPU, **không có internet ra ngoài**.

---

## 13.1 Gốc: một lời gọi `generate()` đi qua bao nhiêu tầng

Trước khi tối ưu bất cứ framework nào, phải biết thời gian có thể mất ở đâu. Sáu tầng,
từ trên xuống:

```
┌─ ① Python / binding ─────────────  vòng lặp, tokenizer, dtype, sampling
│      chi phí: 10-100 us mỗi bước, KHÔNG có trên llama.cpp/C++
├─ ② Framework graph ─────────────   dispatch từng op, cấp phát, launch kernel
│      chi phí: 3.71 us × số kernel  ← ĐO ĐƯỢC trên board này
├─ ③ Thư viện kernel ─────────────   cuBLAS / cuDNN / kernel của TRT / của llama.cpp
│      chi phí: kernel chạy nhanh hay chậm so với trần
├─ ④ CUDA runtime + driver ───────   nvgpu, không tách rời L4T được
├─ ⑤ Bộ nhớ ─────────────────────   66.8 GB/s, CPU và GPU DÙNG CHUNG
└─ ⑥ Silicon ────────────────────   8 SM, 32 tensor core, EMC 2133 MHz
```

**Trần cuối cùng nằm ở ⑤ và không framework nào phá được:**

```
tok/s ≤ 66.8 GB/s ÷ (kích thước model + KV cache)
```

Framework chỉ quyết định bạn đạt **50% hay 85%** của trần đó. Chọn framework không bao
giờ cho bạn 5× nếu bạn đang ở 70% trần — nhưng nó cho bạn 3× nếu bạn đang ở 20% vì
launch overhead.

**Bằng chứng ngay trong repo này** ([`../firmware/jetson/`](../firmware/jetson/)):
model 1.87 MB chạy 1141 tok/s trên Orin, với 117 kernel/token × 3.71 µs =
**0.434 ms/token là launch overhead thuần, đúng 50% tổng thời gian**. Tầng ② ăn một nửa,
tầng ⑤ gần như không đụng tới. Đó là điều bạn *không* đoán được nếu chỉ đọc datasheet.

> **Bài học tổng quát:** model càng nhỏ so với máy, chi phí càng dồn lên tầng ①②.
> Model càng lớn, càng dồn xuống ⑤. Biết mình ở chế độ nào trước khi chọn công cụ.

---

## 13.2 Gốc: Jetson không phải "Linux có GPU rời"

Đây là nguồn gốc của gần như mọi rắc rối cài đặt.

| | Desktop x86 + RTX | Jetson |
|---|---|---|
| Kiến trúc | x86_64 | **aarch64** — wheel x86 vô dụng |
| GPU | rời, qua PCIe | **tích hợp trong SoC**, không PCIe |
| Bộ nhớ | VRAM riêng | **dùng chung DRAM với CPU** |
| Driver | tải rời, tự chọn version | **gắn cứng với L4T**, không thay riêng |
| CUDA | cài version nào cũng được | **khoá theo JetPack** |
| `nvidia-smi` | có | **KHÔNG CÓ** — dùng `tegrastats` / `jtop` |
| DLA | không | Orin **Nano không có**; NX/AGX có 1–2 |

**L4T vs JetPack:**
- **L4T (Linux for Tegra)** = kernel Tegra + driver GPU + BSP. Là *nền*.
- **JetPack** = L4T + CUDA + cuDNN + TensorRT + VPI + OpenCV. Là *bộ SDK* đóng gói trên nền đó.
- Quan hệ khoá cứng: JetPack 6.2 ⇔ L4T r36.4.x ⇔ CUDA 12.6 ⇔ TensorRT 10.x.
  **Không nâng CUDA lẻ mà giữ nguyên L4T.**

### Mười lệnh xác định board bạn đang thực sự có gì

Làm việc này **trước tiên**, mọi lần, trước khi tra bất kỳ hướng dẫn nào trên mạng:

```bash
cat /etc/nv_tegra_release              # ① phiên bản L4T (R36.4.7 = JetPack 6.2)
head -1 /etc/nv_boot_control.conf      # ② model board thật
uname -m && lsb_release -d             # ③ aarch64 + Ubuntu 22.04
nvcc --version                         # ④ CUDA toolkit (12.6.68)
python3 -c "import sys; print(sys.version)"   # ⑤ Python hệ thống (3.10 trên JP6)
dpkg -l | grep -E 'tensorrt|cudnn|cuda-toolkit' | awk '{print $2,$3}'  # ⑥ version thật
nvpmodel -q                            # ⑦ power mode (phải MAXN_SUPER)
sudo jetson_clocks --show | head       # ⑧ clock GPU/EMC thật
free -g && df -h /                     # ⑨ RAM và đĩa còn trống
tegrastats --interval 1000             # ⑩ máy có rảnh không (Ctrl-C để dừng)
```

Ghi kết quả 10 lệnh này vào đầu mọi báo cáo benchmark. Không có nó, con số vô nghĩa —
đúng tinh thần [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md).

> **`nvidia-smi` không tồn tại trên Jetson.** GPU tích hợp không có giao diện đó. Ai đưa
> bạn hướng dẫn Jetson mà dùng `nvidia-smi` là đang copy hướng dẫn desktop — nghi ngờ
> toàn bộ phần còn lại của hướng dẫn ấy.

---

## 13.3 Vì sao `pip install torch` hỏng, và ba con đường đúng

`pip install torch` kéo về wheel **x86 hoặc aarch64-generic build cho GPU rời**. Kết quả
kinh điển:

```python
>>> torch.cuda.is_available()
False                      # 99% là do sai wheel, không phải hỏng driver
```

Ba con đường, theo thứ tự nên thử:

### Con đường A — wheel dựng riêng cho Jetson
NVIDIA và cộng đồng jetson-ai-lab phát hành wheel **khớp đúng JetPack**. Quy tắc:
`torch` + `torchvision` + `torchaudio` phải **cùng một bộ build**, và bộ đó phải khớp
`cu126` ↔ JetPack 6.2.

```bash
# Kiểm JetPack trước (13.2), rồi lấy index tương ứng jp6/cu126
pip3 install --no-cache-dir --extra-index-url <index-jp6-cu126> torch torchvision
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# kỳ vọng: ... True (8, 7)
```

`(8, 7)` = sm_87. Ra `False` → sai wheel, gỡ và làm lại; đừng đi cài driver.

### Con đường B — `jetson-containers` (dusty-nv)
Image dựng sẵn đúng cặp phiên bản cho từng JetPack: `pytorch`, `llama_cpp`, `ollama`,
`tensorrt_llm`, `onnxruntime`, `mlc`… **Đây là lựa chọn mặc định cho người mới**, vì
99% thời gian mất trên Jetson là mất vào ma trận phiên bản, không vào code.

Đổi lại: image lớn (nhiều GB) và bạn vẫn nên biết bên trong nó là gì — nếu không, khi
hỏng bạn không sửa được. Đọc `Dockerfile` của image bạn dùng ít nhất một lần.

### Con đường C — build từ nguồn
Chỉ đáng với thứ build nhanh và ít phụ thuộc. **llama.cpp là ứng viên tốt nhất** — chỉ
cần `nvcc`, ~10–20 phút trên Orin Nano:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc) --config Release
```

`CMAKE_CUDA_ARCHITECTURES=87` là bắt buộc — thiếu nó, hoặc build cho arch khác, bạn mất
thời gian JIT hoặc không chạy được. (Cờ cũ tên `LLAMA_CUBLAS`; bản mới là `GGML_CUDA`.
Kiểm `CMakeLists.txt` của đúng commit bạn đang build.)

Ngược lại, **build PyTorch hay TensorRT-LLM từ nguồn trên Orin Nano 8GB là lựa chọn tồi**:
hàng giờ, và dễ OOM (xem §13.6 về swap).

### Trường hợp board không có internet — đúng board này

`09-so-do-phan-cung.md` ghi: board không ra được `huggingface.co`. Quy trình offline:

```bash
# TRÊN MÁY x86 CÓ MẠNG
curl -LO <đường-dẫn-wheel-aarch64>          # tải thẳng file .whl, đừng pip download
huggingface-cli download <repo> <file.gguf> --local-dir ./gguf
docker pull <image>:<tag> && docker save <image>:<tag> | zstd -o img.tar.zst

# CHUYỂN SANG BOARD
rsync -avP --partial ./*.whl ./gguf ./img.tar.zst user@board:/mnt/nvme/stage/

# TRÊN BOARD
pip3 install --no-index /mnt/nvme/stage/*.whl
zstd -d -c img.tar.zst | docker load
```

Ba lưu ý thực chiến:
1. **Chuyển vào NVMe, không vào eMMC/SD.** Model 5 GB + image 8 GB làm đầy rootfs rất nhanh;
   board này chỉ còn 49 GB.
2. **`rsync --partial`** để nối lại được khi rớt mạng, thay vì `scp` phải làm lại từ đầu.
3. **Ghi lại checksum** (`sha256sum`) hai đầu. File GGUF hỏng nửa chừng cho lỗi rất khó hiểu.

---

## 13.4 Bốn tầng chi phí — framework nào chữa tầng nào

Đây là bảng quan trọng nhất của cả file. Mỗi framework **chỉ chữa được vài tầng**; biết
mình đang tắc ở tầng nào thì việc chọn trở nên hiển nhiên.

| Tầng | Triệu chứng | Đo bằng | Cách chữa |
|---|---|---|---|
| ① Python/binding | GPU rảnh, CPU 1 core 100% | `nsys` thấy khoảng trống giữa kernel | bỏ Python khỏi vòng lặp nóng: C++, hoặc `torch.compile` |
| ② Launch/dispatch | rất nhiều kernel ngắn (< 10 µs) | `nsys` + kernel rỗng để hiệu chuẩn | **CUDA Graphs**, fusion, batch |
| ③ Kernel kém | kernel dài nhưng %peak thấp | `ncu`: `sm__throughput` và `dram__throughput` đều thấp | thư viện tốt hơn (TRT), tile/precision khác |
| ④ Bandwidth | `dram__throughput` ~ 100% | `ncu`, hoặc so với 66.8 GB/s | **quantize**, giảm KV, speculative decoding |

**Cách phân loại chỉ bằng hai con số trong `ncu`** (đã có ở [11.7](11-toi-uu-nvidia.md)):

```
sm__throughput cao   → compute-bound     → tầng ③
dram__throughput cao → memory-bound      → tầng ④  ← LLM decode luôn ở đây khi model đủ lớn
cả hai đều thấp      → latency/launch    → tầng ①②  ← model của repo này ở đây
```

Bây giờ đối chiếu từng framework:

| Framework | ① Python | ② Launch | ③ Kernel | ④ Bandwidth | Thực chất nó là gì |
|---|---|---|---|---|---|
| **PyTorch eager** | ❌ tệ nhất | ❌ mỗi op 1 launch | ✅ cuBLAS/cuDNN tốt | — | thư viện *nghiên cứu*: linh hoạt, không tối ưu inference |
| **torch.compile** | ✅ | ✅ fusion + graph | ✅ | — | biên dịch graph, giữ nguyên Python API |
| **llama.cpp** | ✅ C++ thuần | ⚠️ có CUDA graph | ✅ kernel viết tay cho GGUF | ✅ quantize sẵn | runtime LLM tối giản, mmap GGUF |
| **TensorRT** | ✅ | ✅✅ fusion mạnh nhất | ✅✅ **autotune trên chính máy bạn** | ✅ per-layer precision | trình biên dịch graph tĩnh |
| **TensorRT-LLM** | ✅ | ✅✅ | ✅✅ + paged KV, in-flight batching | ✅✅ INT4 AWQ | TRT + phần LLM chuyên dụng |
| **ONNX Runtime + TRT EP** | ⚠️ | ✅ | ✅ (fallback CUDA EP khi TRT không nuốt được) | ✅ | lớp bọc tiện, thêm overhead |
| **MLC-LLM** | ✅ | ✅ | ✅ TVM auto-schedule | ✅ | sinh kernel bằng compiler |

**Ba điều framework KHÔNG làm được cho bạn:**
1. Không phá được trần `66.8 / model_GB` ở batch = 1. Muốn phá thì phải đổi *thuật toán*
   (speculative decoding), không đổi thư viện.
2. Không sửa được power mode sai. `nvpmodel -m 2` cho **1.9×** — lớn hơn phần lớn khác biệt giữa các framework.
3. Không cứu được model không vừa RAM. 8 GB dùng chung, xem §13.6.

---

## 13.5 Chọn cái gì — bảng quyết định

| Bạn đang làm | Chọn | Vì sao |
|---|---|---|
| Mới bắt đầu, muốn chạy LLM hôm nay | **llama.cpp** | build 20 phút, GGUF sẵn, `llama-bench` đo được ngay |
| Cần API kiểu OpenAI cho app | llama.cpp `llama-server` / ollama | server có sẵn, không phải viết |
| Vision (YOLO/ResNet/Whisper encoder) | **ONNX → TensorRT** | model tĩnh, đúng sở trường TRT, thường 2–4× so với torch |
| Production LLM, cần throughput nhiều luồng | TensorRT-LLM | in-flight batching + paged KV |
| Đang nghiên cứu, cần sửa model | PyTorch (+ `torch.compile` khi đo) | linh hoạt trước, tốc độ sau |
| Model tự viết, muốn hiểu tận gốc | CUDA tay như [`../firmware/jetson/`](../firmware/jetson/) | không có tầng nào che mất bản chất |

**Kỳ vọng số trên board này** (trần tính từ 66.8 GB/s, thực tế 50–70% trần):

| Model | Q4 | trần | kỳ vọng thực |
|---|---:|---:|---:|
| Llama-3.2-1B | 0.7 GB | 95 tok/s | 48–67 |
| Qwen2.5-1.5B | 1.0 GB | 67 | 33–47 |
| Llama-3.2-3B | 1.9 GB | 35 | 18–25 |
| Llama-3.1-8B | 4.8 GB | 14 | 7–10 |

Nếu đo được **thấp hơn 40% trần** → gần như chắc chắn là lỗi cấu hình (power mode, thiếu
`-ngl 99`, tải nền, KV fp32), **không phải** lỗi framework. Kiểm §13.2 trước khi đổi công cụ.

---

## 13.6 Ngân sách 8 GB — tính trước khi tải model

RAM là **dùng chung**: OS + framework + weights + KV + activation, tất cả trong 7.98 GB.

```
weights        = tham số × bit/8            Llama-8B Q4  ≈ 4.8 GB
KV cache       = 2 × L × n_kv_head × d_head × ctx × 2B    8B @ctx4096 ≈ 0.54 GB
activation     ≈ vài trăm MB (batch 1)
CUDA context   ≈ 300-600 MB (chỉ riêng việc khởi tạo)
OS + desktop   ≈ 1.2-2.5 GB  ← chạy headless tiết kiệm ~1 GB
```

Quy tắc nhẩm: **model Q4 nên ≤ 4.5 GB nếu chạy có desktop, ≤ 5.5 GB nếu headless.**

Ba lever khi thiếu:
```bash
sudo systemctl set-default multi-user.target   # bỏ desktop, +1 GB, cần reboot
--cache-type-k q8_0 --cache-type-v q8_0        # KV int8: KV còn ~1/2
--ctx-size 2048                                # KV tỉ lệ thuận với ctx
```

**Swap — chỉ để build, không để chạy.** Build engine TensorRT rất dễ OOM ở 8 GB:
```bash
sudo systemctl disable nvzramconfig            # zram mặc định chiếm RAM thật
sudo fallocate -l 16G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```
Nhưng **inference tràn vào swap là hỏng**: tok/s tụt hàng chục lần vì mỗi token phải đọc
lại weights từ NVMe (~1-2 GB/s) thay vì DRAM (66.8 GB/s) — đúng công thức roofline,
chỉ thay mẫu số. Thấy tok/s tụt đột ngột thì kiểm `free -g` trước tiên.

---

## 13.7 Lab — sáu bài từ gốc lên

Mỗi bài: **mục tiêu → lệnh → phải thấy gì → bẫy**. Bài 0–2 chạy được ngay với thứ đã có
sẵn trên board (`~/jetson-optim/bench/` đã build).

### Lab 0 — Xác lập môi trường đo (30 phút). *Không có bài này thì mọi bài sau vô nghĩa.*

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks && nvpmodel -q     # phải in MAXN_SUPER
tegrastats --interval 1000                                   # GR3D_FREQ ~0%
./bench_roofline 256                                         # 3 lần, lấy median
```
**Phải thấy:** ~66.8 GB/s (read float4), ~1.88 TFLOP/s FP32.
**Bẫy:** đo khi có tải nền → sai **2.4×**. Đo ở mode 0 → sai **1.9×**. Cả hai đều *không*
báo lỗi, chỉ cho số đẹp/xấu sai.

### Lab 1 — Đo chi phí tầng ② bằng kernel rỗng (30 phút)

```bash
./bench_cuda ../model/model.bin 200 66.8
```
**Phải thấy:** `launch overhead ĐO ĐƯỢC: 3.71 us/kernel`, `117 kernel/token`,
`50% thời gian là launch overhead`.
**Câu hỏi:** nếu gộp 117 kernel thành 1 graph thì trần lý thuyết là bao nhiêu? So với
1.59× đo được ở [`11-toi-uu-nvidia.md §11.5`](11-toi-uu-nvidia.md).
**Bẫy:** đây là *chi phí của mọi framework*, không riêng code này. PyTorch eager trên
cùng model sẽ tệ hơn vì cộng thêm tầng ①.

### Lab 2 — Cùng một model.bin, ba runtime (1 giờ)

```bash
cd firmware/jetson
make verify        # đúng chưa: argmax MATCH
make bench         # CUDA:  ~1141 tok/s
make host_verify   # scalar C trên CPU của chính board, cùng file model.bin
```
**Phải thấy:** cùng argmax, tốc độ khác hàng chục lần.
**Ý nghĩa:** bạn có một **baseline không framework**. Mọi framework sau này đo được nhanh
hơn/chậm hơn cái này đều giải thích được bằng bảng 4 tầng ở §13.4.

### Lab 3 — LLM thật bằng llama.cpp, khép lại "việc chưa làm được" (nửa ngày)

Đây chính là mục còn treo trong [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md).

```bash
# build (§13.3 con đường C), model GGUF chuyển sang bằng rsync (§13.3 offline)
./build/bin/llama-bench -m qwen2.5-1.5b-q4_k_m.gguf -ngl 99 -p 512 -n 128 -r 3
```
**Phải thấy:** hai con số tách biệt — `pp512` (prefill, token/s cao) và `tg128`
(decode, thấp hơn nhiều). Đó là [`03-roofline.md`](03-roofline.md) hiện ra trên máy thật.
**So sánh bắt buộc:**
```bash
python3 samples/gpu/roofline.py --hw orin-nano-super --model qwen2.5-1.5b --bits 4.5
```
Đạt 50–70% trần → bình thường. Dưới 40% → quay lại Lab 0.
**Bẫy:** quên `-ngl 99` thì layer chạy trên CPU và bạn đo nhầm CPU. Đây là lỗi phổ biến
nhất, và nó **không báo lỗi** — chỉ chậm.

### Lab 4 — Đo chi phí của chính PyTorch (nửa ngày)

Chạy **đúng một phép matvec** giống `bench_cuda`, bằng PyTorch, rồi so:

```python
import torch, time
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
v = torch.randn(4096, 1,    device='cuda', dtype=torch.float16)
for _ in range(10): y = x @ v
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(1000): y = x @ v
torch.cuda.synchronize(); print((time.perf_counter()-t)/1000*1e6, "us")
```
**Phải thấy:** thời gian mỗi lần lớn hơn hẳn thời gian kernel thật (`ncu` cho biết kernel
bao lâu). Chênh lệch đó chính là tầng ①+②.
**Rồi bọc lại bằng `torch.cuda.CUDAGraph` hoặc `torch.compile(mode="reduce-overhead")`**
và đo lại — bạn sẽ tái tạo được đúng hiệu ứng 1.59× của repo, ở tầng framework.
**Bẫy:** không `torch.cuda.synchronize()` thì bạn đang đo tốc độ *xếp hàng lệnh*, không
phải tốc độ tính. Đây là lỗi benchmark GPU phổ biến nhất trên đời.

### Lab 5 — ONNX → TensorRT cho vision (1 ngày)

```bash
trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16 --verbose
trtexec --loadEngine=model.fp16.engine --iterations=200 --avgRuns=100
```
**Phải thấy:** log liệt kê các layer **bị fusion** (`Conv+BN+ReLU` gộp làm một) và thời
gian autotuning từng layer. So `--fp16` với FP32 và với `--int8`.
**Ý nghĩa:** bạn thấy tận mắt hai thứ TRT làm mà bạn khó tự làm — fusion (bỏ round-trip
DRAM cho tensor trung gian) và autotuning trên **chính** phần cứng này.
**Bẫy:** engine **không portable** — build trên x86 rồi copy sang Orin là không chạy.
Phải build trên board, và build lại khi nâng JetPack.

### Lab 6 — Profiling nghiêm túc (1 ngày)

```bash
nsys profile -t cuda,nvtx -o prof ./app        # thời gian đi đâu ở mức ứng dụng
sudo ncu --set full -k <tên_kernel> ./app      # kernel này vì sao chậm
```
**Bài tập phân loại:** với mỗi workload bạn có, dùng hai chỉ số `sm__throughput` và
`dram__throughput` để xếp nó vào một trong bốn tầng ở §13.4. Viết ra một câu:
*"workload này bị chặn bởi ___, nên bước tối ưu tiếp theo là ___."*
Không viết được câu đó thì chưa được phép tối ưu.

---

## 13.8 Mười bẫy đặc thù Jetson

| # | Bẫy | Hậu quả | Phát hiện |
|---:|---|---|---|
| 1 | `nvpmodel -m 0` tưởng là MAXN | mất **1.9×** | `nvpmodel -q` phải in MAXN_SUPER |
| 2 | Tải nền chiếm GPU | sai **2.4×** | `tegrastats`, GR3D_FREQ |
| 3 | Tin datasheet 102 GB/s | sai **1.53×** | đo `bench_roofline` |
| 4 | `pip install torch` | `cuda.is_available() False` | wheel Jetson đúng JetPack |
| 5 | Copy engine TRT từ máy khác | không chạy / sai | build trên chính board |
| 6 | Quên `-ngl 99` | chậm 5–20×, **không báo lỗi** | log llama.cpp: layer offloaded |
| 7 | `cudaMemcpy` H2D theo thói quen desktop | đốt băng thông 2 lần | `nsys` thấy memcpy; dùng zero-copy |
| 8 | Không `synchronize()` khi bench | số đẹp giả | so với `ncu` |
| 9 | Inference tràn swap | tụt hàng chục lần | `free -g` |
| 10 | Tìm DLA trên Orin Nano | mất thời gian | Nano **không có** DLA |

Ba bẫy đầu đã **đo được** trên chính board này. Nhân lại: 1.9 × 2.4 = **4.6×** sai lệch
chỉ từ hai lỗi vận hành, lớn hơn mọi khác biệt giữa các framework trong bảng §13.4.

> **Hệ quả nghề nghiệp:** khi ai đó khoe "framework X nhanh gấp 3 framework Y trên
> Jetson", câu hỏi đầu tiên không phải "kernel nào tốt hơn" mà là **"hai lần đo có cùng
> power mode, cùng trạng thái tải, cùng ctx, cùng quantization không?"**. Phần lớn so
> sánh trên mạng trượt ở đúng câu đó.

---

## 13.9 Bài tập

1. Chạy 10 lệnh §13.2, viết "fact sheet" cho board bạn. Dán vào đầu mọi báo cáo sau này.
2. Lab 3 tới cùng: một model GGUF thật, `llama-bench`, so với `roofline.py`. Đạt bao
   nhiêu % trần? Nếu < 50%, tìm nguyên nhân bằng bảng §13.8.
3. Quét `--ctx-size` 512 / 2048 / 8192 với cùng model. Vẽ tok/s theo ctx, đối chiếu công
   thức KV ở §13.6. Điểm nào tok/s bắt đầu tụt phi tuyến, và vì sao?
4. Bật/tắt `--cache-type-k q8_0`. Đo **cả** tok/s **lẫn** chất lượng (perplexity). Đáng đổi không?
5. Lab 4: đo chênh lệch giữa thời gian PyTorch báo và thời gian kernel trong `ncu`. Đó là
   tầng ①②. `torch.compile(mode="reduce-overhead")` lấy lại được bao nhiêu %?
6. Chạy benchmark 30 giây và 10 phút liên tục, ghi `tegrastats`. Throttle làm tụt bao nhiêu?
   (Đây vẫn là mục **chưa đo được** trong `09-so-do-phan-cung.md` — làm xong thì cập nhật file đó.)
7. Với một workload bất kỳ của bạn, viết câu: *"bị chặn bởi tầng ___, bước tiếp theo là ___."*

**Tiêu chí "đã hiểu":** nhìn một con số tok/s bất kỳ, bạn nói được ngay nó **nên** là bao
nhiêu (từ roofline), nó **đang** là bao nhiêu % trần, và **tầng nào** đang ăn phần chênh
lệch. Ba câu đó là toàn bộ nghề tối ưu trên Jetson.

---

→ [README.md](README.md) · Nền: [03-roofline.md](03-roofline.md) ·
Kernel: [11-toi-uu-nvidia.md](11-toi-uu-nvidia.md) ·
Số đo board: [09-so-do-phan-cung.md](09-so-do-phan-cung.md)
