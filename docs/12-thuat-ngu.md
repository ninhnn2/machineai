# 12. Thuật ngữ — tra nhanh

Mỗi mục: **là gì** (1–2 câu) → *trong repo này là gì*. Sắp theo bảng chữ cái tiếng Anh vì
thuật ngữ gốc là tiếng Anh. Chưa biết gì thì đọc [`00-nhap-mon.md`](00-nhap-mon.md) trước.

---

### argmax
Chỉ số của phần tử lớn nhất trong vector. Với logits, đó là token model tự tin nhất.
*Tiêu chí PASS của tầng 1: cả PyTorch, C và CUDA phải cùng ra `top=580`. Không dùng
`max|diff| < 1e-5` vì cộng số thực không kết hợp.*

### arithmetic intensity (AI)
Số phép tính trên mỗi byte đọc, đơn vị FLOP/byte. So nó với **machine balance** để biết
memory-bound hay compute-bound. *LLM decode FP16 có AI ≈ 8, balance của Orin là 152 →
memory-bound gấp 19 lần.* → [`03-roofline.md`](03-roofline.md)

### attention
Phép cho token hiện tại "nhìn" các token trước. Là thứ **duy nhất** trộn thông tin *giữa*
các token; chi phí tăng theo độ dài context. *Đo được: 25.6 ms/token trên ESP32 (25%).*

### AWQ / GPTQ / SmoothQuant
Ba phương pháp quantize thông minh hơn "chia đều": AWQ giữ nguyên các kênh quan trọng,
GPTQ bù lỗi theo thứ tự cột, SmoothQuant chuyển độ khó từ activation sang weight.
*Repo dùng round-to-nearest group-wise, đơn giản nhất — đủ vì bảng lớn chịu lỗi tốt.*
→ [`04-quantization.md`](04-quantization.md)

### bandwidth
Tốc độ đọc/ghi bộ nhớ, MB/s hoặc GB/s. **Luôn phải đo, đừng lấy datasheet.**
*Đo được: PSRAM ESP32 60.7 MB/s, SRAM 240 MB/s, DRAM Orin 66.8 GB/s (datasheet 102).*

### bank conflict
Shared memory GPU chia 32 bank; hai thread trong cùng warp đọc hai địa chỉ khác nhau
cùng bank thì bị tuần tự hoá. *Cách chữa cổ điển: đệm mảng thành `[N][33]`.*
→ [`11-toi-uu-nvidia.md §11.3`](11-toi-uu-nvidia.md)

### BPE (Byte-Pair Encoding)
Thuật toán học từ điển token bằng cách ghép dần cặp ký tự hay đi cùng nhau.
*`data/prepare.py` train BPE → `data/bpe4096.json`; bản C ở `firmware/jetson/bpe.h` phải
ra đúng ids như bản Python (tầng 0 của chuỗi kiểm chứng).*

### coalescing
Các thread liền nhau trong một warp đọc các địa chỉ liền nhau → GPU gộp thành ít giao
dịch bộ nhớ. Không coalesce có thể chậm 8–32×. → [`11-toi-uu-nvidia.md §11.3`](11-toi-uu-nvidia.md)

### compute-bound
Đang bận tính, dữ liệu không thiếu. Chữa bằng SIMD / nhiều core / tensor core.
Ngược với **memory-bound**. *Bậc L0–L2 của `matvec_ladder` là compute-bound (bận gỡ
nibble int4), từ L3 trở đi mới chạm bandwidth.*

### CUDA Graph
Ghi lại cả chuỗi kernel một lần rồi phát lại bằng một lệnh, cắt gần hết chi phí launch.
*Đo được 1.59× trên Orin: 1116 → 1774 tok/s.* → [`11-toi-uu-nvidia.md §11.5`](11-toi-uu-nvidia.md)

### decode / prefill
**Prefill** = nạp prompt, xử lý T token cùng lúc → compute-bound, quyết định **TTFT**.
**Decode** = sinh từng token một → memory-bound, quyết định tok/s. Hai chế độ có nút thắt
*ngược nhau*; tối ưu cho cái này có thể vô ích với cái kia.

### DeepStream
**GStreamer + bộ plugin NVIDIA + chuẩn metadata** cho pipeline video thời gian thực. Nó
**không tự suy luận** — plugin `nvinfer` bên trong gọi TensorRT. Sản phẩm thật của nó là
cây metadata (`NvDsBatchMeta`), không phải ảnh có vẽ box.
→ [`14-tensorrt-deepstream.md`](14-tensorrt-deepstream.md)

### dequantize
Đổi int4/int8 về float bằng scale để tính. *`golden.txt` là logits của model **đã**
dequantize — cố ý, để tầng 1 chỉ đo lỗi port, tách khỏi lỗi quantize đo ở tầng 2.*

### DP4A
Lệnh CUDA core làm 4 phép nhân int8 + cộng dồn trong 1 chu kỳ. Đường INT8 không cần
tensor core. → [`11-toi-uu-nvidia.md §11.4`](11-toi-uu-nvidia.md)

### EMC (External Memory Controller)
Bộ điều khiển DRAM trên Tegra/Jetson. Xung của nó chặn trần bandwidth thật.
*Orin Nano Super khoá 2133 MHz → 68.3 GB/s lý thuyết, đo được 66.8 (98%). `cudaDeviceProp.
memoryClockRate` **không đáng tin** trên Tegra.* → [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md)

### engine (`.engine`)
File nhị phân TensorRT sinh ra sau khi biên dịch. **Không portable**: gắn chặt với model
+ phiên bản TensorRT + kiến trúc GPU + precision + shape profile. Đổi một trong số đó là
phải build lại trên chính board.

### FFN / SwiGLU
Khối biến đổi **từng token độc lập**. SwiGLU = `down(silu(gate(x)) · up(x))` — 3 ma trận
thay vì 2, nhánh `gate` học "cho gì đi qua". *Đo được 6.9 ms/token trên ESP32.*

### FlashAttention
Attention 1 lượt, không bao giờ ghi ma trận điểm số `[T,T]` ra bộ nhớ chính; softmax tính
tăng dần. Lợi ích chính là **tiết kiệm bộ nhớ + băng thông**, không phải bớt phép tính.

### fp32 / fp16 / bf16 / fp8
Các định dạng số thực. **bf16** có dải động bằng fp32 nhưng ít chữ số nghĩa hơn fp16 —
nên train ổn định hơn. Nhớ: *dải động* và *độ chính xác* là hai thứ khác nhau.
→ [`10-ly-thuyet-nen.md §10.1`](10-ly-thuyet-nen.md)

### GGUF
Định dạng file model của llama.cpp: weights đã quantize + metadata + tokenizer trong một
file, đọc bằng `mmap`. *Cùng tinh thần với `model.bin` của repo này — một artifact tự mô tả.*

### golden / golden.txt
Bộ logits tham chiếu do PyTorch sinh lúc export. Mọi runtime (C, CUDA, ESP32) phải khớp
argmax với nó. *Là "sự thật" duy nhất của repo — mất nó thì không biết port đúng hay sai.*

### group (quantization)
Số tham số dùng chung một scale. Nhỏ hơn = chính xác hơn nhưng file to hơn.
*Repo dùng group=128 → chi phí thật 4 + 16/128 = **4.125 bit**/tham số.*

### GQA / MQA / MHA
Cách chia sẻ K,V giữa các head. MHA: mỗi head một bộ. MQA: tất cả dùng chung một bộ.
GQA: ở giữa. **Mục đích chính là thu nhỏ KV cache**, không phải giảm phép tính.

### GStreamer / NVMM
GStreamer là khung pipeline đa phương tiện; DeepStream là tập plugin chạy trên nó.
**NVMM** (`video/x-raw(memory:NVMM)`) là loại buffer nằm trong vùng nhớ GPU truy cập được
— giữ buffer trong NVMM suốt pipeline chính là "zero-copy". Chèn nhầm một plugin CPU
(`videoconvert` thay vì `nvvideoconvert`) là lỗi hiệu năng số 1 của người mới DeepStream.

### head (output head)
Ma trận cuối `[V, D]` biến vector ẩn thành V logits. Phải quét **toàn bộ** mỗi token.
*To nhất và tốn nhất: 57.6/102.9 ms trên ESP32. Ở đây nó **tied** với `tok_emb`.*

### JetPack / L4T
**L4T** = kernel Tegra + driver GPU + BSP (cái nền). **JetPack** = L4T + CUDA + cuDNN +
TensorRT + VPI (bộ SDK). Khoá cứng theo nhau: JetPack 6.2 ⇔ L4T r36.4.x ⇔ CUDA 12.6.
*Không nâng CUDA lẻ mà giữ nguyên L4T. Đây là nguồn lỗi cài đặt số 1 trên Jetson.*

### jetson-containers
Bộ image Docker dựng sẵn (dusty-nv) khớp đúng từng phiên bản JetPack: pytorch, llama.cpp,
ollama, tensorrt_llm… *Đáng dùng vì 99% thời gian mất trên Jetson là mất vào ma trận
phiên bản, không phải vào code.*

### KV cache
Bộ nhớ giữ K,V của các token đã sinh để khỏi tính lại. Tốn cả **dung lượng** lẫn
**băng thông**, và lớn theo context. *Llama-8B @ctx4096 tốn 0.54 GB — hơn 10% RAM của
Orin 8GB, chỗ hay bị quên khi tính "có vừa RAM không".* → [`07-kv-cache-engine.md`](07-kv-cache-engine.md)

### launch overhead
Chi phí cố định để khởi động một kernel GPU. *Đo được 3.71 µs/kernel trên Orin, 2.5 µs
trên RTX 4060. Model của repo tốn 117 kernel/token → **50% thời gian là overhead thuần**.*

### logits
Vector V số thực chưa chuẩn hoá, mỗi phần tử là điểm của một token. Qua softmax thành
xác suất.

### LX7 / Xtensa
Lõi CPU của ESP32-S3 (2 lõi, 240 MHz). Không có SIMD kiểu AVX/NEON dùng được dễ dàng từ
C thuần — nên bản ESP32 là scalar C.

### machine balance
`FLOPS đỉnh / bandwidth đỉnh`. Kernel có AI thấp hơn ngưỡng này là memory-bound.
*Orin: 152 FLOP/byte (FP16), 192 OP/byte (INT8).*

### matvec
Ma trận × vector. ~90% thời gian của LLM decode. Mỗi phần tử ma trận đọc 1 lần dùng 1
lần → AI thấp → memory-bound. *`llm.h` có `matvec_q` (int4) và `matvec_q8` (int8).*

### memory-bound
Đang **chờ dữ liệu**, đơn vị tính rảnh. Chữa bằng: đọc ít byte hơn (quantize, model nhỏ
hơn), hoặc đặt dữ liệu ở tầng nhanh hơn. **Decode ở batch=1 gần như luôn memory-bound.**

### mmap / XIP
Ánh xạ file trong flash vào không gian địa chỉ, đọc trực tiếp không cần copy vào RAM
(XIP = execute in place). *Bảng PLE 25M tham số nằm nguyên trong flash theo cách này.*

### MoE (Mixture of Experts)
Nhiều tham số nhưng mỗi token chỉ kích hoạt một phần nhỏ. *Cùng nguyên lý với bảng PLE
của repo: 43.7% tham số tốn 0.024% băng thông.*

### nats
Đơn vị của cross-entropy loss. `ppl = e^nats`, `nats = ln(ppl)`. Thấp hơn = tốt hơn.
*Quy đổi thô: 0.01 nats ≈ 1% perplexity. Repo báo cáo "+0.098 nats = 9.3% ppl".*

### nibble
Nửa byte (4 bit). *Định dạng int4 của repo nhét 2 giá trị vào 1 byte; giá trị thật =
`code − 8`. "Gỡ nibble" (unpack) tốn phép tính — bậc L2 của thang tối ưu tránh nó bằng
cách gỡ **một lần** lúc boot rồi giữ int8.*

### NVDEC / NVENC
Khối phần cứng giải mã / mã hoá video, **tách rời GPU** nên chạy song song không tốn SM.
⚠️ *Jetson Orin **Nano** có NVDEC nhưng **không có NVENC** — mọi pipeline xuất RTSP/ghi
file phải mã hoá bằng CPU. Orin NX/AGX mới có encoder.*

### nvinfer
Plugin DeepStream bọc TensorRT: tiền xử lý → chạy engine → hậu xử lý → gắn metadata.
*Khoá đáng nhớ: `network-mode` (0=FP32/1=INT8/2=FP16), `batch-size`, và `interval` —
lever rẻ nhất, bỏ bớt khung suy luận và để tracker nội suy.*

### nvpmodel / jetson_clocks / tegrastats
Ba lệnh vận hành Jetson: đặt power mode, khoá clock ở mức tối đa, xem trạng thái thực.
*`nvidia-smi` **không tồn tại** trên Jetson. MAXN_SUPER là **mode 2**, không phải 0 —
đặt sai mất 1.9×.*

### occupancy
Tỉ lệ warp đang hoạt động trên mức tối đa của một SM. **Occupancy cao không đồng nghĩa
nhanh** — nó chỉ là điều kiện để che độ trễ bộ nhớ. → [`11-toi-uu-nvidia.md §11.2`](11-toi-uu-nvidia.md)

### ONNX
Định dạng graph model trung gian giữa các framework. *Đường vào chuẩn của TensorRT:
PyTorch → ONNX → engine. Lỗi "unsupported op" xuất hiện ở bước parse — sửa ở khâu export
là rẻ nhất, viết plugin là đắt nhất.*

### perplexity (ppl)
`e^loss`. Hiểu nôm na: "model đang phân vân giữa bao nhiêu lựa chọn". *Dùng để nghiệm thu
quantization: delta < 0.1 là ổn, > 0.5 là có vấn đề.*

### PLE (Per-Layer Embeddings)
Ý tưởng của Google (Gemma 3n): bảng embedding khổng lồ tiêm điều kiện vào **từng lớp**,
mỗi token chỉ tra 1 hàng. *Ý tưởng trung tâm của repo — cho phép 25M tham số nằm ở flash
mà gần như không tốn băng thông. Tiêm ở mỗi lớp hơn tiêm ở đáy 0.046 nats (`ple` vs
`fatembed`).*

### polygraphy
Công cụ so đầu ra TensorRT với ONNX Runtime trên cùng input, và **bisect theo layer** để
tìm layer đầu tiên lệch. *Cách đúng để debug engine sai — thay vì đoán.*

### PSRAM / SRAM / flash (ESP32-S3)
`SRAM 512KB` nhanh nhất, khan hiếm → chứa core. `PSRAM 8MB` trung bình (60.7 MB/s) →
chứa output head. `Flash 16MB` chậm, bao la → chứa bảng PLE. Ba tầng này là toàn bộ bài
toán thiết kế.

### PTQ / QAT
PTQ: quantize **sau** khi train xong (repo dùng cái này). QAT: mô phỏng quantize **trong
lúc** train, chất lượng cao hơn nhưng phải train lại.

### RMSNorm
LayerNorm bỏ phần trừ trung bình và bỏ bias — chỉ chuẩn hoá theo độ lớn. Rẻ hơn, chất
lượng tương đương. *Giữ **fp32** kể cả khi mọi thứ khác 4-bit: nguyên tắc chung là
**không quantize tensor nhỏ**.*

### roofline
Mô hình vẽ trần hiệu năng theo arithmetic intensity: dốc lên = vùng memory-bound, phẳng =
vùng compute-bound. Công cụ để biết **nên tối ưu cái gì**, và quan trọng hơn, **khi nào
nên dừng**. → [`03-roofline.md`](03-roofline.md)

### RoPE
Mã hoá vị trí bằng cách **xoay** vector q,k một góc tỉ lệ vị trí; tích vô hướng chỉ còn
phụ thuộc *khoảng cách* giữa hai token. Không tốn tham số nào. *Repo dùng biến thể
**split-half**, không phải interleaved — hai cách không tương thích và đây là nguồn bug số
1 khi port model.*

### sampling / temperature / top-k
Cách chọn token từ logits. `temperature` <1 làm phân bố nhọn hơn (an toàn, dễ lặp), >1
phẳng hơn (sáng tạo, dễ lảm nhảm). `top-k` giới hạn số ứng viên. *Mặc định repo: 0.8 / 40.*

### seed noise (nhiễu seed)
Chênh lệch kết quả chỉ do khởi tạo ngẫu nhiên khác nhau. **Mọi kết luận phải chạy ≥2 seed
và so chênh lệch với nhiễu.** *"+0.098 nats, nhiễu ±0.006" → tin được; chênh 0.008 thì vô
nghĩa.*

### SIMD (AVX2 / NEON)
Một lệnh xử lý nhiều phần tử. *Bài học đo được: SIMD viết tay được +49% trên x86 nhưng
**0%** trên ARM — vì gcc đã tự vectorise sẵn. Đừng port tối ưu mà không đo lại.*

### SM (Streaming Multiprocessor)
Đơn vị tính của GPU NVIDIA, chứa nhiều CUDA core + scheduler. *Orin Nano Super: 8 SM.*

### speculative decoding
Model nhỏ đoán trước k token, model lớn xác nhận cả k trong **một** lượt đọc trọng số.
**Kỹ thuật duy nhất phá được trần `bandwidth / model_size` ở batch=1.**

### TensorRT
**Trình biên dịch** model, không phải thư viện suy luận thông thường: parse → tối ưu graph
→ **fusion** → chọn precision từng layer → **autotune tactic bằng cách đo trên chính máy
này** → ghi ra `.engine`. Mọi chi phí dồn vào build-time để runtime chỉ còn nạp và chạy.
→ [`14-tensorrt-deepstream.md`](14-tensorrt-deepstream.md)

### tensor core
Đơn vị nhân ma trận chuyên dụng của GPU NVIDIA. Chỉ chạy khi phép toán đúng dạng
ma trận×ma trận đủ lớn và kích thước chia hết theo yêu cầu. *Decode batch=1 là
ma trận×**vector** → tensor core nằm không. Bản CUDA của repo hoàn toàn không dùng.*

### tied embeddings
Dùng chung một ma trận cho embedding đầu vào và output head. Tiết kiệm `V×D` tham số.
*Bẫy đã gặp: `state_dict` liệt kê cả hai key, quên xử lý thì head lặng lẽ ở fp32 và golden
không khớp nữa.*

### token
Đơn vị model làm việc, thường là mảnh từ. *Xem ids thật ở `firmware/jetson/tok_ref.txt`.
Dấu cách nằm **trong** token, số bị băm nhỏ — nên LLM dốt số học.*

### trtexec
CLI có sẵn ở `/usr/src/tensorrt/bin/trtexec` — build engine, đo throughput/latency, và
`--dumpProfile` để biết layer nào ăn thời gian. *Trả lời được 90% câu hỏi về TensorRT mà
không cần viết dòng code nào. Đọc `Enqueue Time` vs `GPU Compute Time`: xấp xỉ nhau nghĩa
là launch-bound.*

### TTFT (time to first token)
Thời gian từ lúc nhận prompt tới token đầu tiên. Do **prefill** quyết định, tách hẳn khỏi
tok/s (do decode quyết định).

### unified memory (Jetson)
CPU và GPU dùng chung DRAM vật lý. *Hệ quả: `cudaMemcpy` H2D nhiều khi là copy thừa —
kiểm tra trước khi tối ưu kernel.*

### W4A16 / W8A8
Ký hiệu "weight 4 bit, activation 16 bit" v.v. *W4A16 hợp với decode (memory-bound, chỉ
cần bớt byte weight). W8A8 hợp với prefill/vision (compute-bound, cần đường tính int8).*

### warp
32 thread chạy đồng bộ trên GPU NVIDIA — đơn vị lập lịch thật sự. *`k_matvec_q4` dùng
**một warp cho một hàng output** và reduce bằng `__shfl_down_sync`, không cần shared
memory. Bẫy: mask `0xffffffff` khẳng định cả 32 lane còn sống — sai nếu có lane `return`
sớm.*

---

→ [README.md](README.md) · Bắt đầu lại từ đầu: [00-nhap-mon.md](00-nhap-mon.md)
