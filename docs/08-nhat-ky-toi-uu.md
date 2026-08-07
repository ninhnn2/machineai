# 5. Học cách tối ưu — nhật ký 0.57 → 9.5 tok/s, và ánh xạ sang Jetson

File [04](02-hieu-model.md) là *model là gì*. File này là *làm sao cho nó nhanh*, đọc
qua lịch sử tối ưu có thật trong [`RESULTS.md`](../RESULTS.md).

Giá trị của repo này không nằm ở ESP32. Nằm ở chỗ tác giả **ghi lại từng bước, kèm
số đo và kèm cả những thứ đã thử mà bỏ đi**. Đó là thứ hiếm.

---

## 5.1 Nhật ký tối ưu — 17× trong 5 bước

| # | Thay đổi | ms/token | tok/s | tăng | Bài học |
|---:|---|---:|---:|---:|---|
| 0 | Bản port C đúng đầu tiên | 1757.2 | 0.57 | — | **Đúng trước, nhanh sau** |
| 1 | Head sang PSRAM + dọn scalar | 193.9 | 4.7 | **9.1×** | Đặt dữ liệu nóng đúng tầng bộ nhớ |
| 2 | Dọn dot/RoPE/attention | 172.9 | — | 1.12× | Bỏ tính toán lặp |
| 3 | Head chạy 2 core | 139.4 | 6.0 | 1.24× | Song song hoá phần chiếm ưu thế |
| 4 | Head int8 + activation int8 | **102.9** | **9.5** | 1.35× | Giảm **bytes đọc**, không phải phép tính |

**Bước 1 chiếm 9.1× trong tổng 17×.** Nó không phải thuật toán mới — chỉ là đặt ma trận
head vào đúng chỗ. Bốn bước còn lại cộng lại chỉ được 1.9×.

> **Đây là quy luật chung, không phải đặc thù ESP32.** Trên Jetson, tương đương của
> bước 1 là: `-ngl 99` (đẩy hết layer lên GPU), dùng đúng backend CUDA, tránh
> `cudaMemcpy` H2D thừa. Người ta hay bỏ qua vì "quá đơn giản", rồi mất tuần tối ưu
> kernel để lấy 20%.

## 5.2 Vì sao dừng lại ở 102.9 ms — bài học quan trọng nhất repo

Profile sau bước 4 ([RESULTS.md:146-147](../RESULTS.md)):

```
ms/token:  head 57.6 | attn 25.6 | ple 8.5 | ffn 6.9 | input 4.4
```

Tác giả không lao vào viết SIMD cho head. Thay vào đó **tính sàn**:

```
head đọc 2.43 MB int8 mỗi token
PSRAM đo được 60.7 MB/s
→ 2.43 / 60.7 = 40 ms là SÀN, không thể phá bằng compute
→ trong 57.6 ms có 40 ms chờ bộ nhớ, chỉ 17.6 ms là tính toán
→ SIMD dù nhanh vô hạn cũng chỉ cắt được 17.6/102.9 ≈ 17%
```

Kết luận trong [RESULTS.md:151-152](../RESULTS.md):

> *"The bigger levers from here are reducing bytes-read (int4 head + SIMD unpack) or
> a smaller/factorised output head (a model change) — **not vectorising harder**."*

**Đây chính xác là bước 5 trong quy trình roofline ở [01-roofline.md §1.5](03-roofline.md).**
Cùng lối suy nghĩ, khác phần cứng 4 bậc độ lớn.

Chúng ta đã lặp lại đúng lối này trên Jetson của bạn: bench đạt 66.8/68.3 GB/s = 98%
trần EMC → dừng tối ưu kernel, nút thắt nằm ở EMC clock
([MEASUREMENTS.md §Phát hiện 5](09-so-do-phan-cung.md)).

---

## 5.3 Bốn kỹ thuật cụ thể — và bản đối chiếu Jetson

### KT1 — Đặt dữ liệu đúng tầng bộ nhớ

ESP32 có 3 tầng, và repo dùng cả 3 một cách có chủ đích
([esp32_llm.ino:1-5](../firmware/esp32_llm/esp32_llm.ino#L1-L5)):

```
FLASH (mmap, chậm, 16MB)  → bảng 25M tham số, đọc 6 hàng/token
PSRAM (60.7 MB/s, 8MB)    → head int8, KV cache, scratch
SRAM  (240 MB/s, 512KB)   → core (thực tế để XIP từ flash, đo thấy đủ nhanh)
```

Chi tiết đáng chú ý: `esp_partition_mmap` ([esp32_llm.ino:136](../firmware/esp32_llm/esp32_llm.ino#L136))
ánh xạ 15MB flash vào không gian địa chỉ. Bảng 25M tham số **không bao giờ được load** —
chỉ những hàng cần mới bị đọc, lazily, bởi phần cứng cache.

| ESP32 | Jetson Orin Nano |
|---|---|
| `esp_partition_mmap` bảng vào flash | `mmap` file GGUF — llama.cpp làm mặc định, model không tốn RAM cho phần chưa dùng |
| Stage head vào PSRAM lúc boot | `-ngl 99` đẩy layer lên GPU |
| Core để XIP ở flash | unified memory — **không cần copy H2D**, dùng `cudaHostAllocMapped` |
| KV cache ở PSRAM | KV cache trong 8GB chung, dùng `--cache-type-k q8_0` |

### KT2 — Giảm bytes đọc, không giảm phép tính

Bước 4 (int8 head) là ví dụ mẫu. Trước đó head lưu int4 và mỗi token phải:
giải nén nibble → nhân với scale fp16 → nhân float. Sau đó:

```c
// esp32_llm.ino:101-117 — giải nén int4→int8 MỘT LẦN lúc boot
static void stage_head_int8(QT *t) {
  head_w8 = ps(head_rows * head_cols);       // int8 trong PSRAM
  for (r...) for (j...) dst[j] = (code - 8); // nibble → int8, một lần duy nhất
}
// esp32_llm.ino:57-61 — mỗi token chỉ còn int8 × int8 → int32
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}
```

Đổi **dung lượng lấy tốc độ**: int8 tốn gấp đôi int4 trong PSRAM (2.43MB thay vì 1.2MB),
nhưng bỏ được toàn bộ khâu giải nén mỗi token. Trên board còn dư PSRAM nên đáng đổi.

> **Nghịch lý cần hiểu:** ở bước này *tăng* bytes đọc lại *nhanh hơn*, vì trước đó nút
> thắt là **compute (giải nén)**, không phải bandwidth. Sau khi đổi, nút thắt chuyển
> sang bandwidth (40ms floor). **Nút thắt di chuyển sau mỗi lần tối ưu — phải profile lại.**

Tương đương Jetson: `Q4_K_M` (giải nén trên GPU mỗi token) vs `Q8_0` (không giải nén).
Trên board bandwidth-bound nặng như Orin Nano thì Q4 vẫn thắng, nhưng **phải đo**.

### KT3 — Song song hoá đúng chỗ

[esp32_llm.ino:69-92](../firmware/esp32_llm/esp32_llm.ino#L69-L92): head
chia đôi hàng cho 2 core LX7. Chỉ head, không phải cả model.

Vì sao chỉ head: nó chiếm 67% thời gian, và các hàng output **hoàn toàn độc lập**
(mỗi hàng là một dot product riêng). Attention thì các bước phụ thuộc nhau.

```c
quantize_act(x, head_cols, head_actq, &head_acts);  // 1 lần, 2 core cùng đọc
head_job_split = head_rows / 2;
xTaskNotifyGive(head_worker);                       // core 0 làm nửa đầu
head_rows_range(y, head_job_split, head_rows);      // core 1 làm nửa sau
ulTaskNotifyTake(pdTRUE, portMAX_DELAY);            // đợi
```

Chú ý `matvec_q_range` trong [llm.h:115](../firmware/common/llm.h#L115)
nhận `row_begin, row_end`. Comment giải thích lý do thiết kế:

> *"Keeping the row range explicit lets platforms parallelize the large output head
> without changing any individual dot product."*

**Thiết kế API cho phép song song hoá mà không đụng vào phép toán.** Đây là nguyên tắc
tốt: giữ kernel thuần, cho phía gọi quyết định phân chia.

Tương đương Jetson: GPU đã song song hoá sẵn ở tầng thread. Điều bạn kiểm soát là
**CUDA streams** và **CUDA graphs** (giảm overhead launch khi batch nhỏ — đáng kể ở
decode vì mỗi layer là một kernel nhỏ).

### KT4 — Bỏ tính toán lặp

Ba ví dụ trong repo, tất cả đều "rẻ tiền" nhưng cộng lại 1.12×:

1. **RoPE tính 1 lần/token** thay vì L×H lần
   ([llm.h:288-296](../firmware/common/llm.h#L288-L296)):
   > *"RoPE frequencies are identical across every head and layer at a position."*
   Với L=6, H=4 → tiết kiệm 23/24 số lần gọi `powf/cosf/sinf`.
2. **Bỏ 7,415 hàng vocab không dùng** ([esp32_llm.ino:153](../firmware/esp32_llm/esp32_llm.ino#L153)):
   tokenizer chỉ học 25,353 mục nhưng vocab padding lên 32,768. Head bỏ qua phần thừa
   → tiết kiệm 23% công việc của phần chiếm ưu thế.
3. **Dùng lại buffer**: `s->trow` chứa hàng table, chết sau khi dựng `s->ple`, được
   tái dụng làm bộ đệm RoPE ([llm.h:291](../firmware/common/llm.h#L291)).

Điểm 2 đáng nhớ nhất: **kiểm xem bạn có đang tính thứ không dùng đến không.** Trên
Jetson tương đương là vocab padding trong TensorRT engine, hoặc batch dimension thừa.

---

## 5.4 Những thứ tác giả THỬ RỒI BỎ — phần hiếm nhất

[RESULTS.md:174-176](../RESULTS.md):

> *"Explicitly staging the remaining 0.29MB quantized core in PSRAM and norms in
> internal RAM saved only 2.0ms/token (1.4%) while adding allocation complexity,
> so that experiment was removed from the polished runtime."*

Và [RESULTS.md:177-182](../RESULTS.md):

> *"There is still bounded exact work—parallel attention, precomputed RoPE frequencies,
> and a one-group-specialized head loop—but the profile caps the entire attention
> opportunity at 26.4ms... They are intentionally deferred rather than presented as
> another large scalar speedup."*

**Hai bài học:**
1. Tối ưu 1.4% mà tăng độ phức tạp → **gỡ ra**. Code đơn giản có giá trị.
2. Biết **trần** của một hướng tối ưu *trước khi* làm nó. Profile cho biết attention
   tối đa chỉ đáng 26.4ms → không cần thử mới biết nó không phải ưu tiên.

Repo còn ghi lại cả một **bug trong chính cách đếm tham số** của tác giả
([README.md:83-88](../README.md#L83-L88)):

> *"That includes a bug I found in my own parameter accounting, which had inflated an
> early number, and the corrected result that followed once I fixed it."*

Và [RESULTS.md:3-5](../RESULTS.md) đánh dấu các run cũ là
`_archive_old_accounting/` với ghi chú *"should not be cited"*.

> **Đây là chuẩn mực nên theo.** Khi bạn đổi cách đo, mọi số cũ thành vô giá trị —
> phải đánh dấu rõ, không lặng lẽ trộn lẫn.

---

## 5.5 Phương pháp khoa học — làm sao biết tối ưu THẬT SỰ có tác dụng

Đây là phần chuyển giao mạnh nhất, và nó không liên quan gì tới phần cứng.

### Thiết kế 5 nhánh đối chứng

[`model.py:10-23`](../src/model.py#L10-L23) — mỗi nhánh trả lời **một
câu hỏi cụ thể**:

| Nhánh | Có gì | Câu hỏi nó trả lời |
|---|---|---|
| `baseline` | không bảng | Mốc so sánh |
| `ple` | đường ống PLE + bảng | Ý tưởng đầy đủ |
| `ple_notable` | đường ống PLE, **không bảng** | *Lợi ích đến từ bảng hay từ đường ống?* |
| `fatembed` | bảng cùng kích thước nhưng tiêm ở **đáy** | *Vị trí tiêm có quan trọng không?* |
| `bigcore` | dùng ngân sách bảng để **làm core to hơn** | *Nếu có bộ nhớ nhanh thì sao?* |

Kết quả ([RESULTS.md:64-71](../RESULTS.md), vocab 4096):

```
ple_notable  8.35   ← TỆ HƠN baseline (8.21)!
fatembed     8.26   ← gần như không khác
ple          8.00   ← tốt hơn
bigcore      6.93   ← tốt hơn nhiều (nhưng không vừa SRAM)
```

`ple_notable` **tệ hơn baseline** chứng minh: đường ống PLE tự nó vô dụng, thậm chí có
hại (tốn core params cho máy móc không sinh lợi). **Toàn bộ lợi ích đến từ bảng.**
Không có nhánh đối chứng này, bạn sẽ không bao giờ biết.

`bigcore` cho biết **cái giá thật**: PLE chỉ lấy lại được ~15% của những gì một core
to hơn cho bạn. Nó không phải phép màu — nó là cách xoay xở khi core bị giới hạn cứng.

> **Bài học:** với mỗi tối ưu bạn định làm, hãy hỏi *"nhánh đối chứng nào sẽ chứng minh
> tôi sai?"* Nếu không nghĩ ra, bạn chưa hiểu tối ưu đó.

### Nhiễu seed

Mọi kết luận đều chạy **2 seed** và báo cáo biên độ: *"+0.098 nats (2 seeds, ±0.006).
~16x the seed noise"* ([RESULTS.md:30-31](../RESULTS.md)).

Chênh lệch 0.098 với nhiễu 0.006 → tin được. Nếu chênh lệch là 0.008 thì vô nghĩa.

**Trên Jetson tương đương là:** chạy benchmark ≥3 lần, báo cáo median và spread. Tôi
đã gặp đúng vấn đề này khi đo board bạn — lần đầu 27.8 GB/s, lần sau 66.8 GB/s, và
chênh lệch không phải nhiễu mà là **tải nền** ([MEASUREMENTS.md §Phát hiện 3](09-so-do-phan-cung.md)).

### Kết quả phản trực giác đáng nhớ

[RESULTS.md:205-209](../RESULTS.md): bảng 25M tham số **chịu 4-bit tốt
hơn** core dày đặc.

```
baseline degrade  +0.079 / +0.088 nats
ple      degrade  +0.055 / +0.061 nats   ← ít hơn
→ PLE giữ 124-128% ưu thế sau khi quantize
```

Giải thích: bảng lớn có dư thừa; mỗi weight ít quan trọng hơn. Model nhỏ dày đặc thì
mỗi weight đều thiết yếu.

**Tổng quát hoá cho Jetson:** model càng lớn càng chịu quantize tốt. Trên 8GB, hãy chọn
**8B-Q4 thay vì 3B-Q8** — cùng dung lượng, chất lượng cao hơn. Nhưng hãy tự đo trên
workload của bạn thay vì tin.

---

## 5.6 Bảng ánh xạ tổng hợp

| Kỹ thuật ở esp32-llm | Vị trí | Tương đương trên Jetson Orin Nano |
|---|---|---|
| Đo bandwidth trước khi tối ưu | `firmware/bandwidth_bench/` | [`bench/bench_roofline.cu`](../samples/gpu/bench_roofline.cu) — đã chạy, 66.8 GB/s |
| Kế toán 3 tầng theo access pattern | [`budget.py`](../src/budget.py) | weights / KV cache / offload; [`bench/roofline.py`](../samples/gpu/roofline.py) |
| Group-wise int4 + fp16 scale | [`quantize.py:30`](../src/quantize.py#L30) | Q4_K_M, AWQ, GPTQ, TRT-LLM INT4 |
| Không quantize tensor nhỏ (norm) | [`quantize.py:59`](../src/quantize.py#L59) | mọi format đều giữ norm fp16/fp32 |
| Golden logits verify port | [`export.py:145`](../src/export.py#L145) | so PyTorch vs TensorRT engine |
| Đo ppl trước/sau mỗi thay đổi | [`host_verify/ppl.c`](../firmware/host_verify/ppl.c) | `llama-perplexity` |
| mmap bảng, không load | [`esp32_llm.ino:136`](../firmware/esp32_llm/esp32_llm.ino#L136) | llama.cpp mmap GGUF mặc định |
| Stage hot weights ở tầng nhanh | [`esp32_llm.ino:101`](../firmware/esp32_llm/esp32_llm.ino#L101) | `-ngl 99`; unified memory, tránh H2D |
| Activation int8 để dùng SIMD | [`llm.h:161-198`](../firmware/common/llm.h#L161-L198) | W8A8 / SmoothQuant để chạm tensor core INT8 |
| Song song hoá phần chiếm ưu thế | [`esp32_llm.ino:69-92`](../firmware/esp32_llm/esp32_llm.ino#L69-L92) | CUDA streams, CUDA graphs |
| Bỏ hàng vocab không dùng | [`esp32_llm.ino:153`](../firmware/esp32_llm/esp32_llm.ino#L153) | cắt vocab, factorized head |
| Tính RoPE 1 lần/token | [`llm.h:288`](../firmware/common/llm.h#L288) | kernel fusion, precomputed rope cache |
| Bảng thưa 1 hàng/token | PLE table | **MoE** — cùng nguyên lý, chỉ 2/8 expert active |
| Softmax 2-pass ổn định | [`llm.h:327`](../firmware/common/llm.h#L327) | FlashAttention biến thành 1-pass online |

---

## 5.7 Lộ trình học đề xuất — 6 tuần

**Tuần 1 — hiểu model**
Đọc `model.py` và `llm.h` song song, dòng đối dòng. Mỗi op PyTorch tìm op C tương ứng.
Làm bài tập mức 1 ở [04-hieu-model.md §4.7](02-hieu-model.md).
*Kết quả: hiểu transformer ở mức có thể tự viết lại bằng C.*

**Tuần 2 — chuỗi kiểm chứng**
Chạy `export.py` → `host_verify/verify.c`. Cố tình phá 3 thứ (RoPE interleaved, quên
tied head, sai thứ tự tensor) và xem golden bắt được không.
*Kết quả: có quy trình verify riêng, dùng được cho mọi port sau này.*

**Tuần 3 — lượng tử hoá**
`quantize.py` với bits ∈ {8,6,5,4,3} và group ∈ {32,64,128}. Vẽ ppl theo dung lượng.
Tự cài lại `quantize_groupwise` không nhìn code.
*Kết quả: hiểu Q4_K_M từ bên trong, không còn là hộp đen.*

**Tuần 4 — roofline trên Jetson**
Đã có: [`bench/bench_roofline.cu`](../samples/gpu/bench_roofline.cu),
[`bench/bench_decode.cu`](../samples/gpu/bench_decode.cu),
[`MEASUREMENTS.md`](09-so-do-phan-cung.md). Làm bài tập ở [01-roofline.md §1.6](03-roofline.md).
*Kết quả: biết trần của board mình, không tin datasheet.*

**Tuần 5 — LLM thật trên Jetson**
`scp` llama.cpp sang board (board không có internet), build với CUDA, chạy `llama-bench`
với 1B/3B/8B ở Q4/Q8. **So mỗi số với trần tính từ roofline.py.** Ghi lại % đạt được.
*Kết quả: xác nhận (hoặc bác bỏ) lý thuyết bằng số thật của mình.*

**Tuần 6 — tối ưu có phương pháp**
Chọn 1 model. Lặp: profile → tìm phần chiếm ưu thế → tính sàn của nó → chỉ tối ưu nếu
sàn còn xa. Ghi nhật ký kiểu §5.1, gồm cả những thứ thử rồi bỏ.
*Kết quả: có phương pháp lặp lại được, không phải mẹo vặt.*

---

## 5.8 Giới hạn cần biết của repo này

Trung thực về chỗ repo **không** dạy được:

- **Không có GPU, không có tensor core, không có kernel song song.** Mọi thứ ở đây là
  scalar C trên 2 core. Kỹ thuật CUDA (shared memory tiling, warp-level primitives,
  tensor core WMMA) phải học chỗ khác.
- **Không có batching.** Mọi thứ là batch=1, decode 1 token. Continuous batching,
  paged attention, prefill/decode disaggregation không xuất hiện.
- **Không có FlashAttention thật.** Attention ở đây là 2-pass naive (đủ tốt vì ctx 512).
- **Model là TinyStories.** Không có instruction tuning, RLHF, tokenizer phức tạp.
- **PLE chưa phải kỹ thuật phổ biến.** Gemma 3n dùng nó; Llama/Qwen thì không. Học nó
  để hiểu *nguyên lý phân tầng bộ nhớ*, đừng kỳ vọng dùng trực tiếp.

Nhưng **phương pháp** thì chuyển giao 100%: đo trước, tính sàn, đối chứng, verify từng
tầng, ghi lại cả thất bại. Đó mới là thứ đáng học từ repo này.
