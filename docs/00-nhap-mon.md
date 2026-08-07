# 0. Nhập môn — 60 phút để hiểu repo này từ số 0

Các file 02–11 giả định bạn đã biết transformer, CUDA, roofline. File này **không giả
định gì cả**. Đọc xong bạn sẽ hiểu được mọi con số trong `RESULTS.md` và biết mình đang
nhìn cái gì khi mở `src/model.py`.

Mọi con số dưới đây lấy từ chính repo, và cuối bài có 3 bài thực hành **chạy được ngay
trên laptop, không cần GPU, không cần board**.

---

## 0.1 Model sinh chữ như thế nào — vòng lặp 5 bước

Language model không "hiểu câu". Nó làm đúng một việc: **đoán token tiếp theo**, rồi
lặp lại.

```
"Once upon a time"
   │
   │ ① TOKENIZE: chữ → số
   ▼
[433, 447, 259, 405]                       ← ids thật, xem firmware/jetson/tok_ref.txt
   │
   │ ② FORWARD: chạy model trên token cuối (405)
   ▼
logits = [ -2.1, 0.4, ..., 4.66, ... ]     ← V=4096 (hoặc 32768) số thực
   │                        ▲
   │                        └─ điểm số cho MỖI token có thể có trong từ điển
   │ ③ SAMPLE: chọn 1 id từ logits
   ▼
580                                        ← "," chẳng hạn
   │
   │ ④ DETOKENIZE + in ra màn hình
   │ ⑤ Đưa 580 vào lại bước ② → lặp
   ▼
"Once upon a time,"
```

**Token là gì.** Không phải chữ cái, không hẳn là từ. Là mảnh từ do thuật toán BPE học
được. Xem thật ở [`firmware/jetson/tok_ref.txt`](../firmware/jetson/tok_ref.txt):

```
433 447 259 405        →  "Once upon a time"     (4 token cho 4 từ)
221 433 447 259 405    →  " Once upon a time"    (dấu cách đầu là token KHÁC)
495 929 1205 221 20 18 221 23 221 25 25 25
                       →  "one two three 42 7 999"   ("42" tách thành "4","2")
```

Ba dòng đó dạy được ba điều: dấu cách nằm *trong* token, số bị băm nhỏ (nên LLM dốt toán),
và cùng một chữ ở vị trí khác nhau ra id khác nhau.

**Logits là gì.** Vector V số thực, chưa chuẩn hoá. Chỉ số của giá trị lớn nhất
(`argmax`) là token model tự tin nhất. Trong bài verify bạn sẽ chạy dưới đây,
`top=580` — cả PyTorch, C và CUDA đều phải ra 580, nếu không là port sai.

**Sample là gì.** Lấy luôn argmax thì văn khô và lặp. Nên người ta bốc thăm có trọng số:
`temperature` (0.8) làm phân bố phẳng hơn hay nhọn hơn, `top-k` (40) chỉ cho 40 ứng viên
mạnh nhất dự thăm. Đó là hai cờ `-t` và `-k` của `generate_cuda`.

> **Điểm cốt lõi của toàn bộ repo:** bước ② chạy **một lần cho mỗi token sinh ra**. Model
> 28.9M tham số sinh 200 token = chạy forward 200 lần. Đó là lý do tốc độ đo bằng
> *tok/s* và vì sao mọi thứ dưới đây xoay quanh "một lần forward tốn bao nhiêu".

---

## 0.2 Bên trong forward chỉ có 4 phép toán

Mở [`src/model.py`](../src/model.py) (323 dòng) hay
[`firmware/common/llm.h`](../firmware/common/llm.h) (385 dòng) sẽ thấy đúng 4 thứ:

| Phép | Là gì | Chiếm bao nhiêu |
|---|---|---|
| **matvec** | ma trận × vector | ~90% thời gian |
| **dot** | tích vô hướng 2 vector | attention |
| **elementwise** | cộng/nhân từng phần tử, SiLU, RMSNorm | rẻ |
| **lookup** | tra 1 hàng của bảng theo id | gần như miễn phí ★ |

**matvec** — cần hiểu đúng một cái này thôi:

```
        cols=3
      ┌         ┐   ┌   ┐     ┌                          ┐
rows=2│ 1  2  3 │ × │ 4 │  =  │ 1·4 + 2·5 + 3·6  =  32   │
      │ 4  5  6 │   │ 5 │     │ 4·4 + 5·5 + 6·6  =  77   │
      └         ┘   └ 6 ┘     └                          ┘

mỗi hàng của ma trận  →  một tích vô hướng  →  một số ở đầu ra
```

Ba quan sát rút ra từ đó, và cả repo dựng trên chúng:

1. Mỗi phần tử ma trận được **đọc đúng một lần** rồi dùng cho **một phép nhân-cộng**.
   Tỉ lệ "tính trên byte đọc" cực thấp → gọi là **memory-bound** (§0.3).
2. Các hàng **độc lập nhau** → chia hàng cho nhiều core/thread là song song hoá tự nhiên.
   Chính là bậc L4 trong `samples/cpu/matvec_ladder.c`.
3. Ma trận to nhất là **output head** `[V, D]`. Với V=32768, D=96 đó là 3.1M tham số,
   phải quét **toàn bộ** mỗi token vì cần đủ V logits.

**lookup (★)** — bảng PLE `[V, L·P]` cũng to, nhưng mỗi token chỉ đọc **1 hàng**. Đây là
toàn bộ mẹo của repo, xem §0.5.

---

## 0.3 Phép chia duy nhất bạn phải thuộc

Mỗi token, máy phải **đọc trọng số từ bộ nhớ ít nhất một lần**. Nên:

```
                bandwidth (byte/giây)
tok/s  ≤  ─────────────────────────────
              kích thước model (byte)
```

Không kernel nào, không compiler nào phá được trần này ở batch = 1. Ba ví dụ thật:

| Máy | Bandwidth đo được | Phần phải đọc | Trần |
|---|---|---|---|
| ESP32-S3, **chỉ riêng output head** | 60.7 MB/s (PSRAM) | 2.43 MB | 40 ms → head không thể nhanh hơn thế |
| Orin Nano Super + Llama-8B Q4 | 66.8 GB/s (DRAM) | 5.35 GB (weights + KV) | **12.5 tok/s** |
| Orin Nano + model của repo | 66.8 GB/s | 1.87 MB | ~35 000 tok/s (thực tế chỉ đạt 1141) |

Dòng cuối là bài học lớn thứ hai: khi model **quá nhỏ** so với máy, nút thắt chuyển sang
chỗ khác — trên Orin, model này chỉ chạy 1141 tok/s và **50% thời gian là chi phí khởi
động kernel**, không phải bộ nhớ. Cùng model, cùng code, nút thắt khác nhau hoàn toàn.

Hai chữ cần phân biệt suốt các file sau:

- **memory-bound** — CPU/GPU rảnh, đang *chờ dữ liệu*. Cách chữa: đọc ít byte hơn
  (quantize, head nhỏ hơn, đặt dữ liệu ở tầng nhanh hơn).
- **compute-bound** — dữ liệu sẵn, đang *bận tính*. Cách chữa: SIMD, nhiều core, tensor core.

Chữa nhầm loại = tốn cả tuần lấy 0%. Đó là nguyên nhân tồn tại của
[`03-roofline.md`](03-roofline.md).

### Tính sàn — thao tác quan trọng nhất của cả repo

Trước khi tối ưu bất cứ thứ gì, chia trước:

```
head trên ESP32 đọc 2.43 MB int8 mỗi token
PSRAM đo được 60.7 MB/s
→ 2.43 / 60.7 = 40 ms  ← SÀN, phép tính nhanh vô hạn cũng không phá được
đo thực tế: 57.6 ms
→ trong đó 40 ms là chờ bộ nhớ, chỉ 17.6 ms là tính toán
→ viết SIMD giỏi cỡ nào cũng chỉ cắt được 17.6/102.9 ≈ 17% tổng thời gian
→ KHÔNG làm. Chuyển hướng: giảm số byte phải đọc.
```

*Biết khi nào ngừng tối ưu* là kỹ năng hiếm hơn biết cách tối ưu. Chi tiết ở
[`08-nhat-ky-toi-uu.md §5.2`](08-nhat-ky-toi-uu.md).

---

## 0.4 Bốn phép tính nhẩm

**(1) Model chiếm bao nhiêu byte?** Số bit thực trên mỗi tham số không phải 4, vì mỗi
nhóm `group` tham số còn kèm 1 scale fp16:

```
bits thực = bits + 16/group = 4 + 16/128 = 4.125
28.9M tham số × 4.125 / 8 = 14.9 MB      (file thật: 14,912,332 B ✓)
```

Đó là lý do `group` nhỏ hơn (32, 64) chính xác hơn nhưng file to hơn.

**(2) ms/token ↔ tok/s.** `tok/s = 1000 / ms`. 102.9 ms → 9.72 tok/s. 0.876 ms → 1141 tok/s.
Số end-to-end luôn thấp hơn số compute-only (9.5 vs 9.72) vì còn in ra serial/màn hình.

**(3) nats ↔ perplexity.** Docs báo cáo chất lượng bằng *nats*; `ppl = e^nats`, và ngược
lại `nats = ln(ppl)`:

```
baseline ppl 12.58 → ln = 2.5321
ple      ppl 11.41 → ln = 2.4345
chênh 0.098 nats  =  9.3% ppl        ← đúng con số RESULTS.md báo
```

Quy đổi thô: **0.01 nats ≈ 1% perplexity**. Thấp hơn là tốt hơn.

**(4) Có đáng tin không?** Luôn hỏi "chênh lệch có lớn hơn nhiễu không":

```
+0.098 nats, nhiễu giữa 2 seed ±0.006  →  gấp ~16 lần nhiễu  →  tin được
+0.008 nats với cùng nhiễu             →  vô nghĩa, đừng báo cáo
```

---

## 0.5 Ba tầng bộ nhớ — ý tưởng làm nên repo

Repo phân loại tham số theo **cách bị truy cập**, không theo tốc độ:

| Tầng | Truy cập mỗi token | Ẩn dụ | Ở ESP32 |
|---|---|---|---|
| **core** | đọc **toàn bộ**, ngẫu nhiên | sách để trên bàn | SRAM 512KB — khan hiếm |
| **stream** | đọc **toàn bộ**, tuần tự 1 lượt | băng cassette tua 1 lần | PSRAM — tốn *bandwidth* |
| **table** | đọc **1 hàng** | từ điển, chỉ tra 1 mục | flash — gần như miễn phí |

Con số của cấu hình dev (chính là `model.bin` có sẵn trong repo):

```
core   1,499,328 params (41.7%)   đọc hết mỗi token
stream   524,288 params (14.6%)   quét 1 lượt mỗi token
table  1,572,864 params (43.7%)   đọc 192 byte/token  = 0.024% băng thông
```

**43.7% số tham số tốn 0.024% băng thông.** Đó là toàn bộ luận điểm. Ở cấu hình deploy
28.9M nó còn cực đoan hơn: 25M tham số (87%) đọc ~450 B/token.

Vì sao được phép làm vậy: bảng đó là **embedding** — model *tra* nó chứ không *tính* trên
nó. Một token chỉ cần đúng hàng của nó. Nên bảng nằm ở flash 16MB (chậm nhưng bao la),
còn phần "suy nghĩ" bé xíu ở SRAM. Đây là ý tưởng **Per-Layer Embeddings** của Google
(Gemma 3n), và cùng nguyên lý với **MoE** (Mixtral, Qwen3-MoE): nhiều tham số, đọc ít
mỗi token.

Cái này **không cho model thông minh hơn**: 28.9M tham số vẫn là model TinyStories, vẫn
chỉ viết tiếp truyện cổ tích, không trả lời câu hỏi. Giới hạn nằm ở 559K tham số "core".

---

## 0.6 Bản đồ repo — file nào làm gì

```
src/            PyTorch: train, quantize, export         ← nơi model ra đời
  model.py      323 dòng, TOÀN BỘ kiến trúc + 5 nhánh ablation
  train.py      157 dòng, vòng lặp train
  quantize.py   139 dòng, int4 group-wise + đo degradation
  export.py     168 dòng, ghi model.bin + golden.txt
  budget.py     kế toán 3 tầng bộ nhớ (chỉ báo cáo, không ảnh hưởng model)

firmware/
  common/llm.h  385 dòng, runtime C thuần — CẢ BA target dùng chung
  model/        model.bin (1.87 MB, cấu hình dev) + golden.txt
  host_verify/  chạy trên laptop: verify.c (đúng/sai) + ppl.c (perplexity)
  jetson/       CUDA: llm_cuda.cuh + verify/bench/generate + Makefile
  esp32_llm/    sketch Arduino cho ESP32-S3
  bandwidth_bench/  đo PSRAM/SRAM/flash thật bằng cycle counter

samples/cpu/    matvec_ladder.c — thang 5 bậc tối ưu, chạy được ngay
samples/gpu/    bench_roofline.cu, bench_decode.cu, roofline.py (không cần GPU)
docs/           bạn đang ở đây
RESULTS.md      mọi số đo + lịch sử, kể cả cái đã thử rồi bỏ
DEPLOY.md       sơ đồ kiến trúc, bảng tensor, byte layout model.bin
```

> **Điều dễ nhầm nhất khi mới clone:** `firmware/model/model.bin` trong repo là **cấu
> hình dev** (V=4096, D=128, 3.6M tham số, 1.87 MB) để lặp nhanh — *không phải* model
> 28.9M trong tiêu đề README. Muốn cái 28.9M thì phải tự train với
> `--vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128`. Chạy bài thực hành dưới đây
> bạn sẽ thấy `head [4096 x 128]`, đúng như vậy là bình thường.

---

## 0.7 60 phút đầu tiên — ba bài chạy được ngay

Không cần GPU, không cần board, không cần cài Python. Chỉ cần `cc` và `make`.

### Bài 1 (5 phút) — "port có đúng không?"

```bash
cc -O3 -o /tmp/verify_c firmware/host_verify/verify.c -lm
/tmp/verify_c firmware/model/model.bin firmware/model/golden.txt
```

Kết quả đúng:

```
logits: C top=580  PyTorch top=580
max abs diff = 0.00001   rms diff = 0.000003
PASS: C matches PyTorch golden
```

**Nhìn gì:** `top=580` khớp nhau. Đây là **tiêu chí PASS thật** — không phải
`max abs diff`. Cộng số thực không có tính kết hợp (`(a+b)+c ≠ a+(b+c)`), nên hai cách
cộng khác thứ tự sẽ luôn lệch vài phần triệu; đòi bit-exact là đòi sai. Cái phải khớp là
**hành vi**: cùng argmax.

Vì sao bài này đứng trước mọi bài tối ưu: **đúng trước, nhanh sau.** Bản port C đầu tiên
của tác giả chạy 0.57 tok/s — chậm 17 lần so với bản cuối, nhưng *đúng*, nên mọi bước sau
mới có mốc để đối chiếu.

### Bài 2 (15 phút) — thang tối ưu, tự chạy trên máy bạn

```bash
make -C samples/cpu run
```

In ra 5 bậc tối ưu **cùng một phép toán** (output head), mỗi bậc kèm ms, GB/s và sai số.
Ví dụ trên một máy x86:

```
L0 int4 + fp32       0.8761 ms    1.00x   llm.h matvec_q -- mốc
L1 int4 + int8act    0.8050 ms    1.09x
L2 int8 staged       0.1216 ms    7.21x   gỡ nibble 1 lần (như ESP32)
L3 + SIMD            0.0671 ms   13.06x   AVX2
L4 + đa luồng        0.0315 ms   27.77x
```

**Nhìn gì:**
1. **L2 là bậc thắng đậm nhất** (7×) và nó *đọc gấp đôi số byte* của L0 — vì đổi từ int4
   sang int8 để khỏi phải gỡ nibble mỗi lần. Đổi dung lượng lấy tốc độ **chỉ đáng khi
   đang compute-bound**. Trên ESP32 đúng bậc này cũng là bậc ăn tiền.
2. **`max|d|` khác 0 từ L1 trở đi là do int8 activation, không phải bug** — nó được
   nghiệm thu bằng perplexity chứ không bằng sai số tuyệt đối.
3. Số của **máy bạn sẽ khác** — và đó mới là bài học. Trên ARM, L3 (SIMD tay) thường
   được **0%** vì gcc đã tự vectorise; trên CPU lai P-core/E-core, dùng hết logical core
   có thể *chậm hơn* dùng 8. Chương trình tự hiệu chuẩn chi phí mở luồng rồi in điểm hoà
   vốn — đọc dòng cuối cùng.

```bash
make -C samples/cpu scaling      # tách phần lợi của SIMD khỏi phần lợi của threads
```

### Bài 3 (10 phút) — trần tốc độ của phần cứng bạn định mua

```bash
python3 samples/gpu/roofline.py --hw orin-nano-super --model llama-3.1-8b --bits 4.8
```

Không cần GPU, không cần cài gì. Nó tính trước cho bạn:

```
weights 4.82 GB + KV cache @ctx4096 0.54 GB = 5.35 GB / 8 GB
trần từ bandwidth   12.5 tok/s   <-- MEMORY-bound
trần từ compute    631.4 tok/s
kỳ vọng đạt được   6.2–8.7 tok/s (50–70% trần)
```

**Nhìn gì:** chênh lệch 50× giữa trần bộ nhớ và trần tính toán — đó là hình ảnh trực quan
nhất của "memory-bound". Và **KV cache chiếm 0.54 GB**, tức hơn 10% ngân sách, thứ hầu
hết người ta quên khi tính "model 4.8GB thì vừa RAM 8GB".

Đổi tham số mà chơi: `--bits 4 / 8 / 16`, `--ctx`, `--hw`. Sau bài này, câu hỏi "con
này chạy nổi Llama-8B không?" trả lời được bằng một phép chia thay vì bằng cảm giác.

---

## 0.8 Tám hiểu lầm hay gặp

| Hiểu lầm | Thực tế |
|---|---|
| "Nhiều tham số hơn = thông minh hơn" | 28.9M ở đây vẫn chỉ viết truyện TinyStories. Năng lực nằm ở 559K core, phần còn lại là bộ nhớ tra cứu. |
| "Tối ưu = viết SIMD/kernel" | Bước ăn 9.1× trên ESP32 chỉ là *đặt ma trận vào đúng tầng bộ nhớ*. 4 bước còn lại cộng lại được 1.9×. |
| "GPU mạnh nên chạy model nhỏ sẽ rất nhanh" | Model nhỏ trên Orin: 50% thời gian là launch overhead. Máy mạnh làm lộ nút thắt *khác*. |
| "Cứ so `max diff < 1e-5`" | Sai với GPU. So bằng argmax / top-k / perplexity. |
| "Datasheet ghi 102 GB/s thì có 102" | Đo được 66.8 (65%), vì EMC khoá 2133 MHz. Mọi tính toán từ datasheet sai 1.53×. |
| "Quantize 4-bit = mất 4 bit/tham số" | Thực tế 4.125 bit, do scale fp16 mỗi nhóm. Và **không quantize tensor nhỏ** (các norm giữ fp32). |
| "Model to thì quantize hỏng nhiều hơn" | Ngược lại: bảng 25M chịu 4-bit **tốt hơn** core dày đặc (+0.055 vs +0.079 nats). Chọn 8B-Q4 hơn 3B-Q8 ở cùng dung lượng. |
| "Đo một lần là đủ" | Cùng board, cùng lệnh: 27.8 rồi 66.8 GB/s. Khác biệt là *tải nền*, không phải nhiễu. Chạy ≥3 lần, báo median. |

---

## 0.9 Đi tiếp đâu

**Nếu chỉ có laptop** (không GPU, không board):
`00` → [`02-hieu-model.md`](02-hieu-model.md) → [`03-roofline.md`](03-roofline.md) →
[`05-kien-truc-phan-cung.md`](05-kien-truc-phan-cung.md) → [`04-quantization.md`](04-quantization.md).
Bài 1 và 2 ở §0.7 chạy lại được sau mỗi file, lần nào cũng thấy thêm thứ mới.

**Nếu có Jetson / GPU NVIDIA:**
thêm [`../DEPLOY.md`](../DEPLOY.md) (deploy + 5 tầng kiểm chứng) →
[`../firmware/jetson/JETSON.md`](../firmware/jetson/JETSON.md) →
[`11-toi-uu-nvidia.md`](11-toi-uu-nvidia.md) → [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md).
Muốn làm việc thật với framework (PyTorch/llama.cpp/TensorRT/DeepStream) thì đi tiếp
[`13-jetson-framework.md`](13-jetson-framework.md) → [`14-tensorrt-deepstream.md`](14-tensorrt-deepstream.md).

**Nếu có ESP32-S3:**
[`../firmware/esp32_llm/README.md`](../firmware/esp32_llm/README.md) →
[`08-nhat-ky-toi-uu.md`](08-nhat-ky-toi-uu.md) (nhật ký 0.57 → 9.5 tok/s).

**Muốn hiểu *vì sao* chứ không chỉ *làm sao*:**
[`10-ly-thuyet-nen.md`](10-ly-thuyet-nen.md) — dấu phẩy động, toán của quantization,
toán của attention, batching, speculative decoding.

**Gặp từ lạ:** [`12-thuat-ngu.md`](12-thuat-ngu.md).

---

→ [README.md](README.md) · Tiếp: [02-hieu-model.md](02-hieu-model.md)
