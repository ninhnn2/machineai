# 10. Lý thuyết nền — thứ quyết định mọi quyết định tối ưu

Tài liệu này giải thích **vì sao** các kỹ thuật ở phần trước hoạt động. Không có nó
thì bạn chỉ đang áp dụng công thức; có nó thì bạn tự suy được kỹ thuật mới khi gặp
phần cứng mới.

---

## 10.1 Số học dấu phẩy động — dải động và độ chính xác là hai thứ khác nhau

Đây là nền của mọi quyết định về precision, và là chỗ bị hiểu sai nhiều nhất.

Một số dấu phẩy động IEEE gồm: 1 bit dấu, `E` bit số mũ, `M` bit phần định trị.

- **Số mũ quyết định DẢI ĐỘNG** (biểu diễn được số lớn/nhỏ tới đâu)
- **Định trị quyết định ĐỘ CHÍNH XÁC** (bao nhiêu chữ số có nghĩa)

| Định dạng | bit | E | M | Dải động | Độ chính xác tương đối |
|---|---:|---:|---:|---|---|
| FP32 | 32 | 8 | 23 | ~1e±38 | ~7 chữ số thập phân |
| **TF32** | 19* | 8 | 10 | **~1e±38** | ~3 chữ số |
| **BF16** | 16 | 8 | 7 | **~1e±38** | ~2 chữ số |
| FP16 | 16 | 5 | 10 | ~6e-5 … 65504 | ~3 chữ số |
| FP8 E4M3 | 8 | 4 | 3 | ~±448 | ~1 chữ số |
| FP8 E5M2 | 8 | 5 | 2 | ~±57344 | <1 chữ số |

\* TF32 chiếm 32 bit trong thanh ghi, chỉ 19 bit có nghĩa. Nó là *chế độ tính*, không
phải kiểu lưu trữ.

**Vì sao BF16 tồn tại dù kém chính xác hơn FP16?** Vì nó có **đúng dải động của FP32**.
Trong huấn luyện, gradient có thể nhỏ tới 1e-8 — FP16 làm chúng thành 0 (underflow),
buộc phải dùng loss scaling. BF16 không cần. Đổi lại chỉ còn 7 bit định trị, nhưng
mạng nơ-ron chịu được nhiễu định trị tốt hơn nhiều so với mất dải động.

**Kiểm chứng bằng đo** ([`samples/cpu/numerics.c`](../samples/cpu/numerics.c) E5,
sai số tương đối sau khi làm tròn về từng định dạng):

| giá trị | FP16 | BF16 | sai số FP16 | sai số BF16 |
|---:|---:|---:|---:|---:|
| 1.0 | 1 | 1 | 0 | 0 |
| 0.1 | 0.0999756 | 0.0996094 | 2.4e−04 | 3.9e−03 |
| 1e−06 | 9.54e−07 | 9.98e−07 | 4.6e−02 | 1.6e−03 |
| **1e−08** | **0** | 9.95e−09 | **1.00 (mất hết)** | 4.7e−03 |
| **1e−10** | **0** | 9.96e−11 | **1.00 (mất hết)** | 4.1e−03 |
| 65504 | 65504 | 65280 | 0 | 3.4e−03 |
| **1e+05** | **inf** | 99840 | **tràn** | 1.6e−03 |
| **1e+30** | **inf** | 9.95e+29 | **tràn** | 4.7e−03 |

Đọc bảng: BF16 sai số tương đối **luôn lớn hơn** FP16 trong dải chung (~4e−03 vs
~4e−04, đúng tỉ lệ 2³ vì kém 3 bit định trị) — nhưng nó **không bao giờ chết**.
FP16 làm `1e−08` thành 0 và `1e+05` thành `inf`.

> **Quy tắc:** cần dải động → BF16/TF32. Cần độ chính xác trong dải hẹp đã biết →
> FP16.
>
> - **Huấn luyện dùng BF16** vì gradient thường cỡ `1e−7 … 1e−9`. FP16 làm chúng
>   thành 0 → phải dùng loss scaling (nhân loss lên 2¹⁵ trước backward, chia lại sau).
>   BF16 bỏ được toàn bộ cơ chế đó.
> - **Suy luận LLM dùng FP16 an toàn** vì RMSNorm giữ activation quanh 1, cách xa cả
>   hai đầu dải. Đó là lý do định dạng suy luận phổ biến là FP16 chứ không phải BF16.

**Trên Orin (Ampere sm_87):** Tensor Core hỗ trợ FP16, BF16, TF32, INT8, INT4.
**Không có FP8** — FP8 chỉ từ Ada (sm_89) và Hopper (sm_90) trở lên. Đừng viết kernel
FP8 cho Jetson Orin.

### Vì sao cộng số thực không kết hợp

`(a+b)+c ≠ a+(b+c)`. Mỗi phép cộng làm tròn về số biểu diễn được gần nhất; thứ tự
khác thì sai số tích luỹ khác.

Cận sai số tương đối của phép cộng `n` số:

| Thuật toán | Cận lý thuyết | Ghi chú |
|---|---|---|
| Tuần tự | `O(n·ε)` | tổng chạy lớn dần, mỗi lần cộng mất thêm bit thấp |
| **Pairwise (cây)** | `O(log₂n · ε)` | đúng cái warp reduction của GPU làm |
| Kahan | `O(ε)` | giữ lại phần bị mất, cộng bù vòng sau |

với `ε` = epsilon máy = `2⁻²⁴ ≈ 5.96e−08` cho FP32.

**Kiểm chứng bằng đo** ([`samples/cpu/numerics.c`](../samples/cpu/numerics.c) E1,
cộng `n` số dương cùng cỡ — trường hợp xấu nhất cho tuần tự):

| n | tuần tự | pairwise | Kahan | tuần/pairwise |
|---:|---:|---:|---:|---:|
| 1 024 | 2.65e−07 | 7.57e−08 | 3.78e−08 | 3.5× |
| 8 192 | 1.04e−06 | 9.56e−08 | 1.80e−08 | 10.9× |
| 65 536 | 1.32e−05 | 1.79e−08 | 1.79e−08 | **736×** |
| 524 288 | 2.68e−04 | 5.02e−08 | 5.02e−08 | **5339×** |

Sai số tuần tự tăng gần **tuyến tính theo n** (×8 mỗi bước ≈ đúng tỉ lệ n), còn
pairwise **gần như không đổi**. Đúng như lý thuyết dự đoán.

Đó chính là lý do ta đo được bản CUDA **gần PyTorch hơn** bản C tuần tự
([JETSON.md §4](../firmware/jetson/JETSON.md)):

```
PyTorch vs scalar C (tuần tự, O(n))  : 7.63e-06
PyTorch vs CUDA (warp tree, O(log n)): 4.05e-06
```

**Song song không chỉ nhanh hơn — với reduction, nó còn chính xác hơn.**

> **Hệ quả khi thiết kế kernel:** nếu bạn cần cộng dồn dài trên CPU, đừng viết vòng
> `for` ngây thơ. Dùng nhiều accumulator độc lập (compiler sẽ vector hoá và tự thành
> pairwise) hoặc Kahan nếu cần chính xác cao. Đây cũng là một trong các lý do
> `dot_i8` trong [`matvec_ladder.c`](../samples/cpu/matvec_ladder.c) tích luỹ vào
> `int32` — số nguyên thì **không có** sai số làm tròn nào cả.

---

## 10.2 Vì sao lượng tử hoá hoạt động

### Phân bố trọng số

Trọng số của một mạng đã huấn luyện gần như luôn phân bố **gần Gauss quanh 0**, đuôi
mỏng. Với phân bố như vậy, chia đều dải `[-max, +max]` thành `2^b` mức là hợp lý:
phần lớn giá trị rơi vào vùng giữa, nơi mật độ mức lượng tử hoá dày nhất về mặt tương đối.

Sai số lượng tử hoá đều với bước `Δ` có phương sai `Δ²/12` (mô hình nhiễu đều).
Sơ đồ mà [`src/quantize.py`](../src/quantize.py) dùng — và LLM nói chung dùng — là
**đối xứng, không zero-point**, với `qmax = 2^(b−1) − 1` và `scale = max|x| / qmax`:

```
Δ = A / qmax,   A = max|x|,   qmax = 2^(b−1) − 1

SNR_dB = 20·log10(qmax) + 10·log10(12) − 20·log10(A/σ)
```

> ⚠️ **Đừng dùng công thức `6.02b + 1.76` hay gặp trên mạng.** Nó là SNR của **sóng
> sin toàn thang** lượng tử hoá bằng `2^b` mức đều — quy ước của xử lý tín hiệu,
> không phải sơ đồ của LLM. Với `b=4`, sơ đồ LLM chỉ có `qmax=7` (15 mức, không phải
> 16), và tín hiệu là Gauss chứ không phải sin. Dùng nhầm sẽ lệch vài dB.

**Kiểm chứng bằng đo** ([`samples/cpu/numerics.c`](../samples/cpu/numerics.c) E2,
Gauss(0,1), n=65536, A/σ = 4.62):

| bits | qmax | công thức dB | **đo được dB** | lệch |
|---:|---:|---:|---:|---:|
| 2 | 1 | −2.51 | 0.29 | **+2.80** |
| 3 | 3 | 7.03 | 7.06 | +0.02 |
| 4 | 7 | 14.39 | **14.40** | +0.01 |
| 5 | 15 | 21.01 | 21.00 | −0.01 |
| 6 | 31 | 27.32 | 27.34 | +0.02 |
| 8 | 127 | 39.57 | 39.56 | −0.00 |

Khớp trong **±0.02 dB** từ 3 bit trở lên. Ở 2 bit lệch 2.8 dB vì `qmax=1` chỉ cho 3
mức `{−1,0,+1}` — **giả thiết nhiễu đều sụp đổ** khi bước lượng tử hoá lớn cỡ tín
hiệu. Đây là giới hạn thật của mô hình, không phải lỗi đo.

Mỗi bit thêm cho đúng **6.02 dB** (`20·log10 2`).

Số hạng `−20·log10(A/σ)` là **hình phạt vì outlier**: một giá trị cực đoan kéo `A`
lên cao thì toàn bộ giá trị còn lại mất độ phân giải. Đây là lý do trung tâm của mọi
kỹ thuật lượng tử hoá hiện đại.

### Vì sao chia NHÓM (group-wise) — đo được

Nếu dùng **một** scale cho cả tensor, một outlier duy nhất phá hỏng toàn bộ. Chia
thành nhóm 32/64/128 phần tử, mỗi nhóm một scale → outlier chỉ phá nhóm của nó.

Thí nghiệm E3: bơm **một** outlier vào 4096 trọng số Gauss, lượng tử hoá 4-bit.
Cột "trừ outlier" bỏ chính phần tử đó ra khỏi phép tính SNR:

| outlier | max/σ | 1 scale (toàn) | **1 scale (trừ)** | g128 (toàn) | **g128 (trừ)** |
|---:|---:|---:|---:|---:|---:|
| 1× | 3.4 | 17.08 | 17.08 | 18.74 | 18.74 |
| 16× | 8.2 | 9.39 | 9.32 | 17.92 | **17.85** |
| 64× | 29.3 | 1.27 | 0.24 | 15.31 | **14.29** |
| 256× | 57.6 | 7.21 | **0.00** | 21.19 | **13.99** |
| 1024× | 63.5 | 18.39 | **0.00** | 32.38 | **13.99** |

Một scale: SNR trên phần "lành" sụp về **0 dB** — nhiễu bằng tín hiệu, trọng số hỏng
hoàn toàn. Group=128: giữ ~14 dB bất kể outlier lớn cỡ nào.

> ### Bẫy đo lường quan trọng
>
> Nhìn cột "toàn bộ" của một-scale: `1.27 → 7.21 → 18.39 dB`. **SNR TĂNG LẠI** khi
> outlier càng lớn. Đó **không phải** chất lượng tốt lên.
>
> SNR = công suất tín hiệu / công suất nhiễu. Outlier khổng lồ chiếm gần hết **tử số**,
> và bản thân nó được biểu diễn *chính xác* (nó đúng bằng `max` nên map thẳng vào
> `qmax`, sai số 0). Chỉ số đẹp lên trong khi **mọi trọng số còn lại đã hỏng**.
>
> **Hệ quả thực hành: với LLM đừng dùng SNR/MSE làm chỉ tiêu nghiệm thu.** Dùng
> **perplexity**. Đó chính là lý do [`src/quantize.py`](../src/quantize.py) đo val
> loss chứ không đo sai số trọng số, và vì sao chuỗi kiểm chứng của repo đặt ppl ở
> tầng 2 riêng biệt.

Cái giá của group nhỏ là overhead lưu scale:

```
bit/weight thực tế = b + (bit_scale / group)
```

Thí nghiệm E4 (65536 trọng số Gauss + 1 outlier 30×, 4-bit):

| group | SNR dB | bit/weight | lợi so 1 scale |
|---:|---:|---:|---:|
| toàn tensor | 1.07 | 4.000 | — |
| 512 | 16.34 | 4.031 | +15.27 dB |
| **128** | **18.27** | **4.125** | **+17.20 dB** |
| 64 | 19.19 | 4.250 | +18.12 dB |
| 32 | 20.16 | 4.500 | +19.09 dB |
| 16 | 21.31 | 5.000 | +20.25 dB |

**Lợi ích lớn nhất nằm ở bước đầu tiên** (toàn tensor → 512: +15.3 dB với chỉ
+0.031 bit). Từ 128 xuống 32 chỉ thêm 1.9 dB nhưng tốn thêm 0.375 bit/weight.
Đó là lý do `group=128` là lựa chọn mặc định của hầu hết định dạng.

Đây cũng là vì sao "Q4" thực tế nặng ~4.5 bit và model 8B Q4_K_M là 4.7 GB chứ không
phải 4.0. Code tham chiếu: [`src/quantize.py:30`](../src/quantize.py#L30).

### Vì sao bảng lớn chịu lượng tử hoá TỐT HƠN

Kết quả đo được ở repo này (và tái hiện lại được):

| arm | degradation 4-bit |
|---|---:|
| baseline (dày đặc) | +0.0522 nats |
| ple (có bảng 1.57M) | **+0.0416 nats** |

Lý do lý thuyết: **dư thừa**. Trong một mạng dày đặc nhỏ, mỗi trọng số mang thông tin
gần như không lặp lại — nhiễu lượng tử hoá cộng thẳng vào lỗi mô hình. Trong một bảng
tra cứu lớn, thông tin phân tán trên nhiều hàng; nhiễu độc lập trên mỗi hàng và trung
bình hoá bớt qua các lớp.

Tổng quát: **tham số càng nhiều so với lượng thông tin cần biểu diễn, càng chịu
lượng tử hoá tốt.** Hệ quả thực dụng: với cùng dung lượng, chọn model lớn hơn quantize
sâu hơn (8B-Q4 > 3B-Q8) — nhưng phải đo trên workload của bạn.

### Activation khó hơn weight

Weight tĩnh, phân bố biết trước, đối xứng quanh 0 → scale-only là đủ.
Activation phụ thuộc input, sau SiLU/GELU thì lệch, và **có outlier theo kênh** rất
mạnh ở LLM lớn (một vài kênh lớn hơn phần còn lại hàng trăm lần).

Đó là lý do:
- **Weight-only (W4A16)** dễ, dùng cho decode (memory-bound bởi weight)
- **W8A8** khó, cần SmoothQuant: chuyển độ khó từ activation sang weight bằng
  `Y = (X/s)·(sW)` với `s` per-channel — toán học tương đương, số học dễ hơn nhiều

---

## 10.3 Toán của attention

```
Attention(Q,K,V) = softmax(QKᵀ / √d_head) · V
```

### Vì sao chia √d_head

`q·k` là tổng của `d` tích. Nếu các thành phần độc lập, kỳ vọng 0, phương sai 1 thì
`q·k` có phương sai `d`, tức độ lệch chuẩn `√d`. Không chia thì với `d=128`, điểm số
có biên độ ~11 — softmax của các số cách nhau 11 gần như là one-hot, **gradient triệt
tiêu** và mô hình không học được.

Chia `√d` đưa phương sai về 1, giữ softmax ở vùng có gradient.

### Softmax ổn định số học

`exp(x)` tràn FP32 khi `x > 88`. Nên luôn tính:

```
softmax(x)_i = exp(x_i − max(x)) / Σ exp(x_j − max(x))
```

Trừ max không đổi kết quả (tử và mẫu cùng nhân `e^{-max}`) nhưng đảm bảo số mũ lớn
nhất là 0. Đây là 2-pass: pass 1 tìm max, pass 2 tính exp.
Code: [`llm.h:327`](../firmware/common/llm.h#L327), [`llm_cuda.cuh` `k_attention`](../firmware/jetson/llm_cuda.cuh).

**FlashAttention chính là biến 2-pass thành 1-pass online**, dùng công thức cập nhật:
khi gặp max mới `m'`, nhân tổng cũ với `exp(m − m')`. Nhờ vậy không bao giờ phải
materialize ma trận `T×T` ra DRAM. FLOP không đổi; **bytes giảm hàng chục lần**.

### Độ phức tạp — và vì sao context dài đắt

| | FLOP | Bytes (KV cache) |
|---|---|---|
| Prefill T token | `O(T²·d)` cho attention + `O(T·d²)` cho projection | ghi `O(T·d)` |
| Decode 1 token tại pos `t` | `O(t·d)` | **đọc lại `O(t·d)` mỗi token** |

Chi phí decode **tăng tuyến tính theo độ dài context** — không phải vì tính toán mà vì
phải đọc lại toàn bộ KV cache mỗi token.

### MHA / MQA / GQA — số học của KV cache

```
KV_bytes = 2 · L · n_kv_heads · d_head · ctx · batch · (bits/8)
```

| Biến thể | n_kv_heads | KV so MHA | Chất lượng |
|---|---|---|---|
| MHA | = n_heads | 1× | mốc |
| MQA | 1 | `1/n_heads` | tụt rõ |
| **GQA** | n_heads / g | `1/g` | gần MHA |

Llama-3 dùng GQA với 32 query head / 8 KV head → KV nhỏ đi 4×. Đây là lý do model
hiện đại chạy được context dài trên phần cứng nhỏ.

Model trong repo này dùng MHA (4 head, `n_kv_heads = 4`) vì `d_model` chỉ 128 — GQA
không đáng ở quy mô đó.

---

## 10.4 Batching — vì sao nó gần như miễn phí, tới một điểm

Nhắc lại arithmetic intensity của decode với batch `B`:

```
FLOP  ≈ 2·P·B          (P = số tham số)
Bytes ≈ P·(bits/8)     (weights đọc MỘT LẦN, dùng cho cả B token)
AI    = 2B / (bits/8) = 16B/bits
```

**Bytes không tăng theo B.** Đó là toàn bộ lý do batching hiệu quả: đọc weights một
lần, dùng cho nhiều token.

Batch tới hạn `B*` là chỗ AI chạm machine balance:

```
B* = balance · bits / 16
```

Với Orin (balance FP16 ≈ 152 đo được) và W4: `B* = 152·4/16 = 38`.

Đo thật xác nhận trên GPU ([`samples/gpu/bench_decode.cu`](../samples/gpu/bench_decode.cu)):
M=1 mất 0.512 ms, M=64 mất 0.544 ms — **64 token với giá của 1**.

Cùng quy luật trên CPU ([`samples/cpu/matvec_ladder.c`](../samples/cpu/matvec_ladder.c)):
đa luồng chỉ thắng từ B≥128.

**Continuous batching** khai thác điều này ở tầng phục vụ: thay vì đợi cả batch xong
mới nhận request mới, ghép request mới vào ngay khi có chỗ trống. **PagedAttention**
giải quyết vấn đề phân mảnh KV cache khi các request có độ dài khác nhau, bằng cách
cấp phát KV theo "trang" cố định thay vì khối liền.

---

## 10.5 Speculative decoding — kỹ thuật duy nhất phá trần bandwidth

Ở batch=1, `tok/s ≤ BW / W_bytes` là trần cứng. Speculative decoding lách được vì
nó **đổi bài toán**: thay vì sinh 1 token mỗi lần đọc weights, sinh `K` token nháp
bằng model nhỏ rồi **verify cả K trong một lần forward** của model lớn.

Vì một forward đọc weights đúng 1 lần bất kể xử lý bao nhiêu token (mục 10.4), verify
K token gần như miễn phí nếu `K < B*`.

**Tăng tốc kỳ vọng.** Gọi `α` là xác suất một token nháp được chấp nhận. Số token
chấp nhận trung bình mỗi vòng (draft K, verify 1 lần):

```
E[accepted] = (1 − α^(K+1)) / (1 − α)
speedup    ≈ E[accepted] / (1 + K·c)
```

`c` = chi phí tương đối của model draft so với model target. Với `α=0.7`, `K=4`,
`c=0.1`: `E ≈ 2.9`, speedup ≈ `2.9/1.4 ≈ 2.1×`.

Điểm quan trọng: **speedup bão hoà theo K**. Tăng K mãi không giúp vì `α^K` tụt nhanh;
tối ưu thường ở `K = 3..5`.

Biến thể: **Medusa** (nhiều head dự đoán token tương lai, không cần model riêng),
**EAGLE** (dự đoán ở không gian đặc trưng thay vì token), **lookahead decoding**.

---

## 10.6 Phân cấp bộ nhớ — vì sao access pattern quan trọng hơn dung lượng

Bộ nhớ không đọc theo byte mà theo **dòng cache** (64 B trên x86, 128 B trên GPU
Ampere; DRAM đọc theo burst 32 B sector).

Hệ quả:

| Pattern | Hiệu suất | Vì sao |
|---|---|---|
| Tuần tự | ~100% | mọi byte trong dòng đều dùng |
| Bước nhảy 64 B | ~1/16 | mỗi dòng chỉ dùng 4/64 byte |
| Ngẫu nhiên trong vùng lớn | rất tệ | thêm TLB miss |
| Ngẫu nhiên nhưng **theo hàng liền** | tốt | đúng trường hợp bảng PLE |

Bảng PLE của repo là ví dụ mẫu: truy cập **ngẫu nhiên theo hàng**, nhưng mỗi hàng
liền mạch 192 byte. Đọc ngẫu nhiên 1 hàng liền mạch rẻ; đọc rải rác 192 byte thì không.

**Đó là lý do kế toán 3 tầng của [`budget.py`](../src/budget.py) phân loại theo
ACCESS PATTERN chứ không theo tốc độ.**

### Ba loại cache miss

- **Compulsory** — lần đầu chạm dữ liệu, không tránh được (trừ prefetch)
- **Capacity** — working set lớn hơn cache; tránh bằng **blocking/tiling**
- **Conflict** — nhiều địa chỉ ánh xạ cùng set; tránh bằng padding

Với LLM decode, phần lớn là compulsory: mỗi token phải đọc toàn bộ weights một lần,
không có gì để tái sử dụng. **Đó là định nghĩa của memory-bound.**

Ngoại lệ đáng chú ý trong repo này: model chỉ 1.87 MB nên **nằm gọn trong L2 2 MB**
của Orin — nên nó KHÔNG bandwidth-bound mà launch-bound. Model thật (4-8 GB) thì
ngược lại hoàn toàn.

---

## 10.7 Bảng tra nhanh — chọn kỹ thuật theo nút thắt

| Nút thắt | Dấu hiệu | Kỹ thuật đúng | Kỹ thuật VÔ ÍCH |
|---|---|---|---|
| Bandwidth | GB/s đạt ≈ peak | quantize sâu hơn, KV int8, GQA, head nhỏ hơn | SIMD, tensor core, occupancy |
| Compute | GB/s thấp, FLOP/s ≈ peak | SIMD/tensor core, thuật toán tốt hơn | giảm bytes |
| Launch/overhead | us/kernel ≈ kernel rỗng | CUDA Graphs, fusion, batch lớn hơn | tối ưu trong kernel |
| Độ trễ (latency) | occupancy thấp, đơn vị rảnh | tăng ILP, tăng occupancy, prefetch | thêm luồng nếu đã bão hoà |
| Dung lượng | OOM | quantize, offload, paged KV | mọi thứ khác |

→ Tiếp: [11-toi-uu-nvidia.md](11-toi-uu-nvidia.md)
