# 4. Hiểu model — đọc `esp32-ai-main` như một cuốn giáo trình

Repo repo này hiếm ở chỗ nó chứa **toàn bộ chuỗi** từ
kiến trúc → train → ablation → quantize → export → runtime C → đo trên silicon,
trong 1936 dòng code. Không có lớp trừu tượng nào che mất bản chất. Đây là lý do nó
là tài liệu học tốt hơn phần lớn tutorial.

File này: **hiểu model**. File [05](08-nhat-ky-toi-uu.md): **hiểu cách tối ưu**.

---

## 4.1 Giải phẫu một decoder-only transformer

Toàn bộ model nằm trong [`src/model.py`](../src/model.py), 323 dòng.
Cấu hình deploy: `V=32768, D=96, L=6, H=4, P=128`.

```
token id (int)
   │
   ├─ tok_emb[token]                          [D=96]        ← tra bảng
   │
   ├─ (PLE) ple_model_proj · x  → [L·P=768] → reshape [6,128] → RMSNorm
   │        ple_table[token]    → [768]     → reshape [6,128]
   │        ple = (proj + table·√P) / √2                     ← điều kiện hoá mỗi lớp
   │
   ├─ × 6 lớp:
   │     x += Attention(RMSNorm(x))          ← trộn thông tin GIỮA các token
   │     x += SwiGLU(RMSNorm(x))             ← xử lý TỪNG token độc lập
   │     x += RMSNorm(ple_proj(gelu(ple_gate(x)) · ple[l]))
   │
   ├─ RMSNorm(x)
   └─ head · x → logits [V=32768]            ← head = tok_emb (tied)
```

**Hai loại phép toán, và đây là chỗ nhiều người nhầm:**

| | Attention | FFN (SwiGLU) |
|---|---|---|
| Trộn thông tin | **giữa các token** | trong từng token |
| Tham số | 4·D² = 36.8K/lớp | 3·D·F |
| Chi phí khi ctx dài | **tăng theo ctx** (KV cache) | không đổi |
| Đo được ở repo này | 25.6 ms/token | 6.9 ms/token |

Attention là thứ duy nhất cho phép token thứ 100 "nhìn thấy" token thứ 1. FFN chỉ
biến đổi từng vector độc lập. Nếu bỏ attention, model thành một MLP áp lên từng token
riêng lẻ — không còn là language model.

---

## 4.2 Từng thành phần: nó là gì và VÌ SAO

### RMSNorm ([model.py:64](../src/model.py#L64))

```python
return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

So với LayerNorm: **bỏ phần trừ trung bình và bỏ bias**. Chỉ chuẩn hoá theo độ lớn.

Vì sao dùng: rẻ hơn (không cần pass tính mean rồi pass tính variance), và thực nghiệm
cho thấy chất lượng tương đương. Llama, Gemma, Qwen đều dùng RMSNorm.

Bản C tương ứng ([llm.h:207](../firmware/common/llm.h#L207)) đúng 5 dòng —
một trong những lý do port sang C khả thi.

> **Chi tiết dễ bỏ qua:** norm giữ nguyên **fp32** kể cả khi mọi thứ khác 4-bit
> ([quantize.py:59](../src/quantize.py#L59): `if p.ndim < 2: continue`).
> Norm chỉ có D=96 tham số mỗi cái — quantize chúng tiết kiệm ~0 byte nhưng gây lỗi
> lớn vì mỗi weight ảnh hưởng tới toàn bộ vector. **Nguyên tắc chung: không quantize
> tensor nhỏ.** llama.cpp cũng vậy, TensorRT cũng vậy.

### RoPE — Rotary Position Embedding ([model.py:74-86](../src/model.py#L74-L86))

Model phải biết token nào đứng trước token nào. Ba cách lịch sử:

1. **Learned positional embedding** (GPT-2): thêm bảng `[seq_len, D]`. Tốn params, không
   ngoại suy được quá seq_len đã train.
2. **Sinusoidal** (Transformer gốc): cộng sin/cos vào embedding. Không tốn params.
3. **RoPE** (Llama/Gemma/Qwen hiện nay): **xoay** vector q và k một góc tỉ lệ với vị trí.

RoPE thắng vì tích vô hướng `q_m · k_n` sau khi xoay **chỉ phụ thuộc `m − n`** — tức
là attention tự động nhận biết *khoảng cách tương đối*, không phải vị trí tuyệt đối.
Không tốn tham số nào.

```python
inv = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))   # tần số giảm dần
freqs = outer(positions, inv)
cos, sin = cos(freqs), sin(freqs)
```

`theta=10000` quyết định "bước sóng" dài nhất. **Đây chính là tham số bạn chỉnh khi
muốn kéo dài context** (RoPE scaling / NTK-aware / YaRN) — tăng theta cho phép model
xử lý ctx dài hơn lúc train.

> **Chi tiết triển khai quan trọng:** repo dùng *split-half* RoPE (chia vector làm đôi,
> [model.py:83](../src/model.py#L83)) chứ không phải *interleaved*
> (cặp phần tử liền kề). Hai cách cho kết quả khác nhau và **không tương thích**.
> Đây là nguồn bug số 1 khi port model giữa các framework. Bản C phải khớp đúng
> ([llm.h:308-317](../firmware/common/llm.h#L308-L317)) — và cách duy
> nhất để biết là golden logits (§4.6).

### SwiGLU ([model.py:108](../src/model.py#L108))

```python
return self.down(F.silu(self.gate(x)) * self.up(x))
```

MLP cổ điển: `down(relu(up(x)))` — 2 ma trận. SwiGLU: **3 ma trận**, trong đó `gate`
và `up` nhân **elementwise** với nhau.

Vì sao tốt hơn: nhánh `gate` học "cho thông tin nào đi qua", nhánh `up` mang nội dung.
Đây là cơ chế cổng (gating) — giống LSTM nhưng không hồi quy. Thực nghiệm (Shazeer,
*GLU Variants Improve Transformer*, 2020) cho thấy với cùng số tham số, SwiGLU thắng.

Cái giá: 3 ma trận thay vì 2 → để giữ nguyên số tham số, `ffn_hidden` phải giảm còn
~2/3. Llama dùng `F ≈ 8/3·D` thay vì `4·D`.

### Tied embeddings ([model.py:158](../src/model.py#L158))

```python
self.head.weight = self.tok_emb.weight  # tied
```

Bảng embedding đầu vào `[V, D]` và ma trận output head `[V, D]` **dùng chung bộ nhớ**.

Tiết kiệm: `32768 × 96 = 3.15M` tham số — với model core chỉ 558K thì đây là khoản
khổng lồ. Ngoài ra tied embedding thường *cải thiện* chất lượng ở model nhỏ.

Nhưng: cũng chính vì tied mà bug xuất hiện. [`export.py:142`](../src/export.py#L142):

```python
# state_dict lists both keys for tied weights; without this the head silently
# stays fp32 and the golden no longer matches the (fully-quantized) C port.
if "head.weight" in dq_sd:
    dq_sd["head.weight"] = dq_sd["tok_emb.weight"]
```

**Bài học:** tied weights xuất hiện 2 lần trong `state_dict()`. Nếu bạn lặp qua
state_dict để quantize, bạn quantize `tok_emb` rồi ghi đè `head` bằng bản fp32 gốc.
Golden sẽ không khớp và bạn sẽ tưởng runtime C sai.

### Attention với KV cache ([llm.h:302-343](../firmware/common/llm.h#L302-L343))

Đây là chỗ dễ hiểu nhất trong bản C, vì nó là decode 1 token, không có batch:

```c
// tính q,k,v cho token hiện tại
MATVEC(&m->qkv[l], s->h, s->qkv);        // [3D]
// áp RoPE tại vị trí pos
// LƯU k,v vào cache tại pos
memcpy(kc + pos*D, k, D*4);
memcpy(vc + pos*D, v, D*4);
// attention: q hiện tại đấu với TẤT CẢ k trong cache (0..pos)
for (t = 0; t <= pos; t++) { dot = q·k[t]; ... }
```

**Đây là toàn bộ ý nghĩa của KV cache:** không tính lại k,v của các token cũ. Đổi lại
phải **lưu** chúng và **đọc lại toàn bộ** mỗi token.

Kích thước ở repo này: `L×S×D×4 × 2 = 6×512×96×4×2 = 2.36 MB`. Trên ESP32 nó nằm ở
PSRAM. Trên Jetson với Llama-8B ctx 8192 con số tương ứng là **1.07 GB** — cùng công
thức, khác 450 lần.

> Softmax ổn định số học: [llm.h:327-342](../firmware/common/llm.h#L327-L342)
> chạy 2 pass — pass 1 tìm max, pass 2 tính `exp(s - max)`. Không có bước trừ max,
> `exp(20)` tràn fp32. **FlashAttention chính là thuật toán biến 2 pass này thành
> 1 pass online** để không phải ghi ma trận điểm ra bộ nhớ.

---

## 4.3 Tham số nằm ở ĐÂU — ý tưởng trung tâm của repo

Đây là phần đáng học nhất, và nó áp dụng nguyên vẹn cho Jetson.

Câu hỏi thông thường: "model có bao nhiêu tham số?" — **sai câu hỏi**.
Câu hỏi đúng: "**mỗi tham số được đọc theo kiểu nào?**"

[`src/budget.py:8-19`](../src/budget.py#L8-L19) chia làm 3 tầng theo
**access pattern**, không phải theo tốc độ:

| Tầng | Truy cập | Ở ESP32 | Ở Jetson |
|---|---|---|---|
| **core** | dày đặc, ngẫu nhiên, **mỗi token** | SRAM 512KB (khan hiếm thật) | phải trong VRAM/RAM |
| **stream** | dày đặc nhưng **quét tuần tự 1 lần/token** | PSRAM/flash — tốn *băng thông*, không tốn *dung lượng nhanh* | tốn bandwidth |
| **table** | **thưa: 1 hàng/token** | flash memory-mapped | có thể để trên disk/CPU RAM |

Model deploy 28.9M tham số phân bổ:

```
core     558 K  (1.9%)   6 lớp attention+FFN+PLE gate   → 273 KB @4-bit → vừa SRAM
stream  3.15 M  (11%)    output head (tied embedding)   → quét toàn bộ mỗi token
table    25 M   (87%)    ple_table[V, L·P]              → chỉ đọc 6 hàng/token (~450 B)
```

**87% số tham số gần như miễn phí** vì mỗi token chỉ chạm 6 hàng của nó.
Đo trên silicon ([RESULTS.md:94](../RESULTS.md)): table tốn **0.12 ms/token**,
head tốn **17.3 ms/token** — bảng lớn gấp 8 lần head nhưng rẻ hơn 144 lần.

> Comment ở [budget.py:20-21](../src/budget.py#L20-L21) là câu quan trọng
> nhất trong repo:
> > *"Treating `stream` as if it were `core` is what made large vocabularies look
> > unaffordable, when in fact they are merely slow."*
>
> Nhầm "chậm" thành "không đủ chỗ" khiến bạn loại bỏ cả một hướng thiết kế tốt.

**Áp vào Jetson.** Cùng cách phân loại, đổi tên tầng:

| Tầng | Jetson Orin Nano 8GB |
|---|---|
| core | trọng số layer — phải nằm trong 8GB, đọc mỗi token → **quyết định trần tok/s** |
| stream | output head (Llama-3 vocab 128K × 4096 = 525M params!) — cũng đọc mỗi token |
| table | không có sẵn trong Llama, **nhưng MoE chính là dạng này** — chỉ 2/8 expert active mỗi token |

Đây là lý do MoE (Mixtral, Qwen3-MoE, DeepSeek) hấp dẫn trên thiết bị nhúng: tham số
nhiều nhưng *đọc ít mỗi token*. Cùng nguyên lý PLE, khác cách thực hiện.

---

## 4.4 PLE — Per-Layer Embeddings, ý tưởng làm nên repo

Ý tưởng: thay vì nhét thêm tham số vào phần *tính toán* (đắt, phải nằm ở bộ nhớ nhanh),
nhét vào một **bảng tra cứu** rồi tiêm vào từng lớp.

```python
# model.py:207-214 — xây per-layer input
ple = self.ple_model_proj(x) * (d_model**-0.5)       # nửa phụ thuộc ngữ cảnh
ple = self.ple_proj_norm(ple.view(B, T, L, P))
table = self.ple_table(idx).view(B, T, L, P)          # nửa tra cứu theo token
ple = (ple + table * (P**0.5)) * (2**-0.5)

# model.py:141-142 — tiêm vào lớp l
g = F.gelu(self.ple_gate(x))
x = x + self.ple_norm(self.ple_proj(g * ple))
```

**Phép nhân `g * ple` mới là điểm mấu chốt** (comment ở [model.py:130-132](../src/model.py#L130-L132)):
đây là *điều kiện hoá* (conditioning), không phải cộng thêm bias. Cả hai thừa số đều
có thể triệt tiêu nhau — model học được "với token này, ở lớp này, bỏ qua nhánh kia".

Nếu chỉ **cộng** `ple` vào residual, nó thành một bias theo token, tác dụng yếu hơn nhiều.

### Ba chi tiết nhỏ, đều load-bearing

1. `* (2**-0.5)` — trung bình hai nguồn nhưng giữ nguyên phương sai. Bỏ đi thì
   activation phình gấp √2 và train kém ổn định.
2. `table * sqrt(ple_dim)` — comment [model.py:211-212](../src/model.py#L211-L212)
   nói thẳng: *"undocumented in Gemma's config but load-bearing"*. Không có nó,
   đóng góp của bảng bị lấn át bởi nhánh projection.
3. `nn.init.zeros_(block.ple_norm.weight)` ([model.py:187](../src/model.py#L187))
   — khởi tạo gain của norm bằng 0 để nhánh PLE **là no-op tại step 0**. Lý do trong
   comment: nếu không, mọi arm bắt đầu từ một hàm khác nhau và bạn đang đo *may mắn
   khởi tạo* chứ không phải *khả năng học*.

> **Bài học chuyển giao:** ba dòng này đều là *scale factor* và *init*. Trong LLM,
> phần lớn "nó không chạy" đến từ scale sai chứ không phải logic sai. Khi port model,
> luôn kiểm phương sai activation ở từng tầng.

---

## 4.5 Bài toán ngân sách — cách so sánh công bằng

Muốn biết PLE có thật sự tốt không, phải so với baseline **cùng ngân sách**. Nhưng
"cùng ngân sách" nghĩa là gì? Repo trả lời: cùng `core` (tham số đọc mỗi token ở bộ
nhớ nhanh), vì đó là thứ khan hiếm.

[`model.py:304-313`](../src/model.py#L304-L313) **binary-search `ffn_hidden`**
cho tới khi mọi arm có core bằng nhau:

```python
lo, hi = 1, 64 * cfg.d_model
while lo < hi:
    mid = (lo + hi + 1) // 2
    cfg.ffn_hidden = mid
    if TinyLM(cfg).param_budget()["core"] <= target_core:
        lo = mid
    else:
        hi = mid - 1
```

Vì sao cần: mỗi arm có "đường ống" khác nhau (PLE tốn thêm `ple_gate`, `ple_proj`,
`ple_model_proj`). Nếu không cân, arm nào ít đường ống hơn sẽ được FFN to hơn, và
bạn chỉ đang đo "FFR to hơn thì tốt hơn" — điều ai cũng biết.

Comment [model.py:272-274](../src/model.py#L272-L274) nói thẳng:
*"Without this the comparison is worthless."*

> **Đây là kỹ năng quan trọng nhất của file này.** Mọi so sánh tối ưu đều cần một
> đại lượng giữ cố định. Trên Jetson: so hai lượng tử hoá thì phải **cùng dung lượng
> GB**, không phải cùng tên model. So 8B-Q4 với 3B-Q8 ở cùng 4.5GB mới là so công bằng.

---

## 4.6 Chuỗi kiểm chứng ba tầng

Repo tách bạch ba loại lỗi, mỗi loại một công cụ. **Đây là mẫu bạn nên sao chép nguyên
vẹn cho Jetson.**

| Tầng | Câu hỏi | Công cụ | Ngưỡng đạt |
|---|---|---|---|
| 1 | Bản port C có **đúng** không? | [`host_verify/verify.c`](../firmware/host_verify/verify.c) so với golden logits | `max abs diff < 1e-5` |
| 2 | Lượng tử hoá làm **hỏng** bao nhiêu? | [`quantize.py`](../src/quantize.py) đo val loss fp32 vs 4-bit | báo cáo nats, 2 seed |
| 3 | int8 activation có **hỏng** thêm không? | [`host_verify/ppl.c`](../firmware/host_verify/ppl.c), build có/không `-DLLM_INT8_ACT` | delta ppl ~0 |

Điểm tinh tế ở [`export.py:12-14`](../src/export.py#L12-L14):

> *"the golden logits are the **4-bit** model's logits: C-vs-PyTorch then isolates
> port correctness from quantization error, which was measured separately."*

Golden được sinh từ model **đã dequantize**, không phải fp32 gốc. Nhờ vậy khi C không
khớp, bạn biết chắc là lỗi port — không phải lỗi lượng tử hoá. **Tách biến.** Nếu golden
là fp32, mọi sai lệch trộn lẫn và bạn không debug được.

Kết quả thật: khớp toàn bộ 32,768 logits, `max abs diff = 0.00001`
([RESULTS.md:116](../RESULTS.md)).

### Dịch sang Jetson

```
Tầng 1: PyTorch fp16 logits  →  TensorRT engine logits    | max diff < 1e-2 (fp16)
Tầng 2: fp16 ppl             →  INT8/INT4 ppl             | delta ppl < 0.1
Tầng 3: ppl trước/sau mỗi tối ưu (KV int8, flash-attn)    | delta ppl ~0
```

llama.cpp có sẵn `llama-perplexity`. TensorRT-LLM có `summarize.py`. Nhưng **golden
logits thì bạn phải tự làm** — và đó là tầng bắt được nhiều bug nhất.

---

## 4.7 Bài tập — theo thứ tự

**Mức 1 — đọc hiểu**
1. Vẽ lại shape của mọi tensor trong `llm_forward` cho cấu hình deploy
   (`V=32768, D=96, L=6, H=4, P=128, S=512`). Đối chiếu với các `ps()` alloc ở
   [esp32_llm.ino:164-176](../firmware/esp32_llm/esp32_llm.ino#L164-L176).
2. Tính bằng tay: KV cache chiếm bao nhiêu byte? So với số bạn tính, `s.kcache`
   cấp phát bao nhiêu?
3. Trong [llm.h:275-296](../firmware/common/llm.h#L275-L296), tìm chỗ
   `s->trow` được **dùng lại** cho mục đích khác. Vì sao an toàn? (comment có giải thích)

**Mức 2 — chạy**
4. `uv run python src/budget.py` — xem 3 tầng cho từng cấu hình ứng viên.
   Sửa `SRAM_BYTES` thành 8GB và `vocab` thành 128256 (Llama-3) rồi chạy lại.
   Tầng nào bùng nổ?
5. `uv run python src/quantize.py --bits 8` và `--bits 3`. Vẽ ppl theo số bit.
6. Build `host_verify/verify.c`, cố tình sửa RoPE trong `llm.h` từ split-half sang
   interleaved. Golden lệch bao nhiêu? Đây là bài tập cho thấy vì sao cần tầng 1.

**Mức 3 — chuyển giao**
7. Viết lại `budget.py` cho Jetson: 3 tầng thành (VRAM-resident / streamed / offloaded),
   nhập Llama-3.1-8B, so với `bench/roofline.py`.
8. Với Llama-3.1-8B: `stream` (output head) là `128256 × 4096 = 525M` tham số =
   **6.5% toàn model**. Ở Q4 nó là 0.3GB đọc mỗi token trong tổng 4.8GB. Tính xem
   nếu dùng vocab 32K thì tok/s tăng bao nhiêu %.

→ Tiếp: [05-hoc-tu-esp32-llm.md](08-nhat-ky-toi-uu.md) — nhật ký tối ưu 0.57 → 9.5 tok/s
