# 14–15. KV Cache và Token Sampling

Chương 6 cho bạn một decoder block sinh ra logits cho **một** token. Hai câu hỏi còn
lại để có một model sinh chữ thật: (14) làm sao sinh token thứ 1000 mà không phải
tính lại từ token thứ 1, và (15) làm sao biến logits (một hàng số) thành **một**
token cụ thể để in ra màn hình.

## 14. KV Cache

### Vấn đề, nhìn từ góc DSP

Attention (chương 3 §3.6, chương 6 §6.3) ở token thứ `pos` cần `K` và `V` của **mọi**
token từ `0` đến `pos`. Không có cache, sinh token thứ 500 nghĩa là tính lại `K,V`
của cả 500 token trước đó — đúng kiểu bạn tính lại toàn bộ buffer FIR mỗi mẫu mới
thay vì trượt cửa sổ. Không ai làm FIR kiểu đó; không ai chạy LLM kiểu đó.

**Giải pháp: cache lại K,V đã tính, chỉ tính K,V của token MỚI.** Đây là "sliding
window" bạn đã quen, chỉ khác: cửa sổ **lớn dần** thay vì trượt (mọi token cũ vẫn
cần dùng lại, không bị bỏ đi).

### Đúng 6 dòng C thực hiện toàn bộ ý tưởng

```c
// llm.h:318-320 — LƯU k,v của token hiện tại vào cache, tại đúng vị trí pos
float *kc = s->kcache + (size_t)l * S * D, *vc = s->vcache + (size_t)l * S * D;
memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
memcpy(vc + (size_t)pos * D, v, D * sizeof(float));

// llm.h:329-335 — ĐỌC LẠI toàn bộ 0..pos, không tính lại, chỉ đọc
for (int t = 0; t <= pos; t++) {
    float *kt = kc + (size_t)t * D + hh * Dh, dot = 0.f;
    for (int i = 0; i < Dh; i++) dot += qh[i] * kt[i];   // chương 3 §3.1 -- dot product
    ...
}
```

**Cái giá:** phải cấp phát bộ nhớ đủ cho **toàn bộ** context trước, và đọc lại toàn
bộ mỗi bước — đây là lý do decode chậm dần khi context dài (attention là `O(T²)` —
chương 3 §3.7). Kích thước cache tính được ngay:

```
KV cache = n_layers × seq_len × d_model × 4 byte × 2 (K và V)

Cấu hình deploy repo này: 6 × 512 × 96 × 4 × 2 = 2.36 MB   -- vừa PSRAM ESP32 dễ dàng
Llama-3.1-8B, ctx=8192   : cùng công thức, ra 1.07 GB       -- công thức GIỐNG HỆT, chỉ đổi số
```

Đây chính là bài học chuyển giao thẳng lên Jetson/PC: KV cache **không phải là
weight**, nó là bộ nhớ **tăng theo độ dài hội thoại**, và trên GPU nó cạnh tranh
VRAM trực tiếp với weight. Đọc [`../07-kv-cache-engine.md`](../07-kv-cache-engine.md)
để có công thức đầy đủ (MHA/MQA/GQA — các cách nén KV cache).

### Thực hành — đo tác động thật của KV cache

```bash
cd firmware/jetson && make bench      # cột "attention" trong output = chi phí đọc KV cache mỗi token
```

So sánh trực tiếp: không có cache, token thứ `pos` tốn `O(pos)` phép tính riêng cho
việc **tính lại** K,V (ngoài phần đọc để attend); có cache, phần đó là `O(1)` — chỉ
tính K,V của **đúng 1 token mới**. Phần đọc lại để attend vẫn là `O(pos)` cả hai
trường hợp — cache không xoá được `O(T²)` của attention, nó chỉ xoá phần **tính lại
K,V** vốn hoàn toàn không cần thiết.

## 15. Token Sampling

Model không "chọn" một token — nó xuất ra 32.768 số (logits, chương 4 §4.1). Việc
biến chuỗi số đó thành **một** token cụ thể là một bước xử lý riêng, tách khỏi
mạng nơ-ron hoàn toàn — đúng như bạn tách "tính toán DSP" khỏi "quyết định điều
khiển" (control logic) trong một hệ nhúng.

### Bước 1 — Softmax: logits → xác suất

```
P(token i) = exp(logit_i) / Σⱼ exp(logit_j)
```

Biến số thực bất kỳ (âm hoặc dương, không giới hạn) thành xác suất hợp lệ (dương,
tổng = 1). Đúng công thức đã dùng trong attention (chương 6 §6.3) — softmax không
phải khái niệm riêng của sampling, nó là **một phép chuẩn hoá dùng lại ở hai chỗ**.

### Bước 2 — Temperature: chỉnh độ "liều lĩnh"

```python
logits = logits / temperature      # model.py:259
```

```
temperature → 0    : phân bố "nhọn" tối đa → luôn chọn token có logit cao nhất → GREEDY, tất định
temperature = 1     : giữ nguyên phân bố model học được
temperature > 1     : phân bố "phẳng" hơn → token yếu có cơ hội cao hơn → nhiều biến thể hơn, dễ lạc đề hơn
```

Trực giác mạch điện: `temperature` giống như hệ số khuếch đại đảo trước một mạch
so sánh — khuếch đại thấp (chia cho số nhỏ) làm mọi ngưỡng rõ ràng, dứt khoát;
khuếch đại cao làm ranh giới mờ đi.

Kiểm tra ngay trong bản C thật, [`generate_cuda.cu:44`](../../firmware/jetson/generate_cuda.cu#L44):

```c
if (temperature <= 0.f) {          // temperature=0 -- xử lý ĐẶC BIỆT, không chia cho 0
    int best = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;   // ARGMAX thuần
    return best;
}
```

### Bước 3 — Top-k: giới hạn số ứng viên

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))     # model.py:261
logits[logits < v[:, [-1]]] = -float("inf")                 # loại mọi token NGOÀI top-k
```

Chỉ giữ lại `k` token có logit cao nhất, loại hẳn phần còn lại (gán `-inf`, softmax
sẽ cho xác suất ≈ 0). Lý do tồn tại: đuôi phân bố xác suất của một vocab 32.768 token
rất dài — luôn có hàng trăm token gần như vô nghĩa với xác suất nhỏ nhưng khác 0.
Không giới hạn, thỉnh thoảng model "trúng xổ số" chọn một token đó, output lạc đề
hoàn toàn. `top_k=40` (mặc định repo này) là một hàng rào an toàn rẻ tiền.

### Toàn bộ pipeline, 6 dòng — [`model.py:255-264`](../../src/model.py#L255-L264)

```python
logits, _ = self(idx_c)                                       # forward -- chương 5,6
logits = logits[:, -1, :] / temperature                         # bước 2
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))          # bước 3
logits[logits < v[:, [-1]]] = -float("inf")
probs = F.softmax(logits, dim=-1)                                # bước 1
idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)       # bốc thăm CÓ TRỌNG SỐ
```

`torch.multinomial` là bước "tung xúc xắc có trọng số" — không chọn xác suất cao
nhất một cách tất định (đó là *greedy*, `temperature=0`), mà bốc ngẫu nhiên theo
đúng phân bố `probs`. Bản C tự viết RNG tuyến tính tính nghịch đảo CDF
([`generate_cuda.cu:70-77`](../../firmware/jetson/generate_cuda.cu#L70)) — cùng ý
tưởng toán, khác công cụ (không có `torch.multinomial` trong C thuần).

### Thực hành — quan sát 3 chế độ sampling trên model thật

```bash
cd src && uv run python sample.py --run ../runs/ple-jetson-s0.pt --temperature 0.0 --samples 1   # greedy
uv run python sample.py --run ../runs/ple-jetson-s0.pt --temperature 0.8 --samples 3              # mặc định
uv run python sample.py --run ../runs/ple-jetson-s0.pt --temperature 1.5 --samples 1               # "liều lĩnh"
```

**Greedy** cho **luôn cùng một output** mỗi lần chạy (tất định — không có bước bốc
thăm). `temperature=0.8` cho các biến thể hợp lý nhưng khác nhau mỗi lần. `1.5`
thường bắt đầu lạc đề hoặc sai ngữ pháp sau vài chục token — đúng dự đoán ở trên.

## Bài tập

1. Tính bằng tay: với `logits = [2.0, 1.0, 0.1]`, tính `softmax` ở `temperature=1`
   và `temperature=0.1`. Xác suất của token đầu tiên thay đổi thế nào?
2. Trong `llm.h`, KV cache cấp phát tĩnh cho `seq_len` tối đa. Nếu bạn muốn hỗ trợ
   context dài hơn `seq_len` hiện tại (512), cần sửa những gì? (Gợi ý: không chỉ là
   tăng một con số — RoPE cũng phụ thuộc `seq_len` qua bảng `cos/sin`.)
3. Implement `top_p` (nucleus sampling — chọn tập token nhỏ nhất có tổng xác suất
   ≥ p, thay vì số lượng cố định `k`) bằng Python, dựa trên `model.py:generate()`
   làm khung. So sánh với `top_k` trên cùng một prompt.
4. Đo thời gian sinh 200 token với KV cache (mặc định) so với **tắt** KV cache (tính
   lại toàn bộ K,V mỗi bước — sửa tạm `llm_forward` hoặc viết bản Python đối chiếu).
   Vẽ đồ thị thời gian theo `pos` cho cả hai — cái nào tuyến tính, cái nào bậc hai?

→ Tiếp: [08-quantization-nhung.md](08-quantization-nhung.md) — model chạy được trên
GPU 6GB VRAM, nhưng ESP32-S3 chỉ có 512KB SRAM. Quantization là cách duy nhất để đi
từ con số đầu tới con số sau.
