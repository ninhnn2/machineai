# 4. Gradient — hướng để cập nhật weight

Chương 2 nói weight "được học". Chương này trả lời chính xác **học bằng cách nào**:
đo weight hiện tại sai bao nhiêu (loss), tính xem sửa weight theo hướng nào thì bớt
sai (gradient), rồi sửa một chút theo hướng đó (gradient descent). Lặp lại hàng
nghìn lần — đó là toàn bộ `train.py`.

## 4.1 Loss Function

**Loss** là một con số duy nhất đo "model đang sai bao nhiêu", tính từ đầu ra model
và đáp án đúng. Hai hàm loss bạn cần biết tên:

| Loss | Công thức | Dùng cho | Ở repo này |
|---|---|---|---|
| **MSE** (Mean Squared Error) | `mean((y_pred - y_true)²)` | hồi quy — dự đoán số thực | không dùng ở đây, dùng trong ví dụ toy bên dưới |
| **Cross Entropy** | `-log(P(đúng))` | phân loại — dự đoán 1 trong N lớp | `F.cross_entropy` — [`model.py:223`](../../src/model.py#L223), N = 32768 token |

Model trong repo này sinh chữ = tại mỗi vị trí, **phân loại** token tiếp theo trong
số 32.768 khả năng (vocab). Cross entropy chỉ nhìn vào xác suất model gán cho **đúng
một** token đúng:

```
loss = -log(P(token đúng))
```

Trực giác: model gán xác suất càng cao cho đáp án đúng, `-log(...)` càng nhỏ (tiệm
cận 0 khi P→1); model càng chắc chắn sai (P→0), loss tiến tới **vô cực**. Đây là lý
do cross entropy "phạt nặng" sự tự tin sai, mạnh hơn nhiều so với MSE.

**Perplexity**, con số bạn thấy khắp `RESULTS.md`, chỉ là `loss` đổi đơn vị:
`ppl = e^loss`. Repo này báo cáo cả hai — xem
[`docs/00-nhap-mon.md §0.4`](../00-nhap-mon.md) để đổi qua lại nhanh.

## 4.2 Derivative — đạo hàm

**Đạo hàm `dy/dx` = độ dốc của hàm `y=f(x)` tại một điểm = "nếu tăng x một chút, y
tăng/giảm bao nhiêu."**

```
f(x) = x²        f'(x) = 2x         tại x=3: độ dốc = 6 -- tăng x một chút thì y tăng NHANH
f(x) = x²        f'(x) = 2x         tại x=0: độ dốc = 0 -- đáy hàm, tăng x không đổi y (tức thời)
```

Ý nghĩa vật lý bạn đã quen: đạo hàm theo thời gian của vị trí là **vận tốc** — "vị
trí đang đổi nhanh cỡ nào." Đạo hàm của loss theo weight cũng vậy, chỉ đổi trục:
**"loss đang đổi nhanh cỡ nào khi ta chỉnh weight này."**

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # 6.0 -- đúng 2*3, PyTorch tự tính đạo hàm, không cần bạn viết công thức
```

Đây là toàn bộ ý nghĩa của "autograd" mà chương 5 sẽ mổ xẻ: PyTorch tính đạo hàm
**tự động**, cho một hàm phức tạp gồm hàng triệu phép toán lồng nhau — bạn không
phải tự đạo hàm tay từng bước.

## 4.3 Gradient

**Gradient là đạo hàm, mở rộng cho hàm nhiều biến** — ở đây là hàm loss theo **mọi**
weight cùng lúc. Với model 559K tham số core, gradient là một vector 559K chiều —
mỗi phần tử trả lời "nếu tôi tăng đúng weight thứ i một chút, loss đổi bao nhiêu."

```
gradient = ∂Loss / ∂W          (mỗi weight một đạo hàm riêng phần — "partial derivative")
```

Ký hiệu `∂` (thay vì `d`) chỉ có nghĩa "đạo hàm theo MỘT biến, giữ mọi biến khác cố
định" — vì loss phụ thuộc hàng triệu weight cùng lúc, không chỉ một.

**Điều quan trọng nhất cần nhớ: gradient trỏ theo hướng làm loss TĂNG nhanh nhất.**
Muốn *giảm* loss, đi **ngược chiều** gradient. Đó là toàn bộ nội dung của §4.4.

## 4.4 Gradient Descent

```
w_mới = w_cũ - learning_rate × gradient
```

Đúng một dòng. Toàn bộ "học máy" của một mạng nơ-ron, về bản chất toán, là lặp lại
đúng phép trừ này hàng triệu lần cho hàng triệu weight.

### Thực hành — tự viết gradient descent, không dùng torch

Học `y = 3x + 2` từ dữ liệu nhiễu, đúng cách bạn từng viết thuật toán LMS
(Least Mean Squares) để cập nhật hệ số bộ lọc thích nghi — **đây thực chất là cùng
một thuật toán**, chỉ khác tên gọi:

```python
import random
random.seed(0)
data = [(x, 3*x + 2 + random.gauss(0, 0.05)) for x in [random.uniform(-2, 2) for _ in range(200)]]

w, b, lr = 0.0, 0.0, 0.05        # khởi tạo -- KHÔNG phải 0,0 cho weight thật (chương 2 §2.4),
                                  # nhưng ở đây bài toán lồi nên an toàn
for epoch in range(60):
    dw = db = 0.0
    for x, y in data:
        pred = w * x + b
        err = pred - y
        dw += 2 * err * x        # ∂loss/∂w  của loss=(pred-y)²
        db += 2 * err            # ∂loss/∂b
    dw /= len(data); db /= len(data)
    w -= lr * dw; b -= lr * db    # ĐÚNG công thức §4.4
```

Kết quả đo được thật, chạy trên máy dùng viết tài liệu này:

```
epoch  0: w=0.3998 b=0.2244 loss=12.53081
epoch  1: w=0.7465 b=0.4232 loss=9.54081
epoch  5: w=1.7302 b=1.0161 loss=3.22238
epoch 20: w=2.8532 b=1.8254 loss=0.06315
epoch 59: w=2.9943 b=1.9986 loss=0.00267
```

`w` hội tụ về 3, `b` về 2 — đúng dữ liệu sinh ra. Đây **chính là** vòng lặp
`for step in range(args.steps)` trong [`train.py:112-121`](../../src/train.py#L112),
chỉ khác model phức tạp hơn (28.9M weight thay vì 2) và optimizer thông minh hơn phép
trừ thẳng (§4.6).

## 4.5 Learning Rate

`learning_rate` (LR) quyết định **bước đi bao xa** mỗi lần cập nhật.

```
LR quá lớn  → w nhảy qua nhảy lại quanh đáy, không bao giờ hội tụ (thậm chí PHÂN KỲ)
LR quá nhỏ  → hội tụ đúng hướng nhưng cực chậm, tốn hàng triệu bước không cần thiết
```

Đổi `lr = 0.05` thành `lr = 2.0` trong ví dụ §4.4 và chạy lại — `loss` sẽ **tăng**
qua từng epoch thay vì giảm. Tự làm thử, đây là trực giác quan trọng nhất cần có
trước khi đọc tiếp.

Repo này **không dùng LR cố định** — nó dùng lịch trình (schedule), viết ở
[`train.py:49-53`](../../src/train.py#L49-L53):

```python
def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup          # WARMUP: tăng dần từ 0 lên peak
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))  # COSINE decay
```

```
LR
 │      ╱‾‾‾╲___
 │    ╱          ‾‾‾╲___
 │  ╱                    ‾‾‾╲___
 │╱                              ‾‾‾___
 └──────────────────────────────────────→ step
   warmup        cosine decay xuống 10% peak
```

**Vì sao warmup:** ở bước đầu, weight còn ngẫu nhiên (chương 2 §2.4), gradient có
thể rất lớn và nhiễu — LR lớn ngay từ đầu dễ đẩy weight đi quá xa, làm hỏng quá trình
train từ trong trứng. Tăng dần cho model "làm quen" trước.

**Vì sao cosine decay:** giai đoạn cuối train, muốn bước nhỏ dần để "chốt" vào đáy
hàm loss chính xác hơn, thay vì tiếp tục nhảy những bước lớn có thể vọt qua đáy.

## 4.6 Optimizer

Trừ thẳng `w -= lr * gradient` (§4.4) là optimizer đơn giản nhất, gọi là **SGD**
(Stochastic Gradient Descent). Nó có vấn đề: gradient mỗi bước nhiễu (được ước lượng
từ một *batch* dữ liệu nhỏ, không phải toàn bộ tập train), nên đường đi zig-zag.

| Optimizer | Ý tưởng thêm vào so với SGD |
|---|---|
| **SGD** | trừ thẳng gradient, không nhớ gì về các bước trước |
| **SGD + Momentum** | cộng thêm "quán tính" — tích luỹ hướng đi trung bình của các bước gần đây, giảm zig-zag |
| **Adam** | giữ trung bình động của **cả gradient lẫn bình phương gradient** — tự động chỉnh LR **riêng cho từng weight** |
| **AdamW** | Adam, nhưng tách *weight decay* ra khỏi phần cập nhật gradient (sửa một lỗi toán trong Adam gốc) |

Repo này dùng **AdamW** — [`train.py:99-103`](../../src/train.py#L99-L103):

```python
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
```

Ba chi tiết đáng chú ý, mỗi cái là một quyết định thiết kế có lý do:

- **`betas=(0.9, 0.95)`**: hệ số trung bình động cho (momentum, bình phương
  gradient). Không phải số ma thuật — đây là cấu hình chuẩn cho train LLM (theo sau
  GPT-3, Llama).
- **`weight_decay=0.1` cho `decay`, `0.0` cho `no_decay`** — weight decay là một
  dạng regularization: kéo weight về 0 một chút mỗi bước, chống *overfitting*. Nhưng
  **không áp cho norm và bảng embedding** ([`train.py:96-98`](../../src/train.py#L96-L98)):
  kéo trọng số của `RMSNorm` (chỉ vài trăm phần tử, ảnh hưởng toàn cục) hay bảng
  lookup (mỗi hàng độc lập, không có khái niệm "quá khớp" theo nghĩa thông thường)
  về 0 không có ý nghĩa và có thể gây hại.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`** — ngay trước
  `opt.step()`. Đây là hàng rào chống *exploding gradient*, xem §4.8.

## 4.7 Vanishing Gradient

Khi gradient phải "truyền" qua nhiều lớp (chương 5 giải thích cơ chế truyền, chain
rule), mỗi lớp nhân gradient với một hệ số. Nếu hệ số đó **liên tục nhỏ hơn 1** qua
hàng chục lớp, gradient co lại theo cấp số nhân — tới lớp đầu tiên, gradient gần như
**bằng 0**. Lớp đó ngừng học, dù loss vẫn còn lớn.

Đây là lý do lịch sử vì sao mạng rất sâu (trước ResNet, 2015) khó train. Ba cách
chữa, cả ba đều **có mặt trong repo này**:

| Cách chữa | Cơ chế | Ở repo này |
|---|---|---|
| **Residual connection** | cộng thẳng đầu vào vào đầu ra mỗi lớp — tạo "đường tắt" cho gradient | `x = x + self.attn(...)` — [`model.py:138`](../../src/model.py#L138), mọi lớp |
| **Normalization** (RMSNorm) | giữ phương sai activation ổn định qua các lớp → giữ gradient ổn định | mỗi lớp đều có `RMSNorm` trước attention/FFN |
| **Init cẩn thận** | (chương 2 §2.4) tránh activation bão hoà ngay từ bước 0 | `std=0.02`, init nhỏ cho lớp ghi residual |

## 4.8 Exploding Gradient

Ngược lại: hệ số nhân **lớn hơn 1** qua nhiều lớp → gradient **phình to** theo cấp số
nhân → bước cập nhật (§4.4) quá lớn → weight nhảy ra khỏi vùng hợp lý → loss thành
`NaN` (Not a Number), train hỏng hoàn toàn, thường không cứu được.

Cách chữa trực tiếp nhất, và là cách repo này dùng — **gradient clipping**:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Ý tưởng: tính `norm` (độ lớn, xem chương 1 §1.1) của **toàn bộ vector gradient**
(gộp mọi weight). Nếu norm đó vượt quá `1.0`, **co toàn bộ vector gradient lại** theo
đúng tỉ lệ để norm bằng đúng 1.0 — giữ nguyên *hướng*, chỉ giảm *độ lớn*. Đây là "cầu
chì" (fuse) của quá trình train: không ngăn được nguyên nhân gây exploding, nhưng
ngăn được hậu quả tàn phá weight.

### Thực hành — tự tạo và quan sát exploding gradient

```python
import torch
torch.manual_seed(0)
x = torch.randn(1, 8)
w = torch.randn(8, 8, requires_grad=True) * 3.0     # khởi tạo CỐ Ý quá lớn (so với std=0.02 chuẩn)

h = x
for layer in range(10):                              # 10 lớp liên tiếp, không residual, không norm
    h = torch.relu(h @ w)
loss = h.sum()
loss.backward()
print("gradient norm của w:", w.grad.norm().item())   # số RẤT lớn -- exploding thật sự
```

So với cùng đoạn code nhưng `w = torch.randn(8, 8, requires_grad=True) * 0.02` (đúng
init chương 2) — gradient norm nhỏ hơn nhiều bậc độ lớn. Đây là bằng chứng bằng số
cho lý do §2.4 khăng khăng "không khởi tạo bằng giá trị lớn tuỳ tiện."

## Bài tập

1. Đổi `lr = 0.05` thành `lr = 2.0` trong ví dụ §4.4. Chạy lại, quan sát `loss` tăng
   thay vì giảm. Tìm giá trị `lr` lớn nhất mà vẫn hội tụ (dò nhị phân bằng tay).
2. Chạy `cd src && uv run python train.py --arm baseline --steps 500 --lr 0.1` (LR
   cao gấp 100 lần mặc định `1e-3`). Loss có thành `NaN` không? Nếu không, giải
   thích vai trò của `clip_grad_norm_` trong việc "cứu" quá trình train.
3. Tắt `clip_grad_norm_` (comment dòng đó trong `train.py`), chạy lại với LR cao ở
   bài 2. So sánh.
4. Đọc lại đoạn code trong Thực hành §4.8. Thêm `RMSNorm` đơn giản
   (`h = h / h.norm()`) sau mỗi lớp và chạy lại — gradient norm thay đổi thế nào?
   Đây là chứng minh bằng số cho lý do RMSNorm giúp chống exploding/vanishing.

→ Tiếp: [05-backpropagation.md](05-backpropagation.md) — `loss.backward()` ở dòng
`train.py:119` thực sự làm gì, từng bước một.
