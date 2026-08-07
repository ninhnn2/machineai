# 5. Backpropagation — `loss.backward()` thực sự làm gì

Chương 4 nói: "tính gradient của loss theo mọi weight, rồi cập nhật." Nhưng model có
28.9 triệu weight, xếp qua 6 lớp lồng nhau — tính gradient của **từng** weight bằng
tay là bất khả thi. Backpropagation là thuật toán giải quyết đúng việc đó: tính
gradient của **mọi** weight trong **một lượt duy nhất**, bằng cách áp dụng đúng một
quy tắc toán học lặp đi lặp lại — quy tắc chuỗi (chain rule).

## 5.1 Computational Graph

Mọi phép tính trong forward pass (từ input tới loss) tạo thành một **đồ thị có
hướng**: mỗi node là một phép toán, mỗi cạnh là một tensor chảy qua.

```
x ──┐
    ├─(×w1)──> h_pre ──(ReLU)──> h ──┐
b1 ─┘                                ├─(×w2)──> y ──┐
                                b2 ──┘              ├─((y-t)²)──> loss
                                              target ┘
```

Đây **chính là** sơ đồ khối bạn vẽ khi thiết kế một chuỗi xử lý tín hiệu (block
diagram DSP) — mỗi khối biến đổi tín hiệu vào thành tín hiệu ra. Khác biệt duy nhất:
mạng nơ-ron cần đi **ngược** qua sơ đồ này để biết mỗi khối cần chỉnh thế nào.
PyTorch **tự xây** đồ thị này khi bạn chạy forward — không cần bạn khai báo tay
(gọi là *dynamic graph* / *define-by-run*, khác với TensorFlow 1.x cũ phải khai đồ
thị tường minh trước).

## 5.2 Chain Rule

Đây là **một quy tắc giải tích duy nhất** làm nền cho toàn bộ backprop. Nếu
`y = f(g(x))`, thì:

```
dy/dx = dy/dg × dg/dx
```

Đạo hàm của hàm hợp = tích các đạo hàm từng chặng. Với đồ thị nhiều bước
(`x → h_pre → h → y → loss`), áp dụng liên tiếp:

```
dLoss/dx = dLoss/dy × dy/dh × dh/dh_pre × dh_pre/dx
```

**Đây là toàn bộ backpropagation.** Không có gì bí ẩn hơn thế: đi từ loss ngược về
đầu vào, ở mỗi bước nhân thêm một đạo hàm cục bộ (đạo hàm của *đúng một phép toán*,
dễ tính), rồi truyền tích luỹ đó tiếp về phía sau.

## 5.3 Forward Pass

**Forward pass** = chạy đồ thị §5.1 theo chiều xuôi, từ input ra loss — đúng như
`llm_forward()` trong [`llm.h:268`](../../firmware/common/llm.h#L268) chạy: embedding
→ 6 lớp attention+FFN+PLE → norm → logits. Trong lúc chạy xuôi, PyTorch **ghi nhớ**
đủ thông tin ở mỗi node (thường là giá trị đầu vào) để tính đạo hàm cục bộ sau này.

## 5.4 Backward Pass

**Backward pass** = đi ngược đồ thị, từ loss về input, ở mỗi node nhân gradient tích
luỹ với đạo hàm cục bộ của node đó (§5.2), rồi truyền tiếp về phía sau.

### Thực hành — tính tay, đối chiếu với autograd, khớp tới 6 chữ số thập phân

Mạng 2 lớp tối giản: `y = w2 · ReLU(w1·x + b1) + b2`, loss = `(y - target)²`.

```python
import torch

x = torch.tensor(2.0)
w1 = torch.tensor(0.5, requires_grad=True); b1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(-0.3, requires_grad=True); b2 = torch.tensor(0.2, requires_grad=True)
target = torch.tensor(1.0)

h_pre = w1 * x + b1
h = torch.relu(h_pre)
y = w2 * h + b2
loss = (y - target) ** 2

loss.backward()
print(w1.grad, b1.grad, w2.grad, b2.grad)
```

**Giờ tính TAY, đúng chain rule §5.2, đi ngược từ loss:**

```python
y_v, h_v, hpre_v = y.item(), h.item(), h_pre.item()

dL_dy    = 2 * (y_v - target.item())            # d(loss)/dy   -- đạo hàm của (y-t)²
dL_dw2   = dL_dy * h_v                           # d(loss)/dw2  = dL/dy * dy/dw2   (dy/dw2 = h)
dL_db2   = dL_dy * 1.0                           # dy/db2 = 1
dL_dh    = dL_dy * w2.item()                     # dy/dh  = w2
dL_dhpre = dL_dh * (1.0 if hpre_v > 0 else 0.0)  # d(ReLU)/d(h_pre) = 1 nếu >0, ngược lại 0
dL_dw1   = dL_dhpre * x.item()                   # d(h_pre)/dw1 = x
dL_db1   = dL_dhpre * 1.0                        # d(h_pre)/db1 = 1
```

Kết quả đo được thật — **hai cách tính khớp tuyệt đối**:

```
AUTOGRAD:  dL/dw1=1.356000  dL/db1=0.678000  dL/dw2=-2.486000  dL/db2=-2.260000
TAY      :  dL/dw1=1.356000  dL/db1=0.678000  dL/dw2=-2.486000  dL/db2=-2.260000
```

**`loss.backward()` không làm gì "thông minh" hơn đoạn tay ở trên** — nó làm đúng
việc đó, tự động, cho một đồ thị có thể có hàng tỷ node thay vì 4 node. Đó là toàn
bộ phép màu: không phải thuật toán mới, mà là **áp dụng chain rule một cách máy móc
và triệt để**.

**Chi tiết cần khắc cốt: đạo hàm của ReLU.** `d(ReLU)/dx = 1 nếu x>0, còn lại = 0`.
Đây là nguồn gốc của một dạng vanishing gradient khác (ngoài chương 4 §4.7): một
neuron ReLU luôn nhận đầu vào âm sẽ **luôn có gradient = 0**, không bao giờ học lại
được — gọi là "dead ReLU". Đây là một trong các lý do model hiện đại (kể cả model
trong repo này) chuyển sang SiLU/GELU (chương 6) — đạo hàm mượt hơn, không có vùng
"chết cứng" tuyệt đối.

## 5.5 Gradient qua từng Layer

Điều làm backprop *hiệu quả* (thay vì chỉ *đúng*): **gradient của một lớp chỉ cần
gradient đã tích luỹ từ lớp sau nó**, không cần biết gì về các lớp xa hơn về phía
sau. Đây gọi là tính chất *cục bộ* của chain rule.

```
Lớp N   (cuối): nhận dLoss/dy_N   trực tiếp từ hàm loss
Lớp N-1        : nhận dLoss/dy_{N-1} = (dLoss/dy_N) × (dy_N/dy_{N-1})  -- CHỈ cần grad lớp N + đạo hàm cục bộ của chính nó
Lớp N-2        : tương tự, chỉ cần grad lớp N-1
...
```

Hệ quả thực dụng: mỗi loại lớp (`nn.Linear`, `RMSNorm`, `SiLU`...) chỉ cần biết
đúng **một công thức** — đạo hàm cục bộ của chính nó — để tham gia vào backprop của
*bất kỳ* mạng nào chứa nó. Đây là lý do bạn có thể lắp ghép các lớp trong
[`model.py`](../../src/model.py) (RMSNorm, Attention, SwiGLU, PLE gate) như lắp
Lego mà không cần tự viết công thức đạo hàm cho toàn mạng — PyTorch tự ghép các đạo
hàm cục bộ lại bằng chain rule.

## 5.6 PyTorch Autograd

`requires_grad=True` (thấy trong mọi ví dụ ở trên) là cách bạn nói với PyTorch:
"ghi nhớ tensor này vào đồ thị §5.1, tôi sẽ cần đạo hàm của nó." Ba điều cần nhớ khi
làm việc với autograd, cả ba đều xuất hiện trong `train.py`:

```python
opt.zero_grad(set_to_none=True)   # train.py:118
```
Gradient **cộng dồn** theo mặc định (hữu ích khi cố tình gộp nhiều batch nhỏ thành
một batch lớn ảo — gradient accumulation). Nếu không xoá trước mỗi bước, gradient
bước này sẽ **cộng thêm** vào gradient bước trước — sai hoàn toàn. Đây là lỗi phổ
biến nhất khi mới viết vòng lặp train.

```python
loss.backward()                    # train.py:119
```
Chạy toàn bộ §5.4 cho cả đồ thị, gán kết quả vào thuộc tính `.grad` của mọi tensor
có `requires_grad=True`.

```python
with torch.no_grad():             # dùng trong evaluate() và generate()
    ...
```
Tắt hẳn việc xây đồ thị §5.1 — dùng khi chỉ cần forward (suy luận, đánh giá), không
cần backward. Tiết kiệm cả bộ nhớ (không cần lưu giá trị trung gian để backward) lẫn
thời gian. `evaluate()` ([`train.py:40`](../../src/train.py#L40)) và
`generate()` ([`model.py:254`](../../src/model.py#L254)) đều bọc trong
`@torch.no_grad()` vì đúng lý do này — suy luận trên chip ESP32/Jetson **không bao
giờ** cần backward, nên bản C ([`llm.h`](../../firmware/common/llm.h)) không có một
dòng nào cho gradient. Đây là lằn ranh giữa *train* và *deploy*: train cần cả forward
lẫn backward, deploy chỉ cần forward.

## 5.7 `optimizer.step()`

Điểm nối giữa chương 5 (đã tính được gradient) và chương 4 (dùng gradient để cập
nhật). `opt.step()` ([`train.py:121`](../../src/train.py#L121)) đọc `.grad` của mọi
weight (vừa được `backward()` điền vào) và áp công thức cập nhật của optimizer đã
chọn (AdamW — chương 4 §4.6) lên từng weight.

```
loss.backward()  →  điền  w.grad  cho MỌI weight       (chương 5, chương này)
opt.step()       →  đọc   w.grad, cập nhật w            (chương 4)
```

Hai lệnh này tách biệt có chủ đích: `backward()` không biết gì về learning rate hay
AdamW; `step()` không biết gì về chain rule. Tách lớp (separation of concerns) —
đúng nguyên tắc bạn đã áp dụng khi thiết kế firmware theo tầng driver/HAL/application.

### Toàn cảnh — một bước train hoàn chỉnh

```python
# train.py:112-121, annotated
for step in range(args.steps):
    x, y = train_b()              # lấy 1 batch dữ liệu
    _, loss = model(x, y)          # FORWARD  (chương 5.3) -- llm_forward tương đương ở bản C
    opt.zero_grad(set_to_none=True)  # xoá gradient cũ     (chương 5.6)
    loss.backward()                # BACKWARD (chương 5.4) -- KHÔNG tồn tại ở bản C, chỉ cần khi TRAIN
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # chống nổ (chương 4.8)
    opt.step()                     # CẬP NHẬT weight       (chương 4.4, dùng AdamW chương 4.6)
```

5 dòng này, lặp lại 11.000 lần (cấu hình deploy), là toàn bộ những gì biến 28.9
triệu con số ngẫu nhiên thành một model biết viết truyện mạch lạc.

## Bài tập

1. Mở rộng ví dụ §5.4 thành 3 lớp thay vì 2. Tính tay gradient của `w1`, đối chiếu
   với autograd.
2. Thay `ReLU` bằng `SiLU` (`x * sigmoid(x)`, dùng trong SwiGLU — chương 6) trong ví
   dụ §5.4. Đạo hàm cục bộ của SiLU phức tạp hơn ReLU — tra công thức, tính tay, rồi
   đối chiếu `torch.nn.functional.silu`.
3. Quên gọi `opt.zero_grad()` trong 3 bước liên tiếp của ví dụ toy ở chương 4. In
   `w.grad` sau mỗi bước — chứng minh bằng số rằng gradient cộng dồn sai như mô tả
   ở §5.6.
4. Trong `model.py`, hàm `generate()` được bọc `@torch.no_grad()`. Xoá decorator đó,
   chạy sinh văn bản 50 token, đo thời gian và bộ nhớ trước/sau. Giải thích chênh
   lệch bằng khái niệm "PyTorch phải lưu gì để backward được".

→ Tiếp: [06-transformer-that.md](06-transformer-that.md) — giờ bạn có đủ 5 viên
gạch (vector, weight, matmul, gradient, backprop) để đọc `model.py` từ đầu tới cuối.
