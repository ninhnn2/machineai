# 1. Vector — dữ liệu trong AI

Mọi thứ AI xử lý — một từ, một token, một pixel, một đoạn âm thanh — trước khi vào
mạng nơ-ron đều bị ép về **một dãy số thực có độ dài cố định**. Dãy số đó là vector.
Chương này xây lại trực giác vector từ đúng chỗ bạn đã có (mảng `float[]`, xử lý tín
hiệu số), rồi cho bạn xem một vector thật, đã học được nghĩa, lấy ra từ chính model
của repo này.

## 1.1 Vector là gì

**Trong toán:** một điểm trong không gian D chiều, hoặc một mũi tên từ gốc toạ độ
tới điểm đó. Viết `v = [v0, v1, ..., v_{D-1}]`.

**Trong lập trình — đây là điều bạn đã biết mà không gọi tên nó:**

```c
float v[128];   // đây CHÍNH LÀ một vector 128 chiều
```

Không khác gì buffer mẫu bạn dùng cho FIR filter hay FFT. Điểm khác duy nhất: trong
DSP, chỉ số của mảng thường mang nghĩa *thời gian* (`x[n]` = mẫu tại thời điểm n).
Trong AI, chỉ số mang nghĩa **một chiều đặc trưng học được** — không ai biết trước
chiều thứ 37 nghĩa là gì, mạng tự học ra cách dùng nó.

**Hai đại lượng mô tả một vector:**

```
magnitude (độ lớn, norm)  = √(v0² + v1² + ... + v_{D-1}²)         -- "vector này mạnh cỡ nào"
direction (hướng)         = v / magnitude(v)                        -- "vector này trỏ về đâu", đã chuẩn hoá độ dài = 1
```

`magnitude` chính là **RMS** bạn đã tính trong xử lý tín hiệu (RMS của tín hiệu =
`magnitude / √D`). Không phải trùng hợp: [`RMSNorm`](../02-hieu-model.md#rmsnorm-modelpy64)
— khối chuẩn hoá xuất hiện ở *mọi* lớp của transformer trong repo này — chính là
phép chia vector cho một đại lượng rất gần `magnitude`:

```python
# src/model.py:71
return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                              đây là 1/RMS(x), không phải 1/magnitude(x) —
#                              chia cho mean thay vì sum nên không phụ thuộc D
```

## 1.2 Vector biểu diễn gì trong AI

| Tên gọi | Vector của cái gì | Chiều dài điển hình | Ở repo này |
|---|---|---|---|
| **Token embedding** | một mảnh từ (token) | 96–4096 | `tok_emb.weight[id]` — [`model.py:156`](../../src/model.py#L156) |
| **Word embedding** | một từ trọn vẹn (tiền-BPE, hiếm dùng nay) | 300 (Word2Vec cổ điển) | không dùng trong repo |
| **Hidden state** | trạng thái của model tại 1 token, 1 lớp | = `d_model` | `s->x[D]` trong [`llm.h`](../../firmware/common/llm.h) |
| **Feature vector** | đặc trưng tổng quát của một đối tượng | tuỳ bài toán | — |
| **Image feature** | đặc trưng của 1 ảnh/vùng ảnh (sau CNN/ViT) | 512–2048 | không có trong repo (repo này là text) |
| **Audio feature** | đặc trưng của 1 khung âm thanh (MFCC, mel-spectrogram) | 13–128 | không có trong repo |

**Điểm chung của tất cả:** ban đầu là số ngẫu nhiên (`nn.init.normal_`,
[`model.py:195`](../../src/model.py#L195)), và **học được nghĩa qua gradient descent**
(chương 4). Không ai lập trình "chiều số 37 nghĩa là giống mèo" — nó tự nổi lên vì
xuất hiện lặp lại trong dữ liệu train.

### Thực hành 1 — nhìn một embedding thật đã học được nghĩa

Model trong `runs/` đã train xong trên TinyStories. Lấy embedding của vài từ và đo
độ giống nhau (§1.3):

```bash
cd src && uv run python3 - <<'EOF'
import torch, torch.nn.functional as F
from tokenizers import Tokenizer

tok = Tokenizer.from_file("../data/bpe4096.json")
ck = torch.load("../runs/ple-jetson-s0.pt", map_location="cpu", weights_only=False)
emb = ck["state"]["tok_emb.weight"]          # [4096, 128] — 4096 vector 128 chiều
print("embedding shape:", tuple(emb.shape))

def tid(w): return tok.encode(" " + w).ids[0]
def cos(a, b): return F.cosine_similarity(emb[a:a+1], emb[b:b+1]).item()

pairs = [("cat","dog"), ("cat","puppy"), ("king","queen"), ("happy","sad"),
         ("cat","the"), ("cat","king")]
for a, b in pairs:
    print(f"{a:6s} vs {b:6s}: cos = {cos(tid(a), tid(b)):+.3f}")
EOF
```

Kết quả đo được thật (seed cố định, tái lập được):

```
cat    vs dog   : cos = +0.705
cat    vs puppy : cos = +0.477
king   vs queen : cos = +0.726
happy  vs sad   : cos = +0.616
cat    vs the   : cos = -0.132
cat    vs king  : cos = +0.222
```

Đọc kết quả: `cat`/`dog` gần nhau (cùng là động vật nuôi, xuất hiện trong ngữ cảnh
giống nhau). `cat`/`the` gần như trực giao, còn hơi âm — một từ nội dung (content
word) và một từ chức năng (function word) hiếm khi thay thế được cho nhau trong
câu. `happy`/`sad` **gần nhau dù nghĩa trái ngược** — bài học quan trọng: cosine
similarity của embedding đo **"xuất hiện trong ngữ cảnh giống nhau"**, không đo
"nghĩa giống nhau". Đừng nhầm hai thứ đó.

## 1.3 Khoảng cách giữa các vector

Ba phép đo, ba câu hỏi khác nhau — chọn sai phép đo là lỗi hay gặp nhất:

| Phép đo | Công thức | Đo cái gì | Dùng khi |
|---|---|---|---|
| **Dot product** | `Σ aᵢbᵢ` | vừa hướng vừa độ lớn | bên trong attention, bên trong mọi lớp Linear |
| **Cosine similarity** | `dot(a,b) / (‖a‖‖b‖)` | **chỉ hướng**, bỏ qua độ lớn | so sánh nghĩa của 2 embedding, tìm kiếm semantic |
| **Euclidean distance** | `√Σ(aᵢ-bᵢ)²` | khoảng cách thật trong không gian | clustering, k-NN, khi độ lớn có ý nghĩa vật lý |

```
dot(a,b)     = ‖a‖ ‖b‖ cos(θ)     θ = góc giữa hai vector
```

Đây là công thức bạn đã dùng — nó **chính là** tích chập tại một điểm trong FIR:
`y[n] = Σ h[k]·x[n-k]` là dot product giữa vector hệ số `h` và cửa sổ tín hiệu
`x[n-k..n]`. Attention (chương 6) không làm gì khác: nó là hàng triệu dot product
giữa vector "query" và vector "key", chỉ khác chỗ cả hai vector đều **học được**
thay vì thiết kế bằng tay như hệ số FIR.

**Vì sao dot product thô không dùng để so "giống nhau":** một vector dài (magnitude
lớn) sẽ cho dot product lớn với mọi thứ, kể cả thứ không liên quan — độ lớn "che"
mất thông tin hướng. Cosine similarity chia cho magnitude của cả hai để loại bỏ
nhiễu đó, chỉ còn lại góc.

### Thực hành 2 — so ba phép đo trên cùng một cặp vector

```python
import torch
a = torch.tensor([3.0, 4.0])          # magnitude 5
b = torch.tensor([6.0, 8.0])          # cùng hướng với a, magnitude 10
c = torch.tensor([4.0, -3.0])         # vuông góc với a

print("dot(a,b)   =", torch.dot(a, b).item())      # 3*6+4*8 = 50 -- LỚN, dễ hiểu lầm là "khác nhau nhiều"
print("cos(a,b)   =", torch.cosine_similarity(a, b, dim=0).item())  # 1.0 -- ĐÚNG: cùng hướng
print("dist(a,b)  =", torch.dist(a, b).item())      # 5.0
print("dot(a,c)   =", torch.dot(a, c).item())      # 0 -- vuông góc, không liên quan
print("cos(a,c)   =", torch.cosine_similarity(a, c, dim=0).item())  # 0.0
```

`a` và `b` cùng hướng (b = 2a) nên **cosine = 1.0 dù chúng khác hẳn về độ lớn** —
đúng cái ta muốn khi so nghĩa hai embedding, vì độ lớn của embedding thường chỉ phản
ánh tần suất xuất hiện của token, không phải "độ mạnh" của nghĩa.

## 1.4 Projection — chiếu vector

Chiếu vector `a` lên vector `b` là tách `a` thành hai phần: một phần **song song**
với `b`, một phần **vuông góc** với `b`.

```
thành phần song song:  a_∥ = (dot(a,b) / dot(b,b)) · b
thành phần vuông góc:  a_⊥ = a - a_∥
```

Đây không phải kiến thức trang trí — nó là **phép toán mà mỗi lớp `nn.Linear` thực
hiện, lặp lại hàng nghìn lần**. Một lớp Linear `y = Wx` tính, với mỗi hàng `wᵢ` của
`W`, giá trị `yᵢ = dot(wᵢ, x)` — đó chính xác là **độ dài của hình chiếu của `x` lên
hướng `wᵢ`** (sai khác một hệ số `‖wᵢ‖`). Nói cách khác: **một lớp Linear là một tập
câu hỏi "x giống hướng này bao nhiêu?"**, mỗi hàng của ma trận trọng số là một câu
hỏi. Đây là cách nghĩ sẽ dùng lại nguyên vẹn ở chương 3 (Matrix Multiplication) và
chương 6 (Attention — Q·K chính là "token này giống câu hỏi kia bao nhiêu?").

## 1.5 Tensor — vector tổng quát hoá

```
Scalar   :  5                         hạng 0 (không chiều)     — 1 con số
Vector   :  [5, 2, 9]                 hạng 1                   — 1 mảng
Matrix   :  [[5,2],[9,1]]             hạng 2                   — bảng 2 chiều
Tensor 3D:  [batch, seq, dim]         hạng 3                   — 1 batch câu, mỗi câu 1 chuỗi vector
Tensor 4D:  [batch, channel, H, W]    hạng 4                   — ảnh (CNN dùng dạng này)
```

Trong `firmware/common/llm.h`, mọi tensor được lưu **phẳng** (flat array) — đúng
cách bạn đã quen trong C, không có khái niệm "tensor" ở tầng runtime:

```c
// llm.h:57 — ple_table thật ra là tensor 2D [V, L*P], nhưng lưu là 1 mảng byte phẳng
QT ple_table;           // [V, L*P]

// truy cập hàng r (= token id) của tensor 2D [rows, cols]:
const uint8_t *row = t->codes + (size_t)r * t->row_bytes;   // llm.h:87
//                                          ^^^^^^^^^^^^ = cols * kích_thước_1_phần_tử
```

`shape` (`[rows, cols]`) chỉ tồn tại ở tầng Python/PyTorch để bạn suy luận; ở tầng C
nó biến mất, chỉ còn lại **stride** — bước nhảy con trỏ để đi từ hàng này sang hàng
kia. Đây chính là con đường bạn sẽ đi mỗi khi debug: PyTorch nói "shape sai", C nói
"đọc nhầm offset" — cùng một lỗi, hai cách nhìn.

## Bài tập

1. Chạy lại Thực hành 1 với 5 cặp từ khác trong TinyStories (`little`, `big`,
   `garden`, `forest`, `mom`, `dad`...). Cặp nào bạn dự đoán đúng độ giống nhau,
   cặp nào bất ngờ?
2. Tính bằng tay `magnitude`, rồi `direction`, của vector `[3, 4, 0]`. Kiểm lại bằng
   `torch.norm` và `F.normalize`.
3. `d_model` của cấu hình deploy là 96 ([`RESULTS.md`](../../RESULTS.md)). Một
   embedding 96 chiều — bạn **không thể** vẽ nó ra giấy. Đọc trước về PCA/t-SNE (sẽ
   không dùng trong repo này, nhưng là công cụ chuẩn để "nhìn" vector cao chiều) và
   giải thích bằng lời tại sao chiếu xuống 2D luôn làm mất thông tin.
4. Trong `llm.h:87` (`deq_row`), tìm dòng tính `row_bytes`. Giải thích bằng công
   thức vì sao `row_bytes = ceil(cols/2)` cho tensor int4 (2 giá trị/byte).

→ Tiếp: [02-weight.md](02-weight.md) — vector nào là **cố định** (weight, học được
một lần) và vector nào **thay đổi mỗi lần chạy** (activation, hidden state).
