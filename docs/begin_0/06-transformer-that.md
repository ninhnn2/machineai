# 6–13. Từ Embedding tới Decoder Block — ráp 5 viên gạch lại thành Transformer

Roadmap liệt kê 8 chủ đề (Embedding, RoPE, Self-Attention, Multi-Head Attention,
RMSNorm, SwiGLU, Residual Connection, Decoder Block). Tin tốt: repo này đã có một
tài liệu viết đúng các chủ đề này ở độ sâu code thật —
[`../02-hieu-model.md`](../02-hieu-model.md). Chương này **không lặp lại** nó, mà
làm hai việc tài liệu đó chưa làm: (1) chỉ đúng đường dẫn — mỗi chủ đề trong roadmap
nằm ở đâu trong đó, và (2) đưa ra bằng chứng cuối cùng rằng transformer **không có gì
bí ẩn** — chỉ là 5 viên gạch của chương 1–5, ráp lại theo một sơ đồ cố định. Bằng
chứng đó là: tự viết lại một khối attention bằng NumPy thuần, và **đo được** nó khớp
với PyTorch tới sai số làm tròn dấu phẩy động.

## 6.1 Bản đồ — chủ đề nào nằm ở đâu

| # | Chủ đề roadmap | Đọc ở | Viên gạch nào từ chương 1–5 |
|---|---|---|---|
| 6 | **Embedding** | [`02-hieu-model.md` §4.2 "Tied embeddings"](../02-hieu-model.md) | chương 1 (vector), chương 2 §2.6 |
| 7 | **Positional Encoding (RoPE)** | [`02-hieu-model.md` §4.2 "RoPE"](../02-hieu-model.md) | chương 1 (vector xoay = phép chiếu có góc) |
| 8 | **Self Attention** | [`02-hieu-model.md` §4.2 "Attention với KV cache"](../02-hieu-model.md) | chương 3 §3.6 (QKᵀ là matmul) |
| 9 | **Multi-Head Attention** | cùng mục trên — mỗi "head" là 1 bản attention song song trên 1 lát cắt của vector | chương 1 (chia vector D chiều thành H lát) |
| 10 | **RMSNorm** | [`02-hieu-model.md` §4.2 "RMSNorm"](../02-hieu-model.md) | chương 1 §1.1 (magnitude) |
| 11 | **SwiGLU** | [`02-hieu-model.md` §4.2 "SwiGLU"](../02-hieu-model.md) | chương 3 (Linear = GEMM), chương 5 (đạo hàm SiLU) |
| 12 | **Residual Connection** | [`02-hieu-model.md` §4.1](../02-hieu-model.md) — dòng `x = x + ...` | chương 4 §4.7 (chống vanishing gradient) |
| 13 | **Decoder Block** | [`02-hieu-model.md` §4.1 "Giải phẫu"](../02-hieu-model.md) | tổng hợp toàn bộ 5 chương |

Đọc `02-hieu-model.md` **ngay bây giờ**, đầy đủ — nó đã trả lời "cái này là gì và vì
sao", với đúng số dòng code (`model.py:xx`) cho từng khối. Quay lại đây sau khi đọc
xong để làm phần thực hành dưới, phần khẳng định bạn **thực sự** hiểu, không chỉ đọc
hiểu.

## 6.2 Sơ đồ tổng — với đúng tên chương đã học

```
token id
   │
   │ ①  tra bảng embedding                              [chương 1+2 §2.6]
   ▼
vector D chiều
   │
   │ ②  RMSNorm  (chia cho magnitude)                    [chương 1 §1.1]
   ▼
   │ ③  Linear (qkv) → Q,K,V   -- GEMM                   [chương 3 §3.4]
   │ ④  xoay Q,K theo vị trí (RoPE)                       [chương 1 §1.4 phép chiếu]
   │ ⑤  Q·Kᵀ, softmax, ·V     -- attention, matmul        [chương 3 §3.6]
   │ ⑥  Linear (proj)                                     [chương 3 §3.4]
   ▼
   +── residual: x = x + kết quả bước ⑥                   [chương 4 §4.7]
   │
   │ ⑦  RMSNorm → Linear → SiLU × Linear → Linear (SwiGLU) [chương 3+5]
   ▼
   +── residual: x = x + kết quả bước ⑦
   │
   │  (lặp lại ②–⑦  × 6 lớp = "Decoder Block" × N)         [đây là "13. Decoder Block"]
   ▼
RMSNorm cuối → Linear (head) → logits                       [chương 3, chương 4 §4.1 loss]
```

Không có bước nào trong sơ đồ trên là kiến thức mới so với chương 1–5. Đây chính là
điều roadmap hứa ở cuối phần 5: *"Sau khi xong 5 chủ đề, có thể đọc và hiểu: Embedding,
Linear Layer, Attention, Residual Connection, RMSNorm, SwiGLU, MLP, Logits."*

## 6.3 Thực hành — tự viết lại Attention bằng NumPy thuần, đo khớp với PyTorch

Đây là bài kiểm tra thật: nếu bạn viết được đoạn dưới **mà không copy**, bạn đã thực
sự hiểu attention, không chỉ nhớ tên các bước.

```python
import numpy as np, torch, math
import torch.nn.functional as F
from model import Config, RMSNorm, Attention, build_rope

torch.manual_seed(0)
D, H, T = 8, 2, 3                       # d_model nhỏ để nhìn rõ, y hệt cấu trúc thật
cfg = Config(d_model=D, n_heads=H, seq_len=T, n_layers=1)

attn = Attention(cfg)                    # lớp PyTorch THẬT, lấy nguyên từ model.py
norm = RMSNorm(D)
x_t = torch.randn(1, T, D)
cos, sin = build_rope(T, D // H, cfg.rope_theta, "cpu")
with torch.no_grad():
    ref = attn(norm(x_t), cos, sin)      # ĐÁP ÁN — chạy bằng code thật của repo

# ---- Giờ tính LẠI từ số 0, chỉ dùng numpy, không gọi bất kỳ nn.Module nào ----
def rmsnorm_np(x, w, eps=1e-6):
    return w * x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)   # chương 1 §1.1

x = x_t.numpy()[0]
xn = rmsnorm_np(x, norm.weight.detach().numpy())

Wqkv = attn.qkv.weight.detach().numpy()          # [3D, D]
qkv = xn @ Wqkv.T                                 # chương 3 §3.4 -- Linear = x @ W^T
q, k, v = np.split(qkv, 3, axis=-1)
Dh = D // H
q = q.reshape(T, H, Dh).transpose(1, 0, 2)        # [H, T, Dh] -- "multi-head" = chia D thành H lát
k = k.reshape(T, H, Dh).transpose(1, 0, 2)
v = v.reshape(T, H, Dh).transpose(1, 0, 2)

def rope_np(x, cos, sin):                          # RoPE split-half -- xem model.py apply_rope
    x1, x2 = np.split(x, 2, axis=-1)
    c, s = cos.numpy()[None, :T, :], sin.numpy()[None, :T, :]
    return np.concatenate([x1*c - x2*s, x2*c + x1*s], axis=-1)

q, k = rope_np(q, cos, sin), rope_np(k, cos, sin)

scale = 1.0 / math.sqrt(Dh)
out = np.zeros((H, T, Dh))
for h in range(H):
    scores = (q[h] @ k[h].T) * scale                          # chương 3 §3.6 -- Q·Kᵀ
    scores += np.triu(np.ones((T, T)), k=1) * -1e30            # causal mask: không nhìn token tương lai
    scores -= scores.max(-1, keepdims=True)                    # softmax ổn định số học
    w_ = np.exp(scores); w_ /= w_.sum(-1, keepdims=True)
    out[h] = w_ @ v[h]                                          # weights · V

out = out.transpose(1, 0, 2).reshape(T, D)
out = out @ attn.proj.weight.detach().numpy().T

print("max abs diff numpy vs PyTorch:", np.abs(out - ref.numpy()[0]).max())
```

Kết quả đo được thật:

```
max abs diff numpy vs PyTorch: 7.72e-08
```

**7.72e-08 không phải "gần đúng" — đó là sai số làm tròn dấu phẩy động thuần tuý**
(đúng chủ đề [`../10-ly-thuyet-nen.md §10.1`](../10-ly-thuyet-nen.md), "vì sao cộng
số thực không kết hợp"). Hai đoạn code, viết độc lập, cho **cùng một câu trả lời**.
Đó là bằng chứng dứt khoát: attention không có phép màu ẩn giấu trong `nn.Module` —
mọi thứ nằm phơi ra trong 25 dòng numpy ở trên.

**Đối chiếu ngược lại bản C thật** — [`llm.h:322-343`](../../firmware/common/llm.h#L322):
vòng lặp 2-pass softmax (tìm max rồi mới `exp`) trong C **chính là** dòng
`scores -= scores.max(...)` ở trên. Ba ngôn ngữ (PyTorch, NumPy tay, C nhúng), một
thuật toán.

## 6.4 Bài tập bổ sung — SwiGLU (chủ đề 11)

Lặp lại đúng bài tập trên cho `SwiGLU` thay vì `Attention` — dễ hơn nhiều (không có
RoPE, không có mask):

```python
from model import SwiGLU
cfg2 = Config(d_model=8, ffn_hidden=16)
ffn = SwiGLU(cfg2)
x_t = torch.randn(1, 3, 8)
with torch.no_grad(): ref = ffn(x_t)

x = x_t.numpy()[0]
def silu(z): return z / (1 + np.exp(-z))            # chương 5 §5.4 -- đạo hàm của hàm này
Wg = ffn.gate.weight.detach().numpy()
Wu = ffn.up.weight.detach().numpy()
Wd = ffn.down.weight.detach().numpy()
g = silu(x @ Wg.T)
u = x @ Wu.T
out = (g * u) @ Wd.T
print(np.abs(out - ref.numpy()[0]).max())            # đo được: 2.98e-08
```

Chú ý dòng `g * u` — phép **nhân elementwise** giữa hai nhánh, không phải cộng. Đây
là "gating" (cổng): nhánh `g` (đã qua SiLU, nằm khoảng gần [0,1] cho phần lớn giá
trị) quyết định "cho bao nhiêu phần trăm của nhánh `u` đi qua." Đúng cơ chế nhân
`g * ple` bạn sẽ gặp lại ở bảng PLE ([`../02-hieu-model.md §4.4`](../02-hieu-model.md)).

## 6.5 Câu hỏi tổng hợp — Decoder Block (chủ đề 13)

Sau khi làm xong §6.3–6.4, tự trả lời — không tra code:

1. `Decoder Block` là gì, bằng đúng 1 câu, dùng từ vựng chương 1–5?
   *(Gợi ý đáp án: một hàm nhận vào 1 vector D chiều, trả về 1 vector D chiều khác,
   ghép từ 2 phép biến đổi có residual — một phép trộn thông tin GIỮA các token
   (attention), một phép biến đổi TRONG từng token (SwiGLU).)*
2. Vì sao xếp chồng N decoder block **không** làm gradient biến mất, dù N có thể lên
   tới hàng chục ở model lớn? (đáp án nằm ở chương 4 §4.7 — residual connection)
3. `d_model=96`, `n_heads=4` → `head_dim=24` (chương 1 §1.5, tensor 3D). Vì sao chia
   nhỏ thành 4 head thay ví giữ nguyên 1 head 96 chiều? (Gợi ý: mỗi head học một
   "kiểu quan hệ" khác nhau giữa token — ví dụ head này chú ý cú pháp, head kia chú ý
   ngữ nghĩa; đây là quan sát thực nghiệm, không phải định lý toán học chặt chẽ.)

## Bài tập

1. Đọc toàn bộ [`../02-hieu-model.md`](../02-hieu-model.md) nếu chưa đọc, đặc biệt
   phần "Ba chi tiết nhỏ, đều load-bearing" ở §4.4 (PLE) — đây là ví dụ thật về cách
   một chi tiết scale/init tưởng nhỏ quyết định cả model có học được hay không, nối
   thẳng về chương 2 §2.4 và chương 4 §4.7-4.8 bạn vừa học.
2. Tự viết lại `RMSNorm` bằng numpy (dễ nhất trong 3 khối), đo khớp với PyTorch.
3. Ghép 3 đoạn numpy (RMSNorm, Attention, SwiGLU) thành **một decoder block hoàn
   chỉnh** có 2 residual connection, chạy nó, so với gọi trực tiếp lớp `Block` trong
   `model.py`. Đây là bài tập khó nhất và đáng làm nhất của cả file này — làm xong,
   bạn đã tự tay dựng lại kiến trúc mà cả ngành AI hiện đại dùng.
4. Tại sao bản C ([`llm.h`](../../firmware/common/llm.h)) không cần lưu computational
   graph (chương 5 §5.1) như PyTorch? Trả lời bằng khái niệm forward/backward đã học.

→ Tiếp: [07-kv-cache-sampling.md](07-kv-cache-sampling.md) — model đã sinh ra
logits, giờ làm sao biến logits thành chữ, và làm sao không phải tính lại từ đầu mỗi
token.
