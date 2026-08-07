# 6. Tối ưu GPU — thiết kế kernel CUDA

Nội dung đầy đủ nằm cạnh code, ở **[`../firmware/jetson/JETSON.md`](../firmware/jetson/JETSON.md)**
(để đọc song song với `llm_cuda.cuh`).

Tóm tắt cái bạn sẽ học ở đó:

| Mục | Nội dung |
|---|---|
| §4 | **Sai số song song** — vì sao CUDA không trùng bit với C, và vì sao nó lại *gần PyTorch hơn* (tree reduction sai số `O(log n)` vs `O(n)`) |
| §5.1 | `k_matvec_q4` — một **warp** cho một hàng output, `__shfl_down_sync` không cần shared memory |
| §5.2 | `k_attention` — một **block** cho một head, softmax 2-pass. FlashAttention là bản 1-pass của đúng kernel này |
| §5.3 | `__restrict__` với buffer alias — bẫy **không crash, chỉ ra số sai** |
| §6 | Launch overhead, tự hiệu chuẩn bằng kernel rỗng |
| §10 | Số đo thật: 1141 tok/s, 50% thời gian là launch overhead |

Hai bẫy CUDA đáng nhớ nhất, cả hai đều **không crash**:

1. **`__restrict__` khi buffer alias.** `k_rmsnorm` gọi in-place (`s->h` vừa vào vừa ra).
   Khai `__restrict__` là hứa không alias → compiler được sắp xếp lại đọc/ghi → sai âm
   thầm. nvcc *có* cảnh báo — đọc warning.
2. **`__shfl_down_sync(0xffffffff, …)` sau `return` sớm.** Mask khẳng định cả 32 lane
   còn sống. Chỉ đúng vì `blockDim.x` là bội của 32 nên cả warp cùng thoát. Launch với
   100 thread thì mask thành lời nói dối và reduction đọc rác từ lane đã chết.

→ Quay lại [README.md](README.md) · Tiếp: [07-kv-cache-engine.md](07-kv-cache-engine.md)
