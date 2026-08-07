# Lộ trình nền tảng: Kỹ sư Embedded → Embedded AI Engineer

Bạn đến từ thế giới firmware: ngắt (interrupt), thanh ghi, DMA, bộ lọc FIR/IIR,
fixed-point Q15, RTOS. Giáo trình này không dạy lại embedded — nó dạy phần **AI**
bằng đúng trực giác bạn đã có, rồi cắm thẳng vào code C và Python **có thật** trong
repo này để bạn thực hành trên máy thật, không phải trên slide.

Toàn bộ đường học dựa trên [`AI_Embedded_Foundation_Roadmap.txt`](AI_Embedded_Foundation_Roadmap.txt)
— roadmap gốc bạn đã có. Tài liệu này viết ra chi tiết từng phần của roadmap đó,
theo đúng thứ tự, và trả lời câu hỏi mà roadmap đặt ra: *đọc xong 5 chủ đề đầu,
bạn phải đọc hiểu được `model.py`, TensorRT, ONNX Runtime, TIDL, CUDA.*

## Vì sao lấy repo này làm bãi thực hành

Đây là một trong rất ít chỗ bạn có thể chạy **cùng một model, cùng một `model.bin`**
qua ba runtime khác hẳn nhau — MCU (ESP32-S3, thuần C, không hệ điều hành), CPU
(x86/ARM), GPU (CUDA) — và đo thật trên từng cái. Với dân embedded, đó chính là điều
kiện lý tưởng: bạn không phải tưởng tượng "weight nằm ở đâu", bạn **đọc file
`model.bin`** và thấy nó nằm ở byte offset nào; không phải tưởng tượng "ma trận nhân
mất bao lâu", bạn build và đo trên chính CPU của mình.

```
src/model.py (PyTorch, dạy AI)  ──export──>  model.bin  ──đọc bởi──>  llm.h (C thuần, dạy embedded)
```

## Cấu trúc

| # | File | Chủ đề trong roadmap | Grounded bằng code nào trong repo |
|---|---|---|---|
| 1 | [01-vector.md](01-vector.md) | 1. Vector — dữ liệu trong AI | `tok_emb.weight` đã train thật, đo cosine similarity |
| 2 | [02-weight.md](02-weight.md) | 2. Weight — kiến thức của mô hình | `model.py` init, `train.bin`→SRAM/PSRAM/flash, `budget.py` |
| 3 | [03-matrix-nhan.md](03-matrix-nhan.md) | 3. Matrix Multiplication | `llm.h` matvec, `samples/cpu/matvec_ladder.c` đo thật trên máy bạn |
| 4 | [04-gradient.md](04-gradient.md) | 4. Gradient | `train.py` optimizer, AdamW, warmup+cosine LR |
| 5 | [05-backpropagation.md](05-backpropagation.md) | 5. Backpropagation | `loss.backward()`, đối chiếu tay vs autograd |
| 6 | [06-transformer-that.md](06-transformer-that.md) | 6–13. Embedding→Decoder Block | cầu nối sang [`../02-hieu-model.md`](../02-hieu-model.md), tự viết 1 decoder block bằng NumPy |
| 7 | [07-kv-cache-sampling.md](07-kv-cache-sampling.md) | 14–15. KV Cache, Token Sampling | `llm.h` KV cache thật, `model.py generate()` |
| 8 | [08-quantization-nhung.md](08-quantization-nhung.md) | 16. Quantization | `quantize.py`, đối chiếu Q15 fixed-point bạn đã biết |
| 9 | [09-runtime-tensorrt-onnx-tidl.md](09-runtime-tensorrt-onnx-tidl.md) | 17. TensorRT/ONNX Runtime/TIDL | cầu nối sang [`../14-tensorrt-deepstream.md`](../14-tensorrt-deepstream.md) + nội dung mới về TIDL |
| 10 | [10-vla-robot.md](10-vla-robot.md) | 18. VLA cho Robot | cầu nối sang [`../15-kernel-den-camera.md`](../15-kernel-den-camera.md) |

Chủ đề 1–5 là nội dung **hoàn toàn mới**, không có ở đâu khác trong `docs/` — đây là
nền toán/lập trình mà mọi file khác trong `docs/00`–`docs/15` giả định bạn đã biết.
Chủ đề 6–10 phần lớn là **cầu nối**: nhắc lại đúng độ sâu cần cho một kỹ sư embedded,
rồi trỏ sang tài liệu đã có sẵn trong repo (đừng học lại hai lần cùng một thứ).

## Cách học — 70/30, đúng như roadmap khuyến nghị

**70% thời gian hiểu toán + thuật toán, 30% đọc code chạy thật.** Với mỗi chương:

1. Đọc phần lý thuyết — luôn có ví dụ số cụ thể, không chỉ công thức trừu tượng.
2. Chạy phần "Thực hành" — lệnh có sẵn, copy–paste được, chạy trên máy bạn.
3. Làm bài tập cuối chương trước khi sang chương sau. Chương sau **giả định** bạn
   đã làm được bài tập chương trước.

Môi trường: `cd src && uv run python ...` cho mọi lệnh Python (xem
[`../../README.md`](../../README.md)); `cc -O3 ...` hoặc `make -C samples/cpu` cho C —
không cần GPU cho 8 trong 10 chương.

## Lộ trình theo tuần (gợi ý, không bắt buộc)

| Tuần | Việc |
|---|---|
| 1 | 01–02 (Vector, Weight) — đọc `model.py` xong tuần này phải hiểu được từng dòng khai báo tensor |
| 2 | 03 (Matrix Multiplication) — làm hết bài tập `matvec_ladder.c`, đây là phép toán ăn 90% thời gian của mọi LLM |
| 3 | 04–05 (Gradient, Backprop) — tự viết một mạng 2 lớp bằng NumPy, so gradient tay với autograd |
| 4 | 06 (Transformer) — đọc `../02-hieu-model.md` song song, tự implement 1 decoder block |
| 5 | 07–08 (KV Cache, Sampling, Quantization) — đọc `../04-quantization.md`, `../07-kv-cache-engine.md` |
| 6 | 09–10 (Runtime, VLA) — đọc `../13`, `../14`, `../15`; nếu có Jetson thì chạy thật |

## Sau khi xong

Bạn sẽ đọc được không chỉ `src/model.py` mà cả `firmware/common/llm.h` — và thấy
chúng **là cùng một thuật toán**, một bên viết bằng tensor, một bên viết bằng vòng
lặp `for`. Đó chính là năng lực lõi của một Embedded AI Engineer: đứng giữa hai thế
giới và dịch qua lại được.

→ [`../README.md`](../README.md) — mục lục đầy đủ của toàn bộ khoá học trên repo này.
