---
name: model-lab
description: Chạy thí nghiệm có kiểm soát trên model PLE TinyLM để HIỂU model, không phải để đạt PPL thấp hơn. Dùng khi được hỏi vì sao model tốt hay xấu, nhanh hay chậm, khi cần so sánh cấu hình (vocab, d_model, layers, lr, batch, quantization), khi muốn biết bottleneck nằm ở đâu, hoặc khi nhắc tới "experiment", "sweep", "ablation", "so sánh cấu hình".
---

# Model Understanding Lab

Mục tiêu không phải đẩy PPL xuống, mà dùng model hiện có như một phòng thí nghiệm
để hiểu từng tầng: tokenizer, embedding, weight, GEMM, attention, gradient,
quantization, runtime, và cuối cùng là silicon.

## Kỷ luật, năm phần, không bỏ phần nào

```
CÂU HỎI  ->  GIẢ THUYẾT  ->  ĐỔI MỘT BIẾN  ->  ĐO  ->  GIẢI THÍCH
```

**Giả thuyết phải viết TRƯỚC khi chạy.** Viết sau khi thấy kết quả thì không còn là
kiểm chứng, chỉ là kể lại. `experiments/lab.py` từ chối chạy nếu thiếu
`--question` và `--hypothesis`, và ghi cả hai vào từng dòng kết quả.

Đoán sai là kết quả tốt. Sweep learning rate đầu tiên có giả thuyết "3e-3 sẽ dao
động hoặc phân kỳ"; thực tế 3e-3 thắng rõ rệt ở ngân sách 2000 bước. Ghi lại chỗ
đoán sai đó có giá trị hơn một bảng số mà ai cũng gật gù.

## Baseline, mọi thứ so với đây

Cấu hình deploy 28.9M, đo trên RTX 2060 ngày 13/8/2026:

```
vocab 32768   d_model 96   layers 6   ffn 66   ple_dim 128
core 558.368  +  stream 3.145.728  +  table 25.165.824  =  28.869.920

11.000 bước x batch 12 x seq 512 = 67,6M token, hết 1213 giây
val 2.1102   ppl 8.25   model.bin 14,91 MB   suy giảm INT4 +0,1078

GPU:   eager 2.037 tok/s   CUDA graph 3.569 tok/s   117 kernel/token
       49% thời gian mỗi token là launch overhead thuần
ESP32: 9,5 tok/s, nghẽn ở băng thông đọc weight
```

Đối chiếu `RESULTS.md`: ngân sách tham số khớp từng chữ số, suy giảm INT4 trùng
khít `+0,1078`. Val loss lệch 0,068 nats nhưng **không so trực tiếp được**, vì
`prepare.py` bây giờ tráo tài liệu trước khi cắt val còn bản trong RESULTS.md thì
không, nên hai tập validation khác nhau.

## Chạy một sweep

```bash
python3 experiments/lab.py --name lr --var lr --values 1e-4,3e-4,1e-3,3e-3 \
  --question "Learning rate ảnh hưởng thế nào tới tốc độ hội tụ?" \
  --hypothesis "1e-4 chậm, 3e-3 dao động, 1e-3 tốt nhất" \
  --fix steps=2000

python3 experiments/lab.py --report        # in lại mọi thí nghiệm đã chạy
```

Kết quả nối thêm vào `experiments/results.jsonl`, mỗi dòng có đủ config, câu hỏi,
giả thuyết, đường cong val, và độ lệch so với baseline.

Hạ `steps` cho cả sweep là hợp lệ: trong sweep vẫn chỉ một biến thay đổi. Harness
ghi lại độ lệch đó và cảnh báo không được so val của sweep với dòng baseline.

## Chi phí, để biết cái nào chạy được ngay

| Loại | Cần train lại | Thời gian trên RTX 2060 |
|---|---|---|
| Quantization: bit, group size | không | vài giây, dùng `src/trace_quant.py` |
| Numerical accuracy PyTorch/C/CUDA | không | dưới 1 phút, `make verify` hai nơi |
| Bottleneck GPU theo stage | không | dưới 1 phút, `make bench` |
| Attention, embedding, một token | không | vài giây, `src/trace_token.py` |
| Learning rate, batch size | có, ngắn | 4 x 230 giây |
| d_model, layers, seq_len | có | 4 x 4 tới 20 phút |
| Vocab | có, kèm prepare lại | 4 x (8 phút prepare + train) |
| Bản deploy đầy đủ | có | 20 phút |

Làm hết nhóm "không cần train" trước. Chúng cho kết quả trong vài phút và trả lời
được kha khá câu hỏi về quantization và runtime.

## Công cụ đo đã có

| Câu hỏi | Công cụ |
|---|---|
| Một token đi qua model thế nào | `src/trace_token.py`, tự kiểm attention với SDPA |
| `loss.backward()` làm gì | `src/trace_learn.py`, chain rule tay so autograd |
| INT4 lấy đi cái gì | `src/trace_quant.py`, giải nén từ chính `model.bin` |
| Model quên bao nhiêu sau fine-tune | `src/eval_loss.py` trên tập val CŨ |
| Dữ liệu của tôi đã vào model chưa | `data/make_probe.py` + `src/probe_check.py` |
| Nghẽn ở compute, bandwidth hay launch | `make -C firmware/jetson bench` |
| Port C và CUDA có đúng không | `make -C firmware/host_verify verify` |

Mỗi công cụ tự kiểm chứng và **thoát mã 1 nếu phép kiểm sai**, nên không thể vô
tình báo cáo số của một model không tồn tại.

## Thứ tự nên đi

```
Phase 1  dữ liệu     tokenizer, vocab, token id, embedding
Phase 2  toán        matvec, GEMM, attention, MLP, logits
Phase 3  huấn luyện  learning rate, batch, gradient, backprop, clipping
Phase 4  kiến trúc   d_model, layers, FFN, bỏ RoPE/RMSNorm/residual
Phase 5  nén         FP32, INT8, INT4, group size
Phase 6  runtime     PyTorch, C, CUDA, sai số, kernel, CUDA graph
Phase 7  nhúng       memory map, băng thông weight, GPU so với ESP32
```

## Đã xong

- Baseline 28.9M, đo đầy đủ.
- Bottleneck GPU theo stage: 3 trong 5 stage là launch-bound.
- Sai số PyTorch so C so CUDA: argmax khớp trên cả hai runtime.
- Sweep learning rate ở 2000 bước.
- Ba demo tương tác đã đăng: mổ bụng một token, nhìn AI học, FP32 tới INT4.

## Cạm bẫy đã cắn thật

**So val loss giữa hai tokenizer là vô nghĩa.** Đổi vocab là đổi đơn vị đo. Chỉ so
trong cùng một bảng.

**Giao thức đo phải giống nhau.** `train.py` báo val ở `seq_len` đang train, còn
`quantize.py` đo ở 256. Cùng một model cho 2.1102 và 2.1568. Luôn hỏi "đo bằng
cách nào" trước khi so hai con số.

**Tập val cắt theo vị trí.** Nối dữ liệu vào cuối corpus mà nó nhỏ hơn tập val thì
toàn bộ rơi vào validation, không có gì vào training, và không có thông báo lỗi.
`prepare.py` đã tráo tài liệu trước khi cắt, nhưng số cũ sinh trước bản sửa thì
không so được với số mới.

**`export.py` ghi đè `firmware/model/` tại chỗ.** Sao lưu nếu cần giữ mốc.

**Model nhỏ trên GPU luôn launch-bound.** Đừng tối ưu kernel trước khi nhìn cột
chẩn đoán trong `make bench`.
