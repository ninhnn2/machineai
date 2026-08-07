# Phương pháp: tối ưu, đo, và chứng minh

Nguồn: [`RESULTS.md`](../../../RESULTS.md), [`docs/03-roofline.md`](../../../docs/03-roofline.md),
[`docs/08-nhat-ky-toi-uu.md`](../../../docs/08-nhat-ky-toi-uu.md),
[`docs/09-so-do-phan-cung.md`](../../../docs/09-so-do-phan-cung.md).

## Quy trình tối ưu (áp cho mọi kiến trúc)

1. **Đúng trước.** Tầng 1 phải `argmax MATCH` rồi mới nói tới tốc độ.
2. **Profile theo stage**, đừng đoán. ESP32: `102.9 ms = head 57.6 | attn 25.6 |
   ple 8.5 | ffn 6.9 | input 4.4`. Jetson: `make bench` in ms/token + MB đọc +
   GB/s đạt + µs/kernel cho từng stage.
3. **Tính sàn của stage lớn nhất** trước khi viết một dòng SIMD/kernel:
   `bytes_đọc / bandwidth_đo_được = sàn`. Ví dụ head: `2.43 MB / 60.7 MB/s = 40 ms`
   sàn trong 57.6 ms → SIMD dù nhanh vô hạn cũng chỉ cắt 17.6/102.9 ≈ 17%.
4. **Nếu lợi ích < ~20%, đừng làm.** Đổi hướng: giảm **bytes đọc** (int4 head, head
   nhỏ hơn / factorised) thay vì vectorise mạnh hơn. Biết khi nào NGỪNG là kỹ năng,
   không phải mẹo.
5. **Đo lại sau mỗi bước**, ghi vào nhật ký kèm số.

## Nhật ký ESP32 — 17× trong 5 bước (bảng tham chiếu)

| # | Thay đổi | ms/token | tok/s | tăng | Bài học |
|---:|---|---:|---:|---:|---|
| 0 | port C đúng đầu tiên | 1757.2 | 0.57 | — | đúng trước, nhanh sau |
| 1 | head sang PSRAM + dọn scalar | 193.9 | 4.7 | **9.1×** | đặt dữ liệu đúng tầng bộ nhớ |
| 2 | dọn dot/RoPE/attention | 172.9 | — | 1.12× | bỏ tính toán lặp |
| 3 | head chạy 2 core | 139.4 | 6.0 | 1.24× | song song hoá phần chiếm ưu thế |
| 4 | head int8 + activation int8 | **102.9** | **9.5** | 1.35× | giảm **bytes đọc**, không phải phép tính |

**Bước 1 chiếm 9.1× trong tổng 17×** — chỉ là đặt ma trận vào đúng chỗ. Tương đương
trên GPU: đẩy hết layer lên device, đúng backend, bỏ `cudaMemcpy` H2D thừa. Người ta
hay bỏ qua vì "quá đơn giản" rồi mất cả tuần tối ưu kernel để lấy 20%.

## Không có tối ưu phổ quát

| | ESP32-S3 | CPU x86 | GPU Orin |
|---|---|---|---|
| Nút thắt | bandwidth PSRAM | compute → cache bandwidth | **launch overhead 50%** |
| SIMD tay | +15% (bị bandwidth chặn) | **+49%** | không áp dụng |
| SIMD tay trên ARM | — | **0%** (gcc đã tự vectorise) | — |
| Song song | gần như luôn đáng | **hỏng nếu > 8 luồng** (CPU lai P/E core) | phải batch ≥128 |

`samples/cpu/matvec_ladder.c` chạy đúng 5 bậc đó trên x86 và ARM và đo từng bậc —
kết luận không giống nhau. Đừng port một tối ưu sang kiến trúc khác mà không đo lại.

## Chứng minh một tối ưu / một ý tưởng có thật

**Nhánh đối chứng.** Với mỗi thay đổi, hỏi: *"nhánh đối chứng nào sẽ chứng minh tôi
sai?"* Không nghĩ ra được = chưa hiểu thay đổi đó. Repo có sẵn 5 nhánh core-matched
(xem `references/architecture.md`) — thêm nhánh mới thì giữ nguyên nguyên tắc:
mọi nhánh **cùng ngân sách core**, chỉ khác ở cách tiêu phần tham số thưa/flash.

**Nhiễu.** Mọi kết luận chạy **≥2 seed**, báo cả biên độ: "+0.098 nats (2 seeds,
±0.006) ~16× nhiễu seed". Chênh 0.008 với nhiễu 0.006 là vô nghĩa — đừng báo cáo.
Benchmark phần cứng: chạy ≥3 lần, báo **median + spread**.

**Tách biến.** `golden.txt` là logits của model đã dequantize → tầng 1 đo lỗi port,
tầng 2 đo lỗi quantize. Đừng gộp hai loại lỗi vào một con số.

## Bẫy đo đã gặp thật (đừng lặp lại)

- **Tải nền.** Cùng board Orin: lần đầu 27.8 GB/s, lần sau 66.8 GB/s. Không phải
  nhiễu — là tiến trình khác đang chạy. Kiểm máy rảnh trước khi đo.
- **Power mode.** MAXN_SUPER là **mode 2** trên Orin Nano Super, không phải mode 0.
- **Datasheet nói dối.** Orin Nano Super: 102 GB/s trên giấy, 66.8 GB/s đo được (65%)
  vì EMC khoá ở 2133 MHz. Mọi trần tok/s tính từ datasheet sai 1.53×.
- **Ngưỡng float kiểu CPU.** `1e-5` không dùng được cho GPU/TensorRT — reduce dạng cây
  đổi thứ tự cộng. Validate bằng argmax / top-k / perplexity.
- **Dùng hết logical core.** Trên CPU lai, dùng mọi core chậm hơn 700× so với dùng 8.
  `make -C samples/cpu scaling` tự hiệu chuẩn.

## Kết quả phản trực giác cần nhớ

- **Bảng lớn chịu quantize tốt hơn core dày đặc.** baseline degrade +0.079/+0.088 nats,
  ple chỉ +0.055/+0.061 → PLE giữ 124–128% ưu thế sau 4-bit. Bảng có dư thừa; model nhỏ
  dày đặc thì mọi weight đều thiết yếu. Tổng quát: **8B-Q4 > 3B-Q8** ở cùng dung lượng
  — nhưng tự đo trên workload của bạn.
- **Vocab lớn làm PLE có lợi hơn**: +0.025 nats @vocab 4096 → +0.098 @vocab 32768.
  Bảng vừa to vừa rẻ chính là chế độ PLE được thiết kế cho.
- **Widening rows bão hoà, thêm rows thì không** (trong dải đã thử).
- **Song song có khi vừa nhanh hơn vừa chính xác hơn** (pairwise vs naive summation).

## Giới hạn đã biết — nói thẳng, đừng giả vờ đã làm

Chưa có prefill theo batch (mọi target decode 1 token/lần); chưa dùng tensor core (CUDA
toàn fp32 CUDA core); chưa chạy LLM thật (Llama/Qwen/Gemma) trên Jetson để đối chiếu
trần roofline; training vẫn bằng PyTorch. **Đã có** CUDA Graphs, đo 1.59× trên Orin
(1116 → 1774 tok/s), xem [`docs/11-toi-uu-nvidia.md`](../../../docs/11-toi-uu-nvidia.md) §11.5.
