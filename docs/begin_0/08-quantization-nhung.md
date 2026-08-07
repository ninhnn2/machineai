# 16. Quantization — cầu nối trực tiếp nhất tới thế giới bạn đã biết

Đây là chương mà kiến thức embedded của bạn **chuyển giao gần như nguyên vẹn**.
Quantization trong AI không phải kỹ thuật mới — nó là **Q-format fixed-point** bạn
đã dùng cho CMSIS-DSP/FFT, áp lên weight của một mạng nơ-ron. Chương này nối hai thế
giới bằng số thật, rồi trỏ sang [`../04-quantization.md`](../04-quantization.md) và
[`../10-ly-thuyet-nen.md §10.2`](../10-ly-thuyet-nen.md) để đọc sâu.

## Từ Q15 tới int4 group-wise — cùng một ý tưởng, khác độ tinh vi

Bạn đã làm việc này: `Q15` lưu số thực trong khoảng `[-1, 1)` bằng `int16`, với
**một scale cố định toàn cục** (`2^15`). Chuyển đổi:

```
Q15:  int_value = round(float_value × 2^15)
      float_value ≈ int_value / 2^15
```

Quantization AI làm **đúng việc này**, chỉ khác 2 chỗ:

| | Q15 (bạn đã biết) | Repo này (`quantize.py`) |
|---|---|---|
| Số bit | 16 | 4 |
| Scale | **cố định toàn cục** (luôn `2^15`) | **tính riêng cho mỗi nhóm** 64/128 giá trị, lưu kèm dưới dạng `fp16` |
| Vùng giá trị | luôn `[-1, 1)` | mỗi nhóm có scale riêng theo giá trị lớn nhất của chính nhóm đó |

Scale "tính riêng cho mỗi nhóm nhỏ" (**group-wise**) là điểm khác biệt cốt lõi — và
đây là toàn bộ nội dung §16 của roadmap ("FP32 → FP16 → INT8 → INT4"). Hàm thật của
repo, [`quantize.py:30-52`](../../src/quantize.py#L30):

```python
def quantize_groupwise(w, bits=4, group=64, fp16_scales=False):
    qmax = 2 ** (bits - 1) - 1                      # = 7 cho 4-bit (dải đối xứng: -7..7)
    scale = x.abs().amax(dim=2, keepdim=True) / qmax  # SCALE RIÊNG cho mỗi nhóm `group` giá trị
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)   # y hệt round(value * 2^15) của Q15
    dq = (q * scale)                                 # dequantize -- đây là giá trị THỰC SỰ được dùng khi suy luận
```

### Thực hành — quantize tay 8 số, đối chiếu công thức

```python
import torch
from quantize import quantize_groupwise

w = torch.tensor([[0.03, -0.51, 0.22, -0.09, 0.44, -0.61, 0.05, -0.02]])   # 1 nhóm 8 giá trị
dq = quantize_groupwise(w, bits=4, group=4)                                # group=4 -> 2 nhóm
print("gốc     :", w.numpy())
print("sau q/dq:", dq.numpy())
```

Kết quả đo được thật:

```
gốc      : [ 0.03 -0.51  0.22 -0.09  0.44 -0.61  0.05 -0.02]
sau q/dq : [ 0.   -0.51  0.219 -0.073  0.436 -0.61  0.087 -0. ]
max sai số: 0.0371
```

**Tính tay để hiểu vì sao:** nhóm đầu `[0.03, -0.51, 0.22, -0.09]`, giá trị lớn nhất
tuyệt đối là `0.51`. `scale = 0.51 / 7 = 0.07286`. Với `0.03`:
`round(0.03 / 0.07286) = round(0.41) = 0` → dequant về **đúng 0**, sai số `0.03` —
**đây chính là câu trả lời cho câu hỏi "vì sao INT4 làm hỏng chất lượng"**: giá trị
nhỏ so với giá trị lớn nhất trong cùng nhóm gần như bị **xoá về 0**. Với `-0.51`
(chính là giá trị lớn nhất): `round(-0.51/0.07286) = round(-7.0) = -7` → dequant về
**đúng -0.51**, sai số = 0. Giá trị lớn nhất trong nhóm luôn được lưu chính xác gần
như tuyệt đối; giá trị nhỏ chịu thiệt.

**Hệ quả trực tiếp cho thiết kế model:** nhóm càng "đồng đều" (không có giá trị
ngoại lai quá lớn), sai số trung bình càng nhỏ — vì cả nhóm dùng chung 1 scale. Đây
là động lực thật đằng sau các kỹ thuật nâng cao (AWQ, GPTQ, SmoothQuant — liệt kê ở
[`../04-quantization.md §2.2`](../04-quantization.md)): chúng đều tìm cách "làm
phẳng" phân bố trước khi lượng tử hoá, để một scale chung không phải hy sinh quá
nhiều giá trị nhỏ.

## Bốn định dạng, và tương đương embedded của mỗi cái

| Định dạng | Bit | Q-format gần nhất bạn biết | Dùng khi |
|---|---:|---|---|
| **FP32** | 32 | không có tương đương — dải động + độ chính xác đều tối đa | train (chương 4, cần gradient chính xác) |
| **FP16/BF16** | 16 | không có tương đương chuẩn MCU | train/suy luận GPU |
| **INT8** | 8 | gần **Q7** (`[-1,1)`, 7 bit phân số) | cân bằng tốc độ/chất lượng, phổ biến nhất |
| **INT4** | 4 (+scale) | không có Q-format chuẩn ở mức này | thiết bị siêu nhỏ — chính là repo này |

Chi phí lưu trữ thật **không phải** đúng "4 bit" — mỗi nhóm còn tốn 1 scale `fp16`
(16 bit) dùng chung:

```
bit thực/tham số = bits + 16/group = 4 + 16/128 = 4.125 bit    (group=128, cấu hình deploy)
28.9M tham số × 4.125 / 8 = 14.9 MB     -- khớp CHÍNH XÁC với model.bin thật: 14,912,332 byte
```

Đây là bài học có thể áp thẳng vào mọi thiết kế fixed-point embedded: **overhead của
metadata (ở đây là scale) không được bỏ qua khi tính ngân sách bộ nhớ.**

## Đo "hỏng bao nhiêu" — không đoán, đo bằng loss

Câu hỏi thực dụng nhất: quantize xong, model tệ đi bao nhiêu? Trả lời bằng cách so
`loss` (chương 4 §4.1) trước/sau, **không** so sai số từng weight riêng lẻ — vì cái
cuối cùng quan trọng là model còn viết đúng hay không, không phải từng con số có
khớp tuyệt đối.

```bash
cd src && uv run python quantize.py --bits 4 --group 64 --tag jetson --seed 0
```

Kết quả thật, đo được trong repo này (xem thêm `RESULTS.md`):

```
baseline  fp32 val 2.2769 | 4-bit val 2.3291 | degradation +0.0522 nats
ple       fp32 val 2.2364 | 4-bit val 2.2779 | degradation +0.0416 nats   <- ÍT hơn baseline!
```

**Kết quả phản trực giác, đáng nhớ nhất chương này:** bảng PLE 25M tham số (to hơn
core dày đặc **44 lần**) lại **chịu 4-bit tốt hơn**. Vì sao: một bảng lớn có dư
thừa (redundancy) — mỗi hàng độc lập, sai một hàng không ảnh hưởng hàng khác. Một
core dày đặc thì **mọi** weight đều tham gia vào **mọi** tính toán — sai một weight
lan ra khắp model. Bài học tổng quát, dùng lại được ở bất kỳ SoC AI nào bạn gặp:
**model càng lớn (đủ dư thừa) càng chịu lượng tử hoá tốt hơn**, không phải ngược lại
như trực giác "nhiều tham số dễ vỡ hơn" hay gợi ý.

## Bài tập

1. Tính tay quantize/dequantize cho nhóm thứ hai trong ví dụ Thực hành
   (`[0.44, -0.61, 0.05, -0.02]`, group=4). Đối chiếu với output thật ở trên.
2. Chạy `uv run python quantize.py --bits 8` và `--bits 3`. Vẽ (hoặc lập bảng) độ
   suy giảm (`deg`) theo số bit. Từ bit nào trở xuống chất lượng "gãy" rõ rệt?
3. Đọc [`../04-quantization.md §2.4`](../04-quantization.md) — phần "Validate". So
   tiêu chí nghiệm thu ở đó (`delta ppl < 0.1`) với tiêu chí nghiệm thu bạn đã dùng
   cho fixed-point DSP (thường là SNR hoặc THD tính bằng dB). Viết một đoạn ngắn quy
   đổi qua lại giữa hai cách đo.
4. Thử `--group 32` và `--group 256`. Model to ra hay nhỏ đi? Chất lượng tốt lên hay
   xấu đi? Giải thích bằng chính công thức `bits + 16/group` ở trên.

→ Tiếp: [09-runtime-tensorrt-onnx-tidl.md](09-runtime-tensorrt-onnx-tidl.md) — model
đã nhỏ gọn, giờ cần một runtime để thực sự chạy nó trên silicon production.
