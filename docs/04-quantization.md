# 2. Quantization — lever lớn nhất khi memory-bound

Từ file 01: decode là memory-bound, `tok/s ∝ 1/bytes`. Quantization là cách trực tiếp nhất
để giảm bytes. FP16 → INT4 = nhanh gần **4×**. Không kernel trick nào cho bạn con số đó.

## 2.1 Cơ chế: group-wise symmetric

Bạn **đã có một implementation tham chiếu rất sạch** trong repo esp32:
[`../src/quantize.py:30`](../src/quantize.py#L30). Đọc nó — 20 dòng,
và đó chính xác là cơ chế đằng sau Q4_K của llama.cpp và INT4 của TensorRT-LLM.

```python
x = w.reshape(out, n_groups, group)        # chia mỗi hàng thành nhóm 64/128 phần tử
qmax  = 2**(bits-1) - 1                    # 7 với 4-bit
scale = x.abs().amax(dim=2) / qmax         # 1 scale fp16 cho mỗi nhóm
q     = clamp(round(x / scale), -qmax, qmax)
dq    = q * scale
```

**Ba tham số quyết định chất lượng/kích thước:**

| Tham số | Ảnh hưởng |
|---|---|
| `bits` | 8 / 5 / 4 / 3. Dưới 4 bit cần thuật toán thông minh hơn round-to-nearest |
| `group` | 32/64/128. Nhỏ hơn = chính xác hơn nhưng tốn overhead scale |
| scale dtype | fp16 (chuẩn). Repo esp32 làm đúng: [`export.py:55`](../src/export.py#L55) làm tròn scale về fp16 **trước** khi tính dq, để golden khớp bit-exact |

**Overhead thật của group=128, 4-bit:** 128 weights × 4 bit = 64 byte + 1 scale fp16 = 2 byte
→ 4.125 bit/weight. Với group=32 → 4.5 bit/weight. Đây là lý do "Q4" thực tế là ~4.5 bit và
model 8B Q4_K_M nặng 4.7GB chứ không phải 4.0GB.

**Symmetric vs asymmetric:** symmetric (chỉ scale) đủ cho weights vì phân bố quanh 0.
Activations sau ReLU/SiLU lệch → cần asymmetric (scale + zero-point). Đây là lý do
weight-only quantization dễ hơn nhiều so với W8A8.

## 2.2 Phân loại các phương pháp

### Weight-only (W4A16) — dùng cho decode
Chỉ nén weights, activations giữ FP16. Vì decode memory-bound bởi weights, đây là lựa chọn đúng.

| Phương pháp | Ý tưởng | Khi nào dùng |
|---|---|---|
| **RTN** (round-to-nearest) | Chính là code trên | Baseline. 8-bit thì đủ tốt |
| **GPTQ** | Quantize từng cột, dùng Hessian nghịch đảo để bù lỗi vào các cột chưa quantize | 4-bit, cần calib set |
| **AWQ** | Nhận ra ~1% kênh "salient" (theo độ lớn activation) gây phần lớn lỗi → scale chúng lên trước khi quantize | 4-bit, thường tốt hơn GPTQ, có kernel nhanh |
| **Q4_K_M** (llama.cpp) | k-quant: 2 tầng scale, mix bit theo tensor (attention.wv và ffn.down giữ 6-bit) | Mặc định thực dụng nhất |

### Weight+activation (W8A8) — dùng cho prefill / vision
Nén cả hai để **thực sự dùng tensor core INT8** (33 TOPS thay vì 16.7 TFLOPS FP16).

| Phương pháp | Ý tưởng |
|---|---|
| **SmoothQuant** | Activation có outlier cực đoan → "chuyển" độ khó từ activation sang weight bằng scale per-channel: `Y = (X/s)(sW)` |
| **TensorRT PTQ** | Calibration với entropy/minmax, chọn scale per-tensor |

> Repo esp32 cũng đã đi con đường này: RESULTS.md dòng 135 — "int8-staged head + int8
> activations" cho 1.35× tốc độ. Cùng nguyên lý, khác phần cứng.

### QAT
Train lại với fake-quant trong loop. Đắt. Chỉ dùng khi PTQ mất > 1-2% accuracy và bạn
buộc phải xuống 3-4 bit. Repo esp32 kết luận không cần QAT (RESULTS.md:210) — đó là kết
luận đúng cho phần lớn trường hợp.

## 2.3 Một phát hiện quan trọng từ repo esp32

RESULTS.md:205-209 — bảng lookup 25M tham số **degrade ÍT hơn** dense core dưới 4-bit
(PLE giữ 124-128% edge). Lý do: bảng lớn có dư thừa, mỗi weight ít critical hơn.

Tổng quát hoá: **model càng lớn/càng dư thừa thì càng chịu quantize tốt.**
- 70B Q4 thường tốt hơn 13B Q8 ở cùng dung lượng
- Model 1B xuống 4-bit hỏng rõ rệt, 8B thì gần như không
- Trên Orin Nano 8GB: ưu tiên **model lớn hơn + quantize sâu hơn**, đừng chạy 3B FP16

## 2.4 Validate — phần bắt buộc, hay bị bỏ

Repo esp32 làm đúng chuẩn và bạn nên copy quy trình này:

1. **Golden logits** — [`export.py:145-164`](../src/export.py#L145-L164):
   forward một prompt cố định, dump logits, yêu cầu engine mới khớp `max_abs_diff < 1e-5`.
   Đây là test **port correctness**, tách bạch với quantization error.
2. **Perplexity** — [`quantize.py:76`](../src/quantize.py#L76) đo val loss
   fp32 vs quantized. Đây là test **quantization quality**.
3. Báo cáo degradation bằng nats/ppl, **nhiều seed** (repo dùng 2 seed).

Trên Jetson:
```bash
# llama.cpp
llama-perplexity -m model-Q4_K_M.gguf -f wiki.test.raw
# so với FP16 baseline. Delta < 0.1 ppl là ổn; > 0.5 là có vấn đề
```

Sai lầm phổ biến: chuyển sang TensorRT INT8, thấy nhanh 3×, ship luôn — không biết
accuracy đã tụt vì calibration set không đại diện.

## 2.5 Bảng quyết định cho Orin Nano Super 8GB

| Mục tiêu | Chọn |
|---|---|
| LLM chat, chất lượng tối đa trong 8GB | 7-8B **Q4_K_M** + INT8 KV cache |
| LLM latency thấp | 1-3B **Q4_K_M** hoặc Q8_0 |
| Vision (YOLO, detection) | TensorRT **INT8** PTQ, entropy calibration |
| Whisper/ASR | TensorRT FP16, INT8 ít lợi vì encoder compute-bound đã dùng tensor core |
| Thử nghiệm nhanh | GGUF Q4_K_M — có sẵn trên HF, không cần calib |

## 2.6 Bài tập

1. Đọc [`quantize.py`](../src/quantize.py) và tự cài lại hàm
   `quantize_groupwise` từ đầu, không nhìn. Kiểm bằng cách so RMSE với bản gốc.
2. Trên model bất kỳ: đo ppl ở FP16 / Q8_0 / Q4_K_M / Q3_K_M. Vẽ ppl vs GB.
   Tìm điểm gãy.
3. Với cùng dung lượng ~4.5GB: so 8B-Q4 vs 3B-Q8 trên cùng bộ prompt. Xác nhận
   (hoặc bác bỏ) luận điểm 2.3 trên workload của bạn.

→ Tiếp: [03-kv-cache-va-engine.md](07-kv-cache-engine.md)
