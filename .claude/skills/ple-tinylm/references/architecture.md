# Kiến trúc model, ba tầng bộ nhớ, và format `model.bin`

Nguồn đầy đủ: [`DEPLOY.md`](../../../DEPLOY.md) §1, [`docs/02-hieu-model.md`](../../../docs/02-hieu-model.md).
File này là cái cần nhớ khi *sửa code*.

## Ba tầng — phân theo cách truy cập, không theo tốc độ

| Tầng | Truy cập | ESP32-S3 | Jetson |
|---|---|---|---|
| **core** | dày đặc, ngẫu nhiên, mỗi token | SRAM 512KB — ngân sách thật sự khan hiếm | VRAM |
| **stream** | dày đặc, quét tuần tự 1 lần/token (output head) | PSRAM — tốn *bandwidth*, không tốn dung lượng SRAM | bandwidth |
| **table** | **thưa: 1 hàng/token** (bảng PLE) | flash mmap — gần như miễn phí | gần như miễn phí |

Định nghĩa và hằng số board ở [`src/budget.py`](../../../src/budget.py) (file này chỉ
*report*; `param_budget()` trong `src/model.py` mới là cái `make_model()` binary-search
vào — **không đổi ngữ nghĩa của nó**).

Ở cấu hình dev (vocab 4096): core 1.50M (41.7%) / stream 0.52M (14.6%) / table 1.57M
(43.7%) → **43.7% tham số tốn 0.024% băng thông**. Ở cấu hình deploy 28.9M: 25M tham số
(87%) đọc ~450 B/token. Cùng nguyên lý với MoE: nhiều tham số, đọc ít mỗi token.

## Nhánh (`--arm`) — mỗi nhánh trả lời một câu hỏi

| Nhánh | Có gì | Câu hỏi |
|---|---|---|
| `baseline` | không bảng | mốc so sánh |
| `ple` | đường ống PLE + bảng | ý tưởng đầy đủ |
| `ple_notable` | đường ống PLE, **không bảng** | lợi ích đến từ bảng hay từ plumbing? |
| `fatembed` | bảng cùng cỡ, tiêm ở **đáy** | vị trí tiêm có quan trọng không? |
| `bigcore` | dùng ngân sách bảng làm core to hơn | nếu có bộ nhớ nhanh thì sao? |

Kết quả (vocab 4096): `ple_notable` 8.35 **tệ hơn** `baseline` 8.21 → plumbing tự nó
vô dụng, toàn bộ lợi ích đến từ bảng. `bigcore` 6.93 → PLE chỉ lấy lại ~15% của một
core to hơn; nó là cách xoay xở khi core bị giới hạn cứng, không phải phép màu.

## Bảng tensor (cấu hình dev: vocab 4096, d_model 128, L 6, ple_dim 64)

```
tok_emb.weight          [4096, 128]     524,288   stream   tied với output head
ple_model_proj.weight    [384, 128]      49,152   core     384 = L×P = 6×64
ple_proj_norm.weight          [64]           64   core     fp32, không quantize
ple_table.weight        [4096, 384]   1,572,864   TABLE    ★ 1 hàng/token
── mỗi lớp ×6 ────────────────────────────────────────────────────────────
  attn_norm [128] fp32 · attn.qkv [384,128] · attn.proj [128,128]
  ffn_norm  [128] fp32 · ffn.gate/up [415,128] · ffn.down [128,415]
  ple_gate  [64,128] · ple_proj [128,64] · ple_norm [128] fp32 (init 0)
                                 241,664/lớp → 1,449,984
out_norm.weight              [128]          128   core     fp32
                                        TỔNG 3,596,480 → model.bin 1.87 MB @4-bit
```

Thành phần mỗi lớp: RMSNorm → attention (RoPE, KV cache) → RMSNorm → SwiGLU FFN, cộng
đường PLE (gate + proj + norm) tiêm **ở mỗi lớp** — đó là điểm khác `fatembed`.

## Format `model.bin`

Sinh bởi [`src/export.py`](../../../src/export.py), đọc bởi `llm_load()` trong
[`firmware/common/llm.h`](../../../firmware/common/llm.h). **Cả ba runtime dùng chung.**

```
offset  size    nội dung
0       4 B     magic = 0x504C4531 ("PLE1")
4       32 B    8 × int32: vocab, d_model, n_layers, n_heads, ffn, ple_dim, seq_len, group
36      4 B     float32 rope_theta
40      ...     65 tensor, THỨ TỰ CỐ ĐỊNH (hard-code phía C)

tensor QUANTIZED:
  4 B                     int32 group (=128)
  rows × ceil(cols/2) B   nibble int4, value = code − 8, 2 giá trị/byte
                          (byte j>>1, nửa thấp nếu j chẵn)
  rows × n_groups × 2 B   scale fp16, n_groups = ceil(cols/group)

tensor FP32 (chỉ các norm):
  n × 4 B raw float32
```

**Ragged, không padding** — nhóm cuối mỗi hàng có thể ngắn hơn `group`. Đó là lý do
file vừa flash 16MB. Byte trên GPU **chính là** byte trong file: `llm_cuda.cuh` upload
thẳng, không repack.

### Khi sửa layout
Sửa `src/export.py` **và** `llm_load()` cùng lúc, rồi chạy lại tầng 0→1 trên **cả ba**
runtime (ESP32 có thể verify gián tiếp qua `host_verify` vì cùng code C). Thứ tự 65
tensor là hợp đồng ngầm giữa Python và C — thêm tensor phải thêm đúng vị trí ở cả hai.

## Bố trí trên ESP32-S3 N16R8

```
SRAM  512KB   core (~273KB @4-bit) — ngân sách sizing toàn bộ thiết kế
PSRAM 8MB     output head 1.64MB copy vào lúc boot (staged int8 = 2.53MB) + scratch
FLASH 16MB    bảng PLE 25M tham số, memory-mapped, ~6 hàng (~450B) mỗi token
```
Model 14,912,332 B nằm trong partition riêng 15,597,568 B tại `0x110000`; app 619KB ở
partition 1MB (`firmware/esp32_llm/partitions.csv`).
