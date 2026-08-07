---
name: ple-tinylm
description: Làm việc với repo PLE TinyLM (esp32-ai-main) — train/export/quantize model, build và chạy runtime trên ESP32-S3 / CPU / Jetson CUDA, đo hiệu năng và validate. Dùng khi động tới src/*.py, firmware/*, samples/*, data/prepare.py, model.bin, golden.txt, hoặc khi được hỏi về tok/s, roofline, PLE, quantize, ablation, deploy lên board.
---

# PLE TinyLM — pipeline, runtime, và cách đo

Một model 28.9M tham số (PLE / Per-Layer Embeddings, train trên TinyStories) chạy
trên **ba runtime từ cùng một `model.bin`**: ESP32-S3, CPU scalar C, GPU CUDA.
Luận điểm trung tâm: phân tham số theo **cách truy cập** chứ không theo tốc độ, nên
25M tham số nằm trong flash mà gần như không tốn băng thông.

```
src/export.py ──> firmware/model/model.bin  ──┬──> firmware/esp32_llm/   Xtensa LX7, C
                  firmware/model/golden.txt  ├──> firmware/host_verify/ CPU, C
                                              └──> firmware/jetson/      CUDA
```

## Quy tắc bất di bất dịch của repo này

1. **Một artifact, một golden.** Mọi runtime đọc chung `firmware/model/model.bin` và
   đối chiếu chung `firmware/model/golden.txt`. Không tạo format riêng cho một target.
   Sửa layout `model.bin` = sửa đồng thời `src/export.py` **và** `llm_load()` trong
   [`firmware/common/llm.h`](../../../firmware/common/llm.h) (65 tensor, thứ tự hard-code).
2. **Đúng trước, nhanh sau.** Không tối ưu khi tầng 1 (verify) chưa `argmax MATCH`.
3. **Đo, đừng tra datasheet.** Mọi con số trong docs/RESULTS.md là số đo thật. Khi
   thêm số mới, ghi rõ phần cứng + điều kiện đo; chỗ chưa đo được thì nói thẳng.
4. **Tính sàn trước khi tối ưu.** Trước khi viết SIMD/kernel, tính giới hạn bandwidth
   của stage đó. Nếu lợi ích < ~20%, đừng làm — đổi hướng sang giảm bytes đọc.
   (Bài học lớn nhất repo: [RESULTS.md](../../../RESULTS.md) §head 57.6ms / sàn 40ms.)
5. **Mỗi kết luận cần một nhánh đối chứng và ≥2 seed.** Chênh lệch phải lớn hơn nhiễu
   seed nhiều lần mới được coi là thật.
6. **Không sửa `param_budget()` trong `src/model.py`** — `make_model()` binary-search
   vào nó để core-match các nhánh; đổi ngữ nghĩa là phá tính so sánh được của mọi run
   đã chạy. Cần báo cáo khác thì sửa `src/budget.py` (chỉ *report*).
7. **Runs cũ dùng accounting cũ** (`runs/_archive_old_accounting/`) — không trích dẫn.

## Chuỗi kiểm chứng 5 tầng — chạy theo thứ tự, đừng gộp

Nguyên tắc: mỗi tầng tách **một loại lỗi**. Gộp thì không biết lỗi ở đâu.

| Tầng | Câu hỏi | Lệnh | Ngưỡng PASS |
|---|---|---|---|
| 0 | tokenizer C == Python? | `make -C firmware/jetson tok` | 18/18 khớp |
| 1 | port có đúng không? | `make -C firmware/jetson verify` | **`argmax MATCH`** |
| 2 | 4-bit hỏng bao nhiêu? | `./firmware/jetson/run.sh quantize` | báo nats, ≥2 seed |
| 3 | nghẽn ở đâu? | `make -C firmware/jetson bench` | so với trần roofline |
| 4 | model viết ra gì? | `make -C firmware/jetson chat` | đọc bằng mắt |

Không có GPU thì tầng 1 chạy bằng CPU:
```bash
cc -O3 -o /tmp/verify_c firmware/host_verify/verify.c -lm
/tmp/verify_c firmware/model/model.bin firmware/model/golden.txt
```

**Tiêu chí PASS là `argmax MATCH`, KHÔNG phải `max|d| < 1e-5`.** Cộng số thực không
kết hợp; CUDA reduce dạng cây sai số `O(log n)`, C tuần tự `O(n)` — nên CUDA thường
*gần PyTorch hơn* bản C. Đặt ngưỡng theo **hành vi** (argmax / top-k / perplexity).

`golden.txt` là logits của model **đã dequantize** ([`src/export.py`](../../../src/export.py)),
nên tầng 1 chỉ đo lỗi *port*, tách hẳn khỏi lỗi *lượng tử hoá* đo ở tầng 2. Tách biến.

## Việc thường gặp → làm gì

| Việc | Bắt đầu ở đâu |
|---|---|
| Train / export / deploy | `references/pipeline.md` |
| Sửa model, thêm tensor, đổi `model.bin` | `references/architecture.md` |
| Tối ưu tốc độ, đọc kết quả bench | `references/methodology.md` |
| Thêm nhánh ablation, báo cáo kết quả | `references/methodology.md` §đối chứng |
| Deploy ESP32 | [`firmware/esp32_llm/README.md`](../../../firmware/esp32_llm/README.md) |
| Deploy Jetson | [`firmware/jetson/JETSON.md`](../../../firmware/jetson/JETSON.md), [`DEPLOY.md`](../../../DEPLOY.md) |
| Framework trên Jetson (torch/llama.cpp/TensorRT/DeepStream) | [`docs/13-jetson-framework.md`](../../../docs/13-jetson-framework.md), [`docs/14-tensorrt-deepstream.md`](../../../docs/14-tensorrt-deepstream.md) |
| Học lý thuyết | [`docs/README.md`](../../../docs/README.md) (tiếng Việt). Người mới: [`docs/00-nhap-mon.md`](../../../docs/00-nhap-mon.md); tra thuật ngữ: [`docs/12-thuat-ngu.md`](../../../docs/12-thuat-ngu.md) |
| Kỹ sư embedded muốn học AI từ gốc | [`docs/begin_0/README.md`](../../../docs/begin_0/README.md) — vector→weight→matmul→gradient→backprop→transformer→quantization→runtime→VLA, grounded bằng code thật trong repo |

## Bốn con số cần thuộc

```
60.7 MB/s   PSRAM sequential read trên ESP32-S3 N16R8 (đo thật)
102.9 ms    per-token trên ESP32 (head 57.6 | attn 25.6 | ple 8.5 | ffn 6.9 | in 4.4)
66.8 GB/s   DRAM Orin Nano Super (datasheet nói 102 → chỉ đạt 65%, EMC khoá 2133MHz)
3.71 us     launch overhead / kernel trên Orin (2.5 us trên RTX 4060)
```

Hệ quả: `tok/s ≤ bandwidth_GB/s / model_GB`. Trên Orin, model này quá nhỏ nên
**50% thời gian mỗi token là launch overhead thuần** — nghẽn hoàn toàn khác ESP32
(bandwidth). Cùng model, cùng code, nút thắt khác nhau: đó là điểm của repo.

## Khi báo cáo số

- Nói rõ phần cứng, power mode, máy có tải nền không, số lần lặp (≥3, báo median).
- ESP32: phân biệt **end-to-end** (~9.5 tok/s, có serial) với **compute-only** (9.72).
  Public number dùng end-to-end.
- "28.9M tham số" = tham số **stored** qua phân tầng bộ nhớ (559K core + 3.1M head +
  25M table). Không bao giờ quote như bội số năng lực.
- Model là TinyStories: nó **viết tiếp truyện**, không trả lời câu hỏi. Giới hạn nằm ở
  core 559K, không phải ở runtime hay PLE.
