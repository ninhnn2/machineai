# Model đã train, commit sẵn trong repo

Thư mục này chứa artifact **đã build sẵn**, để clone về là chạy được ngay ba runtime
(`host_verify/`, `jetson/`, `esp32_llm/`) mà không phải train lại 21 phút.

| File | Kích thước | Là gì |
|---|---:|---|
| `model.bin` | 1.87 MB | model 4-bit, định dạng phẳng mà `llm_load()` mmap thẳng |
| `golden.txt` | 39 KB | logits tham chiếu do PyTorch sinh, để chứng minh port C đúng |
| `golden.npz` | 17 KB | cùng nội dung, dạng numpy, cho script Python |
| `../../runs/ple-jetson-s0.pt` | 14 MB | checkpoint fp32, chỉ cần khi muốn train tiếp / export lại |
| `../../runs/ple-jetson-s0.json` | 1.6 KB | lịch sử loss của lần train này |
| `../../data/bpe4096.json` | 259 KB | tokenizer — bắt buộc phải khớp, đổi tokenizer là hỏng hết |

## ⚠ Đây KHÔNG phải model 28.9M ở README chính

Đọc kỹ chỗ này trước khi flash lên board và thắc mắc sao khác số:

| | Model commit ở đây | Model deploy trong [`RESULTS.md`](../../RESULTS.md) |
|---|---|---|
| Tham số | **3.6M** (core 1.5M + table 1.57M) | **28.9M** (table 25M) |
| `model.bin` | 1.87 MB | 14.9 MB |
| vocab | 4096 | 32768 |
| `d_model` | 128 | 96 |
| `ple_dim` | 64 | 128 |
| Số bước train | 2000 | nhiều hơn hẳn |
| val loss / ppl | 2.2119 / 9.13 | xem RESULTS.md |
| Dùng để | lặp nhanh, học, kiểm chứng port | bản chạy thật trên ESP32-S3 |

Đây đúng là cấu hình `VOCAB=4096 STEPS=2000` mặc định của
[`firmware/jetson/run.sh`](../jetson/run.sh) — cố ý nhỏ để vòng lặp sửa–chạy tính
bằng phút, không phải bằng giờ. Nó **vừa 16MB flash của ESP32-S3 rất thoải mái**,
nhưng nó không phải con số 28.9M mà README quảng cáo.

## Sinh lại từ đầu

```bash
uv run python data/prepare.py --vocab 4096
cd src && uv run python train.py --arm ple --vocab 4096 --steps 2000 --tag jetson --seed 0
uv run python export.py ple-jetson-s0
```

Đo trên MacBook Pro M3: prepare ~6 phút (tải 300MB), train **1277s = 21,3 phút** trên
MPS, export vài giây.

Muốn bản 28.9M thì đổi tham số train (lâu hơn nhiều):

```bash
cd src && uv run python train.py --arm ple --vocab 32768 --d-model 96 \
    --n-layers 6 --ple-dim 128 --steps <nhiều hơn> --tag deploy --seed 0
```

## Kiểm chứng ngay sau khi clone

```bash
make -C firmware/host_verify verify    # PASS: C matches PyTorch golden
```

Chạy được trên mọi nền tảng có trình biên dịch C — xem bảng nền tảng ở
[`README.md`](../../README.md#chạy-trên-nền-tảng-nào).
