# Pipeline: data → train → quantize → export → deploy

Đường dẫn tương đối tính từ gốc repo.

## Môi trường

Hai cách, đừng trộn:

- **Host + uv** (`pyproject.toml`, Python ≥3.12, torch/numpy/tokenizers/tqdm):
  `uv run python src/train.py ...`
- **Container** (không cài gì lên host, cần Docker + `--gpus all`):
  `./firmware/jetson/run.sh <lệnh>` — bind-mount cả repo vào `/work`, artifact vẫn
  ghi ra host bình thường.

`src/*.py` import nhau bằng tên phẳng → **phải `cd src` trước khi chạy**, hoặc dùng
`uv run python src/train.py` từ gốc (train.py xử lý được cả hai; các script khác thì
`run.sh` luôn `cd src` trước).

## 1. Data

```bash
uv run python data/prepare.py --vocab 4096       # hoặc 32768 cho cấu hình deploy
```
Tải TinyStories → train BPE → `data/{bpe4096.json,train.bin,val.bin}`. ~2 phút.

## 2. Train

```bash
cd src && uv run python train.py --arm ple --vocab 4096 --steps 2000 --tag dev --seed 0
```

Cờ đáng nhớ: `--arm {baseline,ple,ple_notable,fatembed,bigcore}`, `--target-core`
(mặc định 1.5M — model tự binary-search chiều rộng để khớp), `--ple-dim`, `--d-model`,
`--n-layers`, `--n-heads`, `--fixed-ffn`, `--vocab`, `--seed`, `--tag`.

Checkpoint ra `runs/<arm>-<tag>-s<seed>.{pt,json}`.

**Cấu hình deploy 28.9M** (train lâu hơn nhiều, đây là cái sinh ra số trong RESULTS.md):
```
--vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128
```
Cấu hình mặc định `vocab 4096 / d_model 128 / ple_dim 64` là để **lặp nhanh**, không
phải cấu hình deploy — ở vocab 4096 lợi thế PLE chỉ +0.025 nats thay vì +0.098.

Chạy nhiều nhánh: `experiments/run_ablation.sh` (5 nhánh core-matched), rồi
`uv run python src/analyze.py`.

## 3. Quantize (tầng 2 của chuỗi kiểm chứng)

```bash
cd src && uv run python quantize.py --tag dev --seed 0        # --bits 4 --group 64
```
Báo cáo degradation theo nats cho từng arm. Thử `--bits 8/3`, `--group 32/64/128` rồi
vẽ ppl theo dung lượng. Chạy ≥2 seed trước khi kết luận.

## 4. Export

```bash
cd src && uv run python export.py ple-dev-s0        # tag của checkpoint
```
Ghi `firmware/model/model.bin` + `golden.txt` (+ `golden.npz`). `golden.txt` là logits
của model **đã dequantize** — cố ý, để tách lỗi port khỏi lỗi quantize.

## 5. Assets cho runtime C

```bash
cd src && uv run python gen_assets.py --vocab 4096 \
    --out ../firmware/jetson/vocab.h --prompt 'Once upon a time'
```
Sinh `vocab.h` (bảng decode id→bytes + BPE merges để tokenize on-device) và in
`PROMPT_IDS`. Chỉ cần làm lại khi đổi vocab. `src/gen_tok_ref.py` sinh `tok_ref.txt`
cho tầng 0.

## 6. Chạy trên từng target

### CPU (không cần gì đặc biệt)
```bash
cc -O3 -o /tmp/verify_c firmware/host_verify/verify.c -lm
/tmp/verify_c firmware/model/model.bin firmware/model/golden.txt
cc -O3 -o /tmp/ppl firmware/host_verify/ppl.c -lm            # perplexity
```

### GPU / Jetson
```bash
cd firmware/jetson
make ARCH=sm_87            # Orin (mặc định); sm_89 = RTX 40xx, sm_86 = RTX 30xx
make tok                   # tầng 0
make verify                # tầng 1
make bench PEAK=66.8       # tầng 3 — PEAK là bandwidth ĐO ĐƯỢC của board bạn
make generate              # hoặc: make chat
```
Đẩy lên board: `./firmware/jetson/deploy.sh user@ip`. Trên Jetson nhớ
`export PATH=/usr/local/cuda/bin:$PATH` và đặt power mode **MAXN_SUPER = mode 2**
(không phải mode 0) — xem [`docs/09-so-do-phan-cung.md`](../../../docs/09-so-do-phan-cung.md).

### ESP32-S3 (N16R8)
Verify bằng CPU trước, rồi:
```bash
arduino-cli compile --fqbn 'esp32:esp32:esp32s3:...,PartitionScheme=custom,PSRAM=opi,...' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build firmware/esp32_llm
arduino-cli upload -p <port> --fqbn '...' --input-dir /tmp/esp32-llm-build firmware/esp32_llm
esptool.py --chip esp32s3 --port <port> --baud 921600 write_flash 0x110000 firmware/model/model.bin
arduino-cli monitor -p <port> --config baudrate=115200
```
FQBN đầy đủ + SHA-256 của artifact + boot diagnostics mong đợi nằm trong
[`firmware/esp32_llm/README.md`](../../../firmware/esp32_llm/README.md). Model chỉ cần
flash lại sau khi export mới; sửa firmware thì không phải ghi lại partition model.

`firmware/bandwidth_bench/` đo PSRAM/SRAM/flash bằng cycle counter Xtensa — chạy nó
trước khi tranh luận về trần tok/s trên chip.

## 7. Samples đo học

```bash
make -C samples/cpu run          # thang 5 bậc tối ưu matvec, x86 + ARM
make -C samples/cpu scaling      # tách lợi ích SIMD khỏi lợi ích threads
make -C samples/cpu verify-theory
nvcc -O3 -arch=sm_87 samples/gpu/bench_roofline.cu -o /tmp/rf && /tmp/rf 256
python3 samples/gpu/roofline.py --hw orin-nano-super --model llama-3.1-8b --bits 4.8
```
