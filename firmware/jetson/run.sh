#!/usr/bin/env bash
# Chạy toàn bộ pipeline trong container -- không cài gì lên host.
#
#   ./firmware/jetson/run.sh build          # dựng image
#   ./firmware/jetson/run.sh prepare        # tải TinyStories + train BPE + token bins
#   ./firmware/jetson/run.sh train          # train arm `ple`
#   ./firmware/jetson/run.sh quantize       # đo degradation 4-bit
#   ./firmware/jetson/run.sh export         # model.bin + golden.txt
#   ./firmware/jetson/run.sh assets         # vocab.h cho generate_cuda
#   ./firmware/jetson/run.sh shell          # vào container để nghịch
#
# Rồi đẩy lên board:  ./firmware/jetson/deploy.sh user@ip
#
# Mọi thứ ghi vào repo qua bind-mount, nên artifact nằm ở host như bình thường.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=esp32-llm-train
# Cấu hình nhỏ để lặp nhanh. Cấu hình deploy 28.9M ở RESULTS.md là
# --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 (train lâu hơn nhiều).
VOCAB="${VOCAB:-4096}"
TAG="${TAG:-jetson}"
STEPS="${STEPS:-2000}"

dk() {
  docker run --rm --gpus all \
    -v "$REPO":/work -w /work \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    "$IMAGE" "$@"
}

case "${1:-}" in
  build)
    docker build -f "$REPO/firmware/jetson/Dockerfile.train" -t "$IMAGE" "$REPO"
    ;;
  prepare)
    dk python data/prepare.py --vocab "$VOCAB"
    ;;
  train)
    # cwd=src vì src/*.py import nhau bằng tên phẳng
    dk bash -c "cd src && python train.py --arm ple --vocab $VOCAB \
        --steps $STEPS --tag $TAG --seed 0"
    ;;
  train-all)
    for arm in baseline ple; do
      dk bash -c "cd src && python train.py --arm $arm --vocab $VOCAB \
          --steps $STEPS --tag $TAG --seed 0"
    done
    dk bash -c "cd src && python analyze.py"
    ;;
  quantize)
    dk bash -c "cd src && python quantize.py --tag $TAG --seed 0"
    ;;
  export)
    dk bash -c "cd src && python export.py ple-$TAG-s0"
    ;;
  assets)
    # vocab.h: token id -> raw UTF-8 bytes, cho generate_cuda (và esp32_llm).
    # In luôn PROMPT_IDS để dán vào generate_cuda.cu nếu muốn đổi prompt.
    dk bash -c "cd src && python gen_assets.py --vocab $VOCAB \
        --out ../firmware/jetson/vocab.h --prompt '${PROMPT:-Once upon a time}'"
    ;;
  shell)
    docker run --rm -it --gpus all -v "$REPO":/work -w /work \
      -u "$(id -u):$(id -g)" -e HOME=/tmp "$IMAGE" bash
    ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
