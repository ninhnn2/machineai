#!/usr/bin/env bash
# Copy runtime CUDA + model đã export lên Jetson, build luôn trên board.
#
#   # cách 1 -- SSH key (khuyến nghị, không cần mật khẩu)
#   ssh-copy-id machineai@100.92.121.20        # làm 1 lần
#   ./firmware/jetson/deploy.sh machineai@100.92.121.20
#
#   # cách 2 -- mật khẩu tự động, đọc từ biến môi trường
#   read -rsp 'Jetson password: ' SSHPASS; echo; export SSHPASS
#   ./firmware/jetson/deploy.sh machineai@100.92.121.20
#
#   # cách 3 -- mật khẩu trong file chỉ mình đọc được
#   printf '%s' 'matkhau' > ~/.jetson_pass && chmod 600 ~/.jetson_pass
#   JETSON_PASS_FILE=~/.jetson_pass ./firmware/jetson/deploy.sh machineai@100.92.121.20
#
#   # tuỳ chọn
#   ./firmware/jetson/deploy.sh <host> --no-build     # chỉ copy
#   REMOTE_DIR=~/esp32-llm ./firmware/jetson/deploy.sh <host>
#
# KHÔNG dùng `sshpass -p 'matkhau'` trực tiếp trên dòng lệnh: mật khẩu hiện trong
# `ps aux` cho mọi user trên máy, và nằm lại trong ~/.bash_history. `sshpass -e`
# (biến môi trường) và `-f` (file) tránh được cả hai.
set -euo pipefail

HOST="${1:-}"
[ -z "$HOST" ] && { sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 1; }
shift || true
BUILD=1
[ "${1:-}" = "--no-build" ] && BUILD=0

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-esp32-llm}"

# --- chọn cách xác thực -------------------------------------------------------
SSH_WRAP=()
if [ -n "${JETSON_PASS_FILE:-}" ]; then
  command -v sshpass >/dev/null || { echo "cần: sudo apt install sshpass"; exit 1; }
  [ -r "$JETSON_PASS_FILE" ] || { echo "không đọc được $JETSON_PASS_FILE"; exit 1; }
  SSH_WRAP=(sshpass -f "$JETSON_PASS_FILE")
  echo "[auth] mật khẩu từ file $JETSON_PASS_FILE"
elif [ -n "${SSHPASS:-}" ]; then
  command -v sshpass >/dev/null || { echo "cần: sudo apt install sshpass"; exit 1; }
  SSH_WRAP=(sshpass -e)          # -e đọc biến SSHPASS, không hiện trong ps
  echo "[auth] mật khẩu từ biến SSHPASS"
else
  echo "[auth] SSH key (không có SSHPASS/JETSON_PASS_FILE)"
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
rsh() { "${SSH_WRAP[@]}" ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
rcp() { "${SSH_WRAP[@]}" scp "${SSH_OPTS[@]}" "$@"; }

# --- kiểm tra trước khi copy --------------------------------------------------
for f in firmware/model/model.bin firmware/model/golden.txt; do
  [ -f "$REPO/$f" ] || { echo "THIẾU $f -- chạy ./firmware/jetson/run.sh export trước"; exit 1; }
done
[ -f "$REPO/firmware/jetson/vocab.h" ] || cat <<'EOF'
CẢNH BÁO: thiếu firmware/jetson/vocab.h -> `make generate` sẽ không build được.
  Sinh bằng: ./firmware/jetson/run.sh assets
EOF

echo "[1/3] kết nối $HOST"
rsh "mkdir -p ~/$REMOTE_DIR/firmware/{jetson,common,model,host_verify}"
rsh 'echo "  board: $(cat /proc/device-tree/model 2>/dev/null | tr -d "\0")"; \
     echo "  power: $(nvpmodel -q 2>/dev/null | head -1)"'

echo "[2/3] copy"
cd "$REPO"
FILES=(firmware/jetson/llm_cuda.cuh firmware/jetson/verify_cuda.cu
       firmware/jetson/bench_cuda.cu firmware/jetson/generate_cuda.cu
       firmware/jetson/Makefile)
[ -f firmware/jetson/vocab.h ] && FILES+=(firmware/jetson/vocab.h)
rcp "${FILES[@]}"                 "$HOST:~/$REMOTE_DIR/firmware/jetson/"
rcp firmware/common/llm.h         "$HOST:~/$REMOTE_DIR/firmware/common/"
rcp firmware/model/model.bin firmware/model/golden.txt \
                                  "$HOST:~/$REMOTE_DIR/firmware/model/"
rcp firmware/host_verify/verify.c firmware/host_verify/ppl.c \
                                  "$HOST:~/$REMOTE_DIR/firmware/host_verify/" 2>/dev/null || true
for d in DEPLOY.md firmware/jetson/JETSON.md; do
  [ -f "$d" ] && rcp "$d" "$HOST:~/$REMOTE_DIR/$(dirname "$d")/" || true
done

if [ "$BUILD" = 1 ]; then
  echo "[3/3] build trên board"
  rsh "export PATH=/usr/local/cuda/bin:\$PATH && cd ~/$REMOTE_DIR/firmware/jetson && make 2>&1 | tail -5"
else
  echo "[3/3] bỏ qua build (--no-build)"
fi

cat <<EOF

Xong. Chạy trên board:
  ssh $HOST
  sudo nvpmodel -m 2 && sudo jetson_clocks     # MAXN_SUPER = mode 2!
  cd ~/$REMOTE_DIR/firmware/jetson
  make verify && make bench && make generate
EOF
