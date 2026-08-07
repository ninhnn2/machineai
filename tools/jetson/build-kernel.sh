#!/usr/bin/env bash
# Cross-compile kernel L4T r36.4.x + OOT modules cho Jetson Orin Nano Super (sm_87, aarch64).
# Chạy trên x86. Kết quả: Image + module đã staging, sẵn sàng scp sang board.
#
#   ./build-kernel.sh                  # LOCALVERSION mặc định -tegra-custom1
#   LV=-tegra-hz1000 ./build-kernel.sh # đổi hậu tố phiên bản cho thí nghiệm khác
#
# Nguyên tắc: mỗi biến thể kernel có LOCALVERSION riêng => /lib/modules riêng =>
# kernel gốc của board KHÔNG bao giờ bị đụng vào. Rollback = đổi 1 dòng extlinux.
set -euo pipefail

ROOT="/home/machineai/jetson-build"
SRC="$ROOT/Linux_for_Tegra/source"
LV="${LV:--tegra-custom1}"
OUT="$ROOT/out${LV}"
STAGE="$ROOT/stage${LV}"
JOBS="$(nproc)"

export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-
export LC_ALL=C

echo "=== [1/6] Sync nguồn sang $OUT (như nvbuild.sh làm) ==="
mkdir -p "$OUT/kernel"
rsync -a --delete "$SRC/kernel/kernel-jammy-src" "$OUT/kernel/"
cp -a "$SRC/kernel/Makefile" "$OUT/kernel/"
for d in nvethernetrm nvgpu nvidia-oot hwpm hardware nvdisplay kernel-devicetree; do
  rsync -a --delete "$SRC/$d" "$OUT/"
done
cp -a "$SRC/Makefile" "$OUT/"

echo "=== [2/6] Build kernel (Image + dtbs + modules in-tree), LOCALVERSION=$LV ==="
time make -C "$OUT/kernel" version="$LV" NPROC="$JOBS"

KSRC="$OUT/kernel/kernel-jammy-src"
test -f "$KSRC/arch/arm64/boot/Image" || { echo "LỖI: không có Image"; exit 1; }

echo "=== [3/6] Build OOT modules (nvgpu, nvidia-oot, hwpm, nvdisplay...) ==="
export KERNEL_HEADERS="$KSRC"
export KERNEL_OUTPUT="$KSRC"
time make -C "$OUT" modules -j"$JOBS"
test -f "$OUT/nvgpu/drivers/gpu/nvgpu/nvgpu.ko" || { echo "LỖI: thiếu nvgpu.ko"; exit 1; }

echo "=== [4/6] Build DTB của NVIDIA ==="
make -C "$OUT" dtbs -j"$JOBS" || echo "(dtbs lỗi — không chặn, board dùng DTB sẵn có)"

echo "=== [5/6] Staging modules vào $STAGE (strip để giảm dung lượng truyền) ==="
rm -rf "$STAGE"; mkdir -p "$STAGE"
make -C "$KSRC" ARCH=arm64 O="$KSRC" LOCALVERSION="$LV" \
     INSTALL_MOD_PATH="$STAGE" INSTALL_MOD_STRIP=1 modules_install -j"$JOBS" >/dev/null
make -C "$OUT" INSTALL_MOD_PATH="$STAGE" INSTALL_MOD_DIR=updates INSTALL_MOD_STRIP=1 \
     modules_install >/dev/null

echo "=== [6/6] Đóng gói ==="
KREL="$(cat "$KSRC/include/config/kernel.release")"
cp "$KSRC/arch/arm64/boot/Image" "$ROOT/Image-$KREL"
tar -C "$STAGE/lib/modules" -czf "$ROOT/modules-$KREL.tar.gz" "$KREL"

echo
echo "kernel.release : $KREL"
echo "Image          : $ROOT/Image-$KREL  ($(du -h "$ROOT/Image-$KREL" | cut -f1))"
echo "modules        : $ROOT/modules-$KREL.tar.gz  ($(du -h "$ROOT/modules-$KREL.tar.gz" | cut -f1))"
echo "số module      : $(find "$STAGE/lib/modules/$KREL" -name '*.ko' | wc -l)"
