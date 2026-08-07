#!/usr/bin/env bash
# Chạy TRÊN BOARD bằng sudo:  sudo bash /tmp/remote-install.sh <kernel-release>
# Cài kernel tự build mà KHÔNG đụng vào kernel gốc. Rollback = chọn "primary" ở menu boot.
set -euo pipefail
KREL="${1:?thiếu kernel release}"
CONF=/boot/extlinux/extlinux.conf

echo "== [1] backup =="
cp -n "$CONF" /root/extlinux.conf.orig 2>/dev/null || true
cp -n /boot/Image /boot/Image.orig-r36447 2>/dev/null || true
sha256sum /boot/Image | tee /root/Image.orig.sha256
ls -la /root/extlinux.conf.orig /boot/Image.orig-r36447

echo "== [2] modules =="
tar -C /lib/modules -xzf "/tmp/modules-$KREL.tar.gz"
depmod -a "$KREL"
echo "  $(find "/lib/modules/$KREL" -name '*.ko' | wc -l) module, depmod xong"

echo "== [3] Image + initrd riêng =="
cp "/tmp/Image-$KREL" "/boot/Image-$KREL"
update-initramfs -c -k "$KREL" 2>&1 | tail -2
ls -la "/boot/Image-$KREL" "/boot/initrd.img-$KREL"

echo "== [4] extlinux: thêm LABEL custom, GIỮ NGUYÊN primary làm fallback =="
APPEND="$(grep -m1 -E '^[[:space:]]*APPEND' "$CONF" | sed -E 's/^[[:space:]]*APPEND[[:space:]]*//')"
python3 - "$CONF" "$KREL" "$APPEND" <<'PY'
import sys, re
conf, krel, append = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(conf).read()
# gỡ entry custom cũ (nếu chạy lại)
txt = re.sub(r'\nLABEL custom\n(?:[ \t]+.*\n)*', '\n', txt)
entry = (f"\nLABEL custom\n"
         f"      MENU LABEL custom kernel {krel}\n"
         f"      LINUX /boot/Image-{krel}\n"
         f"      INITRD /boot/initrd.img-{krel}\n"
         f"      APPEND {append}\n")
txt = txt.rstrip('\n') + '\n' + entry
txt = re.sub(r'^DEFAULT .*$', 'DEFAULT custom', txt, count=1, flags=re.M)
open(conf, 'w').write(txt)
PY
echo "--- extlinux.conf hiệu lực ---"
grep -vE '^[[:space:]]*#' "$CONF" | grep -vE '^[[:space:]]*$'

echo
echo "== XONG. Fallback: reboot, chọn 'primary kernel' trong menu (TIMEOUT 30s) =="
