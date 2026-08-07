# Runbook — build lại, đổi config, và deploy kernel Jetson Orin Nano Super

Hướng dẫn **thao tác từng bước**, khác với [`docs/15`](../../docs/15-kernel-den-camera.md)
(kể lại quá trình + kết quả + bài học). File này là **checklist thực thi** để bạn tự làm
lại toàn bộ vòng: *build → (đổi config) → kiểm tra → deploy → boot → đo → rollback*.

Bốn script trong thư mục này (`build-kernel.sh`, `remote-install.sh`, `cam-bench.sh`,
`latency-bench.sh`) là công cụ; runbook này là trình tự dùng chúng.

```
Board đích : Jetson Orin Nano Super (3767-300-0005)  — board machineai-gw
L4T        : R36.4.7 (JetPack 6.2.x) | kernel 5.15.148-tegra | CUDA 12.6 | TRT 10.3
Host build : x86_64, ≥16 core, cross-compile aarch64
Nguồn dùng : L4T r36 Release v4.4 (khớp đúng kernel 5.15.148 board đang chạy)
```

> **Nguyên tắc an toàn xuyên suốt (board đang chạy hệ thống an toàn thật):**
> 1. **Không bao giờ ghi đè `/boot/Image` gốc.** Kernel mới nằm ở file riêng.
> 2. **Mỗi biến thể một `LOCALVERSION` riêng** ⇒ `/lib/modules/` riêng ⇒ kernel gốc bất
>    khả xâm phạm.
> 3. **Giữ nguyên entry `primary` trong extlinux** làm đường lùi. Rollback = đổi 1 dòng.
>
> Nếu bạn chỉ làm đúng ba điều này, mọi thí nghiệm đều có đường về.

---

## Phần 0 — Chuẩn bị (làm một lần)

### 0.1 Trên HOST (x86)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential bc bison flex libssl-dev libncurses-dev \
  gcc-aarch64-linux-gnu rsync wget tar
aarch64-linux-gnu-gcc --version    # kỳ vọng 11.x
```

Thư mục làm việc (mọi script giả định đúng đường này):

```bash
export ROOT=/home/machineai/jetson-build
mkdir -p "$ROOT" && cd "$ROOT"
```

### 0.2 Trên BOARD — kiểm truy cập + mạng trước khi động vào kernel

```bash
ssh machineai-gw 'uname -r; cat /etc/nv_tegra_release | head -1'
# kỳ vọng: 5.15.148-tegra  ...  và KERNEL_VARIANT: oot  (đặc trưng JetPack 6)
```

Nếu board cần cài gói đo (`rt-tests`, `stress-ng`, gstreamer) mà `apt` báo không phân
giải được tên miền — **đó là DNS, không phải mất mạng** (Tailscale chiếm DNS, xem
[`docs/15 §15.8`](../../docs/15-kernel-den-camera.md)). Tách bệnh trong 2 giây:

```bash
ping -c2 8.8.8.8            # đi được?  -> có mạng
curl -sI https://github.com  # 000?      -> hỏng DNS
# sửa runtime:
sudo resolvectl dns    enP8p1s0 1.1.1.1 8.8.8.8
sudo resolvectl domain enP8p1s0 '~.'
```

---

## Phần 1 — Lấy đúng nguồn (bẫy đầu tiên: 404)

Không phải version nào NVIDIA cũng publish. **v4.7/v4.6/v4.5 = 404**; dùng **v4.4** —
`kernel-jammy-src/Makefile` cho `5.15.148`, **khớp đúng kernel board đang chạy**.

```bash
cd "$ROOT"
wget https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.4/sources/public_sources.tbz2
tar xf public_sources.tbz2
cd Linux_for_Tegra/source
tar xf kernel_src.tbz2
tar xf kernel_oot_modules_src.tbz2
tar xf nvidia_kernel_display_driver_source.tbz2
```

Cấu trúc phải ra (JetPack 6 — **driver NVIDIA là out-of-tree, không nằm trong cây kernel**):

```
Linux_for_Tegra/source/
  kernel/kernel-jammy-src/   kernel Linux
  nvgpu/ nvidia-oot/ nvdisplay/ hwpm/ nvethernetrm/   ← OOT modules, build RIÊNG
  kernel-devicetree/         DTB
  nvbuild.sh  Makefile
```

---

## Phần 2 — Build kernel gốc TRƯỚC (dựng mốc đối chứng)

**Đừng đổi config ở lần đầu.** Build lại đúng kernel gốc và chứng minh nó bằng nhau
trong sai số — nếu không có mốc này thì mọi thí nghiệm sau đều vô nghĩa.

```bash
cd /home/machineai/workspaces/AI/esp32-ai-main/tools/jetson
LV=-tegra-custom1 ./build-kernel.sh
```

Script làm 6 việc: sync nguồn → build kernel (Image + dtbs + module in-tree) → build OOT
modules → build DTB → staging (strip) → đóng gói. Thời gian thật: **kernel ~7m28s,
modules ~1m52s** trên 32 core.

Kết quả cuối script in ra:

```
kernel.release : 5.15.148-tegra-custom1
Image          : /home/machineai/jetson-build/Image-5.15.148-tegra-custom1
modules        : /home/machineai/jetson-build/modules-5.15.148-tegra-custom1.tar.gz
số module      : 1076
```

Đặt biến để dùng ở các bước sau:

```bash
export KREL=5.15.148-tegra-custom1
export KSRC=/home/machineai/jetson-build/out-tegra-custom1/kernel/kernel-jammy-src
export STAGE=/home/machineai/jetson-build/stage-tegra-custom1
```

---

## Phần 3 — (TÙY CHỌN) Đổi cấu hình kernel

Bỏ qua phần này nếu bạn chỉ build lại kernel gốc. Làm phần này khi cần **bật driver**,
**đổi CONFIG_HZ**, **bật PREEMPT_RT**, v.v.

> **Quy tắc vàng khi đổi config: đổi ĐÚNG MỘT thứ mỗi lần, và cấp `LOCALVERSION` mới.**
> Cùng LV mà khác config ⇒ đè lên `/lib/modules` cũ ⇒ mất khả năng đối chứng.

### 3a. Đổi một symbol không cần giao diện (khuyến nghị cho script hoá)

Ví dụ `CONFIG_HZ_250` → `CONFIG_HZ_1000` (bài tập [`docs/15 §15.11`](../../docs/15-kernel-den-camera.md)):

```bash
"$KSRC/scripts/config" --file "$KSRC/.config" \
    -d CONFIG_HZ_250 -e CONFIG_HZ_1000 --set-val CONFIG_HZ 1000
make -C "$KSRC" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- olddefconfig
```

### 3b. Đổi bằng menuconfig (khi dò tìm symbol)

```bash
make -C "$KSRC" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- menuconfig
```

### 3c. Build lại với LOCALVERSION mới

Sau khi sửa `.config`, build **thẳng in-tree** (giữ nguyên `.config` bạn vừa sửa — không
để wrapper `version=` sinh lại defconfig đè lên):

```bash
export LV=-tegra-hz1000                       # <- hậu tố MỚI cho biến thể này
export CROSS_COMPILE=aarch64-linux-gnu- ARCH=arm64
make -C "$KSRC" LOCALVERSION="$LV" -j"$(nproc)" Image modules dtbs

# build lại OOT modules đối với kernel vừa build (BẮT BUỘC — module compile theo kernel)
OUT=/home/machineai/jetson-build/out-tegra-custom1
export KERNEL_HEADERS="$KSRC" KERNEL_OUTPUT="$KSRC"
make -C "$OUT" modules -j"$(nproc)"

# staging + đóng gói lại (như bước [5],[6] của build-kernel.sh)
export KREL=$(cat "$KSRC/include/config/kernel.release")   # vd 5.15.148-tegra-hz1000
export STAGE=/home/machineai/jetson-build/stage$LV
rm -rf "$STAGE"; mkdir -p "$STAGE"
make -C "$KSRC" ARCH=arm64 O="$KSRC" LOCALVERSION="$LV" \
     INSTALL_MOD_PATH="$STAGE" INSTALL_MOD_STRIP=1 modules_install
make -C "$OUT"  INSTALL_MOD_PATH="$STAGE" INSTALL_MOD_DIR=updates INSTALL_MOD_STRIP=1 modules_install
cp "$KSRC/arch/arm64/boot/Image" "/home/machineai/jetson-build/Image-$KREL"
tar -C "$STAGE/lib/modules" -czf "/home/machineai/jetson-build/modules-$KREL.tar.gz" "$KREL"
```

> **Muốn PREEMPT_RT?** Đọc [`docs/15 §15.6`](../../docs/15-kernel-den-camera.md) TRƯỚC.
> Trên board này RT **mất 30–40% throughput VÀ độ trễ xấu nhất còn tệ hơn ~10×**. NVIDIA
> chặn build bằng `preempt_rt_sanity_check`; phải `export IGNORE_PREEMPT_RT_PRESENCE=1`
> để qua — cái tên nói đúng bản chất: bạn đang **bỏ qua cảnh báo**, không bật tính năng.
> Kết luận đã đo: đừng deploy RT lên board này.

---

## Phần 4 — Hai phép kiểm tra BẮT BUỘC trước khi deploy

Đây là ranh giới giữa "build được" và "build đúng". **Không bỏ bước này.**

### 4.1 Config của tôi có khớp kernel đang chạy không?

```bash
ssh machineai-gw 'zcat /proc/config.gz' > /tmp/board-config-running
diff <(sort /tmp/board-config-running) <(sort "$KSRC/.config")
```

- **Build gốc (Phần 2):** chỉ được khác **các dòng do compiler** (`CC_VERSION_TEXT`,
  `GCC_PLUGINS`, `PAHOLE_VERSION`…). Thật tế: **14 dòng, cả 14 đều do compiler.** Nếu
  thấy một dòng chính sách (`CONFIG_HZ`, `CONFIG_PREEMPT…`) khác mà bạn *không* cố ý đổi
  ⇒ **dừng, tìm hiểu lý do.**
- **Build đổi config (Phần 3):** diff phải cho thấy **đúng symbol bạn định đổi** (cộng
  các dòng compiler), không hơn. Đây chính là cách tự xác nhận thay đổi đã ăn.

### 4.2 Module đang nạp có bị thiếu cái nào trong bản build không?

```bash
ssh machineai-gw 'lsmod | awk "NR>1{print \$1}"' | sort > /tmp/loaded.txt
find "$STAGE" -name '*.ko' -printf '%f\n' | sed 's/\.ko$//' | tr - _ | sort -u > /tmp/built.txt
comm -23 /tmp/loaded.txt /tmp/built.txt      # PHẢI RỖNG
```

Kết quả tham chiếu: **158 module đang nạp, 0 thiếu.** Quan trọng nhất là **`r8168`**
(driver NIC có dây) — thiếu nó là **mất mạng và board thì ở xa**.

---

## Phần 5 — Copy artifact sang board

```bash
scp /home/machineai/jetson-build/Image-$KREL              machineai-gw:/tmp/
scp /home/machineai/jetson-build/modules-$KREL.tar.gz     machineai-gw:/tmp/
scp /home/machineai/workspaces/AI/esp32-ai-main/tools/jetson/remote-install.sh machineai-gw:/tmp/
```

---

## Phần 6 — Deploy trên board (có đường lùi)

`remote-install.sh` làm: backup → giải nén module + `depmod` → copy Image + **sinh initrd
riêng** → thêm `LABEL custom` vào extlinux **và giữ nguyên `primary`**.

```bash
ssh machineai-gw
sudo bash /tmp/remote-install.sh 5.15.148-tegra-custom1     # = $KREL
```

### 6.1 Vì sao phải có initrd RIÊNG (nếu quên = không boot)

```
CONFIG_BLK_DEV_NVME=m       ← nvme là MODULE
root=/dev/nvme0n1p1         ← mà root nằm trên NVMe
```

initrd cũ chỉ chứa module cho `5.15.148-tegra`. Kernel mới dùng nó ⇒ không nạp được
`nvme.ko` ⇒ **không mount được root**. Script đã tự chạy `update-initramfs -c -k $KREL`.

### 6.2 Cái bẫy `root=root=` — sửa CHỈ trong entry custom

`cat /proc/cmdline` của board này có lỗi đánh máy **hai chữ `root=`**:

```
root=root=/dev/nvme0n1p1 rw rootwait ...
```

initrd của NVIDIA chịu được (regex quét chuỗi con); initrd chuẩn Ubuntu (`update-initramfs`
sinh ra) thì **không** ⇒ rơi vào initramfs shell. Kiểm entry `custom` sau khi cài, nếu
`APPEND` có `root=root=` thì sửa thành một `root=`, **giữ `primary` nguyên si**:

```bash
sudo nano /boot/extlinux/extlinux.conf
#   LABEL primary  APPEND ... root=root=/dev/nvme0n1p1 ...   ← ĐỪNG ĐỘNG
#   LABEL custom   APPEND ... root=/dev/nvme0n1p1 ...        ← sửa còn 1 root=
```

Xác nhận extlinux đang trỏ đúng:

```bash
grep -vE '^\s*#|^\s*$' /boot/extlinux/extlinux.conf   # DEFAULT custom + có LABEL primary
```

> **`TIMEOUT 30` trong extlinux = 3.0 giây** (đơn vị 1/10 s), không phải 30 giây.

---

## Phần 7 — Reboot và xác nhận

```bash
sudo reboot
# đợi ~30 s rồi:
ssh machineai-gw 'uname -r'          # kỳ vọng: 5.15.148-tegra-custom1
ssh machineai-gw 'cat /proc/sys/kernel/tainted'   # 4096 = TAINT_OOT_MODULE, giống bản NVIDIA
ssh machineai-gw 'dmesg | grep -iE "error|fail" | head'
```

**Nếu KHÔNG SSH lại được sau ~2 phút** → board có thể kẹt ở initramfs. Đường lùi: qua màn
hình/serial console, ở menu boot chọn **`primary kernel`** (TIMEOUT 3 s — canh sẵn), hoặc
xem Phần 9.

---

## Phần 8 — Đo đúng cách (nếu không, số vô nghĩa)

Ba thủ phạm kinh điển ([`docs/09`](../../docs/09-so-do-phan-cung.md)): sai power mode,
clock chưa ghim, có tải nền. **Luôn** làm trước mỗi lần đo:

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks            # MAXN_SUPER + ghim clock
sudo jetson_clocks --show | grep -E '^GPU|^EMC'     # XÁC NHẬN clock đã ghim THẬT
tegrastats --interval 1000                          # XÁC NHẬN máy rảnh
```

Rồi chạy benchmark **3 lần, lấy median**, và **lấy mẫu clock TRONG lúc đo** (một số bất
ngờ gần như luôn là lỗi đo cho tới khi chứng minh ngược lại — vd −35% RT hoá ra do
`jetson_clocks` chưa chạy):

```bash
# độ trễ scheduler (cần sudo):
sudo SECS=30 bash /tmp/latency-bench.sh    # cột Max là con số quyết định
# đường camera:
FRAMES=570 REAL=19.02 bash /tmp/cam-bench.sh /path/cam_front.mp4
```

---

## Phần 9 — Rollback (một dòng)

Về kernel gốc mà không xoá gì (vẫn có thể thử lại sau):

```bash
ssh machineai-gw
sudo sed -i 's/^DEFAULT .*/DEFAULT primary/' /boot/extlinux/extlinux.conf
sudo reboot
# xác nhận:  uname -r  ->  5.15.148-tegra
```

Gỡ hẳn biến thể (khi chắc chắn không cần):

```bash
sudo rm -rf /lib/modules/5.15.148-tegra-custom1 \
            /boot/Image-5.15.148-tegra-custom1 \
            /boot/initrd.img-5.15.148-tegra-custom1
# rồi xoá LABEL custom trong extlinux.conf, đặt lại DEFAULT primary
```

---

## Checklist rút gọn (in ra dán tường)

```
HOST
  [ ] apt: gcc-aarch64-linux-gnu build-essential bc bison flex libssl-dev rsync
  [ ] nguồn v4.4 (KHÔNG v4.5+ = 404), giải nén 3 tarball
  [ ] LV=-tegra-custom1 ./build-kernel.sh           (kernel ~7m, modules ~2m)
  [ ] (đổi config? -> LV MỚI, build in-tree + OOT modules lại)
KIỂM TRA — KHÔNG BỎ
  [ ] diff .config vs /proc/config.gz  -> chỉ khác compiler (+ symbol cố ý đổi)
  [ ] comm -23 loaded built            -> RỖNG (đặc biệt r8168)
DEPLOY
  [ ] scp Image + modules.tar.gz + remote-install.sh -> /tmp
  [ ] sudo bash remote-install.sh $KREL
  [ ] sửa root=root= CHỈ trong LABEL custom, giữ primary
  [ ] grep extlinux: DEFAULT custom, primary còn nguyên
BOOT
  [ ] reboot; uname -r == $KREL; tainted == 4096; dmesg sạch
  [ ] không SSH lại được -> menu boot chọn primary (TIMEOUT 3s)
ĐO
  [ ] nvpmodel -m 2 && jetson_clocks; XÁC NHẬN clock ghim; máy rảnh; 3 lần median
ROLLBACK
  [ ] sed DEFAULT primary; reboot; uname -r == 5.15.148-tegra
```

---

## Bảng sự cố thường gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `public_sources.tbz2` 404 | version chưa publish (v4.5+) | Dùng **v4.4** |
| Board không boot, kẹt initramfs shell | `root=root=` + initrd Ubuntu | Menu boot chọn `primary`; sửa `root=` trong LABEL custom |
| Boot xong mất mạng | thiếu `r8168.ko` trong bản build | Kiểm tra 4.2 phải RỖNG *trước* khi deploy |
| `apt` không phân giải tên | Tailscale chiếm DNS | `resolvectl dns/domain` (Phần 0.2) |
| Build RT: `Failed PREEMPT_RT sanity check` | driver display NVIDIA từ chối | `export IGNORE_PREEMPT_RT_PRESENCE=1` — nhưng đọc §15.6, đừng deploy |
| Số đo tụt bất thường (vd −35%) | `jetson_clocks` chưa chạy / clock chưa ghim | Phần 8; lấy mẫu clock trong lúc đo |
| diff config nhiều dòng lạ | dùng wrapper `version=` sinh lại defconfig, đè `.config` | Build in-tree trực tiếp như §3c |

---

→ Chi tiết + kết quả đo: [`docs/15-kernel-den-camera.md`](../../docs/15-kernel-den-camera.md) ·
Điều kiện đo: [`docs/09-so-do-phan-cung.md`](../../docs/09-so-do-phan-cung.md) ·
Script: [`README.md`](README.md)
