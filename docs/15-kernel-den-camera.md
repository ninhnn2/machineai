# 15. Từ kernel tới camera pipeline — làm thật trên Jetson Orin Nano Super

Ba file trước đi từ **thuật toán** ([03](03-roofline.md)) tới **kernel CUDA**
([11](11-toi-uu-nvidia.md)) tới **framework** ([13](13-jetson-framework.md),
[14](14-tensorrt-deepstream.md)). File này đi xuống tầng cuối cùng: **kernel Linux và
driver**, rồi đi ngược lên tới **đường video**.

Mọi thứ dưới đây **đã làm thật** trên board `machineai-gw` ngày 2026-08-05:
tự build kernel, deploy, boot, đo, so sánh, rồi trả board về nguyên trạng. Board này
đang chạy một hệ thống an toàn xe nâng thật (YOLOv8n TensorRT, 2 camera RTSP), nên mọi
bước đều phải có đường lùi.

```
Board : Jetson Orin Nano Engineering Reference Dev Kit Super (3767-300-0005)
L4T   : R36.4.7 (JetPack 6.2.x) | kernel 5.15.148-tegra | CUDA 12.6.68 | TRT 10.3.0.30
Host  : x86_64, 32 core, aarch64-linux-gnu-gcc 11.4 (Ubuntu)
```

---

## 15.1 Vì sao phải tự build kernel

Ba lý do thật, không phải để nghịch:

1. **Thêm driver.** Cảm biến CSI mới, thiết bị CAN/SPI lạ, module vá lỗi → cần kernel
   có đúng `CONFIG_*` hoặc cần build module ngoài cây nguồn.
2. **Đổi hành vi thời gian thực.** `PREEMPT` → `PREEMPT_RT`, `CONFIG_HZ`, tickless,
   isolcpus. Đây là thứ quyết định một hệ thống an toàn có kịp phản ứng không.
3. **Hiểu cái mình đang chạy.** Không đọc được `.config` thì mọi lý luận về hiệu năng
   đều là đoán.

Và một lý do phản diện quan trọng: **phần lớn thời gian, câu trả lời là ĐỪNG build.**
Phần 15.6 cho thấy một biến thể kernel nghe rất hợp lý lại làm hỏng 40% băng thông.

---

## 15.2 Lấy đúng nguồn — bẫy đầu tiên

```bash
# đường dẫn theo phiên bản; KHÔNG phải phiên bản nào cũng được publish
https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.<X>/sources/public_sources.tbz2
```

Kết quả dò thật (2026-08-05):

| version | HTTP |
|---|---|
| v4.7 (đúng bản board đang chạy) | **404 — NVIDIA chưa publish** |
| v4.6, v4.5 | 404 |
| **v4.4** | **200, 226 MB** |
| v4.3 | 200 |

**Quyết định:** dùng v4.4. Cơ sở: `kernel-jammy-src/Makefile` cho `VERSION=5 PATCHLEVEL=15
SUBLEVEL=148` — **cùng đúng kernel 5.15.148 mà board đang chạy**. Và vì ta build *cả*
kernel *lẫn* toàn bộ OOT modules rồi boot chúng cùng nhau, nội bộ vẫn nhất quán.

Giải nén ra 8 thư mục — nhớ cấu trúc này, JetPack 6 khác hẳn JetPack 5:

```
Linux_for_Tegra/source/
  kernel/kernel-jammy-src/   kernel Linux (in-tree)
  nvgpu/                     driver GPU        ┐
  nvidia-oot/                driver Tegra      │ TẤT CẢ là out-of-tree module,
  nvdisplay/                 driver hiển thị   │ build RIÊNG sau khi có kernel
  hwpm/  nvethernetrm/       counter, ethernet ┘
  kernel-devicetree/         DTB
  nvbuild.sh  Makefile       script build của NVIDIA
```

> **Điểm khác biệt lớn nhất của JetPack 6:** `KERNEL_VARIANT: oot` trong
> `/etc/nv_tegra_release`. Driver NVIDIA **không nằm trong cây kernel nữa**. Build xong
> kernel mới là được nửa việc; nửa còn lại là build lại toàn bộ module NVIDIA đối với
> kernel đó.

---

## 15.3 Build — và cách chứng minh bản build của bạn đúng

Chuỗi lệnh cốt lõi (script đầy đủ ở cuối bài):

```bash
export ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
make -C $OUT/kernel version=-tegra-custom1        # Image + dtbs + modules in-tree
export KERNEL_HEADERS=$OUT/kernel/kernel-jammy-src
make -C $OUT modules -j32                          # nvgpu, nvidia-oot, nvdisplay...
```

`version=` chính là `LOCALVERSION` — NVIDIA truyền `-tegra` ở đó
(`kernel/Makefile`). Đặt hậu tố **riêng** cho mỗi biến thể là quy tắc an toàn số 1:

```
kernel gốc     5.15.148-tegra              /lib/modules/5.15.148-tegra/
bản của tôi    5.15.148-tegra-custom1      /lib/modules/5.15.148-tegra-custom1/
bản RT         5.15.148-rt-tegra-custom1   /lib/modules/5.15.148-rt-tegra-custom1/
```

Ba thư mục module tách biệt ⇒ **kernel gốc không bao giờ bị đụng vào** ⇒ rollback là
đổi một dòng trong `extlinux.conf`.

### Hai phép kiểm tra BẮT BUỘC trước khi deploy

Đây là phần phân biệt "build được" với "build đúng".

**(1) Config của tôi có khớp kernel đang chạy không?**

```bash
ssh board 'zcat /proc/config.gz' > board-config-running
diff <(sort board-config-running) <(sort $KSRC/.config)
```

Kết quả thật: **14 dòng khác, và cả 14 đều do compiler**, không có lấy một lựa chọn
chính sách nào:

```
< CONFIG_CC_VERSION_TEXT="aarch64-buildroot-linux-gnu-gcc ... 11.3.0"   (NVIDIA)
> CONFIG_CC_VERSION_TEXT="aarch64-linux-gnu-gcc (Ubuntu ...) 11.4.0"    (của tôi)
< CONFIG_GCC_PLUGINS=y          > (không có)
< CONFIG_PAHOLE_VERSION=0       > CONFIG_PAHOLE_VERSION=125
```

Từ đây trở đi, mọi khác biệt hiệu năng đo được **không thể** đổ cho "chắc là do config
khác". Đây chính là nguyên tắc golden reference của
[`DEPLOY.md §4.1`](../DEPLOY.md) áp cho kernel.

**(2) Module đang nạp có bị thiếu cái nào trong bản build không?**

```bash
ssh board 'lsmod | awk "NR>1{print \$1}"' | sort > loaded.txt
find $STAGE -name '*.ko' -printf '%f\n' | sed 's/\.ko$//' | tr - _ | sort -u > built.txt
comm -23 loaded.txt built.txt        # phải RỖNG
```

Kết quả: **158 module đang nạp, 0 thiếu** (1076 module build ra). Quan trọng nhất là
`r8168` — driver NIC có dây. Thiếu nó là mất mạng, và board thì ở xa.

---

## 15.4 Deploy có đường lùi — và cái bẫy suýt làm board không boot

### Bốn quy tắc

1. **Không ghi đè `/boot/Image`.** Kernel mới nằm ở file riêng.
2. **Module vào thư mục riêng** theo `LOCALVERSION`.
3. **initrd riêng cho kernel mới.** Bắt buộc, vì:
   ```
   CONFIG_BLK_DEV_NVME=m       <- nvme là MODULE
   root=/dev/nvme0n1p1         <- mà root nằm trên NVMe
   ```
   initrd cũ chỉ chứa module cho `5.15.148-tegra`. Kernel mới dùng nó ⇒ không nạp được
   `nvme.ko` ⇒ **không mount được root ⇒ không boot**.
4. **Giữ nguyên entry cũ làm fallback.**

```ini
LABEL primary                      # NGUYÊN VẸN, không đụng
      LINUX /boot/Image
      INITRD /boot/initrd
LABEL custom                       # thêm mới
      LINUX /boot/Image-5.15.148-tegra-custom1
      INITRD /boot/initrd.img-5.15.148-tegra-custom1
DEFAULT custom
```

### Cái bẫy: `root=root=/dev/nvme0n1p1`

Dòng `APPEND` của board có lỗi đánh máy **hai chữ `root=`**:

```
$ cat /proc/cmdline
root=root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 ...
```

Board vẫn boot suốt bao lâu nay. Vì sao? Mở initrd của L4T ra xem:

```bash
$ grep -n 'root=' init
240: dev_regex='root=\/dev\/[abcdefklmnpsv0-9]*'
246: rootdev=$(echo "${rootdev}" | sed -ne "s/root=\(.*\)/\1/p")
```

Regex của NVIDIA **quét tìm chuỗi con** `root=/dev/...`, mà `root=root=/dev/nvme0n1p1`
có chứa đúng chuỗi đó. **Nó chịu được lỗi này một cách tình cờ.**

initrd chuẩn của Ubuntu (`update-initramfs` sinh ra) thì parse `root=` theo kiểu
`ROOT=${x#root=}` → được `root=/dev/nvme0n1p1` → mount thất bại → **rơi vào initramfs
shell, không boot**.

Cách xử lý đúng: **chỉ sửa trong entry `custom`**, giữ `primary` nguyên si:

```
LABEL primary  APPEND ... root=root=/dev/nvme0n1p1 ...   # nguyên trạng
LABEL custom   APPEND ... root=/dev/nvme0n1p1 ...        # đã sửa
```

> **Bài học tổng quát:** một lỗi cấu hình có thể sống nhiều năm chỉ vì *một* thành phần
> tình cờ khoan dung với nó. Thay thành phần đó ra là lỗi lộ nguyên hình. Trước khi
> đổi bất kỳ mắt xích nào, hãy đọc xem mắt xích cũ đang *thực sự* làm gì.

Và một chi tiết cần biết: **`TIMEOUT 30` trong extlinux là 3.0 giây**, không phải 30
giây — đơn vị là 1/10 giây. Board thật boot lại chỉ mất ~30 s tổng.

---

## 15.5 Kết quả 1 — kernel tự build có bằng kernel NVIDIA không?

Điều kiện hai lần đo **giống hệt**: MAXN_SUPER, `jetson_clocks`, tạm dừng
`forklift_demo`, buffer 256 MB, 3 lần liên tiếp.

| phép đo | 5.15.148-tegra (NVIDIA) | 5.15.148-tegra-custom1 (tự build) | chênh |
|---|---:|---:|---:|
| read scalar | 57.4 / 57.4 / 57.4 | 57.4 / 57.4 / 57.5 | 0% |
| **read float4** | **99.2 / 99.2 / 99.2** | **98.7 / 99.1 / 98.3** | −0.6% |
| triad float4 | 96.1 / 96.2 / 96.1 | 96.2 / 96.2 / 93.9 | trong nhiễu |
| FP16 tensor core | 11.66 TFLOP/s | 11.50 TFLOP/s | −1.4% |
| INT8 tensor core | 15.26 TOPS | 15.32 TOPS | +0.4% |
| module nạp được | 158 | 155 (lúc mới boot) | — |
| `/proc/sys/kernel/tainted` | 4096 | 4096 | giống hệt |

**Bằng nhau trong sai số. Đó chính là kết quả cần có.** Nó không "chán" — nó là bằng
chứng rằng toolchain, config và quy trình deploy của bạn tái tạo được bản gốc. Từ giờ,
đổi **một** thứ trong config thì chênh lệch đo được thuộc về thứ đó.

`tainted=4096` = `TAINT_OOT_MODULE`, giống hệt bản NVIDIA — vì driver NVIDIA cũng là
out-of-tree.

> Đúng nguyên tắc của [`08-nhat-ky-toi-uu.md §5.5`](08-nhat-ky-toi-uu.md): dựng mốc đối
> chứng trước, đổi biến sau.

---

## 15.6 Kết quả 2 — PREEMPT_RT: thí nghiệm cho ra kết quả ngược

Giả thuyết ban đầu rất hợp lý: đây là hệ thống an toàn, đổi sang `PREEMPT_RT` sẽ giảm
độ trễ xấu nhất, đổi lại mất ít throughput. **Cả hai vế đều sai trên board này.**

### Build: NVIDIA chặn ngay từ đầu — và đó là cảnh báo

```bash
./generic_rt_build.sh enable      # script có sẵn trong source của NVIDIA
LV=-rt-tegra-custom1 ./build-kernel.sh
```

```
The kernel you are installing for is a PREEMPT_RT kernel!
*** Failed PREEMPT_RT sanity check. Bailing out! ***
make: *** [Kbuild:348: preempt_rt_sanity_check] Error 1
```

Driver **display** của NVIDIA từ chối biên dịch. Vượt qua được bằng cờ mà chính
`nvbuild.sh` dùng:

```bash
export IGNORE_PREEMPT_RT_PRESENCE=1
```

Tên biến nói đúng bản chất: bạn đang **bỏ qua** một cảnh báo, không phải bật một tính
năng. Hãy nhớ điều này khi đọc kết quả bên dưới.

### Đo độ trễ — `cyclictest`, 150.000 mẫu, chu kỳ 200 µs

| kịch bản | PREEMPT (thường) | **PREEMPT_RT** |
|---|---:|---:|
| máy rảnh — Max | **148 – 186 µs** | **1810 – 2183 µs** |
| máy rảnh — Avg | 3 – 6 µs | 8 – 10 µs |
| tải nặng — Max | 155 – 294 µs | 189 – 397 µs |
| tải nặng — Avg | 7 – 8 µs | 7 – 8 µs |

Lặp lại 2 lần: 2183 µs rồi 1810 µs — **tái lập được, không phải nhiễu lúc mới boot.**
Ghim vào CPU0 và CPU5 đều ra spike (~2098 µs / ~1968 µs) ⇒ **không phải do IRQ affinity
mà là hiện tượng toàn hệ thống.**

### Đo throughput — cùng clock đã ghim

| phép đo | PREEMPT | PREEMPT_RT | chênh |
|---|---:|---:|---:|
| read float4 | 99.2 GB/s | 59.4 GB/s | **−40%** |
| read scalar | 57.4 GB/s | 33.6 GB/s | −41% |
| triad float4 | 96.1 GB/s | 63.9 GB/s | −33% |
| FP16 tensor core | 11.66 TFLOP/s | 7.64 TFLOP/s | **−34%** |
| INT8 tensor core | 15.26 TOPS | 10.54 TOPS | −31% |

### Suýt kết luận sai — và cách tránh

Lần đo RT đầu tiên cho −35%. Trước khi viết kết luận, kiểm điều kiện:

```
GPU MinFreq=306000000  CurrentFreq=306000000        <- GPU đang ở clock TỐI THIỂU
EMC CurrentFreq=2133000000 FreqOverride=0           <- EMC chưa bị ghim
```

`jetson_clocks` chưa được chạy sau lần reboot đó. Số đo **vô hiệu**. Chạy lại sau khi
ghim clock — và lần này lấy mẫu clock **trong lúc** bench để chắc chắn:

```
emc=3199 MHz gpu=1020 MHz   (×6 mẫu, suốt thời gian đo)
```

Vẫn −40%. Lúc đó kết luận mới có giá trị.

> Đây đúng là ba thủ phạm của [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md) tái xuất.
> **Quy tắc: kiểm điều kiện trước, kết luận sau. Một con số bất ngờ gần như luôn là lỗi
> đo, cho tới khi chứng minh được ngược lại.**

### Kết luận

**Trên Jetson Orin Nano + stack driver NVIDIA, PREEMPT_RT mất 30–40% throughput và
KHÔNG cải thiện độ trễ xấu nhất — thực tế còn tệ hơn ~10× khi máy rảnh.**

Giả thuyết (chưa chứng minh, cần `ftrace` để khẳng định): driver GPU/display của NVIDIA
là out-of-tree và không RT-clean — trên RT, spinlock thành mutex ngủ được và IRQ thành
thread, nên những vùng không-preempt dài trong driver biến thành spike. `preempt_rt_
sanity_check` tồn tại chính vì lý do đó.

**Hệ quả thực dụng cho hệ thống an toàn xe nâng này:** đừng dùng RT. Muốn giảm jitter
thì đi hướng khác — `isolcpus` + `SCHED_FIFO` cho luồng quyết định, ghim IRQ, tắt
cpuidle sâu, và quan trọng nhất là **giảm độ trễ ở tầng đường ống video** (15.7).

---

## 15.7 Kết quả 3 — đường camera: NVDEC, NVMM và cái giá của một lần copy

Board này chạy 2 camera RTSP (1280×720 và 2560×1440) qua đường:

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv → BGR → appsink → YOLOv8n TRT
                                       (NVDEC)      (đổi cỡ trên GPU)
```

Đo trên **video thật của chính camera đó** (`samples/cam_front.mp4`, 570 khung,
1280×720 H.264), giải mã hết file với `sync=false`, MAXN_SUPER:

| đường | wall | FPS | so thời gian thực | **CPU** |
|---|---:|---:|---:|---:|
| **A.** NVDEC → ở lại NVMM | 1.39 s | 408.9 | 13.6× | **0.27 core** |
| **D.** NVDEC → `nvvidconv` **giữ NVMM** | 1.42 s | 402.5 | 13.4× | **0.27 core** |
| **B.** NVDEC → `nvvidconv` **kéo về CPU** | 1.60 s | 357.1 | 11.9× | **0.64 core** |
| **C.** giải mã bằng CPU (`avdec_h264`) | 1.78 s | 319.3 | 10.7× | **3.96 core** |

**Ba kết luận, mỗi cái tách được đúng một biến:**

1. **A vs D — `nvvidconv` gần như miễn phí** (402 vs 409 FPS, cùng 0.27 core). Người ta
   hay đổ tội cho bộ chuyển đổi. Sai.
2. **D vs B — thủ phạm là *rời khỏi NVMM***, không phải bộ chuyển đổi. Cùng một plugin,
   chỉ khác caps đầu ra: `video/x-raw(memory:NVMM)` → `video/x-raw`. Giá: **CPU tăng
   2.4×** (0.27 → 0.64 core) và throughput giảm 11%.
3. **A vs C — NVDEC đáng giá 14.7× CPU** (0.27 vs 3.96 core). Trên board 6 core, đường
   CPU đốt gần 4 core chỉ để giải mã *một* luồng 720p. Hai camera là hết máy.

Điều này **sửa lại** cách nói ở [`14-tensorrt-deepstream.md §14.5`](14-tensorrt-deepstream.md):
lỗi không phải là "dùng `videoconvert` thay vì `nvvideoconvert`" mà là **để buffer rơi
khỏi NVMM**. Và trên L4T thuần (không cài DeepStream) plugin tên là **`nvvidconv`** —
`nvvideoconvert` chỉ có khi có DeepStream.

### Ảnh hưởng của power mode lên đường video

| | 15W (mode 0) | MAXN_SUPER (mode 2) |
|---|---:|---:|
| A. NVDEC → NVMM | 402.0 FPS / 0.40 core | 408.9 FPS / **0.27 core** |
| C. CPU decode | 275.0 FPS / 3.93 core | 319.3 FPS / 3.96 core |

**NVDEC gần như không quan tâm power mode** (+1.7%) vì nó là khối phần cứng chuyên
dụng chạy clock riêng — nhưng **CPU tiêu thụ giảm 33%** vì CPU chạy nhanh hơn thì xong
việc sớm hơn. Đường CPU decode thì được +16% throughput. Nghĩa là: **chuyển sang đường
phần cứng có lợi hơn nhiều so với tăng power mode**, và hai thứ đó không thay thế nhau.

### Xác nhận lại: Orin Nano không có bộ mã hoá

```bash
$ ls /dev | grep -E 'nvdec|msenc'
v4l2-nvdec                      # có bộ GIẢI mã
$ gst-inspect-1.0 nvv4l2h264enc
(không có)                      # KHÔNG có bộ MÃ HOÁ
```

Xác nhận trên phần cứng thật điều [`14 §14.7`](14-tensorrt-deepstream.md) cảnh báo.
Thiết kế hệ thống phải theo hướng **gửi metadata, đừng gửi video**.

---

## 15.8 Phát hiện phụ — hai thứ tưởng hỏng mà chỉ là cấu hình

**(1) "Board không có internet" → thật ra chỉ hỏng DNS.**

[`09-so-do-phan-cung.md`](09-so-do-phan-cung.md) ghi board không ra được ngoài. Kiểm lại:

```bash
$ ping -c2 8.8.8.8        →  0% packet loss, 30 ms      # ĐI ĐƯỢC
$ curl -sI https://github.com  →  000                    # nhưng không phân giải được tên
$ resolvectl status
Link 4 (enP8p1s0)
  Current Scopes: none                                   # <- KHÔNG có DNS scope nào
  Protocols: -DefaultRoute ...
```

Tailscale đã chiếm quyền DNS và link Ethernet mất `DefaultRoute`. Sửa runtime:

```bash
sudo resolvectl dns enP8p1s0 1.1.1.1 8.8.8.8
sudo resolvectl domain enP8p1s0 '~.'
```

→ internet hoạt động ngay, `apt-get install rt-tests stress-ng` chạy được.
**"Không có mạng" và "không phân giải được tên" là hai bệnh khác nhau. `ping 8.8.8.8`
tách được chúng trong 2 giây.**

**(2) EMC đã hết bị khoá — trần bandwidth tăng 1.49×.**
Xem [`09 §Phát hiện 5, phần CẬP NHẬT`](09-so-do-phan-cung.md): 66.8 → **99.2 GB/s**.
Trần decode LLM phải tính lại: `tok/s = 99.2 / model_GB`.

---

## 15.9 Quy trình rút gọn — dùng lại được

Bốn script đã dùng nằm ở [`tools/jetson/`](../tools/jetson/) (`build-kernel.sh`,
`remote-install.sh`, `cam-bench.sh`, `latency-bench.sh`).

```bash
# 1. NGUỒN
curl -LO https://developer.download.nvidia.com/embedded/L4T/r36_Release_v4.4/sources/public_sources.tbz2
tar xf public_sources.tbz2 && cd Linux_for_Tegra/source
tar xf kernel_src.tbz2 && tar xf kernel_oot_modules_src.tbz2
tar xf nvidia_kernel_display_driver_source.tbz2

# 2. BUILD (x86, cross)
export ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
make -C $OUT/kernel version=-tegra-custom1                    # 7m28s / 32 core
export KERNEL_HEADERS=$OUT/kernel/kernel-jammy-src
make -C $OUT modules -j$(nproc)                               # 1m52s
make -C $KSRC INSTALL_MOD_PATH=$STAGE INSTALL_MOD_STRIP=1 modules_install
make -C $OUT  INSTALL_MOD_PATH=$STAGE INSTALL_MOD_DIR=updates INSTALL_MOD_STRIP=1 modules_install

# 3. KIỂM TRA TRƯỚC KHI DEPLOY  <- đừng bỏ
diff <(sort board-config-running) <(sort $KSRC/.config)       # chỉ được khác về compiler
comm -23 loaded.txt built.txt                                 # phải rỗng

# 4. DEPLOY (trên board, sudo)
tar -C /lib/modules -xzf modules-$KREL.tar.gz && depmod -a $KREL
cp Image-$KREL /boot/ && update-initramfs -c -k $KREL
#   thêm LABEL custom vào extlinux.conf, GIỮ NGUYÊN primary, DEFAULT custom

# 5. ĐO — mỗi lần đo đều phải:
sudo nvpmodel -m 2 && sudo jetson_clocks && nvpmodel -q       # đúng power mode
sudo jetson_clocks --show | grep -E '^GPU|^EMC'               # XÁC NHẬN clock đã ghim
tegrastats --interval 1000                                    # xác nhận máy rảnh
#   rồi mới chạy benchmark, 3 lần, lấy median

# 6. ROLLBACK
sudo sed -i 's/^DEFAULT .*/DEFAULT primary/' /boot/extlinux/extlinux.conf && sudo reboot
```

Thời gian thật: build 9m20s, deploy 2 phút, reboot 30 s. Một vòng thí nghiệm đầy đủ
(build → deploy → boot → đo → kết luận) mất khoảng **25 phút**.

---

## 15.10 Bảy bài học

1. **Dựng mốc đối chứng trước.** Build lại đúng kernel gốc và chứng minh nó bằng nhau
   trong sai số. Không có bước này thì mọi thí nghiệm sau đều vô nghĩa.
2. **Mỗi biến thể một `LOCALVERSION`.** Rollback thành một dòng thay vì một chuyến đi
   tới chỗ đặt board.
3. **Đọc xem mắt xích cũ đang thực sự làm gì** trước khi thay nó. `root=root=` sống sót
   nhiều năm nhờ một regex khoan dung.
4. **Số bất ngờ = nghi lỗi đo trước.** −35% hoá ra là `jetson_clocks` chưa chạy. Lấy mẫu
   clock *trong lúc* đo, đừng chỉ đo trước/sau.
5. **Cảnh báo của nhà cung cấp là dữ liệu.** `IGNORE_PREEMPT_RT_PRESENCE` đúng nghĩa
   đen là "bỏ qua" — và số đo cho thấy vì sao nó tồn tại.
6. **Tách đúng một biến.** A vs D vs B tách được `nvvidconv` khỏi "rời NVMM" — hai thứ
   luôn bị gộp làm một trong các hướng dẫn trên mạng.
7. **Kết quả ngược kỳ vọng là kết quả có giá trị nhất.** PREEMPT_RT tệ hơn ở *cả hai*
   mặt là thứ không thể tra ra từ tài liệu, chỉ đo mới biết.

---

## 15.11 Bài tập

1. Build lại kernel gốc trên máy bạn, chạy hai phép kiểm tra ở 15.3. Bao nhiêu dòng
   config khác? Có dòng nào **không** phải do compiler không?
2. Đổi **một** thứ: `CONFIG_HZ_250` → `CONFIG_HZ_1000`. Đo lại cyclictest **và**
   `bench_roofline`. Cái nào đổi, cái nào không, vì sao?
3. `isolcpus=4,5` + `taskset` + `chrt -f 80` cho một luồng bận: max latency của luồng đó
   còn bao nhiêu? So với PREEMPT_RT — cách nào rẻ hơn cho cùng mục tiêu?
4. Dựng lại bảng 15.7 trên luồng 2560×1440. Tỉ lệ CPU giữa đường NVDEC và đường CPU
   thay đổi thế nào theo độ phân giải, và vì sao?
5. Thêm `nvvidconv` **resize xuống 640×360** trong khi vẫn ở NVMM. Nó có làm chậm không?
   Đối chiếu với luận điểm của `forklift_demo` là "cv::resize của luồng chính bị bỏ qua".
6. Chạy `ftrace`/`cyclictest --breaktrace=1000` trên kernel RT để tìm thủ phạm thật của
   spike 2 ms. Có đúng là driver NVIDIA không? (Bài này chưa ai làm — làm xong thì cập
   nhật §15.6.)
7. Board có 1 NVDEC. Với 2 camera 1440p@25, NVDEC còn dư bao nhiêu? Đo bằng cách chạy
   N luồng giải mã song song tới khi FPS tụt.

---

→ [README.md](README.md) · Trước: [14-tensorrt-deepstream.md](14-tensorrt-deepstream.md) ·
Số đo board: [09-so-do-phan-cung.md](09-so-do-phan-cung.md)
