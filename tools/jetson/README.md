# Script build/deploy/đo kernel Jetson

Bốn script đã dùng thật để làm [`docs/15-kernel-den-camera.md`](../../docs/15-kernel-den-camera.md)
trên Jetson Orin Nano Super (L4T R36.4.7). Đọc file 15 trước, đây chỉ là công cụ.

> **Cần làm lại từng bước?** Xem [`RUNBOOK.md`](RUNBOOK.md) — checklist thực thi đầy đủ:
> build → đổi config → kiểm tra bắt buộc → deploy → boot → đo → rollback.

| Script | Chạy ở đâu | Việc |
|---|---|---|
| `build-kernel.sh` | **x86 host** | Cross-compile kernel + toàn bộ OOT modules, đóng gói `Image-*` và `modules-*.tar.gz`. `LV=-hậu-tố` chọn biến thể. |
| `remote-install.sh` | **board (sudo)** | Cài modules + Image + sinh initrd riêng + thêm `LABEL custom` vào extlinux, **giữ nguyên `primary` làm đường lùi**. |
| `cam-bench.sh` | **board** | Đo 4 đường giải mã cùng một file H.264: NVDEC/NVMM, NVDEC+nvvidconv giữ NVMM, kéo về CPU, CPU decode. Báo FPS + số core CPU. |
| `latency-bench.sh` | **board (sudo)** | `cyclictest` lúc rảnh và lúc bị `stress-ng` ép tải. Cột **Max** là con số quyết định. |

```bash
# host
LV=-tegra-thu-nghiem ./build-kernel.sh

# board
sudo bash remote-install.sh 5.15.148-tegra-thu-nghiem
# sửa root= trong entry custom nếu cmdline của bạn cũng có lỗi root=root= (xem docs/15 §15.4)
sudo reboot

# rollback
sudo sed -i 's/^DEFAULT .*/DEFAULT primary/' /boot/extlinux/extlinux.conf && sudo reboot
```

**Trước mỗi lần đo** (nếu không thì số vô nghĩa — xem [`docs/09`](../../docs/09-so-do-phan-cung.md)):

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks
sudo jetson_clocks --show | grep -E '^GPU|^EMC'   # XÁC NHẬN clock đã ghim thật
tegrastats --interval 1000                        # XÁC NHẬN máy rảnh
```
