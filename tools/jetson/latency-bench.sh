#!/usr/bin/env bash
# Chạy TRÊN BOARD (cần sudo cho cyclictest -m/-p). Đo ĐỘ TRỄ của scheduler,
# thứ quyết định một hệ thống an toàn có kịp phản ứng hay không.
#
# cyclictest: đặt hẹn giờ mỗi 200us, đo xem thực tế bị đánh thức TRỄ bao nhiêu.
#   max  = trường hợp xấu nhất  <- con số DUY NHẤT quan trọng với hệ thống thời gian thực
#   avg  = trung bình           <- gần như luôn đẹp, đừng để nó đánh lừa
#
# Hai kịch bản, vì độ trễ chỉ lộ ra khi máy BẬN:
#   1. máy rảnh
#   2. máy bị ép tải (stress-ng: CPU + bộ nhớ + I/O), giống lúc xe nâng chạy thật
set -uo pipefail
SECS="${SECS:-30}"
N="${N:-$((SECS*5000))}"     # chu kỳ 200us => 5000 mẫu/giây

echo "kernel: $(uname -r)   PREEMPT_RT: $(grep -q PREEMPT_RT /sys/kernel/realtime 2>/dev/null && echo yes || (uname -v | grep -q PREEMPT_RT && echo yes || echo no))"
echo "power : $(nvpmodel -q 2>/dev/null | head -2 | tail -1)"
echo

echo "--- [1] máy rảnh, $SECS giây ---"
cyclictest -m -p 80 -t 4 -i 200 -l "$N" -q 2>/dev/null | tail -6

echo
echo "--- [2] dưới tải nặng (stress-ng 6 cpu + 2 vm + io), $SECS giây ---"
stress-ng --cpu 6 --vm 2 --vm-bytes 256M --io 2 --timeout $((SECS+8))s >/dev/null 2>&1 &
SPID=$!
sleep 3
cyclictest -m -p 80 -t 4 -i 200 -l "$N" -q 2>/dev/null | tail -6
wait $SPID 2>/dev/null
echo
echo "(Min/Avg/Max theo micro giây. Cột Max là cái quyết định.)"
