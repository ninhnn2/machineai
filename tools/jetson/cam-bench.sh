#!/usr/bin/env bash
# Chạy TRÊN BOARD. Cùng một file H.264 (570 khung, 1280x720, 29.97fps, 19.0s thời gian thực),
# giải mã hết file bằng ba đường khác nhau, sync=false (chạy hết tốc lực).
#
#   A. NVDEC, frame ở lại NVMM       <- đường forklift_demo dùng
#   B. NVDEC rồi nvvidconv kéo về CPU <- "lỗi hiệu năng số 1" trong docs/14
#   C. Giải mã bằng CPU (avdec_h264)  <- không dùng phần cứng
#   D. NVDEC + nvvidconv nhưng GIỮ trong NVMM  <- chứng minh nvvidconv không phải thủ phạm
#
# Đo: wall time, CPU time (user+sys) => FPS và số core CPU bị đốt.
set -uo pipefail
F="${1:?cần file .mp4}"
FRAMES="${FRAMES:-570}"      # 19.019s * 29.97fps
REAL="${REAL:-19.02}"        # thời lượng thực của video

TIMEFORMAT='%R %U %S'

run() {
  local label="$1"; shift
  local t
  t=$( { time gst-launch-1.0 -q "$@" >/dev/null 2>&1; } 2>&1 )
  awk -v l="$label" -v t="$t" -v n="$FRAMES" -v r="$REAL" 'BEGIN{
    split(t,a," "); wall=a[1]; cpu=a[2]+a[3];
    printf "  %-34s  wall %6.2fs   FPS %7.1f   nhanh gấp %5.1fx thời gian thực   CPU %5.2f core\n",
           l, wall, n/wall, r/wall, cpu/wall }'
}

SINK="fpsdisplaysink text-overlay=false video-sink=fakesink sync=false"
D="filesrc location=$F ! qtdemux ! h264parse"

echo "== $(basename "$F") — $FRAMES khung, 1280x720 H.264 =="
run "A. NVDEC -> NVMM"                  $D ! nvv4l2decoder ! $SINK
run "D. NVDEC -> nvvidconv (giữ NVMM)"  $D ! nvv4l2decoder ! nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! $SINK
run "B. NVDEC -> kéo về CPU (BGRx)"     $D ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! $SINK
run "C. CPU decode (avdec_h264)"        $D ! avdec_h264 ! $SINK
