# 14. TensorRT và DeepStream — hai tầng của stack NVIDIA, từ gốc

[`13-jetson-framework.md`](13-jetson-framework.md) trả lời *chọn framework nào*. File này
đi sâu vào hai thứ đặc trưng nhất của hệ sinh thái NVIDIA, và là thứ người ta thực sự
trả tiền để bạn biết: **TensorRT** (trình biên dịch model) và **DeepStream** (pipeline
video thời gian thực).

Quan hệ giữa chúng — nhớ hình này là hiểu một nửa:

```
DeepStream  ─ pipeline video: decode → batch → suy luận → tracking → hiển thị/gửi đi
    │          (GStreamer + plugin NVIDIA. KHÔNG tự suy luận.)
    ├─ nvinfer ──────────► TensorRT   ─ biên dịch model thành engine, chạy engine
    │                          │        (KHÔNG biết gì về video, chỉ biết tensor)
    │                          └─────► CUDA / cuDNN / kernel  ─ phép toán
    └─ nvv4l2decoder ──────► NVDEC     ─ phần cứng giải mã video, KHÔNG tốn GPU
```

**DeepStream không thay thế TensorRT — nó gọi TensorRT.** Ai nói "chuyển từ TensorRT sang
DeepStream" là đang hiểu sai tầng.

---

# PHẦN A — TensorRT

## 14.1 Gốc: TensorRT thực sự là cái gì

Nó là **trình biên dịch**, không phải thư viện suy luận theo nghĩa thông thường.

```
model.onnx  ──►  [ TensorRT builder ]  ──►  model.engine  ──►  [ runtime ]  ──► kết quả
              (chạy MỘT LẦN, chậm: phút→giờ)              (chạy triệu lần, nhanh)
```

Tách bạch **build-time** và **runtime** là ý tưởng trung tâm. Mọi thứ đắt đỏ — thử
nghiệm thuật toán, chọn kiểu dữ liệu, cấp phát bộ nhớ — bị đẩy hết sang build-time và
đóng băng vào file `.engine`. Runtime chỉ còn việc nạp và chạy.

### Sáu việc builder làm (theo thứ tự)

| # | Bước | Nó làm gì | Bạn thấy nó ở đâu |
|---:|---|---|---|
| 1 | **Parse** | đọc ONNX → graph nội bộ | lỗi "unsupported op" xuất hiện ở đây |
| 2 | **Graph optimization** | bỏ node thừa, gấp hằng số (constant folding), bỏ nhánh chết | log `--verbose` |
| 3 | **Layer fusion** | gộp `Conv+BN+ReLU` → **một** kernel | log in tên layer dạng `a + b + c` |
| 4 | **Precision** | chọn FP32/FP16/INT8 **cho từng layer** | `--fp16`, `--int8`, `--best` |
| 5 | **Tactic autotuning** | thử nhiều thuật toán/tile cho mỗi layer, **đo trên chính máy này**, giữ cái nhanh nhất | log `Tactic: ... Time: ...` |
| 6 | **Serialize** | ghi ra `.engine` gắn với GPU + phiên bản này | file nhị phân |

**Bước 3 là nguồn lợi ích lớn nhất và ít người hiểu.** `Conv → BN → ReLU` chưa fusion
phải ghi tensor trung gian ra DRAM rồi đọc lại hai lần. Với ảnh 640×640×64 fp16 đó là
~50 MB đi-về mỗi lần, ở 66.8 GB/s là ~1.5 ms **hoàn toàn lãng phí**. Fusion xoá luôn
round-trip đó. Đây chính là bản tự động của việc gom kernel mà repo này làm tay bằng
CUDA Graphs ([`11-toi-uu-nvidia.md §11.5`](11-toi-uu-nvidia.md)).

**Bước 5 giải thích ba điều gây bực mình:**
- vì sao build lâu (nó đang *benchmark* thật, không phải "biên dịch" theo nghĩa gcc);
- vì sao engine **không portable** — tactic tốt nhất trên sm_87 không phải tốt nhất trên sm_89;
- vì sao build lại khi đổi JetPack/TensorRT là **bắt buộc**, không phải khuyến nghị.

> **Hệ quả cần thuộc:** `.engine` = (model + TensorRT version + GPU arch + precision +
> shape profile). Đổi bất kỳ thành phần nào → build lại. Đừng bao giờ commit `.engine`
> vào git rồi mong nó chạy ở máy khác.

## 14.2 `trtexec` — công cụ bạn sẽ dùng 90% thời gian

Có sẵn trên board: `/usr/src/tensorrt/bin/trtexec`. Không cần viết code để trả lời
"model này chạy nhanh cỡ nào và có lỗi ở đâu".

```bash
export PATH=/usr/src/tensorrt/bin:$PATH

# 1. Build + đo luôn, FP16
trtexec --onnx=model.onnx --saveEngine=m_fp16.engine --fp16 \
        --memPoolSize=workspace:2048M --timingCacheFile=.tcache

# 2. Chỉ đo engine đã có (đo sạch, không lẫn thời gian build)
trtexec --loadEngine=m_fp16.engine --iterations=200 --avgRuns=100 --warmUp=500

# 3. Xem từng layer tốn bao nhiêu — đây là chỗ ra quyết định
trtexec --loadEngine=m_fp16.engine --dumpProfile --separateProfileRun \
        --exportProfile=prof.json --exportLayerInfo=layers.json

# 4. Shape động (bắt buộc cho model có batch/kích thước thay đổi)
trtexec --onnx=model.onnx --saveEngine=m.engine --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:8x3x640x640

# 5. Gộp launch bằng CUDA Graph (đúng chủ đề của repo này)
trtexec --loadEngine=m.engine --useCudaGraph --iterations=200
```

### Đọc output cho đúng

```
[I] Throughput: 412.3 qps
[I] Latency: min = 2.31 ms, mean = 2.42 ms, median = 2.41 ms, max = 3.90 ms
[I] Enqueue Time: mean = 0.31 ms          ← chi phí CPU đẩy việc xuống GPU
[I] GPU Compute Time: mean = 2.38 ms      ← thời gian GPU thật sự tính
```

| Quan sát | Kết luận | Hành động |
|---|---|---|
| `Enqueue` ≈ `GPU Compute` | **launch-bound** — CPU không đẩy kịp | `--useCudaGraph`, tăng batch |
| `max` ≫ `median` | có gai: throttle, tải nền, hoặc lần chạy đầu | Lab 0 của [13.7](13-jetson-framework.md); tăng `--warmUp` |
| FP16 ≈ FP32 | không layer nào thật sự chạy FP16 | xem `layers.json`, kiểm precision từng layer |
| throughput không tăng theo batch | đã memory-bound | roofline, không phải lỗi TRT |

**`--dumpProfile` là thứ phân biệt người biết việc.** Nó cho biết 3 layer nào ăn 70%
thời gian. Tối ưu bất cứ layer nào khác là lãng phí — đúng nguyên tắc "tính sàn trước
khi tối ưu" của [`08-nhat-ky-toi-uu.md §5.2`](08-nhat-ky-toi-uu.md).

## 14.3 Độ chính xác: FP16 và INT8 làm hỏng gì

FP16 gần như luôn an toàn cho vision (dải động đủ, xem
[`10-ly-thuyet-nen.md §10.1`](10-ly-thuyet-nen.md)). INT8 mới cần làm việc thật.

Hai đường vào INT8, khác nhau về bản chất:

| | **PTQ implicit** (calibration) | **QAT / explicit Q-DQ** |
|---|---|---|
| Cách làm | cho TRT chạy ~500–1000 ảnh đại diện, nó tự đo dải động mỗi tensor | ONNX đã chứa sẵn node QuantizeLinear/DequantizeLinear |
| Công sức | thấp | phải train lại / hiệu chỉnh |
| Chất lượng | thường đủ | tốt hơn, kiểm soát được |
| Ai quyết định scale | TensorRT | bạn |

```bash
trtexec --onnx=m.onnx --int8 --calib=calib.cache --saveEngine=m_int8.engine
trtexec --onnx=m.onnx --best --saveEngine=m_best.engine   # TRT tự trộn FP32/FP16/INT8
```

**Dữ liệu calibration phải giống dữ liệu thật** (cùng camera, cùng ánh sáng, cùng
tiền xử lý). Calibrate bằng ảnh COCO rồi chạy trên camera hồng ngoại ban đêm là hỏng.

### Nghiệm thu — theo hành vi, không theo sai số tuyệt đối

Đây là chỗ lặp lại nguyên tắc đã dùng cho ESP32/CUDA ở
[`../DEPLOY.md §4.1`](../DEPLOY.md): **đừng đòi `max|diff| < 1e-5`**, vì fusion đổi thứ tự
phép cộng và cộng số thực không có tính kết hợp.

| Loại model | Tiêu chí PASS |
|---|---|
| Phân loại | top-1 khớp trên tập kiểm; top-5 accuracy tụt < 0.5% |
| Phát hiện đối tượng | mAP@0.5 tụt < 1%, và **số lượng box** không đổi đột ngột |
| LLM | argmax token đầu khớp; perplexity delta < 0.1 |

```bash
# So thẳng đầu ra TRT với ONNX Runtime trên cùng input — công cụ chuẩn để debug
polygraphy run model.onnx --trt --onnxrt --atol 1e-2 --rtol 1e-2
polygraphy run model.onnx --trt --onnxrt --trt-outputs mark all   # tìm layer đầu tiên lệch
```

Dòng thứ hai là kỹ thuật quan trọng: **bisect theo layer**. Khi engine ra kết quả sai,
đừng đoán — tìm layer *đầu tiên* lệch, gần như luôn là một op bị fallback hoặc một
plugin sai.

## 14.4 Khi TensorRT không nuốt được model

Thứ tự xử lý, từ rẻ tới đắt:

1. **Sửa ONNX export** — đơn giản hoá op, `opset` khác, `onnx-simplifier`. Rẻ nhất, hay ăn nhất.
2. **Đổi op tương đương** — thay op lạ bằng tổ hợp op chuẩn trong model gốc.
3. **ONNX Runtime + TRT EP** — layer nào TRT không nuốt được thì **tự động fallback** về
   CUDA EP. Chạy được ngay, nhưng mỗi lần chuyển vùng là một lần đồng bộ → chậm.
4. **Viết plugin TensorRT** (`IPluginV2DynamicExt`) — đắt nhất, chỉ làm khi op đó nằm
   trong 3 layer tốn nhất theo `--dumpProfile`.

> Đo trước, viết plugin sau. Viết plugin cho một layer chiếm 2% thời gian là công sức
> tuần đổi lấy 2% — cùng loại sai lầm với "vectorise harder" ở
> [`08-nhat-ky-toi-uu.md`](08-nhat-ky-toi-uu.md).

---

# PHẦN B — DeepStream

## 14.5 Gốc: DeepStream là GStreamer, không phải thư viện AI

Nếu chỉ nhớ một câu: **DeepStream = GStreamer + một bộ plugin NVIDIA + một chuẩn
metadata.** Không học GStreamer thì không dùng được DeepStream, và mọi lỗi sẽ trông như
phép thuật.

Pipeline điển hình:

```
nvv4l2decoder ─► nvstreammux ─► nvinfer ─► nvtracker ─► nvinfer(SGIE) ─► nvdsosd ─► sink
   (NVDEC)         (gộp batch)   (TensorRT)  (bám vết)   (model phụ)     (vẽ)    (hiển thị/RTSP/file)
      ▲                                                                              
   nhiều camera vào cùng lúc            ─── buffer đi trong NVMM, KHÔNG copy về CPU ───
```

| Plugin | Việc thật sự | Chạy trên |
|---|---|---|
| `nvv4l2decoder` | giải mã H.264/H.265 | **NVDEC** — phần cứng riêng, GPU vẫn rảnh |
| `nvstreammux` | gộp N luồng thành **một batch** | GPU |
| `nvinfer` | tiền xử lý + gọi **TensorRT** + hậu xử lý | GPU |
| `nvtracker` | bám vết đối tượng qua các khung (IOU / NvDCF / DeepSORT) | GPU/CPU |
| `nvdsosd` | vẽ box, chữ | GPU |
| `nvvideoconvert` | đổi định dạng / chuyển NVMM ↔ hệ thống | GPU |

**Ba ý tưởng làm nên hiệu năng của DeepStream:**

1. **Zero-copy qua NVMM.** Buffer nằm trong bộ nhớ GPU-accessible suốt pipeline
   (`video/x-raw(memory:NVMM)`). Mỗi lần buffer bị kéo về CPU rồi đẩy lại — trên
   Jetson bộ nhớ dùng chung nên nó "vẫn chạy", chỉ là **chậm bất ngờ**. Đây là lỗi
   hiệu năng số 1 của người mới.

   > **Đo thật trên Orin Nano** ([`15 §15.7`](15-kernel-den-camera.md)): thủ phạm
   > **không phải** bộ chuyển đổi mà là **việc rời khỏi NVMM**. Cùng plugin
   > `nvvidconv`, chỉ khác caps đầu ra: giữ NVMM tốn 0.27 core, ép về `video/x-raw`
   > tốn 0.64 core (**2.4×**). Và trên L4T thuần plugin tên là **`nvvidconv`** —
   > `nvvideoconvert` chỉ tồn tại khi đã cài DeepStream.
2. **Batch nhiều camera.** `nvstreammux` gộp 8 camera thành batch 8 → một lần chạy
   TensorRT cho cả 8. Nhớ [`09-so-do-phan-cung.md` Phát hiện 7](09-so-do-phan-cung.md):
   trên board này **M=1 và M=64 tốn gần như cùng thời gian**. Batch là thứ biến 1 camera
   thành 8 camera gần như miễn phí.
3. **Metadata là sản phẩm thật.** Ảnh có vẽ box chỉ để người xem. Cái đi vào hệ thống của
   bạn là cây metadata gắn theo mỗi buffer:

```
NvDsBatchMeta                 (cả batch)
└── NvDsFrameMeta             (1 khung của 1 camera: source_id, frame_num, ntp_ts)
    ├── NvDsObjectMeta        (1 vật thể: class_id, rect_params, confidence, object_id ← tracker)
    │   └── NvDsClassifierMeta   (kết quả model phụ: màu xe, loại xe…)
    └── NvDsUserMeta          (tensor thô nếu bạn cần tự hậu xử lý)
```

Học DeepStream nghiêm túc = **học duyệt cây này trong probe callback**, không phải học
chỉnh file config.

## 14.6 Chạy và đọc cấu hình

```bash
ls /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/
deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

Config chia theo group, mỗi group là một plugin. Những khoá thực sự quan trọng:

```ini
[streammux]
batch-size=4                  # PHẢI bằng số nguồn, nếu không lãng phí hoặc nghẽn
batched-push-timeout=40000    # us. Nguồn live: đặt ≈ 1/fps, tránh chờ vô ích
live-source=1                 # camera/RTSP = 1, file = 0

[primary-gie]
onnx-file=model.onnx          # lần đầu: TRT build engine (chậm)
model-engine-file=model.engine # lần sau: nạp thẳng. Xoá file này khi đổi model!
network-mode=2                # 0=FP32  1=INT8  2=FP16
batch-size=4                  # nên khớp streammux
interval=0                    # 0 = suy luận MỌI khung; 1 = bỏ 1 khung ← lever rẻ nhất
gie-unique-id=1

[tracker]
ll-lib-file=.../libnvds_nvmultiobjecttracker.so
tracker-width=640             # phải là bội của 32
```

**`interval` là lever bị đánh giá thấp nhất.** Đặt `interval=1` chạy suy luận 2 khung một
lần; tracker nội suy phần còn lại. Với vật thể chuyển động chậm, chất lượng gần như không
đổi mà tải GPU giảm gần một nửa. Luôn thử `interval` **trước khi** đi tối ưu kernel.

### Đo FPS cho đúng

```ini
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
```

```
**PERF: FPS 0 (29.97)  FPS 1 (29.97)  FPS 2 (29.97)  FPS 3 (29.97)
```

Số trong ngoặc là trung bình. **Nếu tất cả bằng đúng fps của nguồn, bạn chưa đo được
giới hạn của máy** — pipeline đang bị nguồn kìm, không phải bị GPU kìm. Muốn biết trần
thật: bỏ hiển thị (`type=1` fakesink), đặt `sync=0`, dùng file thay vì camera live.

Đây đúng là bẫy "đo nhầm thứ" của [`09-so-do-phan-cung.md`](09-so-do-phan-cung.md), chỉ
đổi ngữ cảnh.

## 14.7 Nghẽn ở đâu — bốn nghi phạm

| Nghi phạm | Triệu chứng | Kiểm bằng | Chữa |
|---|---|---|---|
| **NVDEC** | FPS trần theo số luồng, GPU rảnh | `tegrastats` (NVDEC%) | giảm độ phân giải/số luồng; Orin Nano có **1 NVDEC** |
| **nvinfer** | GR3D_FREQ ~100% | `tegrastats`; `trtexec` riêng model | `interval`, INT8, model nhỏ hơn, batch lớn hơn |
| **Copy CPU↔GPU** | GPU không đầy mà FPS thấp | `nsys`, hoặc soi pipeline có plugin non-NVMM | thay bằng `nvvideoconvert`, giữ NVMM |
| **Hiển thị/encode** | tụt khi bật OSD/sink | thử `fakesink` | tắt OSD lúc đo; xem cảnh báo NVENC dưới đây |

> ⚠️ **Jetson Orin Nano KHÔNG có NVENC** (bộ mã hoá phần cứng). Nano có NVDEC nhưng
> không có encoder — Orin NX/AGX mới có. Hệ quả: mọi pipeline DeepStream xuất RTSP hoặc
> ghi file H.264 **phải mã hoá bằng CPU** (`x264enc`), và đó thường thành nghẽn mới.
> Kiểm ngay:
> ```bash
> gst-inspect-1.0 | grep -E 'nvv4l2h264enc|nvv4l2decoder'
> ```
> Thấy decoder mà không thấy encoder là đúng như mô tả. Thiết kế hệ thống theo hướng
> **gửi metadata (JSON/Kafka) thay vì gửi video** — vừa tránh giới hạn này, vừa đúng
> tinh thần "metadata mới là sản phẩm".

## 14.8 Gỡ lỗi GStreamer — ba lệnh cứu mạng

```bash
GST_DEBUG=3 deepstream-app -c cfg.txt                    # cảnh báo + lỗi
GST_DEBUG=nvinfer:5,nvstreammux:5 deepstream-app -c cfg.txt   # chỉ plugin nghi ngờ
GST_DEBUG_DUMP_DOT_DIR=/tmp deepstream-app -c cfg.txt && dot -Tpng /tmp/*PLAYING*.dot -o p.png
```

Lệnh thứ ba vẽ **pipeline thật đã được thương lượng**, kèm caps từng liên kết. Nhìn ảnh
đó là thấy ngay chỗ nào rơi khỏi `memory:NVMM` — nhanh hơn đọc log hàng giờ.

Lỗi kinh điển và nguyên nhân thật:

| Thông báo | Nguyên nhân thật |
|---|---|
| `Failed to link ... not-negotiated` | caps không khớp: thiếu `nvvideoconvert`, sai định dạng màu |
| kết quả sai sau khi đổi model | quên xoá `model-engine-file` cũ → vẫn nạp engine cũ |
| khởi động rất lâu lần đầu | TRT đang build engine — bình thường, lần sau nhanh |
| `deserialize engine failed` | engine build ở JetPack/GPU khác → build lại trên board |
| FPS thấp mà GPU rảnh | rơi khỏi NVMM, hoặc `batched-push-timeout` quá lớn |

---

# PHẦN C — Phần còn lại của bản đồ

Biết chúng tồn tại để không phát minh lại:

| Công cụ | Dùng khi | Một câu |
|---|---|---|
| **Triton** (`nvinferserver`) | nhiều model, nhiều framework, cần server hoá | DeepStream gọi Triton thay cho TensorRT trực tiếp |
| **VPI** | tiền xử lý ảnh (resize, warp, undistort, stereo) | chạy được trên GPU/PVA/CPU, thống nhất một API |
| **jetson-inference** (dusty-nv) | học nhanh, demo | ví dụ chạy được trong 30 phút |
| **Isaac ROS** | robot, ROS 2 | node ROS đã bọc sẵn TensorRT/VPI |
| **jetson-containers** | mọi thứ trên, khỏi vật lộn phiên bản | xem [13.3](13-jetson-framework.md) |

**Ma trận phiên bản là nguồn lỗi số 1** trên toàn stack này: JetPack ⇔ L4T ⇔ TensorRT ⇔
DeepStream bị khoá chặt với nhau. Board này là JetPack 6.2 / L4T R36.4.7 / CUDA 12.6.68 →
tra đúng bản DeepStream tương ứng **trước khi tải**, đừng cài bản mới nhất theo phản xạ.

---

## 14.9 Lộ trình học — bốn tuần, mỗi tuần một sản phẩm

### Tuần 1 — TensorRT cơ bản
Lấy một ONNX có sẵn (ResNet18 hoặc YOLO nhỏ).
1. `trtexec --onnx=... --fp16 --saveEngine=...` rồi đo bằng `--loadEngine`.
2. So FP32 / FP16 / INT8: throughput, latency, **và** độ chính xác (§14.3).
3. `--dumpProfile`: ba layer nào ăn 70% thời gian?
4. `--useCudaGraph`: cải thiện bao nhiêu? Nếu nhiều → bạn đang launch-bound.
**Sản phẩm:** bảng 3 precision × (FPS, latency, accuracy) + kết luận chọn cái nào và vì sao.

### Tuần 2 — TensorRT sâu
1. Shape động với `--minShapes/--optShapes/--maxShapes`; đo khi shape thật lệch `opt`.
2. `polygraphy run --trt --onnxrt` để so số; cố tình dùng `--fp16` cho model nhạy cảm và
   tìm layer đầu tiên lệch.
3. Build lại với `--timingCacheFile` — thời gian build giảm bao nhiêu?
**Sản phẩm:** một quy trình build+nghiệm thu lặp lại được, viết thành script.

### Tuần 3 — DeepStream cơ bản
1. Chạy sample config 4 nguồn có sẵn.
2. Thay bằng model của tuần 1 (`onnx-file`, `network-mode`, `num-detected-classes`).
3. Quét `interval` = 0/1/2/4: FPS và chất lượng đổi thế nào?
4. Quét số nguồn 1/2/4/8 với `batch-size` khớp: FPS mỗi luồng tụt ở đâu, và **cái gì**
   bão hoà trước — NVDEC hay nvinfer?
**Sản phẩm:** đồ thị FPS theo số luồng + câu trả lời "board này gánh được mấy camera".

### Tuần 4 — DeepStream sâu
1. Viết **probe callback** duyệt `NvDsBatchMeta`, in ra `object_id` + class + toạ độ.
2. Đếm vật thể qua một đường kẻ (dùng `object_id` của tracker) → xuất JSON.
3. Bỏ hiển thị, chỉ gửi metadata (đúng hướng cho Orin Nano không NVENC).
**Sản phẩm:** một app đếm người/xe thật, chạy nhiều camera, xuất JSON — đây là dạng bài
thực tế nhất của nghề Jetson.

---

## 14.10 Bài tập kiểm tra hiểu

1. Vì sao `.engine` không portable? Nêu **ba** thành phần khiến nó gắn với máy.
2. Model FP16 nhanh gấp đôi FP32 trên desktop nhưng chỉ 1.1× trên Orin Nano. Hai giả
   thuyết, và cách phân biệt bằng `--dumpProfile` + `ncu`?
3. DeepStream 4 luồng chạy 30 FPS mỗi luồng; thêm luồng thứ 5 thì tất cả tụt còn 24.
   Ba nghi phạm, và lệnh kiểm từng cái?
4. Vì sao `interval=1` gần như không giảm chất lượng đếm xe, nhưng lại phá hỏng bài toán
   đọc biển số?
5. Pipeline có `videoconvert` (không phải `nvvideoconvert`) ở giữa. Giải thích bằng
   ngôn ngữ băng thông của [`03-roofline.md`](03-roofline.md) tại sao nó chậm — dù trên
   Jetson bộ nhớ vốn dùng chung.
6. `trtexec` báo `Enqueue Time ≈ GPU Compute Time`. Model đang bị chặn bởi tầng nào theo
   bảng bốn tầng ở [13.4](13-jetson-framework.md)? Hai cách chữa?
7. Nối lại với repo này: `bench_cuda` đo 117 kernel/token, 50% là launch overhead.
   TensorRT sẽ chữa được phần nào của 50% đó, và **không** chữa được phần nào?

---

→ [README.md](README.md) · Trước: [13-jetson-framework.md](13-jetson-framework.md) ·
Kernel: [11-toi-uu-nvidia.md](11-toi-uu-nvidia.md) ·
Số đo board: [09-so-do-phan-cung.md](09-so-do-phan-cung.md)
