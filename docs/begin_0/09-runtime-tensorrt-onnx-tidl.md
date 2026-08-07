# 17. TensorRT / ONNX Runtime / TIDL — runtime cho từng loại silicon

Chương 6–8 xây model, thu nhỏ nó, dạy nó sinh chữ. Chương này trả lời: **ai chạy nó
trên chip thật, và tại sao có nhiều "runtime" khác nhau chứ không phải một?** Câu
trả lời ngắn: mỗi runtime là lớp phần mềm biết cách nói chuyện với **một loại mạch
MAC chuyên dụng** (chương 3 §3.8) của một họ silicon cụ thể. TensorRT nói chuyện với
Tensor Core của NVIDIA; TIDL nói chuyện với C7x+MMA của TI. Bản chất công việc —
biên dịch một đồ thị phép toán thành lệnh máy tối ưu cho đúng mạch cứng đó — **giống
hệt nhau** giữa hai runtime, chỉ khác đích đến.

## ONNX — ngôn ngữ chung giữa các runtime

**ONNX** (Open Neural Network Exchange) là định dạng file trung gian — không phải
runtime. Vai trò của nó giống hệt file `.elf`/`.bin` trong thế giới nhúng: một dạng
biểu diễn *di động* để nhiều công cụ đọc được, không gắn với công cụ sinh ra nó.

```
PyTorch (model.py)  ──export──>  ONNX (.onnx)  ──đọc bởi──>  TensorRT / TIDL / ONNX Runtime CPU EP
```

Repo này **không** đi qua ONNX — nó tự viết `export.py` ghi thẳng ra
[`model.bin`](../../src/export.py) và tự viết `llm_load()` trong
[`llm.h`](../../firmware/common/llm.h) để đọc, vì runtime tự viết tay kiểm soát
được từng byte (đúng tinh thần "không có lớp trừu tượng nào che mất bản chất" của
toàn repo). Trong công nghiệp, hầu hết pipeline đi qua ONNX vì nó tránh phải viết
lại `export.py` + `llm.h` cho mỗi model mới.

## TensorRT + ONNX Runtime — đã có tài liệu đầy đủ, đọc ở đây

[`../14-tensorrt-deepstream.md`](../14-tensorrt-deepstream.md) đã viết chi tiết cả
hai, với lệnh chạy thật (`trtexec`), số đo thật trên Jetson Orin Nano, và cả bẫy
thường gặp. **Đọc Phần A của file đó ngay bây giờ** — 6 bước của TensorRT builder
(parse → fusion → precision → autotune → serialize), cách đọc output `trtexec`, và
INT8 calibration (nối thẳng với chương 8 — group-wise quantization bạn vừa học chỉ
là *một cách* trong họ kỹ thuật calibration).

Tóm tắt bằng từ vựng bạn đã có ở chương 3 §3.8: **TensorRT là trình biên dịch biến
đồ thị ONNX thành lệnh máy cho Tensor Core**, đúng cách `gcc -O3` biên dịch C thành
lệnh máy cho ALU — chỉ khác đích là mạch MAC chuyên dụng thay vì ALU đa năng, và quá
trình "biên dịch" bao gồm cả bước đo thời gian chạy thật trên máy bạn (autotuning)
để chọn thuật toán nhanh nhất — thứ `gcc` không làm.

**ONNX Runtime** là lớp trung gian: nó chạy được ONNX trực tiếp (execution provider
CPU) hoặc **giao lại** các phần đồ thị cho TensorRT/khác qua *Execution Provider*
(EP) — layer nào TensorRT không nuốt được thì rơi lại về CPU. Đây chính là mô hình
mà TIDL dùng dưới đây, chỉ đổi tên.

## TIDL — runtime tương đương của TensorRT trên silicon Texas Instruments

Đây là nội dung không có sẵn ở đâu khác trong `docs/` — repo này build cho ESP32 và
NVIDIA, không có board TI để đo thật. Phần này viết ở mức **khái niệm và ánh xạ**,
không có số đo trên silicon như các chương khác. **Luôn tự đo trên đúng board và
đúng phiên bản TIDL của bạn** trước khi tin bất kỳ con số hiệu năng nào, kể cả từ
datasheet — đúng nguyên tắc xuyên suốt repo này
([`../09-so-do-phan-cung.md`](../09-so-do-phan-cung.md)).

**TIDL** (TI Deep Learning) là bộ công cụ suy luận AI của Texas Instruments cho các
SoC dòng Jacinto/Sitara có lõi tăng tốc AI — ví dụ TDA4VM, AM68A, AM69A (C7x DSP +
**MMA**, Matrix Multiply Accelerator — chính là mạch MAC array nói ở chương 3 §3.8).

### Ánh xạ trực tiếp — mỗi khái niệm TensorRT có một khái niệm TIDL tương ứng

| TensorRT (đã học ở `docs/14`) | TIDL | Ý nghĩa chung |
|---|---|---|
| `trtexec --onnx=... --saveEngine=...` | `tidl_model_import` (offline, chạy trên host x86) | biên dịch ONNX/TFLite → artifact riêng cho chip |
| `.engine` file | thư mục artifact (`.bin` + `.params.yaml` + `net.bin`...) | kết quả biên dịch, gắn với đúng phiên bản SDK + đúng chip |
| INT8 calibration (`--calib=...`) | calibration bằng vài trăm ảnh mẫu, TIDL tự đo dải động | **cùng thuật toán PTQ** — chương 8 đã học đúng cơ chế này |
| ONNX Runtime + TRT EP, fallback CPU EP khi op không hỗ trợ | ONNX Runtime + **TIDL Execution Provider**, fallback ARM (Cortex-A) khi op không hỗ trợ | **giống hệt cơ chế**: subgraph nào TIDL nuốt được chạy trên C7x+MMA, phần còn lại chạy CPU |
| Tensor Core (mạch cứng NVIDIA) | **C7x DSP + MMA** (mạch cứng TI) | cùng vai trò: MAC array chuyên GEMM, chương 3 §3.8 |
| build lâu vì autotuning trên chính GPU | import lâu vì lượng tử hoá + tối ưu layout cho chính SoC | cùng lý do: đo/tối ưu **trên đúng silicon đích**, không đoán |

### Quy trình hai giai đoạn — đúng mô hình build-time/runtime của chương 3

```
GIAI ĐOẠN 1 — "import"/"compile" (offline, trên máy x86 phát triển)
   model.onnx + vài trăm ảnh calibration
        │  (tidl_model_import / edgeai-tidl-tools)
        ▼
   artifact: subgraph nào chạy C7x+MMA, subgraph nào rơi lại ARM,
             scale lượng tử hoá mỗi layer (đúng §16 chương 8)

GIAI ĐOẠN 2 — suy luận trên chip TI thật (production)
   ONNX Runtime (hoặc TFLite) + TIDL Execution Provider nạp artifact,
   dispatch: subgraph hỗ trợ → C7x DSP + MMA
             subgraph không hỗ trợ (custom op, shape lạ...) → Cortex-A (ARM)
```

Đây **chính là** hai giai đoạn build-time/runtime đã học ở chương 3 §3.8 và
[`../14-tensorrt-deepstream.md §14.1`](../14-tensorrt-deepstream.md) — công sức đắt
đỏ (lượng tử hoá, chọn layout bộ nhớ tối ưu) dồn vào giai đoạn 1, chạy **một lần**;
giai đoạn 2 chỉ nạp artifact đã biên dịch sẵn và chạy — đúng thiết kế `model.bin`
của chính repo này (`export.py` chạy 1 lần, `llm_load()` chạy mỗi lần khởi động).

### Ba khác biệt thật sự với TensorRT, cần biết trước khi bắt tay vào

1. **Độ chính xác mặc định thấp hơn.** Nhiều dòng TIDL lịch sử tối ưu quanh **INT8
   cố định** (fixed-point) làm trung tâm — gần với thế giới Q-format bạn đã quen ở
   chương 8 hơn là FP16 của GPU. Luôn kiểm tài liệu SDK đúng phiên bản đang dùng để
   biết các mức chính xác được hỗ trợ trên đúng chip của bạn.
2. **Không có "CUDA Graphs".** Launch overhead (chương 3 §3.7, đo được rất lớn trên
   Jetson ở [`../09-so-do-phan-cung.md`](../09-so-do-phan-cung.md) — 50% thời gian
   là launch thuần) là vấn đề của kiến trúc GPU nhiều kernel rời rạc. DSP+MMA thường
   chạy theo mô hình khác (ít lệnh khởi động rời rạc hơn) — nghĩa là bài học "gộp
   kernel" không chuyển giao y hệt, phải tự đo lại nút thắt thật trên chip TI, đúng
   tinh thần chương 3 §3.9: đừng đoán, đo trên đúng silicon.
3. **Toolchain gắn chặt với đúng phiên bản SDK (PSDK RTOS / Processor SDK Linux) và
   đúng chip** — y hệt lời cảnh báo về `.engine` không portable ở
   [`../14-tensorrt-deepstream.md §14.1`](../14-tensorrt-deepstream.md): đổi phiên
   bản SDK gần như chắc chắn phải import lại model.

### Việc cần làm nếu bạn có board TI thật

1. Cài `edgeai-tidl-tools` (repo chính thức của TI trên GitHub) theo đúng SDK version
   của board.
2. Export model của bạn (hoặc thử với model nhỏ trong repo này sau khi tự chuyển
   sang ONNX — bài tập 4 dưới) ra `.onnx`.
3. Chạy `tidl_model_import` với vài trăm ảnh/mẫu calibration đại diện — đúng vai trò
   với `--calib=calib.cache` ở TensorRT (`docs/14 §14.3`).
4. Đo bằng đúng phương pháp đã học ở chương 8: so loss/accuracy **trước và sau**
   lượng tử hoá, không tin số liệu "lý thuyết". Ghi lại phiên bản SDK + chip + điều
   kiện đo, đúng kỷ luật đã thấy xuyên suốt `docs/09` và `docs/15`.

## Bảng quyết định — chọn runtime nào

| Bạn có gì | Chọn |
|---|---|
| NVIDIA Jetson / GPU rời | TensorRT (production) hoặc llama.cpp CUDA backend (nhanh gọn) — xem `docs/14` |
| TI Sitara/Jacinto (TDA4x, AM6xx) | TIDL qua ONNX Runtime EP hoặc TFLite delegate |
| MCU không có accelerator AI (ESP32 kiểu repo này) | **tự viết runtime** (`llm.h`) — không có "framework" nào đủ nhỏ, đây chính là lý do cả repo tồn tại |
| Cần chạy được trên NHIỀU loại chip, không tối ưu tuyệt đối | ONNX Runtime với CPU EP làm baseline, thêm EP riêng cho từng chip khi cần |

## Bài tập

1. Đọc trọn [`../14-tensorrt-deepstream.md`](../14-tensorrt-deepstream.md) Phần A
   nếu chưa đọc — làm Lab tuần 1 ở cuối file đó (`trtexec --fp16`, so FP32/FP16/
   INT8) nếu có GPU NVIDIA.
2. Tra datasheet/tài liệu SDK của đúng chip TI bạn quan tâm (TDA4VM hoặc đời mới
   hơn): C7x+MMA chạy ở clock nào, TOPS INT8 công bố là bao nhiêu? So với con số
   Tensor Core đo **thật** trên Orin Nano ở [`03-matrix-nhan.md §3.8`](03-matrix-nhan.md)
   — chênh lệch giữa số datasheet và số đo thật trên Jetson lớn cỡ nào
   ([`../09-so-do-phan-cung.md`](../09-so-do-phan-cung.md))? Bạn nên kỳ vọng gì
   tương tự khi đo TI thật?
3. Viết một `export_onnx.py` nhỏ chuyển `TinyLM` (từ `model.py`) sang ONNX bằng
   `torch.onnx.export`. Model này có attention + KV cache tuỳ biến — op nào trong đồ
   thị có khả năng không được TensorRT/TIDL hỗ trợ ngay, cần xử lý riêng (tham khảo
   [`../14-tensorrt-deepstream.md §14.4`](../14-tensorrt-deepstream.md))?
4. Viết bảng so sánh 3 dòng: TensorRT, TIDL, và bản C tự viết tay (`llm.h`) của repo
   này — theo 3 tiêu chí: (a) độ portable giữa các phiên bản/chip, (b) độ kiểm soát
   từng byte bộ nhớ, (c) công sức phải bỏ ra để deploy 1 model mới. Không có câu trả
   lời "đúng" tuyệt đối — mục tiêu là thấy rõ đánh đổi.

→ Tiếp: [10-vla-robot.md](10-vla-robot.md) — chủ đề cuối roadmap: ghép model ngôn
ngữ với camera và hành động điều khiển thật, trên đúng phần cứng đã đo trong
`docs/15`.
