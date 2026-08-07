# 18. Vision-Language-Action (VLA) cho Robot

Chủ đề cuối cùng của roadmap, và là nơi mọi chương trước hội tụ. VLA là một model
nhận **ảnh camera** + **câu lệnh ngôn ngữ tự nhiên**, xuất ra **hành động điều
khiển** (góc khớp robot, vận tốc bánh xe...). Với một kỹ sư embedded, đây là bài
toán quen thuộc khoác áo mới: **một vòng lặp điều khiển thời gian thực, trong đó
khối "tính toán quyết định" giờ là một transformer** thay vì một bộ PID hay state
machine.

Chương này không có model VLA chạy được trong repo — nhưng **mọi khối bên trong một
VLA đều là thứ bạn đã xây hoặc đo thật ở các chương trước**, và board Jetson đã đo
trong [`../15-kernel-den-camera.md`](../15-kernel-den-camera.md) chính là loại phần
cứng thật sự chạy VLA trong công nghiệp ngày nay.

## 18.1 Kiến trúc — ba khối, bạn đã biết cả ba

```
Camera ──► [Vision Encoder] ──► vision tokens ──┐
                                                  ├──► [LLM Backbone] ──► [Action Head] ──► lệnh điều khiển
Câu lệnh ("nhặt cốc đỏ") ──► [Tokenizer] ──► text tokens ──┘
```

| Khối | Nó làm gì | Bạn đã học ở đâu |
|---|---|---|
| **Vision Encoder** | ảnh → một dãy vector (thường là CNN hoặc ViT) | chương 1 (vector là gì), chương 3 §3.5 (im2col — CNN là matmul) |
| **Tokenizer** (text) | câu lệnh → dãy token id | [`docs/00 §0.1`](../00-nhap-mon.md), đúng cơ chế `bpe4096.json` bạn đã dùng |
| **LLM Backbone** | trộn thông tin ảnh + chữ, "suy luận" | **toàn bộ chương 6** — decoder block y hệt, chỉ khác đầu vào không chỉ là text token mà còn có vision token |
| **Action Head** | vector cuối cùng → con số điều khiển thật (góc khớp, lực kẹp...) | chương 3 §3.4 (một lớp Linear nữa) — hoặc token hoá hành động thành các "action token" rời rạc, tái dùng đúng cơ chế sampling ở chương 7 §15 |

**Điểm mấu chốt cần thấy:** vision token và text token, sau khi qua Vision Encoder
và Tokenizer, **là cùng một loại đối tượng** — vector D chiều, xếp thành một chuỗi
đưa vào decoder block. Attention (chương 3 §3.6, chương 6) không quan tâm token nào
"đến từ ảnh" hay "đến từ chữ" — nó chỉ thấy một chuỗi vector và tính `Q·Kᵀ` giữa
mọi cặp. Đây là lý do kiến trúc transformer "ăn được" nhiều loại dữ liệu khác nhau
mà không cần đổi thuật toán lõi — chỉ cần một bộ encoder đưa dữ liệu về đúng dạng
vector.

Ba kiến trúc VLA được công bố rộng rãi (RT-2 của Google DeepMind, OpenVLA, và họ
π0/pi-zero của Physical Intelligence) đều theo đúng khung trên, khác nhau ở lựa chọn
vision encoder, cỡ LLM backbone, và cách biểu diễn action (rời rạc hoá thành token so
với hồi quy trực tiếp bằng một "flow head"). Chi tiết từng model thay đổi theo thời
gian — khi tìm hiểu sâu một kiến trúc cụ thể, luôn đọc paper/tài liệu chính thức
mới nhất thay vì tin số liệu tóm tắt.

## 18.2 Ngân sách thời gian thực — đây là chỗ mọi con số đã đo trong repo hội tụ

Một vòng lặp điều khiển robot có **hạn chót cứng** (hard deadline): quyết định phải
ra trước khi trạng thái vật lý (robot, vật thể) thay đổi đáng kể. Đây là khái niệm
bạn đã có sẵn từ RTOS — chỉ khác "task" giờ là cả một forward pass transformer.

Ghép các con số đã **đo thật** trong repo này thành một ngân sách ví dụ, để thấy
VLA "ăn" thời gian ở đâu:

```
┌─ camera → ảnh sẵn sàng ──────────────────────────────────────────┐
│  NVDEC giải mã H.264 720p:  ~0.27 core CPU, gần như miễn phí       │  đo thật —
│  (so với giải mã bằng CPU: 3.96 core — chênh 14.7×)                │  docs/15 §15.7
├─ Vision Encoder (không đo trong repo, tham khảo) ──────────────────┤
│  ảnh → vision tokens: một CNN/ViT nhỏ, vài ms trên GPU nhúng        │
├─ LLM Backbone — CHÍNH LÀ decode loop bạn đã đo ────────────────────┤
│  mỗi token: matvec + attention (chương 3, chương 6)                 │  102.9 ms/token
│  càng nhiều context (ảnh + lệnh) → KV cache càng lớn (chương 7)     │  trên ESP32,
│  → tok/s ≤ bandwidth / model_size (docs/00 §0.3)                    │  hoặc GB/s thật
├─ Action Head ───────────────────────────────────────────────────────┤  đo trên Jetson
│  1 lớp Linear hoặc vài bước sampling (chương 7 §15) — rẻ             │  (docs/09)
└─ Bộ điều khiển thật thực thi lệnh ──────────────────────────────────┘
```

**Bài học trực tiếp từ chính repo, áp thẳng vào VLA:**

1. **Đường dữ liệu (camera → ảnh) rẻ nếu dùng đúng phần cứng, đắt nếu dùng sai.**
   [`docs/15 §15.7`](../15-kernel-den-camera.md) đo được: NVDEC + giữ frame trong
   NVMM tốn 0.27 core CPU; cùng luồng, giải mã bằng CPU tốn 3.96 core — **14.7 lần**.
   Trên một robot chạy pin, batch xử lý nhiều camera, đây là chênh lệch giữa "còn
   CPU để chạy LLM backbone" và "hết sạch CPU chỉ để giải mã video."
2. **LLM backbone của một VLA vẫn tuân thủ đúng trần `tok/s ≤ bandwidth / model_size`**
   ([`../00-nhap-mon.md §0.3`](../00-nhap-mon.md)) — không có phép màu nào phá được
   trần này ở batch=1 (một robot, một quyết định tại một thời điểm), trừ speculative
   decoding (chương 9, roadmap ngoài phạm vi file này — xem
   [`../07-kv-cache-engine.md §3.6`](../07-kv-cache-engine.md)).
3. **Quantization (chương 8) không tuỳ chọn, mà bắt buộc** cho VLA chạy on-device:
   model backbone của VLA thật thường lớn hơn model TinyStories của repo này hàng
   trăm tới hàng nghìn lần — không quantize thì không vừa bộ nhớ của bất kỳ SoC
   nhúng nào.
4. **Runtime (chương 9)** quyết định VLA có kịp deadline hay không, không chỉ có
   chạy được hay không — đúng cách `docs/15` đo được PREEMPT_RT làm **chậm đi** 30–
   40% dù mục tiêu là cải thiện độ trễ (bài học: đo trước khi tin trực giác, ngay cả
   khi trực giác nghe rất hợp lý).

## 18.3 Vì sao đây là bài toán khó nhất trong toàn roadmap

Ba ràng buộc cùng lúc, mỗi cái đã khó riêng lẻ:

| Ràng buộc | Đã học ở đâu | Cái khó khi cộng cả ba |
|---|---|---|
| **Độ chính xác cao** (hiểu đúng ảnh + lệnh, ra hành động an toàn) | chương 4–5 (train), chương 6 (kiến trúc) | model càng lớn càng chính xác — nhưng... |
| **Độ trễ thấp** (kịp deadline điều khiển) | chương 3 §3.7 (roofline), chương 7 (KV cache) | ...model càng lớn càng chậm (bandwidth-bound) |
| **Bộ nhớ/năng lượng giới hạn** (chạy on-device, không cloud) | chương 2 §2.3 (3 tầng bộ nhớ), chương 8 (quantization) | ...và càng lớn càng khó vừa SoC nhúng |

Đây là **chính xác** bài toán mà bảng PLE của repo này giải quyết, ở quy mô nhỏ hơn:
"làm sao có nhiều tham số (chính xác hơn) mà vẫn chạy trên phần cứng nhỏ (nhanh,
gọn)?" Câu trả lời của repo — tách tham số theo **cách bị đọc**, không theo tốc độ
([`../00-nhap-mon.md §0.5`](../00-nhap-mon.md)) — là một trường hợp riêng của bài
toán tổng quát mà cả ngành VLA đang giải: MoE (Mixture of Experts, chỉ kích hoạt vài
expert mỗi lần), model phân cấp (một model nhỏ chạy nhanh cho phản xạ tức thời, một
model lớn chạy chậm hơn cho lập kế hoạch), và nén KV cache tích cực đều là các biến
thể của cùng một ý tưởng cốt lõi.

## 18.4 Việc thực hành duy nhất bạn thực sự làm được với repo này

Không có VLA trong repo, nhưng có **đúng bài toán ngân sách thời gian thực** thu
nhỏ — dùng nó để tập tư duy trước khi chạm vào VLA thật:

1. Đọc [`../15-kernel-den-camera.md`](../15-kernel-den-camera.md) trọn vẹn nếu chưa
   đọc — đây là bài đo camera pipeline + kernel Linux thật trên chính loại board
   (Jetson Orin) chạy VLA trong công nghiệp.
2. Chạy `make -C firmware/jetson bench` — cột `ms/token` chính là đơn vị thời gian
   nhỏ nhất bạn cần cộng dồn khi ước tính ngân sách cho một VLA backbone lớn hơn.
3. Tự lập một bảng ngân sách thời gian thực cho một robot giả định: camera 30fps
   (33ms/khung), cần ra quyết định trước khi khung tiếp theo tới. Từ trần
   `tok/s ≤ bandwidth/model_size` (chương 3, `docs/00 §0.3`) và băng thông board bạn
   định dùng, tính ngược: model backbone lớn nhất bao nhiêu GB thì còn kịp?

## Bài tập

1. Với băng thông đo được trên Orin Nano Super (`docs/09`, xem số EMC/GB·s mới nhất
   sau khi nâng L4T — `docs/09` mục cập nhật), và deadline 33ms (camera 30fps), tính
   kích thước model backbone lớn nhất (GB, tại 4-bit) mà một VLA có thể dùng và vẫn
   ra quyết định trong 1 khung hình, giả sử backbone chỉ cần sinh 1 "action token".
2. So sánh: nếu action được biểu diễn bằng 8 token rời rạc (thay vì 1), thời gian
   tăng thêm bao nhiêu theo mô hình decode tuần tự đã học ở chương 7? Đây có phải
   lý do nhiều VLA hiện đại chọn action head hồi quy trực tiếp (1 bước) thay vì
   sinh nhiều action token tuần tự?
3. Liệt kê: nếu bạn phải nén một VLA backbone để chạy trên ESP32-S3 (theo đúng tinh
   thần repo này), tầng nào của model bạn sẽ đưa vào flash "table" (chương 2 §2.3),
   tầng nào bắt buộc phải ở SRAM "core"? Vision encoder thuộc tầng nào?
4. Đọc lại toàn bộ [`README.md`](README.md) của thư mục này. Bạn đã đi từ "vector là
   một mảng float" tới "ngân sách thời gian thực cho một robot". Viết 5 câu tóm tắt
   con đường đó bằng ngôn ngữ của chính bạn — đây là bài kiểm tra cuối cùng, không
   có đáp án mẫu.

---

Đến đây, bạn đã đi hết roadmap gốc. Bước tiếp theo không nằm trong file text nữa —
nó nằm ở việc **quay lại repo, tự đặt câu hỏi, và tự đo**. Đó là kỹ năng duy nhất mà
không tài liệu nào dạy thay bạn được, và là kỹ năng mà toàn bộ repo này — từ
`RESULTS.md` tới `docs/15` — cố tình phơi bày cách làm, kể cả những lần đo sai và
phải sửa lại.

→ [../README.md](../README.md) — mục lục đầy đủ. → [README.md](README.md) — quay
lại đầu lộ trình này.
