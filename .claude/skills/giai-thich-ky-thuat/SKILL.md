---
name: giai-thich-ky-thuat
description: Cách giải thích một khái niệm kỹ thuật cho kỹ sư nhúng bằng tiếng Việt, dựa trên số đo thật của chính họ. Dùng khi được yêu cầu giải thích, khi nghe "tôi chưa rõ", "chỉ tôi", "dễ hiểu", "giảng cho tôi", khi viết bài cho loạt AI cho kỹ sư nhúng, hoặc khi một câu trả lời đang có nguy cơ thành bài giảng trừu tượng.
---

# Giải thích kỹ thuật cho kỹ sư nhúng

Người đọc đã biết C, con trỏ, DMA, fixed-point, memory map. Họ không thiếu năng
lực, họ thiếu cầu nối. Việc cần làm là bắc cầu từ thứ họ đã có sang thứ mới, chứ
không phải dạy lại từ đầu.

## Bảy bước, theo đúng thứ tự

**1. Một câu trả lời cho "thứ này thật ra là gì".** Không định nghĩa sách vở.

> Model của bạn có 28.869.920 con số. Train là quá trình sửa dần 28,9 triệu con
> số đó cho tới khi model đoán đúng token kế tiếp.

**2. Chứng minh ngay bằng log của chính họ.** Câu trên chỉ là lời nói cho tới khi
có số:

```
step     0 | ppl 33137     <- đoán bậy trong toàn bộ 32.768 từ vựng
step 10999 | ppl  8.25     <- phân vân giữa 8 lựa chọn
```

Chỉ ra thêm một chi tiết mà người đọc tự kiểm được: `ppl 33137` xấp xỉ vocab
32768, nên đó là dấu hiệu khởi tạo đúng. Chi tiết kiểu này biến con số từ "được
kể" thành "kiểm được".

**3. Cho xem code thật, không mô tả code.** Bốn dòng thật trong `train.py` có sức
thuyết phục hơn một đoạn văn tả bốn dòng đó.

**4. Mỗi núm vặn một mục, kèm cái giá phải trả.** Không liệt kê tham số suông. Mỗi
tham số phải trả lời "đổi nó thì được gì và mất gì".

**5. Ẩn dụ ĐẶT SAU cơ chế, không đặt trước.** Giải thích `lr` bằng công thức và
bảng số trước, rồi mới:

> Bạn đi xuống dốc trong sương mù. `lr` là độ dài mỗi bước chân. Bước quá ngắn thì
> cả ngày chưa tới đáy; bước quá dài thì nhảy qua đáy rồi dội lên bờ bên kia.

Ẩn dụ đặt trước sẽ thay thế cơ chế trong đầu người đọc. Đặt sau thì nó chỉ đóng
đinh thứ họ vừa hiểu.

**6. Bảng "triệu chứng, nghĩa là gì, phải làm gì".** Đây là phần người đọc quay
lại tra nhiều nhất:

| Dấu hiệu | Nghĩa là | Phải làm gì |
|---|---|---|
| `train ≈ val`, cả hai giảm | đang học quy luật thật | để chạy tiếp |
| `train ≪ val` | học thuộc lòng | thêm dữ liệu hoặc dừng sớm |
| cả hai đứng yên | hết sức chứa | tăng `d_model` hoặc số lớp |

**7. Kết bằng việc họ làm được ngay**, kèm chi phí thật. Không kết bằng tóm tắt.

## Luật về số

**Mọi con số phải từ máy của họ.** Không lấy từ bài báo, không ước lượng, không
làm tròn cho đẹp. Nếu chưa đo thì chạy đo trước khi viết.

**Nói rõ khi hai con số không so được.** Cùng một model cho val 2.1102 và 2.1568
vì `train.py` đo ở `seq_len` đang train còn `quantize.py` đo ở 256. Bỏ qua chi
tiết này là dạy người đọc so nhầm.

**Ghi lại chỗ mình đoán sai.** Giả thuyết "3e-3 sẽ phân kỳ" bị chính số liệu bác
bỏ. Giữ nguyên chỗ sai đó trong bài có giá trị hơn một bảng số mà ai cũng gật gù,
vì nó dạy người đọc rằng phải đo chứ đừng tin trực giác.

## Không làm những thứ này

**Không dùng dấu gạch ngang dài.** Thay bằng dấu phẩy, hai chấm, ngoặc đơn, hoặc
tách câu. Kiểm bằng `grep -c '—\|–'`, phải ra 0.

**Không tuyên bố mình sắp thành thật.** "Nói thẳng là X" luôn rút gọn được thành
"X". Cùng họ với nó: "thành thật mà nói", "phải nói rằng", "thú thật".

**Không định nghĩa trước khi cho xem hiện tượng.** Sai: "Dot product là tổng các
tích thành phần". Đúng: cho xem vòng `for` ba dòng họ đã viết trăm lần, tính ra
32, rồi mới nói toán học gọi nó là dot product.

**Không in đậm quá ba cụm trong một đoạn.** Nhấn mạnh bốn thứ tức là không nhấn
mạnh thứ nào.

**Không nói "quan trọng" mà không nói vì sao.** Thay "đây là tham số rất quan
trọng" bằng con số cho thấy nó quan trọng.

## Kiểm trước khi gửi

```bash
uv run python docs/begin_0/tools/vn_humanizer.py bai.md --min 90
grep -c '—\|–' bai.md          # phải ra 0
```

Bộ dò bắt các cụm sáo rỗng tiếng Việt, rào đón, kết bài sáo, và mật độ in đậm.
Nó phân biệt được "nói thẳng" đầu câu (tic của người viết) với "comment trong code
nói thẳng" (mô tả hợp lệ), nên đừng sửa mù theo cảnh báo.

## Ví dụ đầy đủ đã dùng

Phần giải thích training ngày 13/8/2026 đi đúng bảy bước trên: mở bằng "28,9
triệu con số", chứng minh bằng `ppl 33137 -> 8.25` từ log của chính họ, cho xem
bốn dòng `train.py`, tách từng núm vặn kèm cái giá (batch 12 vì logits chiếm 0,81
GB trên card 6GB), ẩn dụ sương mù đặt sau bảng số learning rate, bảng triệu chứng
train so với val, và kết bằng thí nghiệm tiếp theo kèm chi phí 80 phút.
