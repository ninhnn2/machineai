# 5. Cùng một phép toán, ba kiến trúc — cái gì đổi, cái gì không

Repo này hiếm ở chỗ **cùng một model, cùng một `model.bin`** chạy trên ba kiến trúc
khác hẳn nhau. Đó là bàn thí nghiệm tốt nhất để học tối ưu: giữ nguyên bài toán,
đổi phần cứng, xem cái gì chuyển giao được.

| | ESP32-S3 | CPU (x86 / ARM) | GPU (Ampere) |
|---|---|---|---|
| Mô hình song song | 2 core, scalar | vài core, SIMD | hàng nghìn thread, SIMT |
| Bộ nhớ nhanh | SRAM 512 KB | L1/L2/L3 cache | shared mem + L2 2 MB |
| Đơn vị chia việc | task | thread + vector lane | warp (32 lane) |
| Chi phí song song | ~0 (task cố định) | mở vùng song song | launch kernel |
| Runtime trong repo | `esp32_llm/` | `llm.h`, `samples/cpu/` | `firmware/jetson/` |

---

## 5.1 Thí nghiệm: thang tối ưu một phép toán

[`samples/cpu/matvec_ladder.c`](../samples/cpu/matvec_ladder.c) chạy **cùng phép toán**
(output head `y[4096] = W[4096,128]·x[128]`) qua 5 bậc tối ưu, tái hiện đúng nhật ký
ESP32 ở [`RESULTS.md`](../RESULTS.md):

```
L0  int4 + fp32        gỡ nibble mỗi lần, nhân float     (llm.h matvec_q)
L1  int4 + int8 act    lượng tử hoá activation           (llm.h matvec_q8)
L2  int8 staged        gỡ nibble MỘT LẦN lúc nạp         (esp32_llm.ino:101)
L3  + SIMD             AVX2 / NEON dot int8
L4  + đa luồng         chia theo hàng                     (2 core LX7)
```

```bash
make -C samples/cpu run
```

### Kết quả đo thật

**x86 — Intel i7-13650HX, AVX2, 8 luồng:**

| bậc | ms | GB/s | so L0 |
|---|---:|---:|---:|
| L0 int4 + fp32 | 0.8712 | 0.3 | 1.00× |
| L1 int4 + int8act | 0.8409 | 0.3 | 1.04× |
| L2 int8 staged | 0.1048 | 5.0 | **8.31×** |
| L3 + SIMD (AVX2) | 0.0702 | 7.5 | 12.41× |
| L4 + 8 luồng | 0.0113 | 46.2 | **76.83×** |

**ARM — Cortex-A78AE (Jetson Orin), NEON+dotprod, 6 luồng:**

| bậc | ms | GB/s | so L0 |
|---|---:|---:|---:|
| L0 int4 + fp32 | 0.5879 | 0.4 | 1.00× |
| L1 int4 + int8act | 0.6663 | 0.4 | **0.88× — CHẬM HƠN** |
| L2 int8 staged | 0.0430 | 12.2 | **13.66×** |
| L3 + SIMD (NEON) | 0.0443 | 11.8 | **13.28× — không giúp gì** |
| L4 + 6 luồng | 0.0106 | 49.4 | 55.43× |

---

## 5.2 Ba điều rút ra — và không cái nào đoán trước được

### (a) Bậc L2 thắng ở MỌI kiến trúc — 8-14×

Gỡ nibble int4 **một lần lúc nạp** thay vì mỗi token là bậc lời nhất trên cả ba máy:
ESP32 được 9.1× ([RESULTS.md](../RESULTS.md)), x86 8.3×, ARM 13.7×.

Nó không phải kỹ thuật cao siêu — chỉ là **đổi dung lượng lấy tốc độ**: int8 tốn gấp
đôi int4 trong RAM, đổi lại bỏ hẳn khâu gỡ nibble.

> **Điều kiện để đáng đổi: bạn đang COMPUTE-bound.** Nhìn cột GB/s: từ L2 trở đi đọc
> gấp đôi bytes mà vẫn nhanh hơn nhiều → lúc đó nút thắt là phép tính gỡ nibble chứ
> không phải băng thông. Nếu đã bandwidth-bound thì đổi ngược lại làm chậm đi.

### (b) SIMD tay giúp trên x86, KHÔNG giúp trên ARM

x86: L2 → L3 nhanh thêm **49%**. ARM: **chậm đi 3%**.

Lý do không phải NEON yếu — mà là **gcc đã tự vector hoá vòng scalar `dot_i8` trên ARM
rồi**. Viết intrinsic tay chỉ lặp lại việc compiler đã làm, còn thêm chi phí prologue.
Trên x86 gcc không nhận ra mẫu int8→int32 nên intrinsic mới có tác dụng.

> **Bài học chuyển giao: kiểm compiler đã tự làm chưa TRƯỚC KHI viết intrinsic.**
> ```bash
> gcc -O3 -march=native -fopt-info-vec matvec_ladder.c 2>&1 | grep vectorized
> objdump -d matvec_ladder | grep -cE "vpmadd|vdot|smlal"   # đếm lệnh SIMD
> ```
> Intrinsic tay khoá bạn vào một kiến trúc và khó đọc. Chỉ viết khi đo được là đáng.

Đây cũng chính là lý do [RESULTS.md:228-233](../RESULTS.md) kết luận SIMD trên ESP32
chỉ đáng ~15%: ở đó nút thắt đã chuyển sang bandwidth.

### (c) Song song hoá có GIÁ CỐ ĐỊNH, và giá đó khác nhau theo kiến trúc

Đây là phát hiện đáng giá nhất của thí nghiệm. Sample tự đo chi phí mở một vùng song
song **rỗng**:

| số luồng | x86 (i7-13650HX, lai P+E) | ARM (Cortex-A78AE, đồng nhất) |
|---:|---:|---:|
| 1 | 0.30 µs | 0.59 µs |
| 2 | 0.70 µs | 1.04 µs |
| 4 | 1.22 µs | 2.13 µs |
| 6 | — | 3.00 µs |
| 8 | 1.50 µs | — |
| 16 | **671 µs** ⚠️ | — |
| 20 | **7047 µs** ⚠️ | — |

**Trên CPU lai của Intel, dùng hết logical core làm overhead nổ 4700 lần.**
i7-13650HX có 6 P-core (12 luồng HT) + 8 E-core = 20 logical. Quá 8 luồng là tràn sang
E-core và HT sibling; OS phải đồng bộ qua các core dị chủng.

Đo L4 theo số luồng:

| luồng | ms | so L0 |
|---:|---:|---:|
| 1 | 0.0707 | 15.7× |
| 8 | 0.0110 | 59.0× |
| **12** | **0.0085** | **71.3×** |
| 20 | 6.9699 | **0.10× — chậm hơn cả L0** |

`omp_get_max_threads()` trả về 20 — **mặc định của OpenMP là lựa chọn tệ nhất**.
Cortex-A78AE đồng nhất nên không có vách này.

> Vì vậy sample **hiệu chuẩn lúc chạy** thay vì hardcode: đo overhead ở 1/2/4/8/…,
> chọn số luồng lớn nhất còn dưới 50 µs. Cùng cách ta đã hiệu chuẩn launch overhead
> của CUDA ở [`firmware/jetson/bench_cuda.cu`](../firmware/jetson/bench_cuda.cu).

---

## 5.3 Quy luật chung: song song có giá, việc phải lớn hơn giá

Cùng một hiện tượng, ba tên gọi:

| Kiến trúc | Chi phí cố định | Đo được | Hệ quả |
|---|---|---:|---|
| CPU | mở vùng song song OpenMP | 1.5 µs (8 luồng) / 7047 µs (20 luồng) | việc < 1.5 µs thì đừng chia |
| GPU | kernel launch | 2.5 µs (RTX 4060) / 3.71 µs (Orin) | model nhỏ → 50% thời gian là overhead |
| MCU | tạo task / notify | ~0 (task tạo 1 lần, chỉ notify) | gần như luôn đáng chia |

Quét batch trong sample cho thấy điểm hoà vốn trên x86 8 luồng:

```
     B   1 luồng ms   8 luồng ms   so sánh
     1       0.0739       6.0937     1 luồng thắng
    ...
    64       5.0606       6.9061     1 luồng thắng
   128      10.5490       6.5990     đa luồng THẮNG
```

Giống hệt bảng M-sweep của [`samples/gpu/bench_decode.cu`](../samples/gpu/bench_decode.cu):
`M=1` và `M=64` mất cùng thời gian trên GPU. **Song song hoá không miễn phí — nó chỉ
miễn phí khi bạn đã trả giá cố định rồi.**

---

## 5.4 Cái gì chuyển giao được giữa các kiến trúc

| Chuyển giao 100% | Phải đo lại mỗi kiến trúc |
|---|---|
| Tư duy roofline (bandwidth vs compute) | Giá trị machine balance |
| Kế toán 3 tầng theo access pattern | Dung lượng từng tầng |
| Giảm bytes đọc > tối ưu phép tính | Điểm hoà vốn cụ thể |
| Profile trước, tối ưu sau | Stage nào chiếm ưu thế |
| Đổi dung lượng lấy tốc độ khi compute-bound | Có đang compute-bound không |
| Verify từng bậc tối ưu | Ngưỡng sai số chấp nhận được |
| Song song có giá cố định | Giá đó là bao nhiêu |

**Không có "mẹo tối ưu" phổ quát.** Cùng model, cùng code, ba kiến trúc cho ba nút
thắt khác nhau:

| | ESP32-S3 | CPU | GPU (Orin) |
|---|---|---|---|
| Nút thắt | bandwidth PSRAM | compute (gỡ nibble) rồi cache bandwidth | **launch overhead 50%** |
| Lever tiếp theo | giảm bytes đọc | tăng batch cho threads đáng | CUDA Graphs |

---

## 5.5 Bài tập

1. Chạy `make -C samples/cpu run` trên máy bạn. Overhead ở bao nhiêu luồng thì nổ?
   Máy bạn có phải CPU lai không? (`lscpu | grep -i "model name"`)
2. Bỏ `-mavx2` khỏi Makefile rồi chạy lại. L3 đổi bao nhiêu? Compiler tự làm được
   đến đâu?
3. Đếm lệnh SIMD trong binary bằng `objdump -d`. So x86 và ARM. Vì sao ARM nhiều hơn
   dù ta không viết intrinsic cho nó?
4. Sửa `dot_i8` thành phiên bản dùng `float` thay vì `int8`. SIMD còn thắng không?
5. Chạy `make -C samples/cpu scaling`. Vẽ đồ thị ms theo số luồng, đánh dấu điểm gãy.

→ Tiếp: [06-toi-uu-gpu.md](06-toi-uu-gpu.md)
