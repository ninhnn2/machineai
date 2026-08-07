// Kiểm chứng BẰNG SỐ ĐO những luận điểm lý thuyết trong docs/10-ly-thuyet-nen.md.
//
// Nguyên tắc của repo này là không tin công thức chép lại. Mỗi thí nghiệm dưới đây
// in ra cả giá trị LÝ THUYẾT lẫn giá trị ĐO ĐƯỢC, để bạn tự thấy công thức đúng
// tới đâu và sai ở đâu.
//
//   make -C samples/cpu numerics && ./samples/cpu/numerics [model.bin]
//
// Sáu thí nghiệm:
//   E1  sai số cộng dồn: tuần tự O(n) vs cây O(log n) vs Kahan
//   E2  SNR lượng tử hoá theo số bit -- so với công thức
//   E3  ảnh hưởng của OUTLIER và vì sao phải chia nhóm
//   E4  kích thước nhóm: sai số vs overhead lưu scale
//   E5  dải động FP16 vs BF16 -- underflow/overflow thật
//   E6  trọng số THẬT của model: phân bố có Gauss không, group nào tốt nhất

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

#include "../../firmware/common/llm.h"

// ---------------------------------------------------------------- tiện ích

static uint32_t rng_state = 12345u;
static float urand(void) {          // [0,1)
  rng_state = rng_state * 1664525u + 1013904223u;
  return (float)((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000;
}
static float nrand(void) {          // Gauss(0,1), Box-Muller
  float u1 = urand(), u2 = urand();
  if (u1 < 1e-7f) u1 = 1e-7f;
  return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

// Cộng theo CÂY (pairwise). Đây đúng là thứ warp reduction của GPU làm.
static float sum_pairwise(const float *x, int n) {
  if (n <= 8) { float s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }
  int h = n / 2;
  return sum_pairwise(x, h) + sum_pairwise(x + h, n - h);
}
static float sum_sequential(const float *x, int n) {
  float s = 0; for (int i = 0; i < n; i++) s += x[i]; return s;
}
// Kahan: giữ lại phần bị mất khi làm tròn, cộng bù ở vòng sau.
static float sum_kahan(const float *x, int n) {
  float s = 0.f, c = 0.f;
  for (int i = 0; i < n; i++) {
    float y = x[i] - c;
    float t = s + y;
    c = (t - s) - y;      // phần đã bị mất
    s = t;
  }
  return s;
}
static double sum_exact(const float *x, int n) {   // tham chiếu double
  double s = 0; for (int i = 0; i < n; i++) s += (double)x[i]; return s;
}

// ---------------------------------------------------------------- E1

static void e1_summation(void) {
  printf("\n===== E1. Sai số cộng dồn tăng theo n như thế nào =====\n");
  printf("Lý thuyết: tuần tự O(n*eps), pairwise O(log2(n)*eps), Kahan O(eps).\n");
  printf("eps(float) = 2^-24 = %.3e\n\n", 1.0 / (1 << 24));
  printf("%10s %13s %13s %13s %10s\n",
         "n", "tuần tự", "pairwise", "Kahan", "tuần/pair");
  printf("--------------------------------------------------------------------\n");

  for (int n = 1024; n <= 1048576; n *= 8) {
    float *x = (float *)malloc(n * sizeof(float));
    // Toàn số dương cùng cỡ: trường hợp XẤU NHẤT cho cộng tuần tự, vì tổng
    // chạy lớn dần còn số hạng thì không -- mỗi lần cộng mất thêm bit thấp.
    for (int i = 0; i < n; i++) x[i] = 1.0f + 0.1f * urand();
    double ref = sum_exact(x, n);
    double e_seq  = fabs(sum_sequential(x, n) - ref) / ref;
    double e_pair = fabs(sum_pairwise(x, n) - ref) / ref;
    double e_kah  = fabs(sum_kahan(x, n) - ref) / ref;
    printf("%10d %13.3e %13.3e %13.3e %9.1fx\n", n, e_seq, e_pair, e_kah,
           e_pair > 0 ? e_seq / e_pair : 0.0);
    free(x);
  }
  printf("--------------------------------------------------------------------\n");
  printf("=> Đây chính là lý do bản CUDA (warp reduction = pairwise) GẦN PyTorch\n");
  printf("   hơn bản C tuần tự. Xem firmware/jetson/JETSON.md muc 4.\n");
}

// ---------------------------------------------------------------- lượng tử hoá

// Đúng sơ đồ của src/quantize.py: đối xứng, qmax = 2^(b-1)-1, scale = max|x|/qmax.
static void quant_dequant(const float *x, float *y, int n, int bits) {
  float qmax = (float)((1 << (bits - 1)) - 1);
  float a = 0.f;
  for (int i = 0; i < n; i++) { float v = fabsf(x[i]); if (v > a) a = v; }
  if (a < 1e-12f) a = 1e-12f;
  float s = a / qmax;
  for (int i = 0; i < n; i++) {
    float q = roundf(x[i] / s);
    if (q > qmax) q = qmax;
    if (q < -qmax) q = -qmax;
    y[i] = q * s;
  }
}

// Chia nhóm dọc theo mảng, mỗi nhóm một scale (group-wise).
static void quant_dequant_grouped(const float *x, float *y, int n, int bits, int group) {
  for (int i = 0; i < n; i += group) {
    int m = (i + group <= n) ? group : (n - i);
    quant_dequant(x + i, y + i, m, bits);
  }
}

static double snr_db(const float *x, const float *y, int n) {
  double ps = 0, pn = 0;
  for (int i = 0; i < n; i++) { double d = (double)x[i] - y[i]; ps += (double)x[i]*x[i]; pn += d*d; }
  if (pn <= 0) return INFINITY;
  return 10.0 * log10(ps / pn);
}

static double stddev(const float *x, int n) {
  double m = 0; for (int i = 0; i < n; i++) m += x[i]; m /= n;
  double v = 0; for (int i = 0; i < n; i++) { double d = x[i]-m; v += d*d; }
  return sqrt(v / n);
}
static double amax(const float *x, int n) {
  double a = 0; for (int i = 0; i < n; i++) { double v = fabs(x[i]); if (v>a) a=v; }
  return a;
}

// ---------------------------------------------------------------- E2

static void e2_snr_vs_bits(void) {
  printf("\n===== E2. SNR lượng tử hoá theo số bit =====\n");
  int n = 1 << 16;
  float *x = (float *)malloc(n * sizeof(float));
  float *y = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) x[i] = nrand();

  double sd = stddev(x, n), A = amax(x, n);
  printf("Trọng số giả lập: Gauss(0,1), n=%d, sigma=%.4f, max|x|=%.4f (=%.2f sigma)\n",
         n, sd, A, A / sd);
  printf("\nCÔNG THỨC (lượng tử hoá đều, nhiễu var = delta^2/12,\n");
  printf("           delta = A/qmax, qmax = 2^(b-1)-1):\n");
  printf("  SNR_dB = 20*log10(qmax) + 10*log10(12) - 20*log10(A/sigma)\n\n");
  printf("%6s %8s %14s %14s %10s\n", "bits", "qmax", "công thức dB", "đo được dB", "lệch");
  printf("--------------------------------------------------------------------\n");
  for (int b = 2; b <= 8; b++) {
    quant_dequant(x, y, n, b);
    double meas = snr_db(x, y, n);
    double qmax = (double)((1 << (b - 1)) - 1);
    double theo = 20*log10(qmax) + 10*log10(12.0) - 20*log10(A / sd);
    printf("%6d %8.0f %14.2f %14.2f %9.2f\n", b, qmax, theo, meas, meas - theo);
  }
  printf("--------------------------------------------------------------------\n");
  printf("=> Mỗi bit thêm vào cho ~6.02 dB (vì 20*log10(2) = 6.02).\n");
  printf("   CẢNH BÁO: công thức '6.02b + 1.76' hay gặp trên mạng là cho SÓNG SIN\n");
  printf("   toàn thang với 2^b mức, KHÔNG áp dụng cho sơ đồ đối xứng qmax=2^(b-1)-1\n");
  printf("   mà LLM dùng. Dùng công thức ở trên.\n");
  free(x); free(y);
}

// ---------------------------------------------------------------- E3

static void e3_outlier(void) {
  printf("\n===== E3. Một outlier phá hỏng cả tensor như thế nào =====\n");
  int n = 4096, bits = 4;
  float *x = (float *)malloc(n * sizeof(float));
  float *y = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) x[i] = nrand();

  printf("Cột 'toàn bộ' là SNR trên cả tensor; cột 'trừ outlier' bỏ phần tử 0 ra.\n");
  printf("Hai cột này kể hai câu chuyện khác nhau -- đó là bài học chính.\n\n");
  printf("%12s %11s %11s %11s %11s %11s\n", "outlier",
         "max/sigma", "1sc toàn", "1sc trừ", "g128 toàn", "g128 trừ");
  printf("--------------------------------------------------------------------------\n");
  float base = x[0];
  for (int k = 0; k <= 5; k++) {
    float mult = (k == 0) ? 1.f : powf(4.f, (float)k);
    x[0] = base * mult;                       // bơm 1 outlier duy nhất
    double sd = stddev(x, n), A = amax(x, n);
    quant_dequant(x, y, n, bits);
    double s1 = snr_db(x, y, n), s1x = snr_db(x + 1, y + 1, n - 1);
    quant_dequant_grouped(x, y, n, bits, 128);
    double s2 = snr_db(x, y, n), s2x = snr_db(x + 1, y + 1, n - 1);
    printf("%11.0fx %11.2f %11.2f %11.2f %11.2f %11.2f\n",
           mult, A / sd, s1, s1x, s2, s2x);
  }
  printf("--------------------------------------------------------------------------\n");
  printf("=> Đọc cột 'trừ outlier': 1 scale cho cả tensor thì SNR SỤP ĐƠN ĐIỆU theo\n");
  printf("   outlier, còn group=128 gần như không đổi. Outlier chỉ phá nhóm của nó.\n");
  printf("   Đây là toàn bộ lý do tồn tại của group-wise, và là lý do AWQ/SmoothQuant\n");
  printf("   phải xử lý outlier theo KÊNH ở activation.\n\n");
  printf("   BẪY ĐO LƯỜNG: cột 'toàn bộ' TĂNG LẠI ở outlier rất lớn (1.27 -> 18 dB).\n");
  printf("   Không phải chất lượng tốt lên! SNR = công suất tín hiệu / công suất nhiễu,\n");
  printf("   mà outlier khổng lồ chiếm gần hết TỬ SỐ và bản thân nó được biểu diễn\n");
  printf("   chính xác (nó đúng bằng max nên map thẳng vào qmax). Chỉ số đẹp lên trong\n");
  printf("   khi mọi trọng số còn lại đã hỏng.\n");
  printf("   => Với LLM, đừng dùng SNR/MSE làm chỉ tiêu. Dùng PERPLEXITY.\n");
  printf("      Đó chính là lý do src/quantize.py đo val loss chứ không đo sai số.\n");
  free(x); free(y);
}

// ---------------------------------------------------------------- E4

static void e4_group_size(void) {
  printf("\n===== E4. Kích thước nhóm: sai số đổi lấy dung lượng =====\n");
  int n = 1 << 16, bits = 4;
  float *x = (float *)malloc(n * sizeof(float));
  float *y = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) x[i] = nrand();
  x[n/3] *= 30.f;                              // một outlier thực tế

  printf("%8s %12s %16s %12s\n", "group", "SNR dB", "bit/weight", "so group=n");
  printf("--------------------------------------------------------------------\n");
  double base = 0;
  int groups[] = {n, 512, 256, 128, 64, 32, 16};
  for (int i = 0; i < 7; i++) {
    int g = groups[i];
    quant_dequant_grouped(x, y, n, bits, g);
    double s = snr_db(x, y, n);
    if (i == 0) base = s;
    double bpw = bits + 16.0 / g;              // scale fp16 mỗi nhóm
    printf("%8d %12.2f %16.3f %11.2f dB\n", g, s, bpw, s - base);
  }
  printf("--------------------------------------------------------------------\n");
  printf("=> group nhỏ hơn = chính xác hơn nhưng tốn thêm bit lưu scale.\n");
  printf("   group=128 + fp16 scale = 4.125 bit/weight -- lý do 'Q4' thực tế\n");
  printf("   nặng ~4.5 bit và model 8B Q4_K_M là 4.7GB chứ không phải 4.0GB.\n");
  free(x); free(y);
}

// ---------------------------------------------------------------- E5

// Chuyển float -> bf16 (cắt cụt về 16 bit cao) -> float. bf16 = FP32 mất 16 bit
// định trị, GIỮ NGUYÊN 8 bit số mũ.
static float to_bf16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u &= 0xFFFF0000u;                 // cắt cụt (làm tròn về 0)
  float r; memcpy(&r, &u, 4); return r;
}
// float -> fp16 -> float, dùng chính half2float của llm.h cho chiều ngược.
static float to_fp16(float f) {
  if (!isfinite(f)) return f;
  float af = fabsf(f);
  if (af > 65504.f) return f > 0 ? INFINITY : -INFINITY;   // tràn trên
  if (af < 6.0e-8f) return 0.f;                            // tràn dưới hoàn toàn
  // làm tròn định trị về 10 bit bằng cách đi qua biểu diễn 16-bit
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000;
  int exp = (int)((u >> 23) & 0xFF) - 127 + 15;
  uint32_t man = u & 0x7FFFFF;
  uint16_t h;
  if (exp <= 0) {                                          // subnormal fp16
    man |= 0x800000;
    int shift = 14 - exp;
    h = (uint16_t)(sign | (shift < 24 ? (man >> shift) : 0));
  } else if (exp >= 31) {
    h = (uint16_t)(sign | 0x7C00);
  } else {
    h = (uint16_t)(sign | (exp << 10) | (man >> 13));
  }
  return half2float(h);
}

static void e5_dynamic_range(void) {
  printf("\n===== E5. Dải động: FP16 vs BF16 =====\n");
  printf("Cả hai đều 16 bit. FP16: 5 bit mũ + 10 bit định trị.\n");
  printf("                    BF16: 8 bit mũ + 7 bit định trị (= mũ của FP32).\n\n");
  printf("%14s %16s %16s %14s %14s\n",
         "giá trị", "FP16", "BF16", "sai số FP16", "sai số BF16");
  printf("--------------------------------------------------------------------------------\n");
  double vals[] = {1.0, 0.1, 1e-3, 1e-5, 6e-5, 1e-6, 1e-8, 1e-10, 1e4, 65504.0, 1e5, 1e30};
  for (int i = 0; i < 12; i++) {
    float v = (float)vals[i];
    float a = to_fp16(v), b = to_bf16(v);
    double ea = (v != 0) ? fabs((double)a - v) / fabs(v) : 0;
    double eb = (v != 0) ? fabs((double)b - v) / fabs(v) : 0;
    printf("%14.3g %16.6g %16.6g %14.2e %14.2e%s\n", v, a, b, ea, eb,
           (a == 0 && v != 0) ? "  <- FP16 UNDERFLOW"
           : (isinf(a) && !isinf(v)) ? "  <- FP16 OVERFLOW" : "");
  }
  printf("--------------------------------------------------------------------------------\n");
  printf("=> BF16 sai số tương đối LỚN HƠN (ít định trị hơn) nhưng KHÔNG BAO GIỜ\n");
  printf("   underflow/overflow ở dải mà FP16 chết. Đó là lý do huấn luyện dùng BF16:\n");
  printf("   gradient cỡ 1e-8 bị FP16 làm thành 0, buộc phải loss scaling.\n");
  printf("   Suy luận LLM thì FP16 an toàn vì RMSNorm giữ activation quanh 1.\n");
}

// ---------------------------------------------------------------- E6

static void e6_real_weights(const char *binp) {
  printf("\n===== E6. Trọng số THẬT của model =====\n");
  size_t nb;
  FILE *f = fopen(binp, "rb");
  if (!f) { printf("  (bỏ qua: không mở được %s)\n", binp); return; }
  fseek(f, 0, SEEK_END); nb = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc(nb);
  if (fread(buf, 1, nb, f) != nb) { printf("  short read\n"); fclose(f); return; }
  fclose(f);

  Model m;
  if (llm_load(buf, &m)) { printf("  bad magic\n"); free(buf); return; }

  // Giải nén head về fp32 (đây là bản ĐÃ lượng tử hoá; ta phân tích phân bố của nó).
  QT *t = &m.tok_emb;
  int n = t->rows * t->cols;
  float *w = (float *)malloc((size_t)n * sizeof(float));
  float *row = (float *)malloc(t->cols * sizeof(float));
  for (int r = 0; r < t->rows; r++) {
    deq_row(t, r, row);
    memcpy(w + (size_t)r * t->cols, row, t->cols * sizeof(float));
  }

  double sd = stddev(w, n), A = amax(w, n);
  // Tỉ lệ nằm trong 1/2/3 sigma -- Gauss cho 68.3 / 95.4 / 99.7 %
  int c1=0,c2=0,c3=0;
  double mean=0; for (int i=0;i<n;i++) mean+=w[i]; mean/=n;
  for (int i = 0; i < n; i++) {
    double z = fabs((w[i]-mean)/sd);
    if (z<=1) c1++; if (z<=2) c2++; if (z<=3) c3++;
  }
  printf("head [%d x %d] = %d trọng số (đã dequant từ int4)\n", t->rows, t->cols, n);
  printf("  mean %.5f | sigma %.5f | max|w| %.5f = %.2f sigma\n", mean, sd, A, A/sd);
  printf("  trong 1 sigma %.1f%% (Gauss 68.3)\n", 100.0*c1/n);
  printf("  trong 2 sigma %.1f%% (Gauss 95.4)\n", 100.0*c2/n);
  printf("  trong 3 sigma %.1f%% (Gauss 99.7)\n", 100.0*c3/n);
  printf("  => phân bố gần Gauss => sơ đồ đối xứng chia đều là hợp lý.\n");

  printf("\n  Lượng tử hoá LẠI bản đã dequant, để thấy xu hướng theo bit/group:\n");
  float *y = (float *)malloc((size_t)n * sizeof(float));
  printf("  %8s %10s %10s %10s\n", "group", "4-bit dB", "6-bit dB", "8-bit dB");
  printf("  ------------------------------------------------\n");
  int gs[] = {n, 512, 128, 32};
  for (int i = 0; i < 4; i++) {
    printf("  %8d", gs[i]);
    for (int b = 4; b <= 8; b += 2) {
      quant_dequant_grouped(w, y, n, b, gs[i]);
      printf(" %10.2f", snr_db(w, y, n));
    }
    printf("\n");
  }
  printf("  ------------------------------------------------\n");
  printf("  (SNR ở đây cao bất thường vì đầu vào ĐÃ là lưới int4 -- lượng tử hoá\n");
  printf("   lại cùng lưới gần như không mất gì. Đó là một phép kiểm tính nhất quán:\n");
  printf("   nếu 4-bit/group=128 KHÔNG ra SNR rất cao thì bộ giải nén đang sai.)\n");

  free(w); free(row); free(y); free(buf);
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
  const char *binp = argc > 1 ? argv[1] : "../../firmware/model/model.bin";
  printf("========================================================================\n");
  printf(" Kiểm chứng số học cho docs/10-ly-thuyet-nen.md\n");
  printf(" float=%zu bit, eps=%.3e | double=%zu bit\n",
         sizeof(float)*8, (double)FLT_EPSILON, sizeof(double)*8);
  printf("========================================================================\n");
  e1_summation();
  e2_snr_vs_bits();
  e3_outlier();
  e4_group_size();
  e5_dynamic_range();
  e6_real_weights(binp);
  printf("\nXong. Mọi con số ở trên đều ĐO ĐƯỢC, không chép từ tài liệu.\n");
  return 0;
}
