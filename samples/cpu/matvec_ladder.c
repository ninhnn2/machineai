// Thang tối ưu một phép toán, đo từng bậc — trên CPU x86 và ARM.
//
// Phép toán: output head của model, y[V] = W[V,D] · x[D]. Đây là stage chiếm ưu
// thế trên ESP32 (67% thời gian) và là bài toán nhỏ gọn nhất để học tối ưu.
//
// Năm bậc, tái hiện đúng nhật ký RESULTS.md của bản ESP32, nhưng chạy được trên
// laptop và trên CPU của Jetson để thấy THANG NÀY KHÁC NHAU THEO KIẾN TRÚC:
//
//   L0  int4 + fp32          gỡ nibble mỗi lần, nhân float      (= llm.h matvec_q)
//   L1  int4 + int8 act      lượng tử hoá activation            (= llm.h matvec_q8)
//   L2  int8 staged          gỡ nibble MỘT LẦN lúc nạp          (= esp32_llm.ino)
//   L3  + SIMD               AVX2 / NEON dot int8
//   L4  + đa luồng           OpenMP, chia theo hàng             (= 2 core LX7)
//
// Mọi bậc phải cho CÙNG kết quả (trong sai số làm tròn của int8) — chương trình
// tự kiểm, vì tối ưu mà sai thì vô nghĩa.
//
//   make -C samples/cpu && ./samples/cpu/matvec_ladder firmware/model/model.bin
//
// Đọc kèm: docs/05-toi-uu-cpu.md

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../../firmware/common/llm.h"

#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_NAME "AVX2"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#ifdef __ARM_FEATURE_DOTPROD
#define SIMD_NAME "NEON+dotprod"
#else
#define SIMD_NAME "NEON"
#endif
#else
#define SIMD_NAME "(không có, biên dịch lại với -mavx2 hoặc -march=native)"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

// ---------------------------------------------------------------- L2: staging
// Gỡ int4 -> int8 MỘT LẦN, giống stage_head_int8() ở esp32_llm.ino:101.
// Đổi dung lượng lấy tốc độ: int8 tốn gấp đôi int4 trong RAM, nhưng mỗi token
// không còn phải gỡ nibble nữa. Đáng đổi khi còn dư RAM và đang compute-bound.
static int8_t *W8 = NULL;
static float  *ROWSCALE = NULL;

static void stage_int8(const QT *t) {
  W8 = (int8_t *)malloc((size_t)t->rows * t->cols);
  ROWSCALE = (float *)malloc((size_t)t->rows * t->n_groups * sizeof(float));
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = W8 + (size_t)r * t->cols;
    for (int j = 0; j < t->cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    for (int g = 0; g < t->n_groups; g++)
      ROWSCALE[(size_t)r * t->n_groups + g] =
          half2float(t->scales[(size_t)r * t->n_groups + g]);
  }
}

// int8 dot, scalar. Đây là hàm mà SIMD sẽ thay thế ở L3.
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

// ---------------------------------------------------------------- L3: SIMD
static inline int32_t dot_i8_simd(const int8_t *a, const int8_t *b, int n) {
#if defined(__AVX2__)
  // Mở rộng int8 -> int16 rồi _mm256_madd_epi16 (nhân + cộng cặp -> int32).
  // Không dùng maddubs_epi16 vì nó cần toán hạng đầu KHÔNG DẤU; weights của ta
  // có dấu (-7..7) nên đường int16 vừa đúng vừa dễ đọc.
  __m256i acc = _mm256_setzero_si256();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i av = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + i)));
    __m256i bv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + i)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  int32_t r = _mm_cvtsi128_si32(s);
  for (; i < n; i++) r += (int32_t)a[i] * (int32_t)b[i];
  return r;

#elif defined(__ARM_NEON)
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
#ifdef __ARM_FEATURE_DOTPROD
  // vdotq_s32: 16 tích int8 + cộng dồn trong MỘT lệnh. Cortex-A78 (Orin) có.
  for (; i + 16 <= n; i += 16)
    acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
#else
  for (; i + 16 <= n; i += 16) {
    int8x16_t av = vld1q_s8(a + i), bv = vld1q_s8(b + i);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(av), vget_low_s8(bv)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(av), vget_high_s8(bv)));
  }
#endif
  int32_t r = vaddvq_s32(acc);
  for (; i < n; i++) r += (int32_t)a[i] * (int32_t)b[i];
  return r;

#else
  return dot_i8(a, b, n);
#endif
}

// ---------------------------------------------------------------- các bậc
typedef void (*impl_fn)(const QT *, const float *, float *, int);

static void L0_int4_fp32(const QT *t, const float *x, float *y, int nthreads) {
  (void)nthreads;
  matvec_q(t, x, y);                      // nguyên xi từ firmware/common/llm.h
}

static void L1_int4_int8act(const QT *t, const float *x, float *y, int nthreads) {
  (void)nthreads;
  matvec_q8(t, x, y);                     // cũng từ llm.h, đường int8-activation
}

static int8_t g_xq[1024];
static float  g_xs;

static void quant_x(const float *x, int n) { quantize_act(x, n, g_xq, &g_xs); }

static void L2_staged_scalar(const QT *t, const float *x, float *y, int nthreads) {
  (void)nthreads;
  quant_x(x, t->cols);
  int G = t->group, NG = t->n_groups;
  for (int r = 0; r < t->rows; r++) {
    const int8_t *w = W8 + (size_t)r * t->cols;
    float acc = 0.f;
    for (int g = 0; g < NG; g++) {
      int a = g * G, b = a + G; if (b > t->cols) b = t->cols;
      acc += (float)dot_i8(g_xq + a, w + a, b - a) * ROWSCALE[(size_t)r * NG + g];
    }
    y[r] = acc * g_xs;
  }
}

static void L3_staged_simd(const QT *t, const float *x, float *y, int nthreads) {
  (void)nthreads;
  quant_x(x, t->cols);
  int G = t->group, NG = t->n_groups;
  for (int r = 0; r < t->rows; r++) {
    const int8_t *w = W8 + (size_t)r * t->cols;
    float acc = 0.f;
    for (int g = 0; g < NG; g++) {
      int a = g * G, b = a + G; if (b > t->cols) b = t->cols;
      acc += (float)dot_i8_simd(g_xq + a, w + a, b - a) * ROWSCALE[(size_t)r * NG + g];
    }
    y[r] = acc * g_xs;
  }
}

static void L4_simd_threads(const QT *t, const float *x, float *y, int nthreads) {
  quant_x(x, t->cols);                    // 1 lần, mọi luồng cùng đọc
  int G = t->group, NG = t->n_groups;
  // Các hàng output ĐỘC LẬP nhau -- đó là lý do chia theo hàng chia được sạch,
  // và cũng là lý do esp32_llm.ino chia head cho 2 core LX7 mà không đụng vào
  // bất kỳ phép dot nào.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
  for (int r = 0; r < t->rows; r++) {
    const int8_t *w = W8 + (size_t)r * t->cols;
    float acc = 0.f;
    for (int g = 0; g < NG; g++) {
      int a = g * G, b = a + G; if (b > t->cols) b = t->cols;
      acc += (float)dot_i8_simd(g_xq + a, w + a, b - a) * ROWSCALE[(size_t)r * NG + g];
    }
    y[r] = acc * g_xs;
  }
}

// ---------------------------------------------------------------- overhead
// Đo chi phí MỞ MỘT VÙNG SONG SONG trên chính máy này, thay vì tin một hằng số.
// Cùng lý do đã hiệu chuẩn launch overhead của CUDA: chi phí này phụ thuộc CPU,
// số luồng, và cách OS xếp luồng -- không có con số phổ quát.
static double measure_omp_overhead_us(int nthreads) {
#ifdef _OPENMP
  volatile int sink = 0;
  const int N = 2000;
  for (int k = 0; k < 200; k++) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < nthreads; i++) sink += i;
  }
  double a = now_ms();
  for (int k = 0; k < N; k++) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < nthreads; i++) sink += i;
  }
  return (now_ms() - a) * 1000.0 / N;
#else
  (void)nthreads; return 0.0;
#endif
}

// Chọn số luồng bằng ĐO, không bằng omp_get_max_threads().
//
// Trên CPU lai (Intel P-core + E-core), dùng hết logical core làm overhead nổ
// tung: đo được 1.7us ở 8 luồng nhưng 6600us ở 20 luồng trên i7-13650HX. Mặc
// định của OpenMP là số tệ nhất. Trên CPU đồng nhất (Cortex-A78 của Orin) thì
// không có vách này -- nên KHÔNG có hằng số đúng cho mọi máy, phải đo.
static int pick_threads(int maxn, int verbose) {
#ifndef _OPENMP
  (void)maxn; (void)verbose; return 1;
#else
  int cand[8], nc = 0;
  for (int n = 1; n <= maxn && nc < 8; n *= 2) cand[nc++] = n;
  if (nc < 8 && cand[nc - 1] != maxn) cand[nc++] = maxn;

  if (verbose)
    printf("\n--- hiệu chuẩn số luồng (overhead vùng song song RỖNG) ---\n");
  int best = 1;
  double best_ovh = 1e30;
  for (int i = 0; i < nc; i++) {
    double us = measure_omp_overhead_us(cand[i]);
    if (verbose) printf("  %3d luồng: %8.2f us%s\n", cand[i], us,
                        us > 50.0 ? "   <-- vách, tránh" : "");
    // Chấp nhận số luồng lớn nhất mà overhead còn dưới 50us.
    if (us < 50.0) { best = cand[i]; best_ovh = us; }
  }
  if (verbose)
    printf("  -> chọn %d luồng (overhead %.2f us)\n", best, best_ovh);
  return best;
#endif
}

// ---------------------------------------------------------------- batch sweep
// Cùng thí nghiệm với bench_decode.cu: tăng dần lượng việc mỗi lần gọi và xem
// song song hoá bắt đầu THẮNG ở đâu. B = số vector x xử lý cùng lúc, tức là
// prefill B token thay vì decode 1 token.
static void batch_sweep(const QT *t, int nthreads, double ovh_us) {
  printf("\n--- quét batch: khi nào đa luồng mới thắng? ---\n");
  printf("%6s %12s %12s %9s   %s\n", "B", "1 luồng ms", "%d luồng ms", "so sánh", "");
  printf("------------------------------------------------------------------------\n");
  int Bs[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  int nB = (int)(sizeof(Bs) / sizeof(Bs[0]));
  float *xs = (float *)malloc((size_t)256 * t->cols * sizeof(float));
  float *ys = (float *)malloc((size_t)256 * t->rows * sizeof(float));
  for (int i = 0; i < 256 * t->cols; i++) xs[i] = sinf(i * 0.11f);

  int G = t->group, NG = t->n_groups;
  for (int bi = 0; bi < nB; bi++) {
    int B = Bs[bi];
    double best1 = 1e30, bestN = 1e30;
    for (int pass = 0; pass < 2; pass++) {
      int nt = pass ? nthreads : 1;
      for (int k = 0; k < 3; k++) {
        double a = now_ms();
        for (int rep = 0; rep < 20; rep++) {
          // Song song trên toàn bộ (B x rows) công việc: mỗi cặp (token, hàng)
          // độc lập nhau, nên đây là cách chia tự nhiên nhất.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt) collapse(2)
#endif
          for (int b = 0; b < B; b++)
            for (int r = 0; r < t->rows; r++) {
              const int8_t *w = W8 + (size_t)r * t->cols;
              float acc = 0.f;
              for (int g = 0; g < NG; g++) {
                int lo = g * G, hi = lo + G; if (hi > t->cols) hi = t->cols;
                acc += (float)dot_i8_simd(g_xq + lo, w + lo, hi - lo)
                       * ROWSCALE[(size_t)r * NG + g];
              }
              ys[(size_t)b * t->rows + r] = acc * g_xs;
            }
        }
        double e = (now_ms() - a) / 20;
        if (pass) { if (e < bestN) bestN = e; }
        else      { if (e < best1) best1 = e; }
      }
    }
    const char *verdict = bestN < best1 * 0.95 ? "đa luồng THẮNG"
                        : bestN > best1 * 1.05 ? "1 luồng thắng (overhead)"
                                               : "hoà";
    printf("%6d %12.4f %12.4f %8.2fx   %s\n", B, best1, bestN, best1 / bestN, verdict);
  }
  printf("------------------------------------------------------------------------\n");
  if (ovh_us > 0)
    printf("Điểm hoà vốn ~ khi việc/lần gọi vượt %.1f us (overhead mở vùng song song).\n",
           ovh_us);
  free(xs); free(ys);
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
  const char *binp = argc > 1 ? argv[1] : "firmware/model/model.bin";
  int reps = argc > 2 ? atoi(argv[2]) : 200;
  int nthreads = argc > 3 ? atoi(argv[3]) : 0;
  int auto_threads = (nthreads <= 0);
#ifndef _OPENMP
  nthreads = 1; auto_threads = 0;
#endif

  size_t nb; uint8_t *buf = read_file(binp, &nb);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }
  QT *t = &m.tok_emb;                     // output head (tied embedding)

#ifdef _OPENMP
  if (auto_threads) nthreads = pick_threads(omp_get_max_threads(), 1);
#endif

  printf("========================================================================\n");
  printf(" matvec ladder | head [%d x %d] group=%d n_groups=%d\n",
         t->rows, t->cols, t->group, t->n_groups);
  printf(" SIMD: %s   luồng: %d   lặp: %d\n", SIMD_NAME, nthreads, reps);
  size_t w4 = (size_t)t->rows * t->row_bytes;
  size_t w8 = (size_t)t->rows * t->cols;
  printf(" weights: %.0f KB @int4  ->  %.0f KB @int8 (staged)\n", w4 / 1024.0, w8 / 1024.0);
  printf("========================================================================\n");

  stage_int8(t);

  float *x = (float *)malloc(t->cols * sizeof(float));
  for (int i = 0; i < t->cols; i++) x[i] = sinf(i * 0.37f) * 1.7f;
  float *ref = (float *)malloc(t->rows * sizeof(float));
  float *y   = (float *)malloc(t->rows * sizeof(float));

  struct { const char *name; impl_fn fn; const char *note; } L[] = {
    {"L0 int4 + fp32",   L0_int4_fp32,   "llm.h matvec_q -- mốc"},
    {"L1 int4 + int8act", L1_int4_int8act, "llm.h matvec_q8"},
    {"L2 int8 staged",   L2_staged_scalar, "gỡ nibble 1 lần (như ESP32)"},
    {"L3 + SIMD",        L3_staged_simd, SIMD_NAME},
    {"L4 + đa luồng",    L4_simd_threads, "chia theo hàng"},
  };
  int NL = (int)(sizeof(L) / sizeof(L[0]));

  L[0].fn(t, x, ref, nthreads);           // tham chiếu

  printf("\n%-20s %10s %9s %9s %10s   %s\n",
         "bậc", "ms", "GB/s", "so L0", "max|d|", "ghi chú");
  printf("------------------------------------------------------------------------\n");
  double t0_ms = 0;
  for (int i = 0; i < NL; i++) {
    L[i].fn(t, x, y, nthreads);           // warm-up + kiểm đúng
    double maxd = 0;
    for (int r = 0; r < t->rows; r++) {
      double d = fabs((double)y[r] - (double)ref[r]);
      if (d > maxd) maxd = d;
    }

    double best = 1e30;
    for (int k = 0; k < 3; k++) {         // lấy lần nhanh nhất, bớt nhiễu OS
      double a = now_ms();
      for (int rep = 0; rep < reps; rep++) L[i].fn(t, x, y, nthreads);
      double e = (now_ms() - a) / reps;
      if (e < best) best = e;
    }
    if (i == 0) t0_ms = best;

    size_t bytes = (i <= 1) ? w4 : w8;    // L0/L1 đọc int4, L2+ đọc int8
    printf("%-20s %10.4f %9.1f %8.2fx %10.2e   %s\n",
           L[i].name, best, bytes / (best * 1e-3) / 1e9, t0_ms / best,
           maxd, L[i].note);
    if (i == NL - 1) {
      double ovh = measure_omp_overhead_us(nthreads);
      printf("%-20s %10.4f %9s %8s %10s   %s\n",
             "  (overhead OMP)", ovh / 1000.0, "-", "-", "-",
             "đo bằng vùng song song RỖNG");
      if (ovh > best * 1000.0 * 0.5)
        printf("  ^^ overhead >= 50%% thời gian làm việc -> L4 chậm là ĐÚNG, không phải bug\n");
    }
  }
  printf("------------------------------------------------------------------------\n");

  printf("\nĐọc bảng:\n");
  printf("  max|d| khác 0 từ L1 trở đi là DO int8 activation, không phải bug.\n");
  printf("  Nó được kiểm bằng perplexity, không bằng sai số tuyệt đối --\n");
  printf("  xem firmware/host_verify/ppl.c và RESULTS.md:142.\n");
  printf("  GB/s: L2+ đọc GẤP ĐÔI bytes của L0 mà vẫn nhanh hơn -> lúc đó nút\n");
  printf("  thắt là COMPUTE (gỡ nibble), không phải bandwidth. Đổi dung lượng\n");
  printf("  lấy tốc độ chỉ đáng khi bạn đang compute-bound.\n");
  printf("  Nếu L3 gần bằng L2 -> đã bandwidth-bound, SIMD không giúp được nữa.\n");
  printf("  L4 CHẬM HƠN L3: việc quá nhỏ so với chi phí mở vùng song song. Cùng\n");
  printf("  hiện tượng với launch overhead của CUDA -- song song có giá cố định.\n");

  batch_sweep(t, nthreads, measure_omp_overhead_us(nthreads));
  return 0;
}
