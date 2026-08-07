// CUDA runtime for the PLE TinyLM -- the Jetson counterpart of ../common/llm.h.
//
// Reads the SAME model.bin that src/export.py writes and the ESP32 flashes, and
// is checked against the SAME golden logits. Only the arithmetic moves to the GPU;
// llm_load() from ../common/llm.h is reused verbatim to parse the header and bind
// tensors on the host, then each tensor is uploaded once.
//
// Why this port is worth doing, on a board that could obviously just run llama.cpp:
// decode at batch=1 is memory-bound (see ../../../jetson-optim/01-roofline.md), so
// every kernel here is a bandwidth exercise, not a math exercise. The profile it
// prints is directly comparable to the ESP32 profile in RESULTS.md.
//
// THE ONE THING THAT DIFFERS FROM THE C PORT: floating-point summation order.
// llm.h accumulates each dot product strictly left-to-right, so it reproduces
// PyTorch to 1e-5. A warp reduction sums in tree order. Float addition is not
// associative, so the GPU CANNOT be bit-identical -- see JETSON.md "Sai số".
#ifndef LLM_CUDA_CUH
#define LLM_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/llm.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------- device types

// Device mirror of QT. Same ragged int4 layout as the flash artifact: no repack,
// no padding -- the bytes on the GPU are the bytes in model.bin.
typedef struct {
  const uint8_t  *codes;
  const uint16_t *scales;
  int rows, cols, group, n_groups, row_bytes;
} QTd;

typedef struct {
  Cfg c;
  QTd tok_emb, ple_model_proj, ple_table;
  const float *ple_proj_norm, *out_norm;
  const float *attn_norm[32], *ffn_norm[32], *ple_norm[32];
  QTd qkv[32], attn_proj[32], gate[32], up[32], down[32], ple_gate[32], ple_proj[32];
} ModelD;

typedef struct {
  float *x, *h, *qkv, *att, *g1, *g2, *ple, *tmpP, *trow, *logits, *scores;
  float *kcache, *vcache;
  float *rope_c, *rope_s;
  int   *d_tok, *d_pos;      // per-token state, read by kernels via pointer
} ScratchD;

// ---------------------------------------------------------------- kernels

// One WARP per output row. Each lane strides over the columns, unpacks its own
// int4 nibbles, multiplies by the group scale, then a tree reduction inside the
// warp. 32 lanes over cols=96 means 3 columns per lane -- short, but the head has
// 25k rows so the grid is huge and the GPU stays full.
//
// Read the addressing carefully: `row_bytes = ceil(cols/2)` and nibble j lives in
// byte j>>1, low half for even j. That is exactly deq_row() in llm.h, just with
// the loop distributed across lanes.
__global__ void k_matvec_q4(const uint8_t *__restrict__ codes,
                            const uint16_t *__restrict__ scales,
                            const float *__restrict__ x, float *__restrict__ y,
                            int rows, int cols, int group, int n_groups, int row_bytes) {
  // INVARIANT: blockDim.x must be a multiple of 32. Then blockIdx.x*blockDim.x is
  // too, so `warp` is UNIFORM across all 32 lanes of a hardware warp -- the early
  // return below retires whole warps, never part of one. That is what makes the
  // 0xffffffff mask in __shfl_down_sync legal: it asserts all 32 lanes are still
  // converged. Launch this with, say, 100 threads and the mask becomes a lie and
  // the reduction reads garbage from retired lanes -- silently, no error.
  // mv() below always launches warps_per_block*32, which upholds this.
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp >= rows) return;

  const uint8_t *row = codes + (size_t)warp * row_bytes;
  const uint16_t *sc = scales + (size_t)warp * n_groups;

  float acc = 0.f;
  for (int j = lane; j < cols; j += 32) {
    uint8_t byte = row[j >> 1];
    int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
    float s = __half2float(__ushort_as_half(sc[j / group]));
    acc = fmaf((float)(code - 8) * s, x[j], acc);
  }
  // butterfly reduction: 5 steps for 32 lanes
  for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) y[warp] = acc;
}

// Dequantize ONE row -- the embedding lookup and the PLE table lookup.
// This is the "table" tier of budget.py: on the ESP32 it is a flash random read,
// here it is a tiny global read. Either way it is nearly free; what matters is
// that we touch one row, not the whole matrix.
//
// The row index arrives as a DEVICE POINTER, not a value. That is what makes the
// whole forward pass capturable into a CUDA graph: graph nodes bake their kernel
// arguments at capture time, so anything that changes per token (the token id,
// the position) has to live in memory the graph reads, not in an argument.
// See llm_cuda_graph_* below.
__global__ void k_deq_row(const uint8_t *__restrict__ codes,
                          const uint16_t *__restrict__ scales,
                          const int *__restrict__ r_dev, float *__restrict__ out,
                          int cols, int group, int n_groups, int row_bytes) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  int r = *r_dev;
  const uint8_t *row = codes + (size_t)r * row_bytes;
  uint8_t byte = row[j >> 1];
  int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
  out[j] = (float)(code - 8) *
           __half2float(__ushort_as_half(scales[(size_t)r * n_groups + j / group]));
}

// RMSNorm over n<=1024 in a single block. Mirrors llm.h:207 exactly:
// out = w * x * rsqrt(mean(x^2) + eps)
//
// NOTE: `x` and `out` are deliberately NOT __restrict__. llm.h calls this
// in-place (rmsnorm(s->h, w, D, s->h) at llm.h:367), and so do we. The kernel is
// safe in-place because every read of x happens before the __syncthreads() that
// precedes any write to out -- but promising no-alias to the compiler would let
// it reorder those and silently corrupt the result. The scalar C version gets
// away with __restrict__-free code for the same reason.
__global__ void k_rmsnorm(const float *x, const float *__restrict__ w,
                          int n, float *out) {
  __shared__ float red[32];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < n; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
  for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t == 0) {
    float s = 0.f;
    for (int i = 0; i < (blockDim.x + 31) / 32; i++) s += red[i];
    red[0] = rsqrtf(s / n + RMS_EPS);
  }
  __syncthreads();
  float inv = red[0];
  for (int i = t; i < n; i += blockDim.x) out[i] = w[i] * x[i] * inv;
}

// Per-layer RMSNorm over the [L, P] PLE slices: one block per layer slice.
__global__ void k_rmsnorm_slices(float *__restrict__ x, const float *__restrict__ w,
                                 int P) {
  extern __shared__ float sm[];
  float *base = x + blockIdx.x * P;
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < P; i += blockDim.x) ss = fmaf(base[i], base[i], ss);
  for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
  if ((t & 31) == 0) sm[t >> 5] = ss;
  __syncthreads();
  if (t == 0) {
    float s = 0.f;
    for (int i = 0; i < (blockDim.x + 31) / 32; i++) s += sm[i];
    sm[0] = rsqrtf(s / P + RMS_EPS);
  }
  __syncthreads();
  float inv = sm[0];
  for (int i = t; i < P; i += blockDim.x) base[i] = w[i] * base[i] * inv;
}

// RoPE frequency table for this position. Computed ONCE per token, not L*H times
// -- the same optimisation llm.h:288 documents ("identical across every head and
// layer at a position").
__global__ void k_rope_freqs(float *c, float *s, int half, float theta,
                             const int *__restrict__ pos_dev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half) return;
  int pos = *pos_dev;
  float f = powf(theta, -2.f * i / (2.f * half));
  c[i] = cosf(pos * f);
  s[i] = sinf(pos * f);
}

// Split-half RoPE applied to q and k in place. NOT interleaved -- see JETSON.md.
__global__ void k_rope_apply(float *q, float *k, const float *c, const float *s,
                             int H, int Dh) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = Dh >> 1;
  if (idx >= H * half) return;
  int h = idx / half, i = idx % half;
  float *qh = q + h * Dh, *kh = k + h * Dh;
  float cc = c[i], sn = s[i];
  float q1 = qh[i], q2 = qh[i + half];
  qh[i] = q1 * cc - q2 * sn; qh[i + half] = q2 * cc + q1 * sn;
  float k1 = kh[i], k2 = kh[i + half];
  kh[i] = k1 * cc - k2 * sn; kh[i + half] = k2 * cc + k1 * sn;
}

__global__ void k_store_kv(const float *k, const float *v, float *kc, float *vc,
                           const int *__restrict__ pos_dev, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D) return;
  int pos = *pos_dev;
  kc[(size_t)pos * D + i] = k[i];
  vc[(size_t)pos * D + i] = v[i];
}

// One BLOCK per attention head, decode step (query length 1).
// Two passes with a max subtraction, exactly like llm.h:327 -- without it exp()
// overflows fp32. FlashAttention is the one-pass online-softmax version of this;
// at ctx 512 the naive form is fine and much easier to read.
__global__ void k_attention(const float *__restrict__ q, const float *__restrict__ kc,
                            const float *__restrict__ vc, float *__restrict__ out,
                            float *__restrict__ scores,
                            const int *__restrict__ pos_dev, int D, int Dh) {
  int h = blockIdx.x;
  int t = threadIdx.x;
  int pos = *pos_dev;
  const float *qh = q + h * Dh;
  float *sc = scores + (size_t)h * (pos + 1);
  float scale = rsqrtf((float)Dh);

  __shared__ float sred[32];
  // pass 1: scores + max
  float local_max = -1e30f;
  for (int p = t; p <= pos; p += blockDim.x) {
    const float *kt = kc + (size_t)p * D + h * Dh;
    float dot = 0.f;
    for (int i = 0; i < Dh; i++) dot = fmaf(qh[i], kt[i], dot);
    dot *= scale;
    sc[p] = dot;
    if (dot > local_max) local_max = dot;
  }
  for (int off = 16; off; off >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
  if ((t & 31) == 0) sred[t >> 5] = local_max;
  __syncthreads();
  if (t == 0) {
    float m = -1e30f;
    for (int i = 0; i < (blockDim.x + 31) / 32; i++) m = fmaxf(m, sred[i]);
    sred[0] = m;
  }
  __syncthreads();
  float maxs = sred[0];

  // pass 2: exp, sum, weighted V
  float local_sum = 0.f;
  for (int p = t; p <= pos; p += blockDim.x) {
    float w = __expf(sc[p] - maxs);
    sc[p] = w;
    local_sum += w;
  }
  for (int off = 16; off; off >>= 1) local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
  if ((t & 31) == 0) sred[t >> 5] = local_sum;
  __syncthreads();
  if (t == 0) {
    float s = 0.f;
    for (int i = 0; i < (blockDim.x + 31) / 32; i++) s += sred[i];
    sred[0] = s;
  }
  __syncthreads();
  float denom = sred[0];

  // accumulate V: one thread per head dim (Dh is small, 24 here)
  for (int i = t; i < Dh; i += blockDim.x) {
    float acc = 0.f;
    for (int p = 0; p <= pos; p++) acc = fmaf(sc[p], vc[(size_t)p * D + h * Dh + i], acc);
    out[h * Dh + i] = acc / denom;
  }
}

// ---- elementwise ----------------------------------------------------------

__global__ void k_add(float *x, const float *h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += h[i];
}

__global__ void k_swiglu(float *g1, const float *g2, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) g1[i] = (g1[i] / (1.f + __expf(-g1[i]))) * g2[i];   // silu(gate)*up
}

// erf-GELU, matching llm.h:213 -- NOT the tanh approximation. They differ by
// ~1e-3 which is enough to fail a golden check.
__global__ void k_gelu_mul(float *g, const float *ple, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) g[i] = 0.5f * g[i] * (1.f + erff(g[i] * 0.70710678f)) * ple[i];
}

__global__ void k_scale(float *x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

// ple = (proj + table*sqrt(P)) * (1/sqrt(2))   -- model.py:214
__global__ void k_ple_mix(const float *proj, const float *table, float *out,
                          int n, float sp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (proj[i] + table[i] * sp) * 0.70710678f;
}

// ---------------------------------------------------------------- host side

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

static QTd upload_qt(const QT *t) {
  QTd d;
  d.rows = t->rows; d.cols = t->cols; d.group = t->group;
  d.n_groups = t->n_groups; d.row_bytes = t->row_bytes;
  size_t nc = (size_t)t->rows * t->row_bytes;
  size_t ns = (size_t)t->rows * t->n_groups * sizeof(uint16_t);
  uint8_t *pc; uint16_t *ps;
  CUDA_CHECK(cudaMalloc(&pc, nc));
  CUDA_CHECK(cudaMalloc(&ps, ns));
  CUDA_CHECK(cudaMemcpy(pc, t->codes, nc, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps, t->scales, ns, cudaMemcpyHostToDevice));
  d.codes = pc; d.scales = ps;
  return d;
}

static const float *upload_f(const float *src, int n) {
  float *p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(p, src, n * sizeof(float), cudaMemcpyHostToDevice));
  return p;
}

// Bytes a tensor occupies on device -- used to report achieved bandwidth per stage.
static size_t qt_bytes(const QTd *t) {
  return (size_t)t->rows * t->row_bytes + (size_t)t->rows * t->n_groups * 2;
}

static size_t llm_cuda_upload(const Model *m, ModelD *d) {
  d->c = m->c;
  size_t total = 0;
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim;

  d->tok_emb = upload_qt(&m->tok_emb);              total += qt_bytes(&d->tok_emb);
  d->ple_model_proj = upload_qt(&m->ple_model_proj); total += qt_bytes(&d->ple_model_proj);
  d->ple_table = upload_qt(&m->ple_table);          total += qt_bytes(&d->ple_table);
  d->ple_proj_norm = upload_f(m->ple_proj_norm, P);
  d->out_norm = upload_f(m->out_norm, D);
  for (int i = 0; i < L; i++) {
    d->attn_norm[i] = upload_f(m->attn_norm[i], D);
    d->ffn_norm[i]  = upload_f(m->ffn_norm[i], D);
    d->ple_norm[i]  = upload_f(m->ple_norm[i], D);
    d->qkv[i]       = upload_qt(&m->qkv[i]);       total += qt_bytes(&d->qkv[i]);
    d->attn_proj[i] = upload_qt(&m->attn_proj[i]); total += qt_bytes(&d->attn_proj[i]);
    d->gate[i]      = upload_qt(&m->gate[i]);      total += qt_bytes(&d->gate[i]);
    d->up[i]        = upload_qt(&m->up[i]);        total += qt_bytes(&d->up[i]);
    d->down[i]      = upload_qt(&m->down[i]);      total += qt_bytes(&d->down[i]);
    d->ple_gate[i]  = upload_qt(&m->ple_gate[i]);  total += qt_bytes(&d->ple_gate[i]);
    d->ple_proj[i]  = upload_qt(&m->ple_proj[i]);  total += qt_bytes(&d->ple_proj[i]);
  }
  return total;
}

static void llm_cuda_alloc(const Cfg *c, ScratchD *s) {
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;
  int H = c->n_heads, Dh = D / H;
#define A(p, n) CUDA_CHECK(cudaMalloc(&s->p, (size_t)(n) * sizeof(float)))
  A(x, D); A(h, F > D ? F : D); A(qkv, 3 * D); A(att, D);
  A(g1, F); A(g2, P > F ? P : F);
  A(ple, L * P); A(tmpP, L * P); A(trow, L * P);
  A(logits, V); A(scores, (size_t)H * S);
  A(kcache, (size_t)L * S * D); A(vcache, (size_t)L * S * D);
  A(rope_c, Dh / 2); A(rope_s, Dh / 2);
#undef A
  CUDA_CHECK(cudaMalloc(&s->d_tok, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&s->d_pos, sizeof(int)));
}

// Đẩy (token, pos) của bước này xuống device.
//
// PHẢI là KERNEL, không được là cudaMemcpyAsync từ pinned host memory.
// Lý do là một bug thật đã bị tầng verify bắt được: memcpyAsync từ pinned là
// BẤT ĐỒNG BỘ THẬT, nên host chạy tiếp và ghi đè ô staging bằng token của bước
// SAU trước khi copy của bước TRƯỚC kịp thực thi. Kết quả: model đọc nhầm token,
// argmax lệch, text sai -- mà không có lỗi CUDA nào cả.
//
// Tham số của một kernel thì được CHỤP LẠI ngay lúc launch (driver chép vào
// command buffer), nên không có cửa sổ race nào. Chi phí: 1 launch thay vì 2
// memcpy -- rẻ hơn, và đúng.
__global__ void k_set_step(int *tok, int *pos, int t, int p) {
  if (threadIdx.x == 0) { *tok = t; *pos = p; }
}

static inline void llm_cuda_set_step(ScratchD *s, int token, int pos,
                                     cudaStream_t stream) {
  k_set_step<<<1, 1, 0, stream>>>(s->d_tok, s->d_pos, token, pos);
}

// Launch a quantized matvec. Warps-per-block = 8 (256 threads).
static inline void mv(const QTd *t, const float *x, float *y, cudaStream_t st) {
  int warps_per_block = 8;
  int blocks = DIV_UP(t->rows, warps_per_block);
  k_matvec_q4<<<blocks, warps_per_block * 32, 0, st>>>(t->codes, t->scales, x, y,
      t->rows, t->cols, t->group, t->n_groups, t->row_bytes);
}

typedef struct { float input_ms, attn_ms, ffn_ms, ple_ms, head_ms; int calls; } ProfileD;

// Event layout: [0]=start [1]=after input, then 3 per layer (attn/ffn/ple),
// then [2+3L]=after head. One event per stage per LAYER, so nothing is
// overwritten and we only synchronize once, at the end of the step.
//   n_events = 3*L + 3
static inline int llm_cuda_n_events(int L) { return 3 * L + 3; }

static cudaEvent_t *llm_cuda_make_events(int L) {
  int n = llm_cuda_n_events(L);
  cudaEvent_t *ev = (cudaEvent_t *)malloc(n * sizeof(cudaEvent_t));
  for (int i = 0; i < n; i++) CUDA_CHECK(cudaEventCreate(&ev[i]));
  return ev;
}

// One decode step. Same sequence of operations as llm_forward() in llm.h, so the
// two can be diffed side by side.
// Thân forward. KHÔNG đụng tới token/pos dạng giá trị -- chúng đã nằm ở
// s->d_tok / s->d_pos, nên toàn bộ chuỗi kernel này CAPTURE được vào graph.
static void llm_cuda_forward_body(ModelD *m, ScratchD *s, ProfileD *prof,
                                  cudaEvent_t *ev, cudaStream_t st) {
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim, F = m->c.ffn;
  int H = m->c.n_heads, Dh = D / H, S = m->c.seq_len;
  int T = 256;

#define MARK(i) if (prof) { cudaEventRecord(ev[i], st); }
  MARK(0);

  // ---- embedding + per-layer input
  k_deq_row<<<DIV_UP(D, T), T, 0, st>>>(m->tok_emb.codes, m->tok_emb.scales,
                                 s->d_tok, s->x,
                                 D, m->tok_emb.group, m->tok_emb.n_groups,
                                 m->tok_emb.row_bytes);
  mv(&m->ple_model_proj, s->x, s->tmpP, st);
  k_scale<<<DIV_UP(L * P, T), T, 0, st>>>(s->tmpP, rsqrtf((float)D), L * P);
  k_rmsnorm_slices<<<L, 128, 32 * sizeof(float), st>>>(s->tmpP, m->ple_proj_norm, P);
  k_deq_row<<<DIV_UP(L * P, T), T, 0, st>>>(m->ple_table.codes, m->ple_table.scales,
                                     s->d_tok,
                                     s->trow, L * P, m->ple_table.group,
                                     m->ple_table.n_groups, m->ple_table.row_bytes);
  k_ple_mix<<<DIV_UP(L * P, T), T, 0, st>>>(s->tmpP, s->trow, s->ple, L * P, sqrtf((float)P));
  k_rope_freqs<<<1, 64, 0, st>>>(s->rope_c, s->rope_s, Dh / 2, m->c.rope_theta, s->d_pos);
  MARK(1);

  for (int l = 0; l < L; l++) {
    // ---- attention
    k_rmsnorm<<<1, 256, 0, st>>>(s->x, m->attn_norm[l], D, s->h);
    mv(&m->qkv[l], s->h, s->qkv, st);
    float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + 2 * D;
    k_rope_apply<<<DIV_UP(H * Dh / 2, T), T, 0, st>>>(q, k, s->rope_c, s->rope_s, H, Dh);
    float *kc = s->kcache + (size_t)l * S * D, *vc = s->vcache + (size_t)l * S * D;
    k_store_kv<<<DIV_UP(D, T), T, 0, st>>>(k, v, kc, vc, s->d_pos, D);
    k_attention<<<H, 128, 0, st>>>(q, kc, vc, s->att, s->scores, s->d_pos, D, Dh);
    mv(&m->attn_proj[l], s->att, s->h, st);
    k_add<<<DIV_UP(D, T), T, 0, st>>>(s->x, s->h, D);
    MARK(2 + 3 * l + 0);

    // ---- SwiGLU FFN
    k_rmsnorm<<<1, 256, 0, st>>>(s->x, m->ffn_norm[l], D, s->h);
    mv(&m->gate[l], s->h, s->g1, st);
    mv(&m->up[l], s->h, s->g2, st);
    k_swiglu<<<DIV_UP(F, T), T, 0, st>>>(s->g1, s->g2, F);
    mv(&m->down[l], s->g1, s->h, st);
    k_add<<<DIV_UP(D, T), T, 0, st>>>(s->x, s->h, D);
    MARK(2 + 3 * l + 1);

    // ---- PLE gate
    mv(&m->ple_gate[l], s->x, s->g2, st);
    k_gelu_mul<<<DIV_UP(P, T), T, 0, st>>>(s->g2, s->ple + (size_t)l * P, P);
    mv(&m->ple_proj[l], s->g2, s->h, st);
    k_rmsnorm<<<1, 256, 0, st>>>(s->h, m->ple_norm[l], D, s->h);
    k_add<<<DIV_UP(D, T), T, 0, st>>>(s->x, s->h, D);
    MARK(2 + 3 * l + 2);
  }

  k_rmsnorm<<<1, 256, 0, st>>>(s->x, m->out_norm, D, s->x);
  mv(&m->tok_emb, s->x, s->logits, st);  // tied head
  MARK(2 + 3 * L);
#undef MARK

  if (prof) {
    int last = 2 + 3 * L;
    CUDA_CHECK(cudaEventSynchronize(ev[last]));
    float t;
    cudaEventElapsedTime(&t, ev[0], ev[1]); prof->input_ms += t;
    // Each layer's stage is measured against the previous marker: the attention
    // stage of layer l ends at its own event and starts at the last event of the
    // previous layer (or the input marker for l=0).
    for (int l = 0; l < L; l++) {
      int prev = (l == 0) ? 1 : (2 + 3 * (l - 1) + 2);
      cudaEventElapsedTime(&t, ev[prev], ev[2 + 3 * l + 0]);     prof->attn_ms += t;
      cudaEventElapsedTime(&t, ev[2 + 3 * l + 0], ev[2 + 3 * l + 1]); prof->ffn_ms += t;
      cudaEventElapsedTime(&t, ev[2 + 3 * l + 1], ev[2 + 3 * l + 2]); prof->ple_ms += t;
    }
    cudaEventElapsedTime(&t, ev[2 + 3 * (L - 1) + 2], ev[last]); prof->head_ms += t;
    prof->calls++;
  }
}

// Eager path, chữ ký như cũ: đẩy (token,pos) rồi phóng 117 kernel trên stream 0.
static void llm_cuda_forward(ModelD *m, int token, int pos, ScratchD *s,
                             ProfileD *prof, cudaEvent_t *ev) {
  llm_cuda_set_step(s, token, pos, 0);
  llm_cuda_forward_body(m, s, prof, ev, 0);
}

// ---------------------------------------------------------------- CUDA Graphs
//
// Bài toán: model này phóng 117 kernel mỗi token, mỗi kernel tốn ~3.7us chỉ để
// KHỞI ĐỘNG trên Orin. Đo được 50% thời gian mỗi token là overhead thuần.
//
// CUDA Graph ghi lại toàn bộ chuỗi phụ thuộc MỘT LẦN, rồi mỗi token chỉ cần một
// lệnh `cudaGraphLaunch`. Driver đã biết trước cả đồ thị nên bỏ được phần lớn
// công việc kiểm tra/dispatch của từng lần phóng.
//
// Điều kiện để capture được, và là lý do phải sửa các kernel ở trên:
//   1. Tham số kernel bị "nướng chín" lúc capture -> thứ đổi mỗi token (token id,
//      pos) phải nằm trong BỘ NHỚ mà kernel đọc, không phải trong argument.
//   2. Không được cấp phát / đồng bộ trong lúc capture.
//   3. Phải capture trên stream KHÔNG PHẢI stream mặc định (dùng non-blocking).
//   4. Kích thước grid/block cố định -- ở đây đúng, vì chúng chỉ phụ thuộc cấu hình.
typedef struct {
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  int built;
} LlmGraph;

static void llm_cuda_graph_build(ModelD *m, ScratchD *s, LlmGraph *g) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking));

  // Chạy nháp một lần ngoài graph: cho cuBLAS/driver cấp phát lazily xong xuôi,
  // vì trong lúc capture mà cấp phát là hỏng capture.
  llm_cuda_set_step(s, 0, 0, g->stream);
  llm_cuda_forward_body(m, s, NULL, NULL, g->stream);
  CUDA_CHECK(cudaStreamSynchronize(g->stream));

  // Capture CHỈ phần thân. k_set_step nằm NGOÀI graph vì tham số của node trong
  // graph bị cố định lúc capture -- token/pos thì đổi mỗi bước. Cái giá là 2 lần
  // launch/token (1 kernel + 1 graph) thay vì 1, so với 117 của đường eager.
  CUDA_CHECK(cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal));
  llm_cuda_forward_body(m, s, NULL, NULL, g->stream);
  CUDA_CHECK(cudaStreamEndCapture(g->stream, &g->graph));
  CUDA_CHECK(cudaGraphInstantiate(&g->exec, g->graph, NULL, NULL, 0));
  g->built = 1;

  size_t n_nodes = 0;
  cudaGraphGetNodes(g->graph, NULL, &n_nodes);
  fprintf(stderr, "[graph] captured %zu node -> 1 lần launch/token\n", n_nodes);
}

// Một bước decode qua graph. Vẫn cần đẩy (token,pos) trước -- 2 memcpy 4 byte,
// không đáng kể so với 117 launch đã bỏ đi.
static inline void llm_cuda_graph_step(ScratchD *s, LlmGraph *g, int token, int pos) {
  llm_cuda_set_step(s, token, pos, g->stream);
  CUDA_CHECK(cudaGraphLaunch(g->exec, g->stream));
}

static inline void llm_cuda_graph_sync(LlmGraph *g) {
  CUDA_CHECK(cudaStreamSynchronize(g->stream));
}

static void llm_cuda_graph_destroy(LlmGraph *g) {
  if (!g->built) return;
  cudaGraphExecDestroy(g->exec);
  cudaGraphDestroy(g->graph);
  cudaStreamDestroy(g->stream);
  g->built = 0;
}

#endif
