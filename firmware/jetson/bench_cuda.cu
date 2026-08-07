// Throughput + per-stage profile for the CUDA port, in the same shape as the
// ESP32 numbers in RESULTS.md so the two can be put side by side.
//
// The extra column here is ACHIEVED BANDWIDTH per stage. That is the whole point:
// decode at batch=1 is memory-bound, so each stage's ms is really "bytes it must
// read / bandwidth". Comparing achieved GB/s against the board's measured peak
// tells you whether a stage is at its floor (stop optimising) or far from it
// (keep going). This is the RESULTS.md:149-152 reasoning, automated.
//
//   make bench && ./bench_cuda ../model/model.bin 200

#include <math.h>
#include <string.h>
#include "llm_cuda.cuh"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

static size_t qtb(const QTd *t) {
  return (size_t)t->rows * t->row_bytes + (size_t)t->rows * t->n_groups * 2;
}

// Do-nothing kernel, used to CALIBRATE launch overhead on this specific board.
__global__ void k_noop(float *p) { if (threadIdx.x == 1 << 30) p[0] = 1.f; }

// Measure us per kernel launch here and now, instead of hardcoding a constant.
// Desktop x86 drives launches in ~2-3 us; Jetson's Cortex-A78 needs ~5-9 us for
// the same work, so any fixed threshold is wrong on one of the two. Calibrating
// makes the diagnosis below portable -- and measuring instead of assuming is the
// whole point of this repo.
static double measure_launch_us(float *dummy) {
  const int N = 2000;
  for (int i = 0; i < 200; i++) k_noop<<<1, 32>>>(dummy);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < N; i++) k_noop<<<1, 32>>>(dummy);
  cudaEventRecord(b);
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms * 1000.0 / N;
}

int main(int argc, char **argv) {
  const char *binp = argc > 1 ? argv[1] : "../model/model.bin";
  int n_gen = argc > 2 ? atoi(argv[2]) : 200;
  // Peak bandwidth of the board, from jetson-optim/bench/bench_roofline.cu.
  // Override on the command line once you have measured your own.
  double peak_gbs = argc > 3 ? atof(argv[3]) : 66.8;

  size_t nbytes; uint8_t *buf = read_file(binp, &nbytes);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }
  Cfg *c = &m.c;

  cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("========================================================================\n");
  printf(" %s  |  model V=%d D=%d L=%d H=%d F=%d P=%d  (%.2f MB)\n",
         p.name, c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
         nbytes / 1e6);
  printf(" peak bandwidth dùng để tính %% : %.1f GB/s\n", peak_gbs);
  printf("========================================================================\n");

  ModelD d; ScratchD s;
  size_t dev_bytes = llm_cuda_upload(&m, &d);
  llm_cuda_alloc(c, &s);
  int L = c->n_layers;

  // ---- bytes each stage must read per token (weights only; the dominant term) ----
  size_t b_head = qtb(&d.tok_emb);
  size_t b_input = (size_t)d.tok_emb.row_bytes            // 1 embedding row
                 + qtb(&d.ple_model_proj)
                 + (size_t)d.ple_table.row_bytes;         // 1 table row  <-- the 25M
                                                          // param table costs ONE row
  size_t b_attn = 0, b_ffn = 0, b_ple = 0;
  for (int l = 0; l < L; l++) {
    b_attn += qtb(&d.qkv[l]) + qtb(&d.attn_proj[l]);
    b_ffn  += qtb(&d.gate[l]) + qtb(&d.up[l]) + qtb(&d.down[l]);
    b_ple  += qtb(&d.ple_gate[l]) + qtb(&d.ple_proj[l]);
  }
  size_t b_total = b_head + b_input + b_attn + b_ffn + b_ple;

  printf("\nTrên device: %.2f MB weights\n", dev_bytes / 1e6);
  printf("Đọc mỗi token (weights): %.3f MB  -> trần lý thuyết %.1f tok/s\n",
         b_total / 1e6, peak_gbs * 1e9 / b_total);
  printf("  bảng PLE %.2f MB nhưng chỉ đọc %zu B/token (%.4f%%) <- luận điểm của repo\n",
         qtb(&d.ple_table) / 1e6, (size_t)d.ple_table.row_bytes,
         100.0 * d.ple_table.row_bytes / qtb(&d.ple_table));

  // ---- warm up + generate greedily ----
  int prompt[] = {1, 500, 1000, 200, 42, 777, 13, 99};
  int plen = (int)(sizeof(prompt) / sizeof(int));
  ProfileD prof = {0, 0, 0, 0, 0, 0};
  cudaEvent_t *ev = llm_cuda_make_events(L);

  int pos = 0;
  for (int i = 0; i < plen; i++) llm_cuda_forward(&d, prompt[i], pos++, &s, NULL, NULL);
  CUDA_CHECK(cudaDeviceSynchronize());

  float *logits = (float *)malloc(c->vocab * sizeof(float));
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  int steps = 0;
  for (int step = 0; step < n_gen && pos < c->seq_len; step++) {
    CUDA_CHECK(cudaMemcpy(logits, s.logits, c->vocab * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    for (int v = 1; v < c->vocab; v++) if (logits[v] > logits[best]) best = v;
    llm_cuda_forward(&d, best, pos++, &s, &prof, ev);
    steps++;
  }
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float total_ms; cudaEventElapsedTime(&total_ms, t0, t1);

  double ms_tok = total_ms / steps;
  printf("\n--- %d token trong %.2f s ---\n", steps, total_ms / 1e3);
  printf("throughput: %.1f tok/s   (%.3f ms/token)\n", 1000.0 / ms_tok, ms_tok);
  printf("đạt %.0f%% trần bandwidth\n",
         100.0 * (1000.0 / ms_tok) / (peak_gbs * 1e9 / b_total));

  // ---- per-stage: ms, bytes, achieved GB/s, % of board peak ----
  // Kernel launches per stage, counted from llm_cuda_forward(). A stage that is
  // launch-bound spends the SAME us per kernel no matter how much work the kernel
  // does -- that flat column is the signature, and it is what separates
  // "launch-bound" from "compute-bound" without needing a profiler.
  float n = (float)prof.calls;
  struct { const char *name; float ms; size_t bytes; int kernels; } st[5] = {
    {"input+PLE",  prof.input_ms / n, b_input, 7},
    {"attention",  prof.attn_ms  / n, b_attn,  7 * L},
    {"ffn",        prof.ffn_ms   / n, b_ffn,   6 * L},
    {"ple gate",   prof.ple_ms   / n, b_ple,   5 * L},
    {"head",       prof.head_ms  / n, b_head,  2},
  };
  int total_k = 0;
  for (int i = 0; i < 5; i++) total_k += st[i].kernels;
  double launch_us = measure_launch_us(s.x);

  printf("\nlaunch overhead ĐO ĐƯỢC trên board này: %.2f us/kernel (kernel rỗng)\n",
         launch_us);
  printf("\n%-12s %9s %7s %9s %9s %8s   %s\n",
         "stage", "ms/token", "%", "MB đọc", "GB/s đạt", "us/kern", "chẩn đoán");
  printf("--------------------------------------------------------------------------------\n");
  for (int i = 0; i < 5; i++) {
    double gbs = st[i].bytes / (st[i].ms * 1e-3) / 1e9;
    double frac = 100.0 * gbs / peak_gbs;
    double us_k = st[i].ms * 1000.0 / st[i].kernels;
    // So với overhead ĐO ĐƯỢC, không phải hằng số cứng. Nếu mỗi kernel chỉ tốn
    // hơn kernel rỗng chưa tới 50% thì stage đó gần như toàn là overhead.
    const char *diag = frac > 70 ? "CHẠM SÀN bandwidth - dừng tối ưu"
                     : (us_k < launch_us * 1.5) ? "LAUNCH-BOUND (~= kernel rỗng)"
                     : frac > 30 ? "còn dư địa vừa phải"
                                 : "compute-bound";
    printf("%-12s %9.3f %6.1f%% %9.3f %9.1f %8.2f   %s\n",
           st[i].name, st[i].ms, 100.0 * st[i].ms / ms_tok,
           st[i].bytes / 1e6, gbs, us_k, diag);
  }
  printf("--------------------------------------------------------------------------------\n");
  double avg_us = ms_tok * 1000.0 / total_k;
  printf("%d kernel/token, %.2f us/kernel trung bình, overhead %.2f us\n",
         total_k, avg_us, launch_us);
  printf("=> %.0f%% thời gian mỗi token là LAUNCH OVERHEAD THUẦN (%.3f / %.3f ms)\n",
         100.0 * launch_us * total_k / (ms_tok * 1000.0),
         launch_us * total_k / 1000.0, ms_tok);
  if (launch_us * total_k > 0.5 * ms_tok * 1000.0) {
    printf("\nModel này LAUNCH-BOUND. Lever duy nhất là CUDA Graphs (gộp %d launch\n", total_k);
    printf("thành 1 lần replay), KHÔNG phải tối ưu bên trong kernel.\n");
    printf("Sàn nếu bỏ hết overhead: %.0f tok/s. Xem JETSON.md bài tập 4.\n",
           1000.0 / (ms_tok - launch_us * total_k / 1000.0));
  }
  // ---- CUDA Graphs: gộp 117 launch thành 1 ----
  printf("\n--- CUDA Graphs ---\n");
  LlmGraph g; memset(&g, 0, sizeof(g));
  llm_cuda_graph_build(&d, &s, &g);

  // Chạy lại đúng số token, đường graph. Không sample lại (đã đo tốc độ ở trên),
  // chỉ đo thời gian mỗi bước để so công bằng với đường eager.
  for (int i = 0; i < 8; i++) llm_cuda_graph_step(&s, &g, prompt[i % plen], i);
  llm_cuda_graph_sync(&g);

  cudaEvent_t gt0, gt1;
  CUDA_CHECK(cudaEventCreate(&gt0)); CUDA_CHECK(cudaEventCreate(&gt1));
  CUDA_CHECK(cudaEventRecord(gt0, g.stream));
  int gsteps = 0;
  for (int step = 0; step < steps && step < c->seq_len - 8; step++) {
    llm_cuda_graph_step(&s, &g, 1 + (step % 100), 8 + step);
    gsteps++;
  }
  CUDA_CHECK(cudaEventRecord(gt1, g.stream));
  CUDA_CHECK(cudaEventSynchronize(gt1));
  float gms; cudaEventElapsedTime(&gms, gt0, gt1);
  double g_ms_tok = gms / gsteps;

  printf("%-14s %12s %12s %10s\n", "", "ms/token", "tok/s", "so eager");
  printf("%-14s %12.4f %12.1f %9s\n", "eager (117 launch)", ms_tok, 1000.0 / ms_tok, "1.00x");
  printf("%-14s %12.4f %12.1f %8.2fx\n", "graph (1 launch)", g_ms_tok,
         1000.0 / g_ms_tok, ms_tok / g_ms_tok);
  printf("  tiết kiệm %.3f ms/token; overhead ước tính %.3f ms (%d x %.2f us)\n",
         ms_tok - g_ms_tok, launch_us * total_k / 1000.0, total_k, launch_us);
  printf("  -> graph thu hồi %.0f%% phần overhead dự đoán\n",
         100.0 * (ms_tok - g_ms_tok) / (launch_us * total_k / 1000.0));
  llm_cuda_graph_destroy(&g);

  printf("\nCách đọc bảng (giống RESULTS.md:149-152):\n");
  printf("  Tìm stage chiếm %% lớn nhất. Nếu nó đã CHẠM SÀN bandwidth thì tối ưu\n");
  printf("  kernel là vô ích - phải giảm bytes đọc (quantize sâu hơn / head nhỏ hơn).\n");
  printf("  Nếu nó XA sàn thì nút thắt là compute hoặc kernel-launch overhead\n");
  printf("  (model này nhỏ -> rất dễ bị launch-bound; xem JETSON.md muc CUDA graphs).\n");
  return 0;
}
