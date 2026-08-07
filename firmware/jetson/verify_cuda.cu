// Tier-1 validation for the CUDA port: run the golden prompt on the GPU and
// compare last-position logits against firmware/model/golden.txt -- the SAME
// reference the scalar C port in ../host_verify/verify.c is checked against.
//
// Also runs the scalar CPU path from ../common/llm.h on the identical weights, so
// you get three numbers in one shot:
//     PyTorch  vs  scalar C   -> should be ~1e-5 (same summation order)
//     PyTorch  vs  CUDA       -> larger, and that is EXPECTED
//     scalar C vs  CUDA       -> isolates the port from quantization
//
// Read JETSON.md "Sai số" before deciding a mismatch is a bug.
//
//   make verify && ./verify_cuda ../model/model.bin ../model/golden.txt

#include <math.h>
#include "llm_cuda.cuh"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

// Compare two logit vectors and report the numbers that actually matter:
// max abs diff, relative diff, and whether the ARGMAX agrees (the only thing
// that changes generated text under greedy decoding).
static void compare(const char *a_name, const float *a,
                    const char *b_name, const float *b, int V) {
  double maxabs = 0, sum2 = 0, scale = 0;
  int argmax_a = 0, argmax_b = 0;
  for (int i = 0; i < V; i++) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (d > maxabs) maxabs = d;
    sum2 += d * d;
    if (fabs(a[i]) > scale) scale = fabs(a[i]);
    if (a[i] > a[argmax_a]) argmax_a = i;
    if (b[i] > b[argmax_b]) argmax_b = i;
  }
  printf("  %-10s vs %-10s | max|d| %.3e | rms %.3e | rel %.2e | argmax %s (%d vs %d)\n",
         a_name, b_name, maxabs, sqrt(sum2 / V), maxabs / (scale > 0 ? scale : 1),
         argmax_a == argmax_b ? "MATCH" : "DIFFER", argmax_a, argmax_b);
  if (argmax_a != argmax_b)
    printf("      ^ argmax differs -> generated text WILL diverge. Investigate.\n");
}

int main(int argc, char **argv) {
  const char *binp  = argc > 1 ? argv[1] : "../model/model.bin";
  const char *goldp = argc > 2 ? argv[2] : "../model/golden.txt";

  size_t nbytes;
  uint8_t *buf = read_file(binp, &nbytes);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic in %s\n", binp); return 1; }
  Cfg *c = &m.c;
  printf("model: V=%d D=%d L=%d H=%d F=%d P=%d group=%d (%.2f MB on disk)\n",
         c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim, c->group,
         nbytes / 1e6);

  // ---- golden reference ----
  FILE *gf = fopen(goldp, "r");
  if (!gf) { perror(goldp); return 1; }
  int plen; if (fscanf(gf, "%d", &plen) != 1) { fprintf(stderr, "bad golden\n"); return 1; }
  int *prompt = (int *)malloc(plen * sizeof(int));
  for (int i = 0; i < plen; i++)
    if (fscanf(gf, "%d", &prompt[i]) != 1) { fprintf(stderr, "bad golden\n"); return 1; }
  float *gold = (float *)malloc(c->vocab * sizeof(float));
  for (int i = 0; i < c->vocab; i++)
    if (fscanf(gf, "%f", &gold[i]) != 1) { fprintf(stderr, "bad golden\n"); return 1; }
  fclose(gf);
  printf("golden: prompt len %d, %d logits\n\n", plen, c->vocab);

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;

  // ---- CPU scalar path (../common/llm.h, unchanged) ----
  Scratch s;
  s.x = (float *)malloc(D * 4); s.h = (float *)malloc((F > D ? F : D) * 4);
  s.qkv = (float *)malloc(3 * D * 4); s.att = (float *)malloc(D * 4);
  s.g1 = (float *)malloc(F * 4); s.g2 = (float *)malloc((P > F ? P : F) * 4);
  s.ple = (float *)malloc(L * P * 4); s.tmpP = (float *)malloc(L * P * 4);
  s.trow = (float *)malloc(L * P * 4); s.logits = (float *)malloc(V * 4);
  s.scores = (float *)malloc(S * 4);
  s.kcache = (float *)malloc((size_t)L * S * D * 4);
  s.vcache = (float *)malloc((size_t)L * S * D * 4);
  for (int i = 0; i < plen; i++) llm_forward(&m, prompt[i], i, &s);
  float *cpu = (float *)malloc(V * sizeof(float));
  memcpy(cpu, s.logits, V * sizeof(float));

  // ---- GPU path ----
  ModelD d; ScratchD sd;
  size_t on_dev = llm_cuda_upload(&m, &d);
  llm_cuda_alloc(c, &sd);
  printf("uploaded %.2f MB of quantized weights to device\n", on_dev / 1e6);
  for (int i = 0; i < plen; i++) llm_cuda_forward(&d, prompt[i], i, &sd, NULL, NULL);
  CUDA_CHECK(cudaDeviceSynchronize());
  float *gpu = (float *)malloc(V * sizeof(float));
  CUDA_CHECK(cudaMemcpy(gpu, sd.logits, V * sizeof(float), cudaMemcpyDeviceToHost));

  printf("\n--- so sánh 3 chiều ---\n");
  compare("PyTorch", gold, "scalar C", cpu, V);
  compare("PyTorch", gold, "CUDA", gpu, V);
  compare("scalar C", cpu, "CUDA", gpu, V);

  printf("\nDiễn giải:\n");
  printf("  PyTorch vs scalar C  ~1e-5  -> port C đúng (cùng thứ tự cộng)\n");
  printf("  PyTorch vs CUDA      lớn hơn -> BÌNH THƯỜNG, warp reduction cộng theo\n");
  printf("                                 thứ tự cây. Xem JETSON.md muc 'Sai số'.\n");
  printf("  Điều PHẢI đúng: argmax MATCH. Nếu argmax lệch, text sinh ra sẽ khác.\n");
  return 0;
}
