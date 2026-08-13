// Sinh text bằng runtime CUDA -- bản Jetson của vòng generate trong esp32_llm.ino,
// cộng thêm thứ ESP32 không có: tự tokenize prompt người dùng gõ vào (bpe.h).
//
//   make generate                                    # prompt mặc định
//   ./generate_cuda -p "The little robot"             # prompt tự nhập
//   ./generate_cuda -i                                # chế độ tương tác
//   ./generate_cuda -p "Tom ran" -n 300 -t 0.9 -k 50 -s 7
//   ./generate_cuda -t 0                              # greedy, tất định
//
// LƯU Ý VỀ MODEL: đây là model TinyStories -- nó VIẾT TIẾP câu chuyện, không trả
// lời câu hỏi. Gõ "What is 2+2?" sẽ ra một mẩu truyện chứ không ra "4". Giới hạn
// đó nằm ở core 1.5M tham số, không phải ở runtime. Prompt hợp là mở đầu truyện.
//
// Cần vocab.h có bảng encode:
//   python src/gen_assets.py --vocab 4096 --encoder --out firmware/jetson/vocab.h

#include <math.h>
#include <string.h>
#include "llm_cuda.cuh"
#include "vocab.h"
#include "bpe.h"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

// Ghi raw bytes của token, y hệt emit() trong esp32_llm.ino:26.
static void emit(int tok) {
  if (tok < 0 || tok >= VOCAB_N) return;
  fwrite(VOCAB_BLOB + VOCAB_OFF[tok], 1, VOCAB_OFF[tok + 1] - VOCAB_OFF[tok], stdout);
  fflush(stdout);
}

// Top-k sampling trên host. Logits phải copy về host để chọn token rồi, nên làm
// trên CPU là hợp lý -- V=4096 thì rẻ hơn nhiều so với 117 kernel launch.
// temperature <= 0 => greedy (argmax), tất định.
static int sample(float *logits, int V, float temperature, int top_k,
                  unsigned int *rng) {
  if (temperature <= 0.f) {
    int best = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  for (int i = 0; i < V; i++) logits[i] /= temperature;

  float cut = -1e30f;
  if (top_k > 0 && top_k < V) {
    float *tmp = (float *)malloc(V * sizeof(float));
    memcpy(tmp, logits, V * sizeof(float));
    for (int i = 0; i < top_k; i++) {
      int b = i;
      for (int j = i + 1; j < V; j++) if (tmp[j] > tmp[b]) b = j;
      float t = tmp[i]; tmp[i] = tmp[b]; tmp[b] = t;
    }
    cut = tmp[top_k - 1];
    free(tmp);
  }

  float maxv = -1e30f;
  for (int i = 0; i < V; i++) if (logits[i] >= cut && logits[i] > maxv) maxv = logits[i];
  double sum = 0;
  for (int i = 0; i < V; i++)
    sum += (logits[i] >= cut) ? (double)expf(logits[i] - maxv) : 0.0;

  *rng = *rng * 1664525u + 1013904223u;
  double r = ((*rng >> 8) & 0xFFFFFF) / (double)0x1000000 * sum;
  double acc = 0;
  for (int i = 0; i < V; i++) {
    if (logits[i] < cut) continue;
    acc += expf(logits[i] - maxv);
    if (acc >= r) return i;
  }
  return 0;
}

// Một lượt: nạp prompt rồi sinh tiếp. KV cache reset mỗi lượt (pos về 0), nên mỗi
// prompt là một câu chuyện độc lập -- đúng với model TinyStories, vốn không phải
// model hội thoại có ngữ cảnh nhiều lượt.
static void run_once(ModelD *d, ScratchD *s, Cfg *c, float *logits,
                     const char *text, int n_gen, float temp, int top_k,
                     unsigned *seed, int use_graph) {
  int ids[512];
  int plen = bpe_encode(text, ids, 512);
  if (plen < 0) { fprintf(stderr, "prompt quá dài\n"); return; }
  if (plen == 0) { fprintf(stderr, "prompt rỗng\n"); return; }
  if (plen >= c->seq_len) { fprintf(stderr, "prompt vượt seq_len %d\n", c->seq_len); return; }

  int pos = 0, tok = 0;
  for (int i = 0; i < plen; i++) {
    tok = ids[i];
    emit(tok);
    llm_cuda_forward(d, tok, pos++, s, NULL, NULL);
  }

  // Đường decode qua CUDA Graph. bench_cuda đo được 117 launch/token, và khoảng
  // một nửa thời gian mỗi token là chi phí phóng kernel thuần. Graph ghi cả chuỗi
  // một lần rồi mỗi token chỉ còn 2 launch (k_set_step + graph).
  //
  // Graph chỉ dựng cho phần decode, không cho prefill: prefill chạy trước khi
  // biết prompt dài bao nhiêu, và mỗi token prompt cũng đi qua đúng thân đó nên
  // dựng graph sớm không thêm được gì.
  LlmGraph g; memset(&g, 0, sizeof(g));
  if (use_graph) llm_cuda_graph_build(d, s, &g);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  int steps = 0;
  for (int step = 0; step < n_gen && pos < c->seq_len; step++) {
    // g->stream tạo bằng cudaStreamNonBlocking nên nó KHÔNG tự đồng bộ với
    // stream mặc định mà cudaMemcpy dưới đây chạy trên. Thiếu dòng sync này,
    // logits đọc được là của token TRƯỚC, và văn bản sinh ra vẫn trông hợp lý
    // nên lỗi rất khó thấy.
    if (use_graph) llm_cuda_graph_sync(&g);
    CUDA_CHECK(cudaMemcpy(logits, s->logits, c->vocab * sizeof(float),
                          cudaMemcpyDeviceToHost));
    tok = sample(logits, c->vocab, temp, top_k, seed);
    emit(tok);
    if (use_graph) llm_cuda_graph_step(s, &g, tok, pos++);
    else           llm_cuda_forward(d, tok, pos++, s, NULL, NULL);
    steps++;
  }
  if (use_graph) llm_cuda_graph_sync(&g);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  if (use_graph) llm_cuda_graph_destroy(&g);
  float ms; cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);

  printf("\n");
  fprintf(stderr, "[%d prompt + %d sinh | %.1f tok/s | %.3f ms/token | %s]\n",
          plen, steps, steps * 1000.f / ms, ms / steps,
          use_graph ? "graph" : "eager");
}

static void usage(void) {
  fprintf(stderr,
    "dùng: generate_cuda [model.bin] [tuỳ chọn]\n"
    "  -p \"text\"  prompt        (mặc định \"Once upon a time\")\n"
    "  -n N        số token sinh (200)\n"
    "  -t T        temperature   (0.8; 0 = greedy, tất định)\n"
    "  -k K        top_k         (40)\n"
    "  -s S        seed          (1234)\n"
    "  -i          chế độ tương tác: gõ prompt, Enter; dòng trống hoặc Ctrl-D để thoát\n"
    "  --no-graph  tắt CUDA Graph, quay về 117 launch/token (để so tốc độ)\n");
}

int main(int argc, char **argv) {
  const char *binp = "../model/model.bin";
  const char *prompt = "Once upon a time";
  int n_gen = 200, top_k = 40, interactive = 0, use_graph = 1;
  float temp = 0.8f;
  unsigned seed = 1234u;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-' && strstr(argv[i], ".bin")) { binp = argv[i]; continue; }
    if (!strcmp(argv[i], "-p") && i + 1 < argc) { prompt = argv[++i]; continue; }
    if (!strcmp(argv[i], "-n") && i + 1 < argc) { n_gen = atoi(argv[++i]); continue; }
    if (!strcmp(argv[i], "-t") && i + 1 < argc) { temp = (float)atof(argv[++i]); continue; }
    if (!strcmp(argv[i], "-k") && i + 1 < argc) { top_k = atoi(argv[++i]); continue; }
    if (!strcmp(argv[i], "-s") && i + 1 < argc) { seed = (unsigned)atoi(argv[++i]); continue; }
    if (!strcmp(argv[i], "-i")) { interactive = 1; continue; }
    if (!strcmp(argv[i], "--no-graph")) { use_graph = 0; continue; }
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
    fprintf(stderr, "tham số lạ: %s\n", argv[i]); usage(); return 1;
  }

  size_t nbytes; uint8_t *buf = read_file(binp, &nbytes);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic trong %s\n", binp); return 1; }
  Cfg *c = &m.c;

  cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  fprintf(stderr, "%s | V=%d D=%d L=%d H=%d F=%d P=%d | %.2f MB | temp=%.2f top_k=%d\n",
          p.name, c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
          nbytes / 1e6, temp, top_k);
  if (c->vocab != VOCAB_N) {
    fprintf(stderr, "LỖI: vocab.h có %d token, model có %d. "
                    "Chạy lại gen_assets.py --vocab %d --encoder\n",
            VOCAB_N, c->vocab, c->vocab);
    return 1;
  }

  ModelD d; ScratchD s;
  llm_cuda_upload(&m, &d);
  llm_cuda_alloc(c, &s);
  float *logits = (float *)malloc(c->vocab * sizeof(float));

  if (!interactive) {
    printf("\n>>> ");
    run_once(&d, &s, c, logits, prompt, n_gen, temp, top_k, &seed, use_graph);
    return 0;
  }

  fprintf(stderr,
     "\nGõ phần mở đầu một câu chuyện rồi Enter. Dòng trống hoặc Ctrl-D để thoát.\n"
     "Đây là model TinyStories: nó VIẾT TIẾP, không trả lời câu hỏi.\n"
     "Thử: 'The little robot'  |  'Lily found a key'  |  'One rainy day'\n\n");
  char line[1024];
  for (;;) {
    fprintf(stderr, "prompt> ");
    fflush(stderr);
    if (!fgets(line, sizeof(line), stdin)) break;
    line[strcspn(line, "\r\n")] = 0;
    if (line[0] == 0) break;
    printf("\n>>> ");
    run_once(&d, &s, c, logits, line, n_gen, temp, top_k, &seed, use_graph);
    printf("\n");
  }
  fprintf(stderr, "tạm biệt\n");
  return 0;
}
