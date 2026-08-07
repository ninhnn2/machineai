// Kiểm bộ mã hoá BPE bằng C khớp với `tokenizers` của Python.
//
// Cùng tinh thần golden logits: đừng tin bản port, hãy so với tham chiếu. Sai
// tokenizer thì model sinh ra thứ vô nghĩa còn runtime CUDA vẫn "đúng" hoàn toàn
// -- một loại bug rất khó truy nếu không tách riêng ra kiểm.
//
//   make check_tok        # sinh tham chiếu bằng container rồi so
//
// Đọc file tham chiếu do src/gen_tok_ref.py tạo:
//   <số dòng>
//   <n_ids> <id...>  <TAB>  <text>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vocab.h"
#include "bpe.h"

// File tham chiếu ghi newline thành "\n" hai ký tự (nếu không thì nó cắt dòng).
// Phải hoàn nguyên trước khi mã hoá, nếu không ta đang so hai chuỗi KHÁC nhau
// và test sẽ báo sai trong khi bộ mã hoá vẫn đúng.
static void unescape(char *s) {
  char *r = s, *w = s;
  while (*r) {
    if (r[0] == '\\' && r[1]) {
      switch (r[1]) {
        case 'n': *w++ = '\n'; r += 2; continue;
        case 't': *w++ = '\t'; r += 2; continue;
        case '\\': *w++ = '\\'; r += 2; continue;
        default: break;
      }
    }
    *w++ = *r++;
  }
  *w = 0;
}

int main(int argc, char **argv) {
  const char *ref = argc > 1 ? argv[1] : "tok_ref.txt";
  FILE *f = fopen(ref, "r");
  if (!f) { perror(ref); return 1; }

  int n_cases;
  if (fscanf(f, "%d\n", &n_cases) != 1) { fprintf(stderr, "bad ref\n"); return 1; }

  char line[4096];
  int pass = 0, fail = 0;
  for (int c = 0; c < n_cases && fgets(line, sizeof(line), f); c++) {
    char *tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = 0;
    char *text = tab + 1;
    text[strcspn(text, "\r\n")] = 0;
    unescape(text);

    int want[512], n_want = 0;
    for (char *t = strtok(line, " "); t && n_want < 512; t = strtok(NULL, " "))
      want[n_want++] = atoi(t);

    int got[512];
    int n_got = bpe_encode(text, got, 512);

    int ok = (n_got == n_want);
    for (int i = 0; ok && i < n_got; i++) ok = (got[i] == want[i]);

    if (ok) { pass++; continue; }
    fail++;
    printf("FAIL: \"%s\"\n", text);
    printf("  python (%d): ", n_want);
    for (int i = 0; i < n_want; i++) printf("%d ", want[i]);
    printf("\n  C      (%d): ", n_got);
    for (int i = 0; i < n_got; i++) printf("%d ", got[i]);
    printf("\n");
  }
  fclose(f);

  printf("\n%d/%d khớp", pass, pass + fail);
  if (fail) {
    printf("  -- %d SAI\n", fail);
    printf("Nghi trước tiên: luật tách mảnh regex trong bpe_next_piece().\n");
    return 1;
  }
  printf("  -- bộ mã hoá C khớp Python hoàn toàn\n");
  return 0;
}
