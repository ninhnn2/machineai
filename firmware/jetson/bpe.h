// BPE encoder bằng C thuần, để thiết bị tự tokenize text người dùng gõ vào.
// Không cần Python, không cần thư viện, không đụng UTF-8.
//
// Cần vocab.h sinh bởi:  src/gen_assets.py --vocab <V> --encoder --out vocab.h
//   BYTE_TOK[256]     byte thô -> id của piece 1 ký tự trong bảng byte-level
//   MERGE_A/B/C[]     luật merge THEO THỨ TỰ RANK (index = rank)
//
// Thuật toán đúng như HuggingFace `tokenizers`:
//   1. tách text thành các mảnh theo regex GPT-2 (BPE không merge qua ranh giới mảnh)
//   2. trong mỗi mảnh: mỗi byte -> 1 symbol, rồi lặp gộp cặp có rank NHỎ NHẤT
//
// Bước 1 hay bị bỏ quên. Thiếu nó thì "the cat" có thể merge chữ cuối của "the"
// với space, ra chuỗi id khác hẳn -- và model sinh ra thứ vô nghĩa mà bạn sẽ đổ
// lỗi cho runtime. Kiểm bằng: make check_tok
//
// Regex GPT-2 gốc dùng \p{L}/\p{N} (Unicode). Ở đây cài bản ASCII, đúng tuyệt đối
// với dữ liệu TinyStories; ký tự phi-ASCII rơi vào nhánh "không phải chữ/số" và
// vẫn mã hoá được (byte-level không bao giờ trượt), chỉ có thể tách mảnh khác đi.
#ifndef BPE_H
#define BPE_H

#include <string.h>

#ifndef N_MERGES
#error "vocab.h thiếu bảng encode -- chạy gen_assets.py với --encoder"
#endif

static int bpe_is_alpha(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static int bpe_is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int bpe_is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Độ dài mảnh tiếp theo bắt đầu tại s, theo regex GPT-2:
//   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
static int bpe_next_piece(const char *s, int n) {
  const unsigned char *p = (const unsigned char *)s;
  if (n <= 0) return 0;

  if (p[0] == '\'' && n >= 2) {                       // contractions
    char c1 = (char)(p[1] | 0x20);
    if (c1 == 't' || c1 == 's' || c1 == 'm' || c1 == 'd') return 2;
    if (n >= 3) {
      char c2 = (char)(p[2] | 0x20);
      if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
          (c1 == 'l' && c2 == 'l')) return 3;
    }
  }

  int i = 0;
  int lead_space = (p[0] == ' ' && n >= 2) ? 1 : 0;   // " ?" đứng trước
  i = lead_space;

  if (i < n && bpe_is_alpha(p[i])) {
    while (i < n && bpe_is_alpha(p[i])) i++;
    return i;
  }
  if (i < n && bpe_is_digit(p[i])) {
    while (i < n && bpe_is_digit(p[i])) i++;
    return i;
  }
  if (i < n && !bpe_is_space(p[i]) && !bpe_is_alpha(p[i]) && !bpe_is_digit(p[i])) {
    while (i < n && !bpe_is_space(p[i]) && !bpe_is_alpha(p[i]) && !bpe_is_digit(p[i])) i++;
    return i;
  }

  // Chỉ còn khoảng trắng. `\s+(?!\S)` giữ lại 1 space cho mảnh sau nếu còn chữ.
  i = 0;
  while (i < n && bpe_is_space(p[i])) i++;
  if (i > 1 && i < n) i--;                            // nhường 1 space cho mảnh kế
  return i > 0 ? i : 1;
}

// Merge trong MỘT mảnh. syms[] vào là id từng byte, ra là id sau khi gộp.
static int bpe_merge_piece(int *syms, int n) {
  for (;;) {
    int best_rank = N_MERGES, best_i = -1;
    for (int i = 0; i + 1 < n; i++) {
      // Tìm rank nhỏ nhất áp dụng được. N_MERGES ~ vài nghìn nên quét thẳng;
      // prompt ngắn nên tổng chi phí không đáng kể so với 117 kernel/token.
      for (int r = 0; r < best_rank; r++) {
        if (MERGE_A[r] == syms[i] && MERGE_B[r] == syms[i + 1]) {
          best_rank = r; best_i = i;
          break;                                       // r tăng dần -> đây là min
        }
      }
    }
    if (best_i < 0) return n;
    syms[best_i] = MERGE_C[best_rank];
    memmove(&syms[best_i + 1], &syms[best_i + 2], (size_t)(n - best_i - 2) * sizeof(int));
    n--;
  }
}

// Mã hoá `text` thành ids. Trả về số token, hoặc -1 nếu tràn `max_out`.
static int bpe_encode(const char *text, int *out, int max_out) {
  int n = (int)strlen(text), pos = 0, n_out = 0;
  int syms[1024];

  while (pos < n) {
    int plen = bpe_next_piece(text + pos, n - pos);
    if (plen <= 0) break;
    if (plen > (int)(sizeof(syms) / sizeof(syms[0]))) return -1;

    for (int i = 0; i < plen; i++)
      syms[i] = BYTE_TOK[(unsigned char)text[pos + i]];
    int m = bpe_merge_piece(syms, plen);

    if (n_out + m > max_out) return -1;
    for (int i = 0; i < m; i++) out[n_out++] = syms[i];
    pos += plen;
  }
  return n_out;
}

#endif
