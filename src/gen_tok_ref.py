"""Sinh tham chiếu tokenizer để kiểm bộ mã hoá BPE bằng C.

Cùng vai trò với golden.txt của export.py, nhưng cho tokenizer: nếu bản C tách
mảnh hoặc gộp cặp khác Python thì model nhận sai ids và sinh ra thứ vô nghĩa,
trong khi runtime CUDA vẫn hoàn toàn đúng. Tách riêng ra kiểm thì bắt được ngay.

Định dạng (đọc bởi firmware/jetson/check_tok.c):
    <số case>
    <id id id ...><TAB><text>
"""

import argparse
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

# Chọn để chạm vào từng nhánh của regex GPT-2 trong bpe_next_piece():
# chữ, số, dấu câu, contraction, khoảng trắng đầu/cuối/liên tiếp, xuống dòng.
CASES = [
    "Once upon a time",
    "Once upon a time,",
    " Once upon a time",
    "The cat sat on the mat.",
    "Lily's mom said, \"Don't do that!\"",
    "I'm sure they're going to be 100% happy... aren't they?",
    "one two three 42 7 999",
    "a  b   c    d",
    "Hello\nworld\n\nagain",
    "trailing spaces   ",
    "   leading spaces",
    "!!!???,,,...",
    "CamelCase and snake_case and kebab-case",
    "x",
    " ",
    "",
    "She had 3 apples, 2 oranges and 1 banana.",
    "Tom said: 'hi'",
    "It was a very very very long day for the little girl named Lily.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "firmware", "jetson",
                                                  "tok_ref.txt"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(HERE, "..", "data", f"bpe{args.vocab}.json"))
    cases = [c for c in CASES if c != ""]   # chuỗi rỗng: fgets không đọc được, bỏ

    with open(args.out, "w") as f:
        f.write(f"{len(cases)}\n")
        for text in cases:
            ids = tok.encode(text).ids
            f.write(" ".join(map(str, ids)) + "\t" + text.replace("\n", "\\n") + "\n")

    print(f"wrote {args.out}: {len(cases)} case")
    for text in cases[:3]:
        print(f"  {text!r} -> {tok.encode(text).ids}")


if __name__ == "__main__":
    main()
