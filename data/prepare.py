"""Download a slice of TinyStories, train a small BPE, emit uint16 token bins.

The vocab is deliberately fixed across every ablation arm: cross-entropy is only
comparable between models that share a tokenizer.
"""

import argparse
import os
import sys

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
RAW = os.path.join(HERE, "tinystories_slice.txt")
VOCAB_SIZE = 4096  # overridden by --vocab; 4096 keeps the original bin names
# ~300MB of stories is ~75M tokens: enough to overtrain a 1M-param core well past
# its compute-optimal point, which is the regime we actually care about.
SLICE_BYTES = 300 * 1024 * 1024
VAL_FRACTION = 0.005


def load_custom(path):
    """Đọc corpus của người dùng: một file .txt, hoặc một thư mục chứa .txt.

    Tài liệu được ngăn cách bằng <|endoftext|>. Nếu corpus chưa có dấu ngăn đó,
    mỗi file được coi là một tài liệu và dấu ngăn được chèn vào giữa các file.
    """
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.endswith((".txt", ".md")))
        if not files:
            sys.exit(f"không tìm thấy .txt/.md nào trong {path}")
    else:
        files = [path]
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            parts.append(fh.read())
    text = "<|endoftext|>".join(parts)
    if "<|endoftext|>" not in text:
        text += "<|endoftext|>"
    print(f"đọc {len(files)} file, {len(text)/1e6:.1f}MB từ {path}")
    return text


def download():
    if os.path.exists(RAW) and os.path.getsize(RAW) >= SLICE_BYTES * 0.99:
        print(f"already have {RAW}")
        return
    print(f"downloading first {SLICE_BYTES / 1e6:.0f}MB of TinyStories...")
    got = 0
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got >= SLICE_BYTES:
                    break
                if got % (25 << 20) < (1 << 20):
                    print(f"  {got / 1e6:.0f}MB", flush=True)
    print(f"done, {got / 1e6:.0f}MB")


def train_tokenizer(text, retrain=False):
    path = os.path.join(HERE, f"bpe{VOCAB_SIZE}.json")
    if os.path.exists(path) and not retrain:
        print(f"already have {path}")
        print("  ! dùng lại tokenizer CŨ. Nếu corpus đã đổi thì đây gần như chắc")
        print("  ! chắn là sai: chạy lại với --retrain-tokenizer.")
        return Tokenizer.from_file(path)
    print(f"training BPE vocab={VOCAB_SIZE}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # 40MB is plenty to fit a 4k merge table; using the full slice just burns time.
    tok.train_from_iterator([text[: 40 * 1024 * 1024]], trainer=trainer)
    tok.save(path)
    return tok


def main():
    global VOCAB_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--input", default=None,
                    help="file .txt hoặc thư mục chứa .txt/.md dùng thay TinyStories")
    ap.add_argument("--suffix", default=None,
                    help="ghi ra train{suffix}.bin thay vì tên suy từ vocab, "
                         "để không đè lên dataset đang có")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="giữ nguyên thứ tự tài liệu; chỉ dùng khi bạn CẦN tập val "
                         "là phần đuôi corpus")
    ap.add_argument("--retrain-tokenizer", action="store_true",
                    help="train lại BPE dù đã có file, bắt buộc khi đổi corpus")
    args = ap.parse_args()
    VOCAB_SIZE = args.vocab
    # vocab 4096 keeps the original train.bin/val.bin; others get suffixed names
    # so both datasets coexist and train.py can pick by --vocab.
    suffix = args.suffix if args.suffix is not None else (
        "" if VOCAB_SIZE == 4096 else f"_v{VOCAB_SIZE}")

    if args.input:
        text = load_custom(args.input)
    else:
        download()
        with open(RAW, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Drop the trailing partial story left by the byte-slice.
        text = text[: text.rfind("<|endoftext|>") + len("<|endoftext|>")]

    tok = train_tokenizer(text, retrain=args.retrain_tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    print(f"eot id = {eot}")

    print("encoding...")
    docs = text.split("<|endoftext|>")

    # Tráo thứ tự tài liệu TRƯỚC khi mã hoá.
    #
    # Tập val được cắt theo VỊ TRÍ (arr[-n_val:]), nên nếu không tráo thì mọi thứ
    # bạn nối vào cuối corpus sẽ rơi hết vào validation. Với TinyStories thuần thì
    # vô hại vì corpus đồng nhất; nhưng khi ghép dữ liệu riêng vào cuối, và dữ liệu
    # đó nhỏ hơn tập val, thì 100% nó nằm trong val và 0% trong train. Model không
    # học được gì, còn val loss thì vọt lên, và không có thông báo lỗi nào cả.
    if not args.no_shuffle:
        import random as _r
        _r.Random(1234).shuffle(docs)
        print(f"  đã tráo {len(docs):,} tài liệu (seed 1234) trước khi cắt train/val")
    ids = []
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i : i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            ids.extend(enc.ids)
            ids.append(eot)
        print(f"  {i + len(batch)}/{len(docs)} docs, {len(ids) / 1e6:.1f}M tokens", flush=True)

    dtype = np.uint16 if VOCAB_SIZE <= 65536 else np.uint32
    arr = np.array(ids, dtype=dtype)
    assert arr.max() < VOCAB_SIZE
    n_val = int(len(arr) * VAL_FRACTION)
    arr[:-n_val].tofile(os.path.join(HERE, f"train{suffix}.bin"))
    arr[-n_val:].tofile(os.path.join(HERE, f"val{suffix}.bin"))
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {len(text) / len(arr):.2f} bytes/token")


if __name__ == "__main__":
    sys.exit(main())
