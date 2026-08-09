#!/usr/bin/env python3
"""Đo loss của một checkpoint trên một tập .bin bất kỳ.

Dùng để trả lời câu quan trọng nhất khi fine-tune: model học thêm được dữ liệu
mới, nhưng QUÊN mất bao nhiêu thứ nó từng biết? Muốn biết thì phải đo loss trên
tập validation CŨ, trước và sau khi học thêm.

    cd src && uv run python eval_loss.py --run ../runs/x.pt --bin ../data/val.bin
"""

import argparse
import math
import os

import numpy as np
import torch

from model import Config, TinyLM
from train import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--bin", required=True, help="ví dụ ../data/val.bin")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=None, help="mặc định = seq_len của model")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    device = get_device()
    ck = torch.load(a.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    sl = a.seq_len or ck["cfg"]["seq_len"]
    data = np.memmap(a.bin, dtype=np.uint16, mode="r")
    if len(data) < sl + 2:
        raise SystemExit(f"{a.bin} quá ngắn ({len(data)} token) cho seq_len={sl}")

    # Hạt cố định: hai lần chạy trên cùng file luôn lấy đúng các cửa sổ như nhau,
    # nếu không thì chênh lệch trước/sau có thể chỉ là nhiễu do lấy mẫu khác chỗ.
    rng = np.random.default_rng(a.seed)
    total, n = 0.0, 0
    for _ in range(a.batches):
        ix = rng.integers(0, len(data) - sl - 1, a.batch_size)
        x = np.stack([data[i:i + sl] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + sl] for i in ix]).astype(np.int64)
        with torch.no_grad():
            _, loss = model(torch.from_numpy(x).to(device),
                            torch.from_numpy(y).to(device))
        total += loss.item()
        n += 1

    ce = total / n
    print(f"{os.path.basename(a.run):<28} {os.path.basename(a.bin):<18} "
          f"CE {ce:.4f}   ppl {math.exp(ce):8.2f}   ({n*a.batch_size} cửa sổ, seq_len={sl})")


if __name__ == "__main__":
    main()
