#!/usr/bin/env python3
"""Đo mất mát thật khi ép weight FP32 xuống INT4, xuất JSON cho demo.

Dùng ĐÚNG hàm lượng tử của repo (`export.py:quant_pack`), không viết lại, để con
số hiện trên demo là con số runtime C thật sự đọc.

Phép kiểm quan trọng nhất: giải nén một hàng từ chính `firmware/model/model.bin`
đã commit, theo đúng cách `deq_row()` trong `llm.h` làm, rồi so với bản
dequantize bên Python. Hai bên phải khớp. Nếu không khớp thì demo đang vẽ một
model khác với model đang chạy trên chip.

Dùng:
    cd src && uv run python trace_quant.py --run ../runs/ple-jetson-s0.pt \\
        --bin ../firmware/model/model.bin --out ../trace_quant.json
"""

import argparse
import json
import struct
import sys

import numpy as np
import torch

from export import GROUP, MAGIC, quant_pack

TOL = 1e-6


def deq_row_like_c(codes, scales, cols, group):
    """Giải nén một hàng int4 y hệt deq_row() trong llm.h.

    Mỗi byte chứa 2 giá trị: nibble thấp là phần tử chẵn, nibble cao là lẻ, và
    giá trị thật = (nibble - 8) * scale của nhóm. Viết lại ở đây bằng numpy để
    chứng minh mình hiểu đúng định dạng, chứ không phải tin vào Python.
    """
    out = np.zeros(cols, dtype=np.float32)
    for j in range(cols):
        byte = codes[j >> 1]
        nib = (byte & 0x0F) if (j % 2 == 0) else (byte >> 4)
        out[j] = (float(nib) - 8.0) * float(scales[j // group])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--bin", required=True, help="firmware/model/model.bin đã commit")
    ap.add_argument("--out", default="trace_quant.json")
    ap.add_argument("--row", type=int, default=708, help="hàng nào của tok_emb (708 = ' cat')")
    a = ap.parse_args()

    ck = torch.load(a.run, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    W = ck["state"]["tok_emb.weight"]           # [V, D] fp32
    D = W.shape[1]

    # ---------------------------------------------------------- lượng tử thật
    packed, scales, dq = quant_pack(W, group=GROUP)
    # quant_pack trả về mảng PHẲNG, phải tự cắt ra hàng cần xem
    row_bytes = (D + 1) // 2
    n_groups = (D + GROUP - 1) // GROUP
    row_fp32 = W[a.row].tolist()
    row_dq = dq[a.row].tolist()
    row_codes = np.asarray(packed)[a.row * row_bytes:(a.row + 1) * row_bytes]
    row_scales = np.asarray(scales)[a.row * n_groups:(a.row + 1) * n_groups]

    # mã int4 dạng số nguyên có dấu, để hiện trên demo
    nib = []
    for j in range(D):
        b = int(row_codes[j >> 1])
        nib.append(((b & 0x0F) if j % 2 == 0 else (b >> 4)) - 8)

    # ------------------------------------------- kiểm: giải nén như C có khớp
    mine = deq_row_like_c(row_codes, row_scales, D, GROUP)
    err_c = float(np.abs(mine - np.asarray(row_dq)).max())

    # -------------------------------- kiểm: model.bin trên đĩa cùng một tensor
    raw = open(a.bin, "rb").read()
    magic, = struct.unpack_from("<I", raw, 0)
    err_bin = None
    if magic == MAGIC:
        # tok_emb là tensor lượng tử đầu tiên (export.py:89), ngay sau header.
        # Header: magic uint32 + 8 int32 cấu hình + rope_theta float32 = 40 byte.
        # Lần đầu tôi tính 36 và bỏ quên rope_theta, phép kiểm bên dưới bắt được
        # ngay: lệch 0.334 thay vì 0. Đó chính là việc của nó.
        off = 4 + 8 * 4 + 4
        g, = struct.unpack_from("<i", raw, off); off += 4
        rows, cols = W.shape
        row_bytes = (cols + 1) // 2
        n_groups = (cols + g - 1) // g
        c0 = np.frombuffer(raw, dtype=np.uint8, count=row_bytes,
                           offset=off + a.row * row_bytes)
        s_off = off + rows * row_bytes
        s0 = np.frombuffer(raw, dtype=np.float16, count=n_groups,
                           offset=s_off + a.row * n_groups * 2)
        from_disk = deq_row_like_c(c0, s0, cols, g)
        err_bin = float(np.abs(from_disk - np.asarray(row_dq)).max())

    # ------------------------------------------------- quét group size và bit
    def rms_err(w, bits, group):
        x = w.reshape(-1, w.shape[-1]).float()
        cols = x.shape[1]
        q = torch.zeros_like(x)
        lim = 2 ** (bits - 1) - 1
        for gi in range((cols + group - 1) // group):
            lo, hi = gi * group, min((gi + 1) * group, cols)
            seg = x[:, lo:hi]
            sc = (seg.abs().amax(dim=1, keepdim=True) / lim).clamp_min(1e-8).half().float()
            q[:, lo:hi] = torch.clamp(torch.round(seg / sc), -lim, lim) * sc
        e = (q - x)
        return float(e.pow(2).mean().sqrt()), float(x.pow(2).mean().sqrt())

    sample = W[:512]
    sweep = []
    for bits in (2, 3, 4, 8):
        for group in (16, 32, 64, 128, 256):
            r, base = rms_err(sample, bits, group)
            bytes_per_w = bits / 8 + 2 / group          # code + scale fp16 mỗi nhóm
            sweep.append({"bits": bits, "group": group,
                          "rms": round(r, 6), "rel": round(r / base, 5),
                          "bytes_per_weight": round(bytes_per_w, 4),
                          "mb_28m": round(28.9e6 * bytes_per_w / 1e6, 2)})

    # ------------------------------------------ giá trị nhỏ bị nghiền thành 0
    g0 = np.asarray(row_fp32[:GROUP])
    killed = int((np.asarray(row_dq[:GROUP]) == 0).sum())
    trace = {
        "config": {"d_model": D, "vocab": cfg["vocab_size"], "group": GROUP, "bits": 4,
                   "row": a.row, "row_label": "tok_emb[708] = ' cat'"},
        "row": {"fp32": [round(v, 5) for v in row_fp32[:32]],
                "codes": nib[:32],
                "dq": [round(v, 5) for v in row_dq[:32]],
                "err": [round(abs(row_fp32[i] - row_dq[i]), 5) for i in range(32)]},
        "group0": {"scale": round(float(row_scales[0]), 6),
                   "absmax": round(float(np.abs(g0).max()), 5),
                   "step": round(float(row_scales[0]), 6),
                   "zeroed": killed, "of": GROUP},
        "sweep": sweep,
        "checks": {"giai_nen_kieu_C_vs_python": float(f"{err_c:.3g}"),
                   "model_bin_tren_dia_vs_python":
                       None if err_bin is None else float(f"{err_bin:.3g}")},
    }
    fails = [k for k, v in trace["checks"].items() if v is not None and v > TOL]
    trace["checks_pass"] = not fails

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, separators=(",", ":"))

    print(f"tok_emb: [{W.shape[0]}, {D}] fp32, lượng tử int4 group={GROUP}")
    print(f"hàng {a.row} (' cat'), nhóm 0: |max| = {trace['group0']['absmax']}, "
          f"scale = {trace['group0']['scale']}")
    print(f"  {killed}/{GROUP} giá trị trong nhóm 0 bị nghiền thành đúng 0")
    print()
    print("kiểm chứng:")
    for k, v in trace["checks"].items():
        mark = "✓" if (v is not None and v <= TOL) else ("-" if v is None else "✗")
        print(f"  {mark} {k:<32} {v}")
    print()
    print("  bit  group   RMS lỗi   byte/weight   28.9M ->")
    for s in sweep:
        if s["group"] in (64, 128):
            print(f"  {s['bits']:>3}  {s['group']:>5}   {s['rms']:.6f}   "
                  f"{s['bytes_per_weight']:.4f}        {s['mb_28m']:.1f} MB")
    print(f"\nghi: {a.out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
