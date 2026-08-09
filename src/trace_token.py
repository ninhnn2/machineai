#!/usr/bin/env python3
"""Bóc từng bước một token đi qua model, xuất ra JSON cho demo tương tác.

Nguyên tắc: mọi con số trong demo phải là số THẬT lấy từ checkpoint, và phải
truy ngược được về một công thức và một dòng code. Không có số nào được dựng sẵn.

Chỗ khó: `F.scaled_dot_product_attention` trong model.py:104 làm softmax bên
trong kernel và không trả ra bảng attention weights. Muốn hiện bảng đó thì phải
tự tính lại bằng tay:

    scores = Q·Kᵀ / √head_dim   ->  mask nhân quả  ->  softmax  ->  @ V

Bản tự tính này BẮT BUỘC phải khớp output của SDPA, nếu không demo đang vẽ một
model khác với model thật. Script tự kiểm điều đó và thoát mã 1 nếu lệch.

Dùng:
    cd src && uv run python trace_token.py --run ../runs/ple-jetson-s0.pt \\
        --tokenizer ../data/bpe4096.json --prompt "the cat is" --out ../trace.json
"""

import argparse
import json
import math
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, TinyLM, apply_rope
from train import get_device

TOL = 2e-5          # sai số cho phép giữa bản tự tính và bản của PyTorch


def jl(t):
    """Tensor -> list float đã làm tròn, để JSON không phình."""
    return [round(float(v), 5) for v in t.flatten().tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The cat sat on the")
    ap.add_argument("--layer", type=int, default=0, help="lớp nào để soi attention")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--out", default="trace.json")
    a = ap.parse_args()

    device = torch.device("cpu")          # CPU cho số ổn định, trace không cần nhanh
    ck = torch.load(a.run, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(a.tokenizer)

    ids = tok.encode(a.prompt).ids
    if not ids:
        sys.exit("prompt rỗng sau khi tokenize")
    idx = torch.tensor([ids])
    T = len(ids)
    H, Dh = cfg.n_heads, cfg.head_dim
    last = T - 1

    # ---------------------------------------------------------------- bắt tensor
    grab = {}
    blk = model.blocks[a.layer]

    def hook_qkv(_m, inp, out):
        grab["qkv_in"] = inp[0].detach()      # x đã qua RMSNorm
        grab["qkv_out"] = out.detach()

    def hook_attn(_m, inp, out):
        grab["attn_out"] = out.detach()       # sau proj

    h1 = blk.attn.qkv.register_forward_hook(hook_qkv)
    h2 = blk.attn.register_forward_hook(hook_attn)

    def hook_block(_m, _i, out):
        grab["block_out"] = out.detach()
    h3 = blk.register_forward_hook(hook_block)

    with torch.no_grad():
        logits, _ = model(idx)
    for h in (h1, h2, h3):
        h.remove()

    # ---------------------------------------------------- kiểm 1: Q = Wq · x
    x_norm = grab["qkv_in"][0]                       # [T, D]
    W = blk.attn.qkv.weight                          # [3D, D]
    manual_qkv = x_norm @ W.T
    err_qkv = (manual_qkv - grab["qkv_out"][0]).abs().max().item()

    q, k, v = grab["qkv_out"][0].split(cfg.d_model, dim=1)
    q = q.view(T, H, Dh).transpose(0, 1)             # [H, T, Dh]
    k = k.view(T, H, Dh).transpose(0, 1)
    v = v.view(T, H, Dh).transpose(0, 1)
    qr = apply_rope(q.unsqueeze(0), model.cos, model.sin)[0]
    kr = apply_rope(k.unsqueeze(0), model.cos, model.sin)[0]

    # ------------------------------------- kiểm 2: attention tự tính vs SDPA
    scores = (qr @ kr.transpose(-2, -1)) / math.sqrt(Dh)          # [H, T, T]
    mask = torch.full((T, T), float("-inf")).triu(1)
    weights = F.softmax(scores + mask, dim=-1)
    manual_o = weights @ v                                        # [H, T, Dh]
    sdpa_o = F.scaled_dot_product_attention(
        qr.unsqueeze(0), kr.unsqueeze(0), v.unsqueeze(0), is_causal=True)[0]
    err_attn = (manual_o - sdpa_o).abs().max().item()
    err_rowsum = (weights.sum(-1) - 1).abs().max().item()

    # ---------------------------------------------------- kiểm 3: logits khớp
    with torch.no_grad():
        logits2, _ = model(idx)
    err_logits = (logits - logits2).abs().max().item()

    checks = {
        "qkv_thu_cong_vs_pytorch": err_qkv,
        "attention_thu_cong_vs_sdpa": err_attn,
        "tong_hang_softmax_lech_1": err_rowsum,
        "logits_lap_lai": err_logits,
    }
    fail = [k_ for k_, v_ in checks.items() if not (v_ < TOL)]

    # ---------------------------------------------------------------- top-k
    lg = logits[0, last]
    probs = F.softmax(lg, dim=-1)
    top = torch.topk(lg, a.topk)
    topk = [{"id": int(i), "text": tok.decode([int(i)]),
             "logit": round(float(l), 4), "prob": round(float(probs[int(i)]), 6)}
            for l, i in zip(top.values, top.indices)]

    temps = {}
    for t_ in (0.1, 0.8, 1.5):
        p = F.softmax(lg / t_, dim=-1)
        temps[str(t_)] = [round(float(p[e["id"]]), 6) for e in topk]

    with torch.no_grad():
        emb = model.tok_emb(idx)[0, last]
    D = cfg.d_model

    trace = {
        "prompt": a.prompt,
        "checkpoint": a.run.split("/")[-1],
        "config": {"d_model": D, "n_layers": cfg.n_layers, "n_heads": H,
                   "head_dim": Dh, "vocab": cfg.vocab_size, "ple_dim": cfg.ple_dim},
        "tokens": [{"id": int(i), "text": tok.decode([int(i)])} for i in ids],
        "layer": a.layer, "head": a.head,
        "embedding": {"vector": jl(emb), "norm": round(float(emb.norm()), 4),
                      "macs": 0, "note": "tra bảng, không nhân gì cả"},
        "qkv": {
            "q": jl(qr[a.head, last]), "k": jl(kr[a.head, last]), "v": jl(v[a.head, last]),
            "macs": 3 * D * D,
            "shape_W": [3 * D, D],
        },
        "attention": {
            "scores": jl(scores[a.head, last][:T]),
            "weights": jl(weights[a.head, last][:T]),
            "out": jl(manual_o[a.head, last]),
            "scale": round(1 / math.sqrt(Dh), 5),
            "macs": 2 * T * Dh,
        },
        "hidden": {"vector": jl(grab["block_out"][0, last]),
                   "norm": round(float(grab["block_out"][0, last].norm()), 4)},
        "logits": {"topk": topk, "vocab": cfg.vocab_size,
                   "macs": D * cfg.vocab_size,
                   "min": round(float(lg.min()), 3), "max": round(float(lg.max()), 3)},
        "temperature": temps,
        "checks": {k_: float(f"{v_:.3g}") for k_, v_ in checks.items()},
        "checks_pass": not fail,
    }

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, separators=(",", ":"))

    print(f"prompt   : {a.prompt!r} -> {T} token {[t['text'] for t in trace['tokens']]}")
    print(f"lớp {a.layer} head {a.head}, d_model={D}, vocab={cfg.vocab_size}")
    print("\nkiểm chứng (phải nhỏ hơn %.0e):" % TOL)
    for k_, v_ in checks.items():
        print(f"  {'✓' if v_ < TOL else '✗'} {k_:<32} {v_:.3e}")
    print(f"\ntop-1: {topk[0]['text']!r}  logit {topk[0]['logit']}  p={topk[0]['prob']:.4f}")
    print(f"ghi   : {a.out}")
    if fail:
        print("\n✗ KHÔNG ĐẠT: " + ", ".join(fail), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
