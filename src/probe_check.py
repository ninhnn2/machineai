#!/usr/bin/env python3
"""Kiểm chứng dữ liệu riêng đã vào model hay chưa, bằng mồi và đối chứng.

Đọc manifest do data/make_probe.py sinh ra, rồi với mỗi sự kiện:

  * cho model sinh tiếp từ câu hỏi và xem đáp án đúng có xuất hiện không;
  * đo NLL (negative log-likelihood) trung bình của đúng chuỗi đáp án.

Sự kiện MỒI có trong corpus, sự kiện ĐỐI CHỨNG thì không. Chỉ số đáng tin không
phải là điểm của nhóm mồi, mà là KHOẢNG CÁCH giữa hai nhóm. Nhóm mồi cao mà đối
chứng cũng cao thì model chỉ đang đoán, và con số đó không chứng minh được gì.

Dùng:
    cd src && uv run python probe_check.py --run ../runs/ple-probe-s0.pt \\
        --manifest /tmp/probe/probe.json --tokenizer ../data/bpe2048.json
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, TinyLM
from train import get_device


def nll_of(model, tok, prompt, answer, device):
    """NLL trung bình mỗi token của `answer` khi đã cho `prompt`.

    Thấp = model thấy chuỗi đó là chuyện đương nhiên, tức nhiều khả năng đã gặp
    trong lúc train.
    """
    p_ids = tok.encode(prompt).ids
    a_ids = tok.encode(" " + answer).ids
    if not a_ids:
        return float("nan")
    ids = torch.tensor([p_ids + a_ids], device=device)
    with torch.no_grad():
        logits, _ = model(ids)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    for k, tid in enumerate(a_ids):
        pos = len(p_ids) + k - 1          # vị trí dự đoán ra token thứ k
        total += logp[pos, tid].item()
    return -total / len(a_ids)


def recalled(model, tok, prompt, answer, device, n_tokens=8):
    """Sinh tham lam (greedy) rồi xem model có trả lời ĐÚNG đáp án không.

    Phải so bằng ranh giới từ, không so chuỗi con. Câu hỏi "The TQ-87 board has"
    mà model trả "sixty-four" thì đó là đáp án SAI, nhưng phép so chuỗi con lại
    tính đúng vì "four" nằm trong "sixty-four". Lỗi kiểu đó thổi phồng điểm nhớ
    mồi, tức là làm hỏng đúng cái mà công cụ này sinh ra để đo.
    """
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    out = _greedy(model, ids, n_tokens)
    text = tok.decode(out[0].tolist())
    tail = text[len(tok.decode(ids[0].tolist())):]
    # đáp án phải đứng NGAY đầu phần sinh ra, và kết thúc ở ranh giới từ
    ok = bool(re.match(r"\s*" + re.escape(answer) + r"(?![\w-])", tail, flags=re.I))
    return ok, tail.strip()


def _greedy(model, ids, n):
    for _ in range(n):
        with torch.no_grad():
            logits, _ = model(ids[:, -256:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="checkpoint .pt")
    ap.add_argument("--manifest", required=True, help="probe.json từ make_probe.py")
    ap.add_argument("--tokenizer", required=True, help="bpeN.json đã dùng để train")
    ap.add_argument("--show", type=int, default=5, help="in bao nhiêu ví dụ mỗi nhóm")
    a = ap.parse_args()

    device = get_device()
    ck = torch.load(a.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(a.tokenizer)
    man = json.load(open(a.manifest, encoding="utf-8"))

    result = {}
    for name in ("probe", "control"):
        hits, nlls, ex = 0, [], []
        for f in man[name]:
            ok, tail = recalled(model, tok, f["prompt"], f["answer"], device)
            hits += ok
            nlls.append(nll_of(model, tok, f["prompt"], f["answer"], device))
            if len(ex) < a.show:
                ex.append((f["prompt"], f["answer"], tail, ok))
        n = len(man[name])
        result[name] = {"hits": hits, "n": n,
                        "acc": hits / n,
                        "nll": sum(nlls) / len(nlls), "examples": ex}

    p, c = result["probe"], result["control"]
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  DỮ LIỆU CỦA BẠN ĐÃ VÀO MODEL CHƯA                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  checkpoint : {os.path.basename(a.run)}")
    print(f"  tokenizer  : {os.path.basename(a.tokenizer)}\n")
    print(f"  {'nhóm':<12}{'nhớ đúng':>14}{'NLL đáp án':>14}")
    print(f"  {'-'*40}")
    print(f"  {'MỒI':<12}{p['hits']:>7}/{p['n']:<6}{p['nll']:>14.3f}   (có trong corpus)")
    print(f"  {'ĐỐI CHỨNG':<12}{c['hits']:>7}/{c['n']:<6}{c['nll']:>14.3f}   (không có)")

    gap_acc = p["acc"] - c["acc"]
    gap_nll = c["nll"] - p["nll"]
    print(f"\n  chênh lệch : {gap_acc*100:+.0f} điểm phần trăm, NLL thấp hơn {gap_nll:+.3f}")

    if p["acc"] >= 0.7 and gap_acc >= 0.5:
        verdict, code = "DỮ LIỆU ĐÃ VÀO MODEL", 0
    elif gap_acc >= 0.25 or gap_nll >= 0.5:
        verdict, code = "CÓ DẤU VẾT, NHƯNG YẾU (train thêm hoặc lặp mồi nhiều hơn)", 0
    else:
        verdict, code = "KHÔNG CHỨNG MINH ĐƯỢC: model chỉ đang đoán", 1
    print(f"  KẾT LUẬN   : {verdict}\n")

    for name, lab in (("probe", "MỒI"), ("control", "ĐỐI CHỨNG")):
        print(f"  ── ví dụ {lab} " + "─" * 34)
        for prompt, ans, tail, ok in result[name]["examples"]:
            mark = "✓" if ok else "✗"
            print(f"   {mark} \"{prompt}\" -> \"{tail[:38]}\"   (cần: {ans})")
        print()
    return code


if __name__ == "__main__":
    sys.exit(main())
