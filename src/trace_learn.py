#!/usr/bin/env python3
"""Ghi lại toàn bộ quá trình học của một mạng 9 tham số, xuất JSON cho demo.

Mạng nhỏ tới mức in được TẤT CẢ trọng số lên màn hình:

    x[2] --> Linear(2->2) --> ReLU --> Linear(2->1) --> y      9 tham số

Vì sao nhỏ như vậy: mục tiêu không phải giải bài toán nào, mà là nhìn thấy từng
con số gradient và từng bước cập nhật. Với 1,5 triệu tham số thì không nhìn được
gì cả.

Điều script này chứng minh: `loss.backward()` không phải phép màu. Nó đi ngược đồ
thị tính toán và áp dụng chain rule ở từng nút. Để chứng minh, script tính
gradient BẰNG TAY theo công thức đạo hàm, rồi so với autograd của PyTorch. Hai
bên phải khớp tới 1e-6, nếu không thì thoát mã 1.

    L      = (y - t)^2
    dL/dy  = 2(y - t)
    dL/dW2 = dL/dy * h          dL/db2 = dL/dy
    dL/dh  = dL/dy * W2
    dL/dz  = dL/dh * [z > 0]    (đạo hàm ReLU)
    dL/dW1 = dL/dz (x)^T        dL/db1 = dL/dz

Dùng:
    cd src && uv run python trace_learn.py --steps 40 --out ../trace_learn.json
"""

import argparse
import json
import sys

import torch

TOL = 1e-6


def r(t, n=4):
    if isinstance(t, torch.Tensor):
        t = t.detach().flatten().tolist()
    if isinstance(t, (list, tuple)):
        return [round(float(v), n) for v in t]
    return round(float(t), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="trace_learn.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    W1 = (torch.randn(2, 2) * 0.8).requires_grad_(True)
    b1 = torch.zeros(2, requires_grad=True)
    W2 = (torch.randn(1, 2) * 0.8).requires_grad_(True)
    b2 = torch.zeros(1, requires_grad=True)
    params = [W1, b1, W2, b2]

    x = torch.tensor([0.8, -0.5])
    target = torch.tensor([1.0])

    steps, worst = [], 0.0
    for it in range(a.steps):
        # ---- forward, giữ lại mọi giá trị trung gian để demo hiện được
        z = W1 @ x + b1              # trước ReLU
        h = torch.relu(z)
        y = W2 @ h + b2
        loss = (y - target).pow(2).sum()

        # ---- backward của PyTorch
        for p in params:
            if p.grad is not None:
                p.grad = None
        loss.backward()
        auto = {"W1": r(W1.grad, 6), "b1": r(b1.grad, 6),
                "W2": r(W2.grad, 6), "b2": r(b2.grad, 6)}

        # ---- backward TỰ TÍNH bằng chain rule
        with torch.no_grad():
            dy = 2 * (y - target)                   # dL/dy
            gW2 = dy.view(1, 1) * h.view(1, 2)      # dL/dW2
            gb2 = dy.clone()
            dh = dy.view(1) * W2.view(2)            # dL/dh
            dz = dh * (z > 0).float()               # qua ReLU
            gW1 = dz.view(2, 1) * x.view(1, 2)      # dL/dW1
            gb1 = dz.clone()
        man = {"W1": r(gW1, 6), "b1": r(gb1, 6), "W2": r(gW2, 6), "b2": r(gb2, 6)}

        with torch.no_grad():
            diff = max(
                (W1.grad - gW1).abs().max().item(), (b1.grad - gb1).abs().max().item(),
                (W2.grad - gW2).abs().max().item(), (b2.grad - gb2).abs().max().item())
        worst = max(worst, diff)

        rec = {
            "step": it,
            "W1": r(W1), "b1": r(b1), "W2": r(W2), "b2": r(b2),
            "z": r(z), "h": r(h), "y": r(y), "loss": round(loss.item(), 6),
            "grad_auto": auto, "grad_manual": man, "grad_diff": float(f"{diff:.3g}"),
            "relu_killed": [bool(v <= 0) for v in z.detach().tolist()],
        }

        # ---- cập nhật: w = w - lr * grad
        with torch.no_grad():
            for p in params:
                p -= a.lr * p.grad
        rec["W1_after"] = r(W1)
        steps.append(rec)

    ok = worst < TOL
    out = {
        "config": {"lr": a.lr, "seed": a.seed, "steps": a.steps,
                   "x": r(x), "target": r(target), "n_params": 9},
        "steps": steps,
        "loss_curve": [s["loss"] for s in steps],
        "max_grad_diff": float(f"{worst:.3g}"),
        "checks_pass": ok,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"mạng 9 tham số, x={r(x)}, target={r(target)}, lr={a.lr}")
    print(f"loss: {steps[0]['loss']:.6f} -> {steps[-1]['loss']:.6f} sau {a.steps} bước")
    print()
    print("chain rule tự tính so với loss.backward():")
    print(f"  sai lệch lớn nhất qua {a.steps} bước x 9 tham số: {worst:.3e}")
    print(f"  {'✓ KHỚP' if ok else '✗ LỆCH'} (ngưỡng {TOL:.0e})")
    print()
    print("  bước  loss        dL/dW2[0]   auto        tay")
    for s in steps[:3] + steps[-1:]:
        print(f"  {s['step']:>4}  {s['loss']:<11.6f} "
              f"{s['grad_auto']['W2'][0]:>10.6f}  {s['grad_manual']['W2'][0]:>10.6f}")
    print(f"\nghi: {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
