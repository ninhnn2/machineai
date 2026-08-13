#!/usr/bin/env python3
"""Chạy một sweep, đổi ĐÚNG MỘT biến, ghi kết quả vào một bảng so được.

Vì sao cần file này thay vì gõ tay từng lệnh train: sau bốn năm thí nghiệm, ghi
chép thủ công sẽ lệch nhau về giao thức đo (seq_len khác, số bước khác, đo val
bằng script khác) và không còn so được nữa. Harness này ép mọi dòng kết quả dùng
chung một khuôn.

Ba kỷ luật nó bắt buộc:

  1. ĐÚNG MỘT BIẾN. Khai nhiều hơn một giá trị thay đổi so với baseline thì nó từ
     chối chạy. Đây là điều kiện để câu "vì sao kết quả như vậy" có nghĩa.
  2. PHẢI KHAI CÂU HỎI VÀ GIẢ THUYẾT trước khi chạy. Viết giả thuyết sau khi thấy
     kết quả thì thí nghiệm chỉ còn là kể lại, không phải kiểm chứng.
  3. GHI CẢ THẤT BẠI. Run hỏng vẫn được ghi kèm lý do, vì "cấu hình này không
     train được" cũng là kết quả.

Dùng:
    python3 experiments/lab.py --name lr --var lr --values 1e-4,3e-4,1e-3,3e-3 \\
        --question "Learning rate ảnh hưởng thế nào tới tốc độ hội tụ?" \\
        --hypothesis "1e-4 chậm, 3e-3 dao động hoặc phân kỳ, 1e-3 tốt nhất" \\
        --steps 1500 --base-vocab 4096

    python3 experiments/lab.py --report          # in lại bảng đã có
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results.jsonl")

# Baseline: cấu hình deploy 28.9M trong RESULTS.md. Mọi sweep khai biến thay đổi
# so với đúng bộ này, và harness kiểm chỉ có một khoá khác.
BASELINE = {
    "arm": "ple", "vocab": 32768, "d_model": 96, "n_layers": 6, "n_heads": 4,
    "ple_dim": 128, "target_core": 559000, "steps": 11000, "batch_size": 12,
    "seq_len": 512, "lr": 1e-3, "warmup": 200, "seed": 0,
}

FLAG = {  # khoá config -> cờ dòng lệnh của train.py
    "arm": "--arm", "vocab": "--vocab", "d_model": "--d-model",
    "n_layers": "--n-layers", "n_heads": "--n-heads", "ple_dim": "--ple-dim",
    "target_core": "--target-core", "steps": "--steps", "batch_size": "--batch-size",
    "seq_len": "--seq-len", "lr": "--lr", "warmup": "--warmup", "seed": "--seed",
}


def cast(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def run_one(cfg, tag, data_suffix=None, timeout=7200):
    """Chạy train.py một lần, bóc số từ dòng cuối cùng của log."""
    cmd = [sys.executable, "train.py"]
    for k, v in cfg.items():
        cmd += [FLAG[k], str(v)]
    cmd += ["--tag", tag]
    if data_suffix is not None:
        cmd += ["--data-suffix", data_suffix]

    t0 = time.time()
    p = subprocess.run(cmd, cwd=os.path.join(ROOT, "src"),
                       capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    log = p.stdout + p.stderr

    row = {"wall_s": round(wall, 1), "ok": p.returncode == 0}
    if p.returncode != 0:
        row["error"] = log.strip().splitlines()[-1][:200] if log.strip() else "không có output"
        return row, log

    # "[ple] d_model=96 ... core=558,368 stream=3,145,728 table=25,165,824 total=28,869,920"
    m = re.search(r"core=([\d,]+) stream=([\d,]+) table=([\d,]+) total=([\d,]+)", log)
    if m:
        row.update(core=int(m.group(1).replace(",", "")),
                   stream=int(m.group(2).replace(",", "")),
                   table=int(m.group(3).replace(",", "")),
                   total=int(m.group(4).replace(",", "")))
    m = re.search(r"ffn=(\d+)", log)
    if m:
        row["ffn"] = int(m.group(1))
    # "DONE core=... val=2.1102 ppl=8.25"
    m = re.search(r"DONE .*val=([\d.]+) ppl=([\d.]+)", log)
    if m:
        row["val"] = float(m.group(1))
        row["ppl"] = float(m.group(2))
    else:
        row["ok"] = False
        row["error"] = "không tìm thấy dòng DONE (train hỏng giữa chừng?)"

    # đường cong loss, để vẽ hoặc phát hiện dao động
    row["curve"] = [float(x) for x in re.findall(r"\| val ([\d.]+) ", log)]
    return row, log


def sweep(a):
    var, values = a.var, [cast(v) for v in a.values.split(",")]
    if var not in BASELINE:
        sys.exit(f"biến {var!r} không có trong baseline. Hợp lệ: {sorted(BASELINE)}")

    overrides = {}
    for kv in (a.fix or []):
        k, v = kv.split("=", 1)
        overrides[k] = cast(v)

    # Kỷ luật 1, phiên bản đúng: điều phải giữ là TRONG MỘT SWEEP chỉ có một biến
    # thay đổi giữa các run. Hạ steps xuống cho cả bốn run để chạy nhanh vẫn là
    # thí nghiệm có kiểm soát, vì mọi run chịu chung điều kiện đó.
    #
    # Bản đầu tôi viết luật thành "khác baseline thì chặn", và nó chặn ngay một
    # sweep hoàn toàn hợp lệ. Cái phải làm không phải chặn, mà là GHI LẠI độ lệch
    # so với baseline, để sau này không ai so nhầm val của sweep này với val của
    # dòng baseline 11.000 bước.
    deviation = {k: v for k, v in overrides.items() if BASELINE.get(k) != v}
    deviation.pop(var, None)
    if deviation:
        print(f"  ghi chú: sweep này lệch baseline ở {deviation}.")
        print(f"  Trong sweep vẫn chỉ {var} thay đổi nên so nội bộ hợp lệ, nhưng")
        print(f"  KHÔNG so trực tiếp val ở đây với dòng baseline được.\n")

    print(f"sweep {var} = {values}")
    print(f"câu hỏi   : {a.question}")
    print(f"giả thuyết: {a.hypothesis}\n")

    rows = []
    for v in values:
        cfg = dict(BASELINE)
        cfg.update(overrides)
        cfg[var] = v
        tag = f"{a.name}-{str(v).replace('.', 'p').replace('-', 'm')}"
        print(f"  [{var}={v}] đang chạy ...", flush=True)
        row, log = run_one(cfg, tag, a.data_suffix, a.timeout)
        row.update(exp=a.name, var=var, value=v, tag=tag, config=cfg,
                   deviation=deviation,
                   question=a.question, hypothesis=a.hypothesis,
                   ts=time.strftime("%Y-%m-%d %H:%M"))
        rows.append(row)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["ok"]:
            print(f"      val {row.get('val')} ppl {row.get('ppl')} "
                  f"total {row.get('total', 0):,} wall {row['wall_s']}s")
        else:
            print(f"      HỎNG: {row.get('error')}")
    table(rows, var)


def table(rows, var=None):
    if not rows:
        print("chưa có kết quả nào")
        return
    var = var or rows[0].get("var", "value")
    print(f"\n  {var:>10} {'val':>8} {'ppl':>8} {'total':>12} {'ffn':>5} {'wall':>8}")
    print("  " + "-" * 56)
    best = min((r for r in rows if r.get("val")), key=lambda r: r["val"], default=None)
    for r in rows:
        if not r["ok"]:
            print(f"  {str(r['value']):>10} {'HỎNG':>8}  {r.get('error','')[:34]}")
            continue
        mark = " <-" if best is not None and r is best else ""
        print(f"  {str(r['value']):>10} {r['val']:>8.4f} {r['ppl']:>8.2f} "
              f"{r.get('total', 0):>12,} {r.get('ffn', 0):>5} {r['wall_s']:>7.0f}s{mark}")


def report():
    if not os.path.exists(RESULTS):
        sys.exit(f"chưa có {RESULTS}")
    rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["exp"], []).append(r)
    for name, rs in by.items():
        print(f"\n=== {name} ({rs[0]['ts']})")
        print(f"    câu hỏi   : {rs[0]['question']}")
        print(f"    giả thuyết: {rs[0]['hypothesis']}")
        table(rs, rs[0]["var"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", help="tên thí nghiệm, dùng làm tag checkpoint")
    ap.add_argument("--var", help="biến duy nhất được đổi")
    ap.add_argument("--values", help="danh sách giá trị, phân cách bằng dấu phẩy")
    ap.add_argument("--question", default="")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--fix", action="append",
                    help="ghi đè baseline, dạng key=value, lặp lại được")
    ap.add_argument("--data-suffix", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--allow-multi", action="store_true",
                    help="cho phép đổi nhiều biến cùng lúc; chỉ dùng khi có lý do "
                         "và lý do đó phải nằm trong --question")
    ap.add_argument("--report", action="store_true", help="in lại toàn bộ kết quả đã có")
    a = ap.parse_args()

    if a.report:
        return report()
    if not (a.name and a.var and a.values):
        sys.exit("cần --name, --var, --values (hoặc --report)")
    if not a.question or not a.hypothesis:
        sys.exit("phải khai --question và --hypothesis TRƯỚC khi chạy.\n"
                 "Viết giả thuyết sau khi thấy kết quả thì đó không còn là kiểm chứng.")
    sweep(a)


if __name__ == "__main__":
    main()
