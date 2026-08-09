#!/usr/bin/env python3
"""Sinh corpus có MỒI KIỂM CHỨNG, để trả lời được câu "dữ liệu của tôi đã vào model chưa".

Vấn đề: sau khi train xong, nhìn val loss giảm không chứng minh được dữ liệu
RIÊNG của bạn đã được học. Loss giảm chỉ nói model học được tiếng Anh nói chung.

Cách làm ở đây: nhét vào corpus một số sự kiện bịa hoàn toàn, không thể có trong
bất kỳ dữ liệu nào khác, dạng

    The ZK-14 board has thirty-two kilobytes of memory.

rồi sau khi train, hỏi model "The ZK-14 board has" và xem nó có trả lời đúng
"thirty-two" không.

Điểm mấu chốt là NHÓM ĐỐI CHỨNG. Script sinh gấp đôi số sự kiện rồi chỉ đưa một
nửa vào corpus; nửa còn lại giữ lại làm đối chứng. Không có bước này thì một
model chỉ biết đoán bừa số nào cũng đúng sẽ trông y như thành công. Có nó thì
kết luận mới đứng được:

    mồi 38/40 đúng, đối chứng 1/40 đúng  ->  dữ liệu ĐÃ vào model
    mồi 20/40 đúng, đối chứng 19/40 đúng ->  model chỉ đang đoán

Dùng:
    python3 data/make_probe.py --out /tmp/probe --facts 40 --repeat 150
"""

import argparse
import json
import os
import random

# Mã thiết bị bịa: hai chữ cái + số. Không trùng từ thật nào, nên nếu model đọc
# được chúng thì chỉ có thể do đã thấy trong corpus này.
LETTERS = ["ZK", "QX", "VB", "JM", "KP", "WT", "FN", "HD", "LR", "GS",
           "NC", "YB", "PD", "XR", "MZ", "TQ", "BV", "CJ", "DW", "EH"]

# Đáp án viết bằng CHỮ, không phải chữ số: chữ số dễ bị tokenizer cắt vụn và
# model có thể đoán trúng ngẫu nhiên trong dải hẹp.
AMOUNTS = ["four", "eight", "twelve", "sixteen", "twenty", "thirty-two",
           "forty", "sixty-four", "eighty", "ninety-six", "one hundred",
           "two hundred", "five hundred", "seven hundred"]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "silver",
          "golden", "black", "white"]

CITIES = ["Danang", "Kyoto", "Bergen", "Ravenna", "Tromso", "Oaxaca",
          "Galway", "Hobart", "Rijeka", "Kanazawa"]

# Ba khuôn câu khác nhau, để việc nhớ được không phụ thuộc một cấu trúc duy nhất.
TEMPLATES = [
    ("The {code} board has {amount} kilobytes of memory.",
     "The {code} board has", "{amount}"),
    ("The {code} sensor is {color}.",
     "The {code} sensor is", "{color}"),
    ("The {code} team works in {city}.",
     "The {code} team works in", "{city}"),
]

# Câu nền: model cần học tiếng Anh cơ bản, nếu không nó không sinh nổi câu nào.
NAMES = ["Minh", "Lan", "Nam", "Mai", "Tom", "Ann", "Ben", "Kim"]
THINGS = ["a red ball", "a small cat", "a blue kite", "a toy car", "a paper boat"]
PLACES = ["the park", "the garden", "the beach", "school", "the river"]


def make_facts(n, rng):
    """Sinh n sự kiện đôi một khác nhau về mã thiết bị."""
    codes = [f"{l}-{i}" for l in LETTERS for i in range(10, 100)]
    rng.shuffle(codes)
    facts = []
    for i in range(n):
        code = codes[i]
        tpl, prompt, ans = TEMPLATES[i % len(TEMPLATES)]
        if "amount" in tpl:
            val = rng.choice(AMOUNTS)
        elif "color" in tpl:
            val = rng.choice(COLORS)
        else:
            val = rng.choice(CITIES)
        key = "amount" if "amount" in tpl else ("color" if "color" in tpl else "city")
        facts.append({
            "code": code,
            "sentence": tpl.format(code=code, **{key: val}),
            "prompt": prompt.format(code=code),
            "answer": val,
        })
    return facts


def filler(n, rng):
    out = []
    for _ in range(n):
        nm, th, pl = rng.choice(NAMES), rng.choice(THINGS), rng.choice(PLACES)
        out.append(f"One day {nm} found {th} near {pl}. {nm} was very happy. "
                   f"{nm} played with {th} until the sun went down. "
                   f"Then {nm} went home and slept well.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/tmp/probe", help="thư mục kết quả")
    ap.add_argument("--facts", type=int, default=40, help="số sự kiện mồi (đối chứng cũng bằng ngần này)")
    ap.add_argument("--repeat", type=int, default=150, help="mỗi sự kiện mồi lặp bao nhiêu lần")
    ap.add_argument("--filler", type=int, default=12000, help="số đoạn văn nền")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    allf = make_facts(a.facts * 2, rng)
    probe, control = allf[:a.facts], allf[a.facts:]

    # Trộn mồi vào giữa văn nền chứ không dồn một chỗ, để model không chỉ học
    # được "đoạn cuối file thì nói về board".
    docs = filler(a.filler, rng)
    for f in probe:
        for _ in range(a.repeat):
            docs.append(f["sentence"])
    rng.shuffle(docs)

    corpus = os.path.join(a.out, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as fh:
        fh.write("<|endoftext|>".join(docs))

    manifest = os.path.join(a.out, "probe.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"probe": probe, "control": control,
                   "repeat": a.repeat, "seed": a.seed}, fh, indent=2)

    mb = os.path.getsize(corpus) / 1e6
    print(f"corpus   : {corpus}  ({mb:.2f} MB, {len(docs):,} tài liệu)")
    print(f"manifest : {manifest}")
    print(f"  mồi      {len(probe)} sự kiện, mỗi cái lặp {a.repeat} lần -> CÓ trong corpus")
    print(f"  đối chứng {len(control)} sự kiện                          -> KHÔNG có trong corpus")
    print(f"\nví dụ mồi     : {probe[0]['sentence']}")
    print(f"ví dụ đối chứng: {control[0]['sentence']}  (model không được thấy câu này)")


if __name__ == "__main__":
    main()
