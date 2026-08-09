#!/usr/bin/env python3
"""Chấm điểm "giọng người" cho văn bản kỹ thuật TIẾNG VIỆT.

Vì sao có file này: scorer của skill content-humanizer chấm 60/100 điểm bằng ba
danh sách regex TIẾNG ANH (delve, leverage, crucial, "it's important to note").
Văn bản tiếng Việt không khớp từ nào nên luôn được điểm tuyệt đối ở ba mục đó.
Đấy là "chưa đo được", không phải "đã sạch". File này thay ba mục đó bằng danh
sách tương đương tiếng Việt.

Hai khác biệt nữa so với bản gốc:

  * Bóc code block, frontmatter và bảng markdown TRƯỚC khi chấm. Bản gốc đếm
    `---` của frontmatter, `|---|` của bảng và cờ CLI `--vocab` như em dash,
    nên trừ điểm oan.
  * Có exit code để chặn trong CI: 0 đạt, 1 dưới ngưỡng.

Dùng:
    python3 vn_humanizer.py bai-viet.md
    python3 vn_humanizer.py bai-viet.md --json
    python3 vn_humanizer.py --sample          # xem nó bắt được gì
    python3 vn_humanizer.py bai.md --min 80   # chặn CI dưới 80 điểm
"""

import argparse
import json
import re
import statistics
import sys

# ---------------------------------------------------------------------------
# 1. TỪ NGỮ SÁO RỖNG  (tương đương delve / leverage / crucial / landscape)
#
# Tiêu chí đưa vào danh sách: cụm nghe "trang trọng" nhưng bỏ đi thì câu không
# mất thông tin nào. Đó chính là dấu hiệu nó được thêm vào để lấp chỗ.
# ---------------------------------------------------------------------------
SAO_RONG = [
    # thổi phồng tầm quan trọng
    "đóng vai trò then chốt", "đóng vai trò quan trọng", "đóng vai trò cốt lõi",
    "vô cùng quan trọng", "cực kỳ quan trọng", "hết sức quan trọng",
    "không thể thiếu", "chìa khoá thành công", "chìa khóa thành công",
    "yếu tố quyết định",
    # mở bài rỗng
    "trong bối cảnh hiện nay", "trong thời đại ngày nay", "trong kỷ nguyên số",
    "ngày càng trở nên phổ biến", "ngày càng phát triển mạnh mẽ",
    "không thể phủ nhận rằng", "không thể không nhắc đến",
    # tính từ tiếp thị
    "mang tính cách mạng", "đột phá", "vượt trội", "tối ưu nhất",
    "toàn diện", "mạnh mẽ", "đa dạng và phong phú", "phong phú và đa dạng",
    "hàng đầu", "tiên phong", "đáng kể",
    # động từ sáo
    "tận dụng tối đa", "khai thác tối đa", "nâng cao hiệu quả",
    "thúc đẩy sự phát triển", "góp phần quan trọng",
    "đi sâu vào tìm hiểu", "tìm hiểu sâu hơn về", "khám phá sâu hơn",
    "vượt qua thách thức", "giải quyết triệt để",
    # ẩn dụ mòn
    "bức tranh toàn cảnh", "hành trình khám phá", "làn sóng công nghệ",
    "cuộc cách mạng công nghiệp",
]

# ---------------------------------------------------------------------------
# 2. RÀO ĐÓN  (tương đương "it's important to note that")
#
# Người viết thật rào đón khi thật sự không chắc. Máy rào đón ở mọi câu.
# ---------------------------------------------------------------------------
RAO_DON = [
    "điều quan trọng cần lưu ý là", "điều quan trọng cần nhớ là",
    "cần lưu ý rằng", "cần nhấn mạnh rằng", "đáng chú ý là", "đáng lưu ý là",
    "có thể nói rằng", "có thể thấy rằng", "có thể khẳng định rằng",
    "nhìn chung có thể thấy", "nói một cách tổng quát",
    "trong nhiều trường hợp", "trong hầu hết các trường hợp",
    "ở một mức độ nào đó", "một cách nào đó",
    "khỏi phải nói", "không cần phải nói",
    "như chúng ta đã biết", "như đã đề cập ở trên", "như đã trình bày ở trên",
    "tuy nhiên cần lưu ý", "song cần lưu ý",
]

# ---------------------------------------------------------------------------
# 3. KẾT BÀI SÁO  (tương đương "In conclusion, we explored...")
# ---------------------------------------------------------------------------
KET_SAO = [
    "hy vọng bài viết này", "hy vọng qua bài viết", "hi vọng bài viết này",
    "chúc bạn thành công", "chúc các bạn thành công",
    "trên đây là toàn bộ", "trên đây là những", "bài viết trên đã",
    "tóm lại có thể thấy", "kết luận lại thì",
    "hãy cùng chờ đón", "đừng quên theo dõi",
]

# ---------------------------------------------------------------------------
# 4. CHUYỂN ĐOẠN MÁY MÓC — chỉ tính khi LẶP nhiều, một hai lần là bình thường
# ---------------------------------------------------------------------------
CHUYEN_DOAN = [
    "hơn nữa", "bên cạnh đó", "ngoài ra", "đồng thời",
    "chính vì vậy", "chính vì thế", "do đó", "vì vậy", "vì thế",
    "đầu tiên", "tiếp theo", "cuối cùng",
]


# ---------------------------------------------------------------------------
# 5. GIẢ BỘ THẲNG THẮN
#
# Tic đặc trưng của văn AI: tuyên bố rằng mình sắp nói thật, thay vì nói luôn.
# Người viết thật chỉ nói điều cần nói. Câu "nói thẳng là X" gần như luôn rút
# gọn được thành "X" mà không mất gì, đó là dấu hiệu nó chỉ để lấy giọng.
# ---------------------------------------------------------------------------
GIA_THANG = [
    "nói thẳng", "nói thật", "thành thật mà nói", "thú thật",
    "phải nói rằng", "phải thừa nhận rằng", "nói cho công bằng",
    "nói không ngoa", "chẳng giấu gì", "thẳng thắn mà nói",
]

MAX_DIEM = {"sao_rong": 20, "rao_don": 15, "ket_sao": 10, "gia_thang": 5,
            "chuyen_doan": 10, "nhip_cau": 20, "in_dam": 10, "gach_ngang": 10}

MAU_AI = """Trong bối cảnh hiện nay, trí tuệ nhân tạo đóng vai trò then chốt và
ngày càng trở nên phổ biến. Không thể phủ nhận rằng đây là một công nghệ mang
tính cách mạng. Điều quan trọng cần lưu ý là các doanh nghiệp cần tận dụng tối đa
tiềm năng của nó. Hơn nữa, việc ứng dụng AI góp phần quan trọng vào việc nâng cao
hiệu quả. Bên cạnh đó, nó còn giúp vượt qua thách thức của chuyển đổi số. Nhìn
chung có thể thấy, đây là giải pháp toàn diện và mạnh mẽ. Tóm lại có thể thấy,
hy vọng bài viết này đã mang lại cho bạn bức tranh toàn cảnh về chủ đề này."""

MAU_NGUOI = """Tôi mất ba tiếng mới hiểu vì sao con số không khớp. Hoá ra bảng
PLE đọc 6 hàng mỗi token chứ không phải 1. Đo lại trên board: 0.12 ms, không phải
0.9 ms như tôi đoán. Sai số nằm ở chỗ tôi đếm nhầm số lớp. Bài học: đừng tin con
số mình tự nhẩm khi có sẵn lệnh để đo."""


def boc_van_xuoi(text):
    """Giữ lại phần văn xuôi. Bỏ frontmatter, code block, bảng, code inline.

    Đây là bước bản tiếng Anh thiếu, và vì thiếu nên nó đếm `--vocab` của lệnh
    CLI như một em dash trong câu văn.
    """
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)   # frontmatter
    text = re.sub(r"```.*?```", "", text, flags=re.S)            # code block
    text = re.sub(r"`[^`\n]+`", "", text)                        # code inline
    text = "\n".join(l for l in text.split("\n")
                     if not l.lstrip().startswith("|"))          # bảng
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.M)        # dấu # của tiêu đề
    return text


def _dem_cum(text_low, cum_list, dau_cau_only=False):
    """dau_cau_only: chỉ tính khi cụm MỞ ĐẦU một câu.

    Cần thiết cho nhóm "giả bộ thẳng thắn". "Nói thẳng là X" ở đầu câu là tic
    của người viết; còn "Comment trong code nói thẳng: ..." chỉ là mô tả một
    comment, hoàn toàn hợp lệ. Không phân biệt thì công cụ báo nhầm, và một
    công cụ hay báo nhầm thì không ai còn tin nữa.
    """
    hits = []
    for c in cum_list:
        if dau_cau_only:
            n = len(re.findall(r"(?:^|[.!?:;\n]\s*|^\s*[-*]\s*)" + re.escape(c),
                               text_low, flags=re.M))
        else:
            n = text_low.count(c)
        if n:
            hits.append((c, n))
    return sorted(hits, key=lambda x: -x[1])


def cham_tu_vung(t, low, cum_list, max_diem, moc, dau_cau_only=False):
    """moc = số lần / 1000 từ để mất hết điểm."""
    hits = _dem_cum(low, cum_list, dau_cau_only)
    tong = sum(n for _, n in hits)
    so_tu = max(1, len(re.findall(r"\S+", t)))
    mat_do = tong / (so_tu / 1000)
    diem = max(0, round(max_diem * (1 - min(1.0, mat_do / moc))))
    return {"diem": diem, "max": max_diem, "so_lan": tong,
            "moi_1000_tu": round(mat_do, 2), "cum": hits[:6]}


def cham_nhip_cau(t):
    cau = [c.strip() for c in re.split(r"[.!?…]+", t) if len(c.strip()) > 1]
    if len(cau) < 5:
        return {"diem": MAX_DIEM["nhip_cau"], "max": MAX_DIEM["nhip_cau"],
                "do_lech": 0, "ghi_chu": "quá ít câu để đo"}
    dai = [len(re.findall(r"\S+", c)) for c in cau]
    lech = statistics.pstdev(dai)
    # câu dài đều tăm tắp là dấu hiệu máy; người viết dài ngắn xen kẽ
    diem = MAX_DIEM["nhip_cau"] if lech >= 8 else round(MAX_DIEM["nhip_cau"] * lech / 8)
    return {"diem": diem, "max": MAX_DIEM["nhip_cau"], "do_lech": round(lech, 1),
            "dai_trung_binh": round(statistics.mean(dai), 1)}


def cham_in_dam(t):
    doan = [p for p in t.split("\n\n") if p.strip()]
    xau = [p for p in doan if len(re.findall(r"\*\*[^*]+\*\*", p)) >= 3]
    ty_le = len(xau) / max(1, len(doan))
    diem = max(0, round(MAX_DIEM["in_dam"] * (1 - min(1.0, ty_le / 0.25))))
    return {"diem": diem, "max": MAX_DIEM["in_dam"], "doan_qua_dam": len(xau),
            "tong_doan": len(doan)}


def cham_gach_ngang(t):
    n = t.count("—") + t.count("–")
    so_tu = max(1, len(re.findall(r"\S+", t)))
    mat_do = n / (so_tu / 1000)
    diem = max(0, round(MAX_DIEM["gach_ngang"] * (1 - min(1.0, mat_do / 5))))
    return {"diem": diem, "max": MAX_DIEM["gach_ngang"], "so_dau": n,
            "moi_1000_tu": round(mat_do, 2)}


def cham(text):
    t = boc_van_xuoi(text)
    low = t.lower()
    s = {
        "sao_rong":   cham_tu_vung(t, low, SAO_RONG,   MAX_DIEM["sao_rong"],   6),
        "rao_don":    cham_tu_vung(t, low, RAO_DON,    MAX_DIEM["rao_don"],    4),
        "ket_sao":    cham_tu_vung(t, low, KET_SAO,    MAX_DIEM["ket_sao"],    2),
        "gia_thang":  cham_tu_vung(t, low, GIA_THANG,  MAX_DIEM["gia_thang"],  1.5,
                                  dau_cau_only=True),
        "chuyen_doan": cham_tu_vung(t, low, CHUYEN_DOAN, MAX_DIEM["chuyen_doan"], 20),
        "nhip_cau":   cham_nhip_cau(t),
        "in_dam":     cham_in_dam(t),
        "gach_ngang": cham_gach_ngang(t),
    }
    tong = sum(v["diem"] for v in s.values())
    return {"diem": tong, "muc": s,
            "so_tu_van_xuoi": len(re.findall(r"\S+", t))}


TEN = {"sao_rong": "Từ sáo rỗng", "rao_don": "Rào đón", "ket_sao": "Kết bài sáo", "gia_thang": "Giả bộ thẳng thắn",
       "chuyen_doan": "Chuyển đoạn máy móc", "nhip_cau": "Nhịp câu dài ngắn",
       "in_dam": "In đậm quá tay", "gach_ngang": "Gạch ngang"}


def in_bao_cao(kq, nguon):
    d = kq["diem"]
    verdict = ("giọng người ✅" if d >= 85 else
               "cần sửa vài chỗ 🟡" if d >= 70 else
               "đọc như máy viết 🔴")
    print(f"\n  NGUỒN : {nguon}")
    print(f"  ĐIỂM  : {d}/100   ({kq['so_tu_van_xuoi']} từ văn xuôi)")
    print(f"  KẾT   : {verdict}\n")
    print("  ── Chi tiết ────────────────────────────────")
    for k, v in kq["muc"].items():
        bar = "█" * round(v["diem"] / v["max"] * 10)
        print(f"  {TEN[k]:<22} {v['diem']:>2}/{v['max']:<2} {bar}")
    print("\n  ── Cụm bắt được ───────────────────────────")
    co = False
    for k in ("sao_rong", "rao_don", "ket_sao", "gia_thang", "chuyen_doan"):
        for cum, n in kq["muc"][k].get("cum", []):
            print(f'  {n:>2}x  "{cum}"   [{TEN[k]}]')
            co = True
    if not co:
        print("  không có cụm nào trong danh sách")
    if kq["muc"]["in_dam"]["doan_qua_dam"]:
        print(f"  {kq['muc']['in_dam']['doan_qua_dam']} đoạn có >=3 cụm in đậm")
    if kq["muc"]["gach_ngang"]["so_dau"]:
        print(f"  {kq['muc']['gach_ngang']['so_dau']} dấu gạch ngang dài")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Chấm điểm giọng người cho văn bản kỹ thuật tiếng Việt.")
    ap.add_argument("file", nargs="?", help="đường dẫn file .md")
    ap.add_argument("--json", action="store_true", help="in thêm JSON")
    ap.add_argument("--sample", action="store_true",
                    help="chạy trên hai mẫu đối chứng (máy viết vs người viết)")
    ap.add_argument("--min", type=int, default=None,
                    help="dưới ngưỡng này thì thoát mã 1, để chặn trong CI")
    a = ap.parse_args()

    if a.sample or not a.file:
        for ten, mau in [("MẪU MÁY VIẾT", MAU_AI), ("MẪU NGƯỜI VIẾT", MAU_NGUOI)]:
            in_bao_cao(cham(mau), ten)
        return 0

    kq = cham(open(a.file, encoding="utf-8").read())
    in_bao_cao(kq, a.file)
    if a.json:
        print(json.dumps(kq, ensure_ascii=False, indent=2))
    if a.min is not None and kq["diem"] < a.min:
        print(f"  ✗ {kq['diem']} < ngưỡng {a.min}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
