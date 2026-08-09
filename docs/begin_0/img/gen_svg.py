"""Sinh hình minh hoạ hình học cho chương 1.

Mọi toạ độ được TÍNH từ công thức, không đặt tay, nên hình luôn khớp với phép toán
mô tả trong bài. Đổi a, b ở dưới là hình tự vẽ lại đúng.
"""
import math, os, sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- khung vẽ
CSS = """
  :root{--bg:#ffffff;--grid:#e6eaef;--axis:#9aa5b1;--ink:#1f2937;--muted:#6b7280;
        --a:#0366d6;--b:#57606a;--par:#1a7f37;--perp:#cf222e;--fill:#0366d612}
  @media (prefers-color-scheme: dark){
    :root{--bg:#0d1117;--grid:#1c2330;--axis:#54606f;--ink:#e6edf3;--muted:#9198a1;
          --a:#58a6ff;--b:#8b949e;--par:#3fb950;--perp:#ff7b72;--fill:#58a6ff1a}
  }
  .bg{fill:var(--bg)} .grid{stroke:var(--grid);stroke-width:1}
  .axis{stroke:var(--axis);stroke-width:1.5}
  .lbl{font:500 15px -apple-system,'Helvetica Neue',Arial,sans-serif;fill:var(--ink)}
  .sm{font:400 13px -apple-system,'Helvetica Neue',Arial,sans-serif;fill:var(--muted)}
  .tick{font:400 11px -apple-system,'Helvetica Neue',Arial,sans-serif;fill:var(--muted)}
  .va{stroke:var(--a);stroke-width:2.6;fill:none}
  .vb{stroke:var(--b);stroke-width:2.6;fill:none}
  .vpar{stroke:var(--par);stroke-width:3;fill:none}
  .vperp{stroke:var(--perp);stroke-width:2.6;fill:none}
  .dash{stroke:var(--perp);stroke-width:1.4;stroke-dasharray:5 4;fill:none;opacity:.75}
  .ray{stroke:var(--b);stroke-width:1.2;stroke-dasharray:4 4;opacity:.55;fill:none}
  .ta{fill:var(--a)} .tb{fill:var(--b)} .tpar{fill:var(--par)} .tperp{fill:var(--perp)}
"""

def head(w, h, title):
    m = "".join(
        f'<marker id="ar{n}" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="7.5" '
        f'markerHeight="6" orient="auto"><path d="M0 0 L10 4 L0 8 z" fill="var(--{v})"/></marker>'
        for n, v in [("a", "a"), ("b", "b"), ("p", "par"), ("q", "perp")])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{title}"><title>{title}</title>'
            f'<style>{CSS}</style><defs>{m}</defs><rect class="bg" width="{w}" height="{h}"/>')

class Frame:
    """Hệ toạ độ toán -> pixel, trục y lật lại cho đúng chiều."""
    def __init__(self, ox, oy, s):
        self.ox, self.oy, self.s = ox, oy, s
    def __call__(self, p):
        return (self.ox + p[0] * self.s, self.oy - p[1] * self.s)

def grid(F, xs, ys, w, h):
    o = []
    for i in range(xs[0], xs[1] + 1):
        x = F((i, 0))[0]
        o.append(f'<line class="grid" x1="{x}" y1="8" x2="{x}" y2="{h-8}"/>')
    for j in range(ys[0], ys[1] + 1):
        y = F((0, j))[1]
        o.append(f'<line class="grid" x1="8" y1="{y}" x2="{w-8}" y2="{y}"/>')
    ax0, ay0 = F((0, 0))
    o.append(f'<line class="axis" x1="8" y1="{ay0}" x2="{w-8}" y2="{ay0}"/>')
    o.append(f'<line class="axis" x1="{ax0}" y1="8" x2="{ax0}" y2="{h-8}"/>')
    return "".join(o)

def vec(F, p, cls, mk, frm=(0, 0)):
    x1, y1 = F(frm); x2, y2 = F(p)
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#ar{mk})"/>'

def txt(x, y, s, cls="lbl", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'

def arc(F, u, v, r_units, cls="sm"):
    """Cung tròn bán kính r (đơn vị toán) giữa hai hướng u, v tại gốc."""
    au, av = math.atan2(u[1], u[0]), math.atan2(v[1], v[0])
    p1 = F((r_units * math.cos(au), r_units * math.sin(au)))
    p2 = F((r_units * math.cos(av), r_units * math.sin(av)))
    r = r_units * F.s
    sweep = 1 if (au - av) % (2 * math.pi) < math.pi else 0
    return (f'<path d="M{p1[0]:.1f} {p1[1]:.1f} A{r:.1f} {r:.1f} 0 0 {sweep} '
            f'{p2[0]:.1f} {p2[1]:.1f}" fill="none" stroke="var(--muted)" stroke-width="1.4"/>')

def right_angle(F, corner, d1, d2, size=0.28):
    """Ô vuông đánh dấu góc 90 độ tại `corner`, hai cạnh theo d1 và d2 (đã chuẩn hoá)."""
    n1 = [c / math.hypot(*d1) for c in d1]
    n2 = [c / math.hypot(*d2) for c in d2]
    p = [corner,
         (corner[0] + n1[0] * size, corner[1] + n1[1] * size),
         (corner[0] + (n1[0] + n2[0]) * size, corner[1] + (n1[1] + n2[1]) * size),
         (corner[0] + n2[0] * size, corner[1] + n2[1] * size)]
    pts = " ".join(f"{F(q)[0]:.1f},{F(q)[1]:.1f}" for q in p)
    return f'<polygon points="{pts}" fill="none" stroke="var(--muted)" stroke-width="1.4"/>'


# =================================================================== 1. PROJECTION
def projection():
    a, b = (1, 3), (4, 2)
    k = (a[0]*b[0] + a[1]*b[1]) / (b[0]**2 + b[1]**2)
    par = (k*b[0], k*b[1])
    perp = (a[0]-par[0], a[1]-par[1])
    assert abs(perp[0]*b[0] + perp[1]*b[1]) < 1e-12, "a_perp phải vuông góc với b"

    W, H, s = 620, 400, 74
    F = Frame(72, H - 58, s)
    o = [head(W, H, "Chiếu vector a lên hướng b"), grid(F, (0, 5), (0, 4), W, H)]

    # tia kéo dài của b, để thấy a_par nằm trên đường đó
    far = F((b[0]*1.28, b[1]*1.28))
    o.append(f'<line class="ray" x1="{F((0,0))[0]}" y1="{F((0,0))[1]}" x2="{far[0]}" y2="{far[1]}"/>')
    # tam giác nền
    tri = " ".join(f"{F(q)[0]:.1f},{F(q)[1]:.1f}" for q in [(0, 0), par, a])
    o.append(f'<polygon points="{tri}" fill="var(--fill)" stroke="none"/>')

    o.append(vec(F, b, "vb", "b"))
    o.append(vec(F, a, "va", "a"))
    o.append(vec(F, par, "vpar", "p"))
    o.append(vec(F, a, "vperp", "q", frm=par))          # a_perp vẽ từ ngọn a_par tới ngọn a
    o.append(right_angle(F, par, b, perp))
    o.append(arc(F, a, b, 0.62))

    for p, c in [(a, "ta"), (b, "tb"), (par, "tpar")]:
        x, y = F(p)
        o.append(f'<circle cx="{x}" cy="{y}" r="3.4" class="{c}"/>')

    xa, ya = F(a);   o.append(txt(xa - 8, ya - 12, "a = (1, 3)", "lbl", "end"))
    xb, yb = F(b);   o.append(txt(xb + 10, yb + 6, "b = (4, 2)", "lbl"))
    xp, yp = F(par); o.append(txt(xp + 8, yp + 24, "a∥ = (2, 1)", "lbl"))
    mx, my = F(((a[0]+par[0])/2, (a[1]+par[1])/2))
    o.append(txt(mx + 14, my - 2, "a⊥ = (-1, 2)", "lbl"))
    ax, ay = F((0.86*math.cos(math.radians(63.4)), 0.86*math.sin(math.radians(63.4))))
    o.append(txt(ax + 4, ay + 4, "θ = 45°", "sm"))

    o.append(txt(F((0,0))[0] - 10, F((0,0))[1] + 18, "O", "sm", "end"))
    for i in range(1, 6):
        x, y = F((i, 0)); o.append(txt(x, y + 16, str(i), "tick", "middle"))
    for j in range(1, 5):
        x, y = F((0, j)); o.append(txt(x - 8, y + 4, str(j), "tick", "end"))

    o.append(txt(W - 14, 26, "a∥ = (a·b / b·b) · b = ½ · (4, 2)", "sm", "end"))
    o.append(txt(W - 14, 46, "a⊥ = a − a∥,  a⊥ · b = 0", "sm", "end"))
    o.append("</svg>")
    return "".join(o)


# =================================================================== 2. DOT & ANGLE
def dot_angle():
    a, b = (1, 3), (4, 2)
    W, H, s = 620, 380, 70
    F = Frame(72, H - 56, s)
    o = [head(W, H, "Dot product và góc giữa hai vector"), grid(F, (0, 5), (0, 4), W, H)]
    o.append(vec(F, b, "vb", "b"))
    o.append(vec(F, a, "va", "a"))
    o.append(arc(F, a, b, 0.62))
    o.append(arc(F, a, b, 0.70))
    xa, ya = F(a); o.append(txt(xa - 8, ya - 12, "a = (1, 3)", "lbl", "end"))
    xb, yb = F(b); o.append(txt(xb + 10, yb + 6, "b = (4, 2)", "lbl"))
    ax, ay = F((0.95*math.cos(math.radians(63.4)), 0.95*math.sin(math.radians(63.4))))
    o.append(txt(ax + 6, ay + 4, "θ = 45°", "sm"))
    o.append(txt(F((0,0))[0] - 10, F((0,0))[1] + 18, "O", "sm", "end"))
    for i in range(1, 6):
        x, y = F((i, 0)); o.append(txt(x, y + 16, str(i), "tick", "middle"))
    for j in range(1, 5):
        x, y = F((0, j)); o.append(txt(x - 8, y + 4, str(j), "tick", "end"))
    o.append(txt(W - 14, 26, "a · b = Σ aᵢbᵢ = 1·4 + 3·2 = 10", "sm", "end"))
    o.append(txt(W - 14, 46, "‖a‖ · ‖b‖ · cos θ = √10 · √20 · cos 45° = 10", "sm", "end"))
    o.append(txt(W - 14, 66, "hai vế bằng nhau, đó là cả ý nghĩa của dot product", "sm", "end"))
    o.append("</svg>")
    return "".join(o)


# =================================================================== 3. NORMALIZE
def normalize():
    """a và b cùng hướng, khác độ lớn. Sau normalize cả hai về đúng một điểm.

    Tỉ lệ thật (1000 lần) không vẽ được trên giấy, nên hình dùng hệ số nhỏ hơn và
    nói rõ điều đó thay vì dán nhãn 1000 lên một mũi tên dài gấp đôi.
    """
    u = (math.sqrt(0.5), math.sqrt(0.5))       # vector đơn vị, hướng 45 độ
    a = (1.55*u[0], 1.55*u[1])
    b = (3.05*u[0], 3.05*u[1])

    W, H, s_ = 620, 372, 64
    F = Frame(126, 292, s_)
    o = [head(W, H, "Normalize: bỏ độ lớn, giữ hướng"), grid(F, (0, 4), (0, 3), W, H)]

    cx, cy = F((0, 0))
    o.append(f'<circle cx="{cx}" cy="{cy}" r="{s_}" fill="none" stroke="var(--axis)" '
             f'stroke-width="1.3" stroke-dasharray="5 4"/>')
    far = F((3.5*u[0], 3.5*u[1]))
    o.append(f'<line class="ray" x1="{cx}" y1="{cy}" x2="{far[0]}" y2="{far[1]}"/>')

    o.append(vec(F, b, "vb", "b"))
    o.append(vec(F, a, "va", "a"))
    o.append(vec(F, u, "vpar", "p"))
    for p_, c in [(a, "ta"), (b, "tb"), (u, "tpar")]:
        x, y = F(p_); o.append(f'<circle cx="{x}" cy="{y}" r="3.6" class="{c}"/>')

    xb, yb = F(b); o.append(txt(xb + 12, yb + 4, "b = k · a   (k > 1)", "lbl"))
    xa, ya = F(a); o.append(txt(xa + 12, ya + 16, "a", "lbl"))
    xu, yu = F(u); o.append(txt(xu + 14, yu + 20, "a/‖a‖ = b/‖b‖", "lbl"))
    o.append(txt(cx + s_ + 6, cy - 6, "‖v‖ = 1", "sm"))
    o.append(txt(cx - 10, cy + 18, "O", "sm", "end"))

    o.append(txt(W - 14, H - 54, "a và b cùng hướng, chỉ khác độ lớn", "sm", "end"))
    o.append(txt(W - 14, H - 34, "chia cho ‖v‖ thì cả hai về ĐÚNG một điểm", "sm", "end"))
    o.append(txt(W - 14, H - 14, "trên đường tròn đơn vị: chỉ còn lại hướng", "sm", "end"))
    o.append("</svg>")
    return "".join(o)


for name, fn in [("vector-projection", projection),
                 ("vector-dot-angle", dot_angle),
                 ("vector-normalize", normalize)]:
    p = os.path.join(OUT, name + ".svg")
    open(p, "w", encoding="utf-8").write(fn())
    print("viết", p, os.path.getsize(p), "byte")
