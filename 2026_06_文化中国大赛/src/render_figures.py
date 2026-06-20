#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_figures.py · 把 output/ 里的数字人文数据渲染成 PNG 配图(供 PPTX 插图)

读取:
  output/timeline.json   → fig_timeline.png   (版本流变时间轴)
  output/diffusion.json  → fig_diffusion.png  (四层传播下沉金字塔)
  output/wordcloud.json  → fig_wordcloud.png   (教化主题词云)
  output/network.json    → fig_network.png     (互文知识网络, networkx 布局)
另外自制:
  fig_cover.png    (封面水墨底纹)
  fig_ending.png   (结尾页底纹)

全部用 matplotlib(+networkx),CPU 离线即可。中文用系统 CJK 字体(Hiragino/Songti),
不依赖 torch/cv2。`python render_figures.py` 自测:渲染全部 PNG 并校验非空。

古风配色: 棕 #5a2d0c / 赭 #8b4513 / 米 #f7f3eb / 驼 #c19a6b
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
import matplotlib.patheffects as pe

# ---- 路径 ----
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUTDIR = os.path.join(PROJ, "output")

# ---- 中文字体(按优先级回退) ----
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",   # 宋体, 古风
    "/System/Library/Fonts/Hiragino Sans GB.ttc",      # 黑体
    "/System/Library/Fonts/STHeiti Medium.ttc",
]
_SERIF_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def _pick(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None


_FONT_PATH = _pick(_FONT_CANDIDATES)
_SERIF_PATH = _pick(_SERIF_CANDIDATES)
if _FONT_PATH is None:
    raise RuntimeError("未找到可用中文字体, 无法渲染中文 PNG")


def fp(size, serif=False, bold=False):
    path = _SERIF_PATH if (serif and _SERIF_PATH) else _FONT_PATH
    weight = "bold" if bold else "normal"
    return FontProperties(fname=path, size=size, weight=weight)


# ---- 配色 ----
C_DARK = "#5a2d0c"
C_BROWN = "#8b4513"
C_CREAM = "#f7f3eb"
C_CAMEL = "#c19a6b"
C_PAPER = "#fbf7ee"
C_INK = "#3a2410"


def _load(name):
    with open(os.path.join(OUTDIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ============================================================
# 1. 版本流变时间轴
# ============================================================
def render_timeline(name="fig_timeline.png"):
    data = _load("timeline.json")
    n = len(data)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(C_PAPER)
    ax.set_facecolor(C_PAPER)

    xs = list(range(n))
    y_axis = 0.0
    # 主时间轴线
    ax.plot([-0.5, n - 0.5], [y_axis, y_axis], color=C_CAMEL, lw=4, zorder=1)

    for i, item in enumerate(data):
        up = (i % 2 == 0)
        y_node = y_axis
        s = 1.0 if up else -1.0
        # 节点圆
        ax.scatter([i], [y_node], s=260, color=C_BROWN, edgecolors=C_DARK,
                   linewidths=2, zorder=3)
        ax.scatter([i], [y_node], s=70, color=C_PAPER, zorder=4)
        # 连线(到色块底部)
        ax.plot([i, i], [y_node, s * 0.62], color=C_CAMEL, lw=1.6, ls="--", zorder=2)
        # 三段竖排, 互不重叠: 形态色块(近轴) → 朝代 → 年代(最外)
        y_form = s * 0.95
        y_era = s * 1.55
        y_year = s * 1.92
        ax.text(i, y_form, item["form"], fontproperties=fp(10, bold=True),
                color="#fff", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=C_CAMEL, ec=C_BROWN, lw=1))
        ax.text(i, y_era, item["era"], fontproperties=fp(13, bold=True),
                color=C_DARK, ha="center", va="center")
        ax.text(i, y_year, item["year"], fontproperties=fp(9),
                color="#7a6a55", ha="center", va="center")

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_title("《颜氏家训》版本流变时间轴 · 抄本 → 刻本 → 坊刻 → 排印 → 数字化",
                 fontproperties=fp(15, bold=True), color=C_DARK, pad=14)
    return _save(fig, name)


# ============================================================
# 2. 传播下沉金字塔(倒金字塔: 上窄=精英, 下宽=大众)
# ============================================================
def render_diffusion(name="fig_diffusion.png"):
    data = _load("diffusion.json")
    n = len(data)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.patch.set_facecolor(C_PAPER)
    ax.set_facecolor(C_PAPER)

    shades = ["#5a2d0c", "#8b4513", "#a9692f", "#c19a6b"]
    top_w = 0.30      # 顶层(士族)窄
    bot_w = 0.96      # 底层(民间)宽
    layer_h = 1.0
    total_h = n * layer_h

    for i, item in enumerate(data):
        # 从上到下逐层变宽
        frac_top = i / n
        frac_bot = (i + 1) / n
        w_top = top_w + (bot_w - top_w) * frac_top
        w_bot = top_w + (bot_w - top_w) * frac_bot
        y_hi = total_h - i * layer_h
        y_lo = total_h - (i + 1) * layer_h
        cx = 0.5
        poly = Polygon(
            [(cx - w_top / 2, y_hi), (cx + w_top / 2, y_hi),
             (cx + w_bot / 2, y_lo), (cx - w_bot / 2, y_lo)],
            closed=True, fc=shades[i % len(shades)], ec=C_PAPER, lw=3, zorder=2)
        ax.add_patch(poly)
        ycen = (y_hi + y_lo) / 2
        ax.text(cx, ycen + 0.13, item["level"], fontproperties=fp(15, bold=True),
                color="#fff", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground=C_INK)])
        ax.text(cx, ycen - 0.20,
                f"{item['audience']} · {item['channel']}",
                fontproperties=fp(9.5), color="#fdf6ea", ha="center", va="center")
        # 右侧示例注释
        ax.annotate(item["example"], xy=(cx + w_bot / 2, ycen),
                    xytext=(1.18, ycen), fontproperties=fp(9), color=C_DARK,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_CAMEL, lw=1))

    # 大传统→小传统 箭头
    ax.annotate("", xy=(0.04, 0.2), xytext=(0.04, total_h - 0.2),
                arrowprops=dict(arrowstyle="-|>", color=C_BROWN, lw=3))
    ax.text(-0.02, total_h - 0.2, "大传统", fontproperties=fp(11, bold=True),
            color=C_DARK, ha="right", va="center", rotation=90)
    ax.text(-0.02, 0.2, "小传统", fontproperties=fp(11, bold=True),
            color=C_DARK, ha="right", va="center", rotation=90)

    ax.set_xlim(-0.18, 2.05)
    ax.set_ylim(-0.3, total_h + 0.9)
    ax.axis("off")
    ax.set_title("传播下沉路径 · 从士族家学走向民间日用",
                 fontproperties=fp(15, bold=True), color=C_DARK, pad=12)
    return _save(fig, name)


# 词云铺满率统计(render_wordcloud 写, _selftest 读取并断言)
LAST_WORDCLOUD_FILL = {"drawn": 0, "total": 0}


# ============================================================
# 3. 教化主题词云(权重→字号, 无需 wordcloud 库)
#    采用矩形 bbox 螺旋放置 + 真实碰撞检测(基于估算文本框), 保证不重叠且铺满。
# ============================================================
def render_wordcloud(name="fig_wordcloud.png", topk=30):
    import math
    import random
    data = _load("wordcloud.json")[:topk]
    weights = [d["weight"] for d in data]
    wmin, wmax = (min(weights), max(weights)) if weights else (0.0, 1.0)

    FIG_W, FIG_H = 11.0, 6.2
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(C_PAPER)
    ax.set_facecolor(C_PAPER)

    palette = ["#5a2d0c", "#8b4513", "#a9692f", "#9c4a1a", "#7a3b12", "#b5763c"]
    rng = random.Random(42)

    # 用 axes 坐标 [0,1]^2。把字号(pt)换算成 axes 高度比例:
    #   1 pt = 1/72 inch; axes 高 = FIG_H inch(占满 fig)。
    placed = []  # (x, y, half_w, half_h) in axes coords

    def overlaps(x, y, hw, hh):
        pad = 0.004
        for (px, py, phw, phh) in placed:
            if abs(x - px) < (hw + phw + pad) and abs(y - py) < (hh + phh + pad):
                return True
        return False

    drawn = 0
    for idx, d in enumerate(data):
        norm = (d["weight"] - wmin) / (wmax - wmin + 1e-9)
        size = 13 + 33 * norm          # pt
        txt = d["text"]
        char_h = size / 72.0 / FIG_H    # 单字高(axes 比例)
        # 单字近似正方, 多字宽度按字数
        hh = 0.62 * char_h
        hw = 0.62 * char_h * len(txt)
        placed_ok = False
        for step in range(900):
            # 阿基米德螺旋, x 方向略拉伸适配宽画布
            t = step * 0.30
            r = 0.0042 * step
            jx = (rng.random() - 0.5) * 0.02
            jy = (rng.random() - 0.5) * 0.02
            x = 0.5 + r * 1.55 * math.cos(t) + jx
            y = 0.5 + r * math.sin(t) + jy
            if (hw < x < 1 - hw) and (hh < y < 1 - hh) and not overlaps(x, y, hw, hh):
                placed.append((x, y, hw, hh))
                ax.text(x, y, txt,
                        fontproperties=fp(size, serif=True, bold=(norm > 0.55)),
                        color=palette[idx % len(palette)], ha="center", va="center")
                drawn += 1
                placed_ok = True
                break

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("教化主题词云 · TF-IDF 字-bigram 提取",
                 fontproperties=fp(15, bold=True), color=C_DARK, pad=10)
    p = _save(fig, name)
    LAST_WORDCLOUD_FILL["drawn"] = drawn
    LAST_WORDCLOUD_FILL["total"] = len(data)
    print(f"     (词云放置 {drawn}/{len(data)} 词)")
    return p


# ============================================================
# 4. 互文知识网络(networkx spring 布局)
# ============================================================
def render_network(name="fig_network.png"):
    import networkx as nx
    net = _load("network.json")
    G = nx.Graph()
    for nd in net["nodes"]:
        G.add_node(nd["id"], group=nd["group"], era=nd["era"])
    for ed in net["edges"]:
        G.add_edge(ed["source"], ed["target"],
                   relation=ed["relation"], strength=ed["strength"])

    fig, ax = plt.subplots(figsize=(10, 7.2))
    fig.patch.set_facecolor(C_PAPER)
    ax.set_facecolor(C_PAPER)

    pos = nx.spring_layout(G, seed=7, k=1.4, iterations=200)

    group_color = {"核心": C_BROWN, "家训": "#a9692f", "蒙书": C_CAMEL}
    # 边(粗细=strength)
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lw = 1.0 + 4.5 * d["strength"]
        ax.plot([x0, x1], [y0, y1], color="#cbb793", lw=lw, zorder=1, alpha=0.85)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, d["relation"], fontproperties=fp(8), color="#7a6a55",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="#fbf7ee", ec="none", alpha=0.9),
                zorder=2)
    # 节点
    for nd in G.nodes(data=True):
        name_id, attr = nd
        x, y = pos[name_id]
        is_core = attr["group"] == "核心"
        r = 0.16 if is_core else 0.12
        col = group_color.get(attr["group"], C_CAMEL)
        circ = Circle((x, y), r, fc=col, ec=C_DARK, lw=2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y + 0.02, name_id,
                fontproperties=fp(12 if is_core else 10, bold=is_core),
                color="#fff", ha="center", va="center", zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground=C_INK)])
        ax.text(x, y - 0.05, attr["era"], fontproperties=fp(7),
                color="#fdf6ea", ha="center", va="center", zorder=4)

    # 图例(空图时 pos 为空, 用默认坐标避免 max()/min() 空序列崩溃)
    ys = [p[1] for p in pos.values()]
    xs = [p[0] for p in pos.values()]
    leg_y = (max(ys) if ys else 0.0) + 0.30
    leg_x0 = min(xs) if xs else 0.0
    for i, (grp, col) in enumerate(group_color.items()):
        lx = leg_x0 + i * 0.42
        ax.add_patch(Circle((lx, leg_y), 0.045, fc=col, ec=C_DARK, lw=1))
        ax.text(lx + 0.08, leg_y, grp, fontproperties=fp(9), color=C_DARK,
                ha="left", va="center")

    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    ax.set_title("互文知识网络 · 一条家训脉络的传承",
                 fontproperties=fp(15, bold=True), color=C_DARK, pad=10)
    return _save(fig, name)


# ============================================================
# 5/6. 封面 & 结尾底纹(纯装饰)
# ============================================================
def _ink_background(ax):
    """画几道淡墨竹/印章感的装饰。"""
    rng = __import__("random").Random(9)
    # 米色底
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                fc=C_CREAM, ec="none", zorder=0))
    # 右下角朱印
    ax.add_patch(FancyBboxPatch((0.80, 0.06), 0.13, 0.13,
                                boxstyle="round,pad=0.005",
                                fc="#9c2a1a", ec="#7a1d10", lw=2, zorder=2, alpha=0.92))
    ax.text(0.865, 0.125, "典\n籍", fontproperties=fp(13, serif=True, bold=True),
            color="#f7e9d8", ha="center", va="center", zorder=3, linespacing=0.95)
    # 几条淡墨横纹
    for i in range(6):
        y = 0.12 + i * 0.14
        ax.plot([0.05, 0.55 + rng.random() * 0.2], [y, y],
                color=C_CAMEL, lw=1.0, alpha=0.18, zorder=1)


def render_cover(name="fig_cover.png"):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(C_CREAM)
    ax.set_facecolor(C_CREAM)
    _ink_background(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return _save(fig, name)


def render_ending(name="fig_ending.png"):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(C_DARK)
    ax.set_facecolor(C_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # 居中朱印 + 淡纹
    for i in range(8):
        x = 0.1 + i * 0.1
        ax.plot([x, x], [0.1, 0.9], color=C_CAMEL, lw=1.0, alpha=0.10)
    ax.axis("off")
    return _save(fig, name)


FIGURES = [
    ("fig_timeline.png", render_timeline),
    ("fig_diffusion.png", render_diffusion),
    ("fig_wordcloud.png", render_wordcloud),
    ("fig_network.png", render_network),
    ("fig_cover.png", render_cover),
    ("fig_ending.png", render_ending),
]


def render_all():
    print("=== 渲染 PNG 配图 ===")
    paths = []
    for name, func in FIGURES:
        p = func(name)
        print(f"  [OK] {p}")
        paths.append(p)
    print("=== 全部 PNG 渲染完毕 ===\n")
    return paths


def _selftest():
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    check(_FONT_PATH is not None, f"找到中文字体: {_FONT_PATH}")
    paths = render_all()
    for name, _ in FIGURES:
        p = os.path.join(OUTDIR, name)
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        check(exists and size > 2000, f"{name} 生成且非空 ({size} bytes)")

    # 词云铺满率: 螺旋碰撞放置不应大量丢词(否则字号/螺旋参数回归).
    # 默认语料应近乎全放; 设 ≥90% 阈值, 既能抓真实回归又不脆.
    drawn = LAST_WORDCLOUD_FILL["drawn"]
    total = LAST_WORDCLOUD_FILL["total"]
    fill = drawn / total if total else 0.0
    check(total > 0 and fill >= 0.90,
          f"词云铺满率 {drawn}/{total} = {fill:.0%} (阈值≥90%)")

    print("\n" + ("✅ render_figures 自测通过" if ok else "❌ render_figures 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
