# -*- coding: utf-8 -*-
"""
报告图表生成器 · 低空智能感知城市违建巡查系统
为《项目报告书》生成 4 张可插入 Word 的高清 PNG:
  1. fig_architecture.png  端-边-云系统总体架构框图
  2. fig_workflow.png       四阶段实施流程图
  3. fig_business.png       商业模式 + 成本收益对比图
  4. fig_crossmatrix.png    6 学科交叉融合矩阵图

纯 matplotlib(已装), 无需 GPU / 联网。中文用 macOS 自带 CJK 字体。
运行: python3 gen_figures.py            生成全部图到 ../output/figures/
      python3 gen_figures.py --selftest 仅自测(生成后校验文件非空、尺寸正确)
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")  # 无界面后端, 适合离线批量出图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm
import numpy as np

# ---------------------------------------------------------------------------
# 中文字体: 在 macOS 自带字体里按优先级挑一个可用的, 找不到则警告但不崩
# ---------------------------------------------------------------------------
_CJK_CANDIDATES = ["Arial Unicode MS", "Songti SC", "STHeiti",
                   "Heiti TC", "Hiragino Sans GB", "STFangsong"]


def setup_cjk_font():
    avail = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for c in _CJK_CANDIDATES:
        if c in avail:
            chosen = c
            break
    if chosen is None:
        print("[WARN] 未找到 CJK 字体, 中文可能显示为方块", file=sys.stderr)
        chosen = "DejaVu Sans"
    matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"
    return chosen


# 统一配色 (端-边-云三层 + 强调色)
COL_CLOUD = "#2E5AAC"   # 云端 深蓝
COL_EDGE = "#3F8E5B"    # 边缘 绿
COL_DEVICE = "#C8741A"  # 端 橙
COL_ACCENT = "#B23A48"  # 强调 红
COL_GRID = "#888888"
COL_BOX_BG = "#F4F6FA"
DPI = 200


def _box(ax, x, y, w, h, text, facecolor, edgecolor=None, fontsize=11,
         textcolor="white", bold=True, rounding=0.04):
    """画一个圆角填充框 + 居中文字, 坐标为 axes 数据坐标(0-100)。"""
    edgecolor = edgecolor or facecolor
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.2,rounding_size={rounding*100}",
                       linewidth=1.4, facecolor=facecolor, edgecolor=edgecolor,
                       mutation_aspect=1.0, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            fontweight="bold" if bold else "normal", zorder=3,
            wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=COL_GRID, label=None, lw=2.0,
           style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=18, linewidth=lw, color=color,
                        linestyle=ls, zorder=1)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 2, (y1 + y2) / 2, label, ha="left",
                va="center", fontsize=8.5, color=color, style="italic")


def _blank_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ===========================================================================
# 图 1: 端-边-云 系统总体架构框图
# ===========================================================================
def fig_architecture(out_path):
    fig, ax = _blank_ax((10, 7.6))
    ax.text(50, 97, "图1  端-边-云三层系统总体架构", ha="center", va="top",
            fontsize=15, fontweight="bold", color="#222222")

    # ---- 云端层 (顶部) ----
    ax.add_patch(Rectangle((4, 70), 92, 21, facecolor="#EAF0FB",
                           edgecolor=COL_CLOUD, lw=1.6, zorder=0))
    ax.text(7, 88.5, "云端管控平台", fontsize=12.5, fontweight="bold",
            color=COL_CLOUD, va="center")
    cloud_mods = ["任务调度", "数据管理", "数字孪生", "决策支持"]
    for i, m in enumerate(cloud_mods):
        _box(ax, 8 + i * 22, 72.5, 18, 11, m, COL_CLOUD, fontsize=11)

    # ---- 边缘层 (中部) ----
    ax.add_patch(Rectangle((4, 41), 92, 19, facecolor="#EAF6EE",
                           edgecolor=COL_EDGE, lw=1.6, zorder=0))
    ax.text(7, 57.5, "边缘计算节点", fontsize=12.5, fontweight="bold",
            color=COL_EDGE, va="center")
    edge_mods = ["航线规划", "实时检测", "轨迹跟踪"]
    for i, m in enumerate(edge_mods):
        _box(ax, 11 + i * 28, 43.5, 22, 10.5, m, COL_EDGE, fontsize=11)

    # ---- 端层 (底部) ----
    ax.add_patch(Rectangle((4, 9), 92, 22, facecolor="#FBF0E6",
                           edgecolor=COL_DEVICE, lw=1.6, zorder=0))
    ax.text(7, 28.5, "飞行器端", fontsize=12.5, fontweight="bold",
            color=COL_DEVICE, va="center")
    dev_mods = ["多光谱\n传感器", "高清可见\n光相机", "激光雷达\n(选配)",
                "机载计算\n单元"]
    for i, m in enumerate(dev_mods):
        _box(ax, 8 + i * 22, 11, 18, 12, m, COL_DEVICE, fontsize=10.5)

    # ---- 层间链路箭头 + 标注 ----
    _arrow(ax, 50, 70, 50, 60.2, color=COL_CLOUD, lw=2.6)
    ax.text(52, 65, "4G / 5G / 专网", fontsize=9.5, color=COL_CLOUD,
            va="center", fontweight="bold")
    _arrow(ax, 50, 41, 50, 31.2, color=COL_EDGE, lw=2.6)
    ax.text(52, 36, "遥控 / 图传", fontsize=9.5, color=COL_EDGE,
            va="center", fontweight="bold")

    # 上行回传箭头 (虚线)
    _arrow(ax, 12, 31.2, 12, 41, color=COL_DEVICE, lw=1.6, ls="--")
    _arrow(ax, 12, 60.2, 12, 70, color=COL_EDGE, lw=1.6, ls="--")
    ax.text(13.5, 35.5, "影像上行", fontsize=8, color=COL_DEVICE,
            rotation=90, va="center")
    ax.text(13.5, 65, "结果上行", fontsize=8, color=COL_EDGE,
            rotation=90, va="center")

    ax.text(50, 3, "数据闭环: 采集 → 边缘实时检测 → 云端复核与归档 → 执法对接",
            ha="center", va="center", fontsize=9.5, color="#555555",
            style="italic")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ===========================================================================
# 图 2: 四阶段实施流程图
# ===========================================================================
def fig_workflow(out_path):
    fig, ax = _blank_ax((11.5, 6.2))
    ax.text(50, 97, "图2  系统实施流程(四阶段)", ha="center", va="top",
            fontsize=15, fontweight="bold", color="#222222")

    stages = [
        ("阶段一\n任务规划",
         ["接收巡查任务", "空域申请", "航线自动规划", "风险预评估"], COL_CLOUD),
        ("阶段二\n现场执行",
         ["起飞前检查", "自主/半自主飞行", "实时数据采集", "边缘端智能检测"], COL_DEVICE),
        ("阶段三\n数据处理",
         ["影像回传", "云端变化检测", "违建图斑生成", "合规性校验"], COL_EDGE),
        ("阶段四\n成果输出",
         ["违建报告生成", "证据链归档", "执法系统对接", "复查任务下发"], COL_ACCENT),
    ]

    x0 = 3.0
    col_w = 22.0
    gap = 1.7
    for si, (title, steps, col) in enumerate(stages):
        cx = x0 + si * (col_w + gap)
        # 阶段标题框
        _box(ax, cx, 78, col_w, 12, title, col, fontsize=11.5)
        # 步骤小框 (竖排)
        for j, s in enumerate(steps):
            sy = 66 - j * 14
            _box(ax, cx + 1.5, sy, col_w - 3, 10.5, s, COL_BOX_BG,
                 edgecolor=col, textcolor="#222222", fontsize=9.5,
                 bold=False)
            if j < len(steps) - 1:
                _arrow(ax, cx + col_w / 2, sy, cx + col_w / 2, sy - 3.5,
                       color=col, lw=1.4)
        # 阶段间大箭头
        if si < len(stages) - 1:
            _arrow(ax, cx + col_w, 84, cx + col_w + gap, 84,
                   color="#444444", lw=2.6)

    ax.text(50, 4, "闭环管理: 复查任务下发后回流至阶段一, 形成发现-取证-处置-复查全链路",
            ha="center", va="center", fontsize=9.5, color="#555555",
            style="italic")
    # 回流虚线箭头(从最右回到最左)
    _arrow(ax, x0 + 3 * (col_w + gap) + col_w / 2, 10.5,
           x0 + col_w / 2, 10.5, color=COL_ACCENT, lw=1.6, ls="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ===========================================================================
# 图 3: 商业模式 + 成本收益对比图 (双子图)
# ===========================================================================
def fig_business(out_path):
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # ---- 左: 商业模式三层 (SaaS 订阅 + 按架次计费 + 定制) ----
    axl = fig.add_subplot(gs[0, 0])
    axl.set_xlim(0, 100)
    axl.set_ylim(0, 100)
    axl.axis("off")
    axl.text(50, 96, "商业模式: 政府购买SaaS + 按架次计费", ha="center",
             va="top", fontsize=12.5, fontweight="bold", color="#222222")

    tiers = [
        ("定制版", "特定区域/需求深度定制", COL_ACCENT, 70, 18),
        ("专业版", "按架次计费 · 含飞手+检测报告", COL_EDGE, 44, 26),
        ("基础版", "年度SaaS订阅 · 任务/航线/存储", COL_CLOUD, 14, 34),
    ]
    # 文字框靠左收窄, 右侧留出递进箭头通道, 避免与文字重叠
    for name, desc, col, y, h in tiers:
        _box(axl, 10, y, 64, h, "", col, fontsize=11)
        axl.text(42, y + h - 6, name, ha="center", va="center",
                 fontsize=12, color="white", fontweight="bold")
        axl.text(42, y + h / 2 - 3, desc, ha="center", va="center",
                 fontsize=9, color="white")
    axl.annotate("", xy=(82, 88), xytext=(82, 8),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#666666"))
    axl.text(90, 48, "增值\n递进", ha="center", va="center", fontsize=9.5,
             color="#666666", rotation=90)

    # ---- 右: 成本收益对比 (传统人工 vs 低空智能) ----
    axr = fig.add_subplot(gs[0, 1])
    metrics = ["单次成本\n指数", "覆盖效率\n(km²/单位)", "年运营成本\n(万元)"]
    # 用相对/代表值: 成本类越低越好, 效率越高越好
    trad = [100, 2, 350]      # 传统人工 (单次成本归一100, 效率2, 年成本均值350)
    drone = [32, 15, 75]      # 低空智能 (降低~68%, 效率~15, 年成本均值75)
    x = np.arange(len(metrics))
    bw = 0.36
    b1 = axr.bar(x - bw / 2, trad, bw, label="传统人工巡查",
                 color="#9AA7B8", edgecolor="#4A5568")
    b2 = axr.bar(x + bw / 2, drone, bw, label="低空智能巡查",
                 color=COL_CLOUD, edgecolor="#1B3A6B")
    axr.set_yscale("log")  # 量纲差异大, 用对数轴更可读
    axr.set_ylim(1, 700)   # 顶部留白, 防止最高柱(350)的数值标签被裁切
    axr.set_xticks(x)
    axr.set_xticklabels(metrics, fontsize=10)
    axr.set_ylabel("相对量值(对数轴)", fontsize=10)
    axr.set_title("成本收益对比(代表值)", fontsize=12.5, fontweight="bold")
    axr.legend(fontsize=9.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    axr.grid(axis="y", ls=":", alpha=0.5)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axr.text(bar.get_x() + bar.get_width() / 2, h * 1.05,
                     f"{h:g}", ha="center", va="bottom", fontsize=8.5)
    # 关键结论标注
    axr.text(0.5, 0.02,
             "成本↓60-80% · 效率↑5-10倍 · 取证由数天→实时",
             transform=axr.transAxes, ha="center", va="bottom", fontsize=9,
             color=COL_ACCENT, fontweight="bold")

    fig.suptitle("图3  商业模式与成本收益", fontsize=14.5, fontweight="bold",
                 y=1.02)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ===========================================================================
# 图 4: 6 学科交叉融合矩阵图 (热力矩阵: 学科 × 系统模块)
# ===========================================================================
def fig_crossmatrix(out_path):
    disciplines = ["航空/飞行器", "计算机视觉/AI", "城乡规划",
                   "公共管理/法学", "经济管理", "地理信息/GIS"]
    modules = ["飞行平台\n选型", "智能感知\n检测", "空域/空间\n约束",
               "执法/合规\n流程", "商业模式\n成本收益", "空间分析\n坐标投影"]
    # 贡献强度矩阵 0-3 (0=无 1=弱 2=中 3=强), 行=学科 列=模块
    M = np.array([
        [3, 1, 3, 0, 1, 1],   # 航空/飞行器
        [0, 3, 1, 1, 0, 2],   # 计算机视觉/AI
        [1, 1, 3, 2, 1, 2],   # 城乡规划
        [0, 1, 2, 3, 1, 0],   # 公共管理/法学
        [1, 0, 1, 1, 3, 0],   # 经济管理
        [1, 2, 2, 1, 0, 3],   # 地理信息/GIS
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=3, aspect="auto")

    ax.set_xticks(np.arange(len(modules)))
    ax.set_yticks(np.arange(len(disciplines)))
    ax.set_xticklabels(modules, fontsize=10)
    ax.set_yticklabels(disciplines, fontsize=10.5)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    labels = {0: "—", 1: "弱", 2: "中", 3: "强"}
    for i in range(len(disciplines)):
        for j in range(len(modules)):
            v = int(M[i, j])
            tc = "white" if v >= 2 else "#333333"
            ax.text(j, i, labels[v], ha="center", va="center",
                    fontsize=11, color=tc, fontweight="bold")

    # 网格线
    ax.set_xticks(np.arange(-0.5, len(modules), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(disciplines), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8,
                        pad=0.02)
    cbar.ax.set_yticklabels(["无", "弱", "中", "强"], fontsize=9.5)
    cbar.set_label("学科对模块的贡献强度", fontsize=10)

    ax.set_title("图4  六学科交叉融合矩阵(学科 × 系统模块)",
                 fontsize=14, fontweight="bold", pad=14)
    # 每学科贡献总分(右侧条注释思路改为标题下说明)
    row_sum = M.sum(axis=1)
    note = "各学科贡献度合计: " + " / ".join(
        f"{d.split('/')[0]}={int(s)}" for d, s in zip(disciplines, row_sum))
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ===========================================================================
ALL_FIGS = [
    ("fig_architecture.png", fig_architecture),
    ("fig_workflow.png", fig_workflow),
    ("fig_business.png", fig_business),
    ("fig_crossmatrix.png", fig_crossmatrix),
]


def generate_all(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in ALL_FIGS:
        p = os.path.join(out_dir, name)
        fn(p)
        paths.append(p)
        print(f"[OK] 生成 {name}  ({os.path.getsize(p)} bytes)")
    return paths


def selftest(out_dir):
    """生成全部图并校验: 文件存在、非空(>5KB)、PNG 头正确、可被 PIL 打开且尺寸>500px。"""
    from PIL import Image
    paths = generate_all(out_dir)
    ok = True
    for p in paths:
        if not os.path.exists(p):
            print(f"[FAIL] 缺失: {p}"); ok = False; continue
        size = os.path.getsize(p)
        if size < 5000:
            print(f"[FAIL] 过小({size}B): {p}"); ok = False; continue
        with open(p, "rb") as f:
            head = f.read(8)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            print(f"[FAIL] 非PNG头: {p}"); ok = False; continue
        with Image.open(p) as im:
            w, h = im.size
        if w < 500 or h < 400:
            print(f"[FAIL] 尺寸不足 {w}x{h}: {p}"); ok = False; continue
        print(f"[PASS] {os.path.basename(p)}  {w}x{h}px  {size}B")
    print("=== SELFTEST", "PASS ===" if ok else "FAIL ===")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出目录(默认 ../output/figures)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(here, "..", "output", "figures")
    out_dir = os.path.abspath(out_dir)

    setup_cjk_font()
    if args.selftest:
        sys.exit(0 if selftest(out_dir) else 1)
    else:
        generate_all(out_dir)
        print(f"全部图已生成到: {out_dir}")
