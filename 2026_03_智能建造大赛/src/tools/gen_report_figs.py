#!/usr/bin/env python3
"""
gen_report_figs.py · 生成"性能评估报告"所需的三类图表(占位/合成值,函数化,真数据可直接替换)

为什么:report 的 §2(PR曲线)/§2(分桶召回)/§3(FPS-精度帕累托) 这三张图,
  评委一眼看的是"完成度 + 30FPS 硬门槛证明"。GPU/Orin 数据本地拿不到,
  但**图表骨架、坐标、配色、30FPS 红线、帕累托标注**全部可以本地定稿,
  等 eval.py / trt_infer_orin.py 跑出真值后,只把数据字典换掉即可,无需改绘图代码。

  所有数据都来自显式的 PLACEHOLDER_* 字典,图上水印 "PLACEHOLDER" + 标题带 *合成占位值*。
  三个绘图函数 plot_bucket_recall / plot_pr_curves / plot_fps_accuracy 均接收数据参数,
  真数据(eval.py 的分桶召回 dict、各类 P/R 数组、trt_infer_orin.py 的 FPS 表)直接传入即可。

依赖: matplotlib, numpy。无 GPU。
用法:
  python3 gen_report_figs.py             # 生成 3 张图到 ../../output/figs
  python3 gen_report_figs.py --selftest  # 自测(校验图确实生成且非空,不依赖外网)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = SRC_DIR.parent / "output" / "figs"

# 30 FPS 是赛题硬出局线
FPS_REDLINE = 30.0

# ---------------------------------------------------------------------------
# 占位数据(清楚标注 PLACEHOLDER;真数据来自 eval.py / trt_infer_orin.py)
# ---------------------------------------------------------------------------
# (1) 按 COCO 像素面积分桶的召回(占位演示值;真值来自 eval.py 输出)。
#     注: 这里的分桶标签(tiny<8px / small 8-32px / medium / large)是为方案配图
#     更细地拆出"极小目标"而设的演示口径;eval.py 实际只产 small(<32^2)/medium/large 三桶。
#     plot_bucket_recall 接收任意 {bucket:{config:recall}} dict,直接传 eval.py 的三桶输出即可。
PLACEHOLDER_BUCKET_RECALL = {
    # bucket : {配置: recall}
    "tiny (<8px)":    {"baseline n": 0.41, "+P2": 0.58, "+P2+NWD+Glint": 0.67},
    "small (8-32px)": {"baseline n": 0.63, "+P2": 0.74, "+P2+NWD+Glint": 0.81},
    "medium":         {"baseline n": 0.82, "+P2": 0.85, "+P2+NWD+Glint": 0.86},
    "large":          {"baseline n": 0.90, "+P2": 0.90, "+P2+NWD+Glint": 0.91},
}

# (2) 各类 PR 曲线(eval.py 每类的 precision/recall 序列)。这里用合成单调曲线占位。
def _placeholder_pr(ap_target, seed):
    """造一条 AP≈ap_target 的合成 PR 曲线(recall 升、precision 降)。"""
    rng = np.random.RandomState(seed)
    recall = np.linspace(0, 1, 50)
    # precision 随 recall 单调下降,起点高,终点≈ap_target 附近
    precision = 1.0 - (1.0 - ap_target) * recall ** 1.6
    precision = np.clip(precision - np.abs(rng.normal(0, 0.004, recall.shape)), 0, 1)
    # COCO 风格精度包络:从右往左取累计最大,使曲线随 recall 单调不增
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.trapezoid(precision, recall))
    return recall, precision, ap

# 注:经 COCO 精度包络后显示 AP 会较 target 略高,这里 target 已下调使显示值落在意图区间。
PLACEHOLDER_PR = {
    "落水人员 swimmer": _placeholder_pr(0.62, 1),
    "船只 boat":        _placeholder_pr(0.85, 2),
    "浮标 buoy":        _placeholder_pr(0.74, 3),
}

# (3) FPS-精度权衡(trt_infer_orin.py 端到端 FPS × eval.py mAP)。
#     每点: (FPS_端到端含编码, mAP, 标签, 是否过30线)
PLACEHOLDER_FPS_ACC = [
    # (fps, mAP, label, prec)
    (52.0, 0.71, "n@640 INT8", "INT8"),
    (44.0, 0.74, "n+P2@640 INT8", "INT8"),
    (33.0, 0.77, "n+P2@768 INT8", "INT8"),
    (38.0, 0.76, "n+P2@640 FP16", "FP16"),
    (26.0, 0.79, "n+P2@768 FP16", "FP16"),   # 跌破 30 红线的反例
    (21.0, 0.80, "s+P2@768 FP16", "FP16"),
]


def _setup_cjk_font():
    import matplotlib
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        installed = set()
    for f in [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Heiti TC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "PingFang SC",
    ]:
        if not installed or f in installed:
            matplotlib.rcParams["font.sans-serif"] = [f]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return f
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


def _watermark(ax, text="PLACEHOLDER · 合成占位值 · 待真数据替换"):
    ax.text(0.5, 0.5, "PLACEHOLDER", transform=ax.transAxes, ha="center", va="center",
            fontsize=44, color="gray", alpha=0.12, rotation=24, zorder=0)
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#888", alpha=0.9)


# ---------------------------------------------------------------------------
# 三个绘图函数:全部接收数据参数,真数据直接传入
# ---------------------------------------------------------------------------
def plot_bucket_recall(bucket_recall, out_path):
    """分桶召回分组柱状图。bucket_recall: {bucket: {config: recall}}。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = list(bucket_recall.keys())
    configs = list(next(iter(bucket_recall.values())).keys())
    n_cfg = len(configs)
    x = np.arange(len(buckets))
    width = 0.8 / n_cfg
    colors = ["#9aa7b1", "#4d96ff", "#2ec27e"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    _watermark(ax)
    for i, cfg in enumerate(configs):
        vals = [bucket_recall[b][cfg] for b in buckets]
        bars = ax.bar(x + (i - (n_cfg - 1) / 2) * width, vals, width,
                      label=cfg, color=colors[i % len(colors)], zorder=3)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(buckets)
    ax.set_ylabel("Recall @ IoU0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("图1 · 按目标像素面积分桶的召回(小目标不漏检的证据)*合成占位值*")
    ax.legend(title="配置(逐项消融)", fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(0.0, -0.16, "tiny/small 是落水人员核心尺寸区间 — P2 + NWD + GT-Glint 链路在此带来最大增益",
            transform=ax.transAxes, fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pr_curves(pr_data, out_path):
    """各类 PR 曲线。pr_data: {class_name: (recall_arr, precision_arr, ap)}。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#ff4d4d", "#4dd2ff", "#ffb84d"]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    _watermark(ax)
    aps = []
    for i, (cls, (rec, prec, ap)) in enumerate(pr_data.items()):
        ax.plot(rec, prec, color=colors[i % len(colors)], lw=2,
                label=f"{cls}  (AP@0.5={ap:.3f})")
        aps.append(ap)
    ax.axhline(0.5, color="#bbb", ls="--", lw=0.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(f"图2 · 各类 Precision-Recall 曲线  (mAP@0.5≈{np.mean(aps):.3f}) *合成占位值*")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(0.0, -0.13, "救援场景 recall 优先:落水人员曲线右端(高 recall 区)是验收重点",
            transform=ax.transAxes, fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fps_accuracy(fps_acc, out_path, redline=FPS_REDLINE):
    """FPS-精度权衡散点 + 30FPS 红线 + 帕累托前沿。fps_acc: [(fps, mAP, label, prec_mode)]。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)
    _watermark(ax)

    style = {"INT8": ("o", "#2ec27e"), "FP16": ("s", "#4d96ff")}
    for fps, mAP, label, mode in fps_acc:
        mk, col = style.get(mode, ("^", "#888"))
        passed = fps >= redline
        ax.scatter(fps, mAP, marker=mk, s=120,
                   facecolor=col if passed else "white",
                   edgecolor=col, linewidths=1.6, zorder=4,
                   alpha=0.95 if passed else 0.9)
        ax.annotate(label, (fps, mAP), textcoords="offset points",
                    xytext=(6, 6), fontsize=8,
                    color="#333" if passed else "#c0392b")

    # 30FPS 红线
    ax.axvline(redline, color="#e74c3c", ls="--", lw=1.8)
    ax.text(redline + 0.4, ax.get_ylim()[0], f" 30FPS 硬出局线", color="#e74c3c",
            fontsize=10, va="bottom", rotation=90)
    # 左侧(<30)阴影 = 出局区
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < redline else 0, redline,
               color="#e74c3c", alpha=0.06, zorder=0)

    # 帕累托前沿(过线点中,FPS↑且 mAP↑不被支配者)
    cand = [(f, m, l) for f, m, l, _ in fps_acc if f >= redline]
    cand.sort(key=lambda t: -t[0])  # FPS 降序
    front = []
    best_m = -1
    for f, m, l in cand:
        if m > best_m:
            front.append((f, m)); best_m = m
    if len(front) >= 2:
        front.sort()
        fx, fy = zip(*front)
        ax.plot(fx, fy, color="#555", ls="-", lw=1.2, alpha=0.6,
                label="帕累托前沿(过30线内最优)")

    # 图例(手工)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ec27e",
               markeredgecolor="#2ec27e", markersize=10, label="INT8(过线=实心)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4d96ff",
               markeredgecolor="#4d96ff", markersize=10, label="FP16"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#c0392b", markersize=10, label="未过30线(空心红字)"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8)

    ax.set_xlabel("端到端 FPS(含画框+H265软编码,Orin Nano)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("图3 · FPS-精度权衡 + 30FPS 红线 + 帕累托前沿(选最优工作点)*合成占位值*")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(out_dir=OUT_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _setup_cjk_font()
    paths = {
        "bucket_recall": out_dir / "report_fig1_bucket_recall.png",
        "pr_curves":     out_dir / "report_fig2_pr_curves.png",
        "fps_accuracy":  out_dir / "report_fig3_fps_accuracy.png",
    }
    plot_bucket_recall(PLACEHOLDER_BUCKET_RECALL, paths["bucket_recall"])
    plot_pr_curves(PLACEHOLDER_PR, paths["pr_curves"])
    plot_fps_accuracy(PLACEHOLDER_FPS_ACC, paths["fps_accuracy"])
    return paths


def _selftest():
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  OK  " if c else "  XX  ") + m)
        ok = ok and c

    _setup_cjk_font()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1) 分桶召回
        p1 = td / "f1.png"; plot_bucket_recall(PLACEHOLDER_BUCKET_RECALL, p1)
        check(p1.exists() and p1.stat().st_size > 2000, f"分桶召回图生成且非空({p1.stat().st_size}B)")
        # 2) PR 曲线 + AP 单调性
        p2 = td / "f2.png"; plot_pr_curves(PLACEHOLDER_PR, p2)
        check(p2.exists() and p2.stat().st_size > 2000, f"PR 曲线图生成且非空({p2.stat().st_size}B)")
        for cls, (rec, prec, ap) in PLACEHOLDER_PR.items():
            check(0 < ap < 1 and prec[0] >= prec[-1], f"{cls} AP={ap:.3f} 合理且 precision 单调不增")
        # 3) FPS-精度 + 红线逻辑
        p3 = td / "f3.png"; plot_fps_accuracy(PLACEHOLDER_FPS_ACC, p3)
        check(p3.exists() and p3.stat().st_size > 2000, f"FPS-精度图生成且非空({p3.stat().st_size}B)")
        n_pass = sum(1 for f, *_ in PLACEHOLDER_FPS_ACC if f >= FPS_REDLINE)
        n_fail = len(PLACEHOLDER_FPS_ACC) - n_pass
        check(n_pass >= 1 and n_fail >= 1, f"占位点含过线{n_pass}/未过线{n_fail}(红线判定有对照)")

        # 真数据可替换:传一个最小自定义 dict 不报错
        custom = {"tiny": {"A": 0.5, "B": 0.6}}
        pc = td / "fc.png"; plot_bucket_recall(custom, pc)
        check(pc.exists(), "绘图函数接收自定义数据 dict(真数据可替换)")

    print("\n" + ("OK report 图表生成器 自测通过" if ok else "XX 自测未通过"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    paths = generate(a.out)
    for k, p in paths.items():
        print(f"[OK] {k}: {p}  ({os.path.getsize(p)} bytes)")
    print(f"\n共 3 张报告图表 → {a.out}")


if __name__ == "__main__":
    main()
