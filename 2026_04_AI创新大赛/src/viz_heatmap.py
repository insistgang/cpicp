#!/usr/bin/env python3
"""
viz_heatmap.py · AOI 异常热力图可视化(华为赛题一,定位/可解释性证据)

PatchCore 输出的不只是 image-level 异常分,还有 **patch-level 异常图**:每个 patch
到 memory bank 的最近邻距离 → 上采样回原图,得到"缺陷在哪"的热力图(官方要求输出
缺陷定位)。本模块用 matplotlib 画:
  (A) 正常件 vs 4 类缺陷件的 原图 / patch 异常热力叠加 对比大图
  (B) 测试集 image-level 异常分分布(正常 vs 缺陷)直方图 + 阈值线
  (C) P-R / ROC 曲线(真实指标)

数据来自 run_real_pipeline(真实合成图 + 经典 CPU 特征的真实 patch 异常)。
纯 matplotlib(Agg 后端,无显示器也能存图)。

`python viz_heatmap.py`            生成全部图到 output/
`python viz_heatmap.py --selftest` 自测(小规模生成 + 校验 PNG 非空)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anomaly_score import nn_distance
from aoi_metrics import roc_auc, pr_at_threshold, threshold_sweep

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# 中文标签缺字体时会乱码,但坐标轴/数值正常;统一用英文+缺陷英文名,稳妥
plt.rcParams["axes.unicode_minus"] = False
_KIND_EN = {"scratch": "Scratch", "spot": "Spot", "missing": "Missing", "discolor": "Discolor"}


def patch_anomaly_map(patch_feats_grid, bank):
    """patch_feats_grid: (gh,gw,D) → (gh,gw) 每 patch 到 bank 最近邻距离(异常图)。"""
    gh, gw, D = patch_feats_grid.shape
    flat = patch_feats_grid.reshape(-1, D)
    d = nn_distance(flat, bank)
    return d.reshape(gh, gw)


def _upsample(amap, size):
    """patch 异常图最近邻上采样到 size×size(纯 numpy,无 cv2)。"""
    gh, gw = amap.shape
    yi = (np.arange(size) * gh / size).astype(int).clip(0, gh - 1)
    xi = (np.arange(size) * gw / size).astype(int).clip(0, gw - 1)
    return amap[yi][:, xi]


def fig_overlay(backend, bank, normal_img, defect_samples, vmax, out_path):
    """A: 正常件 + 各类缺陷件的 原图/热力叠加 对比图。
    defect_samples: list[(img, kind, bbox)]。"""
    n = 1 + len(defect_samples)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    if n == 1:
        axes = axes.reshape(2, 1)

    def show_pair(col, img, title, bbox=None):
        size = img.shape[0]
        grid = backend.patch_features(img)
        amap = _upsample(patch_anomaly_map(grid, bank), size)
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=11)
        axes[0, col].axis("off")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            axes[0, col].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="lime", lw=1.6))
        axes[1, col].imshow(img)
        hm = axes[1, col].imshow(amap, cmap="jet", alpha=0.55, vmin=0, vmax=vmax)
        axes[1, col].set_title(f"anomaly heatmap\nmax={amap.max():.3f}", fontsize=10)
        axes[1, col].axis("off")
        return hm

    show_pair(0, normal_img, "Normal (OK)")
    last_hm = None
    for c, (img, kind, bbox) in enumerate(defect_samples, start=1):
        last_hm = show_pair(c, img, f"Defect: {_KIND_EN.get(kind, kind)}\n(GT box green)", bbox)

    if last_hm is not None:
        cbar = fig.colorbar(last_hm, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("patch nn-distance (anomaly)")
    fig.suptitle("AOI PatchCore anomaly localization: Normal vs Defects (classic-CPU features)",
                 fontsize=13)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_score_dist(scores, labels, thr, out_path):
    """B: image-level 异常分分布直方图(正常 vs 缺陷)+ 阈值线。"""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Normal", color="#2c7fb8")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Defect", color="#d95f0e")
    ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold={thr:.3f}")
    ax.set_xlabel("image-level anomaly score (max patch nn-distance)")
    ax.set_ylabel("count")
    ax.set_title(f"Anomaly score distribution (AUC={roc_auc(scores, labels):.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def fig_curves(scores, labels, out_path):
    """C: ROC + P-R 曲线(真实指标)。"""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    sweep = threshold_sweep(scores, labels, n=200)
    rec = [s["recall"] for s in sweep]
    prec = [s["precision"] for s in sweep]
    # ROC:用 sweep 的 (FPR, TPR);FPR 需从混淆量算,这里用阈值扫直接估
    s = scores
    order = np.argsort(-s)
    y = labels[order]
    P = max(1, int((labels == 1).sum()))
    N = max(1, int((labels == 0).sum()))
    tpr = np.cumsum(y == 1) / P
    fpr = np.cumsum(y == 0) / N
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(fpr, tpr, color="#1b9e77", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"ROC (AUC={roc_auc(scores, labels):.3f})")
    ax1.grid(alpha=0.3)

    ax2.plot(rec, prec, color="#7570b3", lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)
    fig.suptitle("AOI few-shot anomaly detection: real metrics on synthetic workpieces", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def generate_all(pipeline_result=None, out_dir=OUT_DIR):
    """用 run_real_pipeline 的结果生成全部图。无结果则现跑一次。"""
    from run_real_pipeline import run
    from synth_aoi import make_normal, make_defect, DEFECT_KINDS

    if pipeline_result is None:
        pipeline_result = run(n_normal=100, n_defect=30, n_test_normal=400,
                              n_test_defect=200, size=160, grid=8, save=True, verbose=False)
    arr = pipeline_result["_arrays"]
    backend, bank = arr["backend"], arr["bank"]
    scores, labels = arr["test_scores"], arr["test_labels"]
    thr = pipeline_result["threshold"]

    os.makedirs(out_dir, exist_ok=True)

    # 选 1 张正常 + 每类 1 张缺陷做叠加图(用与训练不重叠的展示 seed)
    # 展示件 seed 段(88800..)落在 gen_dataset 各 seed 段之外,且对每个 kind 用其
    # 固定下标派生 seed —— 不能用内置 hash(),那是按进程随机化的,会导致每次运行展示件不同。
    size = 160
    normal_img = make_normal(size, seed=99999)
    defect_samples = []
    for ki, k in enumerate(DEFECT_KINDS):
        dimg, bbox, kk = make_defect(size, seed=88800 + ki, kind=k)
        defect_samples.append((dimg, kk, bbox))

    # 统一热力图色标:用正常 patch 距离的高分位 + 缺陷峰值的折中,使正常偏冷、缺陷偏热
    vmax = float(np.percentile(
        np.concatenate([patch_anomaly_map(backend.patch_features(d[0]), bank).ravel()
                        for d in defect_samples]), 97))

    p_a = fig_overlay(backend, bank, normal_img, defect_samples, vmax,
                      os.path.join(out_dir, "heatmap_overlay.png"))
    p_b = fig_score_dist(scores, labels, thr, os.path.join(out_dir, "score_distribution.png"))
    p_c = fig_curves(scores, labels, os.path.join(out_dir, "roc_pr_curves.png"))
    return [p_a, p_b, p_c]


def _selftest():
    import sys
    from run_real_pipeline import run
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    r = run(n_normal=40, n_defect=20, n_test_normal=80, n_test_defect=60,
            size=128, grid=8, save=False, verbose=False)
    arr = r["_arrays"]

    # patch 异常图:缺陷件峰值应高于正常件峰值
    from synth_aoi import make_normal, make_defect
    backend, bank = arr["backend"], arr["bank"]
    nmap = patch_anomaly_map(backend.patch_features(make_normal(128, seed=99)), bank)
    dmap = patch_anomaly_map(backend.patch_features(make_defect(128, seed=99, kind="spot")[0]), bank)
    check(dmap.max() > nmap.max(), f"缺陷件 patch 异常峰值更高({dmap.max():.3f}>{nmap.max():.3f})")

    # 上采样形状
    up = _upsample(nmap, 128)
    check(up.shape == (128, 128), f"异常图上采样到原图尺寸 {up.shape}")

    # 全部图生成且非空。落盘到独立的 selftest 子目录,**不覆盖** output/ 下标准规模
    # 交付物 PNG(那些由 `python viz_heatmap.py` 用 600 测试集生成,与 README 对应)。
    selftest_dir = os.path.join(OUT_DIR, "selftest")
    paths = generate_all(pipeline_result=r, out_dir=selftest_dir)
    for p in paths:
        check(os.path.exists(p) and os.path.getsize(p) > 1000, f"PNG 非空:{os.path.basename(p)} ({os.path.getsize(p)} bytes)")

    print("\n" + ("✅ viz_heatmap 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        paths = generate_all()
        print("✓ 已生成异常可视化:")
        for p in paths:
            print(f"   {os.path.abspath(p)}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
