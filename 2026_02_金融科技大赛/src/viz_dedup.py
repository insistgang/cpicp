#!/usr/bin/env python3
"""
viz_dedup.py · 去重结果可视化(matplotlib):相似度矩阵热图 + 检出可疑对拼图

赛题#23 官方要求"检测汇总视图,支持查看高相似可疑交易对应的影像"。本模块用 matplotlib
把 pipeline 的真实像素去重结果可视化成一张图:
  - 左:面签照片两两余弦相似度矩阵热图(对角=自身;亮块=高相似可疑对/同组)
  - 右:Top 跨客户套用可疑对的影像并排拼图(真实 PNG 缩略图 + 客户/相似度标注)

输入为 pipeline.run_pipeline_real_images 的返回 dict(含 S 相似度矩阵、suspicious、embs 等)。
纯 matplotlib + PIL,Mac 可跑,产出真实 PNG。
"""
import argparse
import os

# 中文字体(matplotlib 默认无中文,显式注册避免豆腐块)
_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def _setup_cn_font():
    import matplotlib
    from matplotlib import font_manager
    for c in _FONT_CANDIDATES:
        if os.path.exists(c):
            try:
                font_manager.fontManager.addfont(c)
                name = font_manager.FontProperties(fname=c).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue
    return None


def visualize(res, out_path="dedup_viz.png", image_dir=None, max_pairs=4):
    """渲染去重可视化。res = run_pipeline_real_images 的返回 dict。
    image_dir: 若给出,可疑对拼图用真实缩略图;默认从 res['image_dir'] 取。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    _setup_cn_font()
    S = np.asarray(res["S"])
    suspicious = res.get("suspicious", [])
    image_dir = image_dir or res.get("image_dir")
    image_ids = res.get("image_ids", [])
    id2idx = {img: i for i, img in enumerate(image_ids)}

    # 优先展示跨客户套用(最严重),不足则补同客户重复
    cross = [p for p in suspicious if p["type"] == "cross_customer_misuse"]
    others = [p for p in suspicious if p["type"] != "cross_customer_misuse"]
    show_pairs = (cross + others)[:max_pairs]

    n_pair_rows = max(1, len(show_pairs))
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(n_pair_rows, 3, width_ratios=[1.5, 1, 1],
                          hspace=0.35, wspace=0.15)

    # ---- 左:相似度矩阵热图 ----
    ax_hm = fig.add_subplot(gs[:, 0])
    im = ax_hm.imshow(S, cmap="viridis", vmin=-0.2, vmax=1.0, aspect="auto")
    ax_hm.set_title(f"面签照片相似度矩阵热图  (N={S.shape[0]})", fontsize=12)
    ax_hm.set_xlabel("影像索引"); ax_hm.set_ylabel("影像索引")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="余弦相似度")

    # ---- 右:可疑对影像并排拼图 ----
    def load_thumb(img_id):
        if image_dir:
            p = os.path.join(image_dir, img_id)
            if os.path.exists(p):
                return np.asarray(Image.open(p).convert("RGB").resize((96, 96)))
        return np.full((96, 96, 3), 200, np.uint8)

    if show_pairs:
        for r, p in enumerate(show_pairs):
            for c, key in enumerate(("img_a", "img_b")):
                ax = fig.add_subplot(gs[r, 1 + c])
                ax.imshow(load_thumb(p[key]))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{p['cust_'+('a' if c==0 else 'b')]}", fontsize=9)
                if c == 0:
                    badge = "[!]跨客户套用" if p["type"] == "cross_customer_misuse" else "同客户重复"
                    color = "#c0392b" if p["type"] == "cross_customer_misuse" else "#d68910"
                    ax.set_ylabel(f"{badge}\nsim={p['score']:.3f}", fontsize=9,
                                  color=color, rotation=0, ha="right", va="center", labelpad=42)
    else:
        ax = fig.add_subplot(gs[:, 1:])
        ax.text(0.5, 0.5, "当前阈值下无可疑对", ha="center", va="center", fontsize=13)
        ax.axis("off")

    auc = res.get("auc", float("nan"))
    nc = res.get("n_cross", 0); ns = res.get("n_same", 0)
    dthr = res.get("det_threshold")
    thr_str = f"阈值={dthr:.3f} | " if dthr is not None else ""
    fig.suptitle(f"金融影像相似度去重检测 · 真实像素(经典特征 baseline)  "
                 f"AUC={auc:.3f} | {thr_str}[!]跨客户套用 {nc} 对 | 同客户重复 {ns} 对",
                 fontsize=13, y=0.99)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def _selftest():
    import sys, tempfile
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    with tempfile.TemporaryDirectory() as td:
        # 用真实流水线产出 res(真实像素),再可视化
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from synth_images import generate
        from pipeline import run_pipeline_real_images
        out = os.path.join(td, "synth")
        generate(out, n_groups=20, reuse_frac=0.5, seed=2, size=(128, 128))
        res = run_pipeline_real_images(out, threshold=None, pca_dim=32, verbose=False)
        check(res is not None, "真实像素流水线产出 res")

        png = os.path.join(td, "viz.png")
        path = visualize(res, out_path=png)
        check(os.path.exists(png) and os.path.getsize(png) > 5000,
              f"可视化 PNG 生成且非空({os.path.getsize(png)} 字节)")

        # 验证是合法 PNG
        from PIL import Image
        im = Image.open(png)
        check(im.format == "PNG" and im.size[0] > 200,
              f"PNG 合法可读(format={im.format}, size={im.size})")

        # 无可疑对的退化场景也不崩
        res2 = dict(res); res2["suspicious"] = []; res2["n_cross"] = 0; res2["n_same"] = 0
        png2 = os.path.join(td, "viz_empty.png")
        visualize(res2, out_path=png2)
        check(os.path.getsize(png2) > 1000, "无可疑对场景也能出图(不崩)")

    print("\n" + ("✅ viz_dedup 自测通过" if ok else "❌ viz_dedup 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="去重结果可视化(相似度热图+可疑对拼图)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--images", help="synth_images 输出目录(含 manifest.csv)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", default="dedup_viz.png")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.images:
        ap.error("需 --images 或 --selftest")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import run_pipeline_real_images
    res = run_pipeline_real_images(a.images, threshold=a.threshold)
    if res:
        path = visualize(res, out_path=a.out)
        print(f"📊 去重可视化已保存: {path}")


if __name__ == "__main__":
    import sys
    main()
