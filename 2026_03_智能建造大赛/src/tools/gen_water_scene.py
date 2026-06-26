#!/usr/bin/env python3
"""
gen_water_scene.py · 程序化合成"低空俯拍海面"图 + 实跑 augment_water 增广对比配图

为什么需要它:
  真实 SeaDronesSee 数据需向命题方申请、且训练/导出需 GPU——本地拿不到。
  但"技术方案/性能报告"里"水面反光误报抑制链"那一章**需要 before/after 增广配图**来讲清:
  GT-Anchored Glint 在非GT水面区域贴高斯高光斑,逼模型学"亮≠目标"。
  本脚本用纯 PIL+numpy **程序化合成**一张带:
    - 渐变海面 + Perlin式波纹噪声 + 条带状太阳反光带(模拟低空俯拍真实观感)
    - 3 类小目标:落水人员(暖色小点)、船只(白色短矩形)、浮标(橙色圆点),并产出 YOLO 标签
    - 若干天然碎浪高光(干扰项)
  再调用 src/augment_water.add_glint(真实算法,非随机) 产出 _after,拼成 before|after 对比 PNG。

  ⚠️ 合成图 = 配图/流程演示用途,清楚标注 "SYNTHETIC — for方案配图,非训练数据"。
     真实数据到位后,同一 add_glint 接口可直接作用于真实帧,无需改代码。

依赖: PIL, numpy, matplotlib(拼图标注)。无 cv2/torch/GPU。
用法:
  python3 gen_water_scene.py                 # 生成默认 1 组对比图到 ../../output/figs
  python3 gen_water_scene.py --selftest      # 纯内存自测(不写盘也校验像素逻辑)
  python3 gen_water_scene.py --n 3 --seed 7  # 多组、指定种子
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 让 "from augment_water import add_glint" 可用(tools/ 的上级是 src/)
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
from augment_water import add_glint  # 复用真实增广算法

OUT_DIR = (SRC_DIR.parent / "output" / "figs")

# 官方 3 类
CLASS_NAMES = ["落水人员", "船只", "浮标"]
CLASS_NAMES_EN = ["swimmer", "boat", "buoy"]


def _value_noise(H, W, scale, rng):
    """轻量 value-noise(低分辨率随机场 + 双线性上采样),近似海面波纹,纯 numpy。"""
    h = max(2, H // scale)
    w = max(2, W // scale)
    low = rng.rand(h, w).astype(np.float32)
    # 双线性放大到 (H,W)
    ys = np.linspace(0, h - 1, H)
    xs = np.linspace(0, w - 1, W)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = low[y0][:, x0] * (1 - wx) + low[y0][:, x1] * wx
    bot = low[y1][:, x0] * (1 - wx) + low[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def synth_sea(H=720, W=1280, rng=None, sun_band=True):
    """合成一张俯拍海面 RGB(uint8) + GT 框(YOLO 归一化 [cls,cx,cy,w,h])。"""
    rng = rng if rng is not None else np.random.RandomState(0)
    # 1) 海面基色:深青蓝 → 远处略亮的竖直渐变
    base_top = np.array([28, 64, 92], np.float32)     # 近(顶部)更暗
    base_bot = np.array([40, 96, 120], np.float32)    # 远(底部)略亮
    grad = np.linspace(0, 1, H)[:, None, None]
    img = (base_top[None, None, :] * (1 - grad) + base_bot[None, None, :] * grad)
    img = np.repeat(img, W, axis=1)

    # 2) 波纹:两层 value-noise 叠加,做明暗起伏
    n = 0.6 * _value_noise(H, W, 36, rng) + 0.4 * _value_noise(H, W, 11, rng)
    n = (n - n.mean())
    img += (n[..., None] * 46.0)

    # 3) 太阳反光带:斜向高亮条带(真实低空俯拍常见,正是 false-positive 来源)
    if sun_band:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # 一条从左上到右下的亮带
        d = np.abs((xx / W) - (yy / H) - 0.05)
        band = np.exp(-(d ** 2) / (2 * 0.05 ** 2)) * 70.0
        # 带内再叠碎浪闪点
        sparkle = (_value_noise(H, W, 5, rng) > 0.86).astype(np.float32)
        img += band[..., None]
        img += (band * sparkle)[..., None] * 1.4

    img = np.clip(img, 0, 255)

    # 4) 撒 3 类小目标 + 收集 YOLO 标签
    boxes = []

    def add_blob(cx, cy, rw, rh, color, soft=0.6):
        x0 = max(0, int(cx - rw)); x1 = min(W, int(cx + rw))
        y0 = max(0, int(cy - rh)); y1 = min(H, int(cy + rh))
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = np.exp(-(((xs - cx) / (rw + 1e-3)) ** 2 + ((ys - cy) / (rh + 1e-3)) ** 2) / (2 * soft ** 2))
        for c in range(3):
            img[y0:y1, x0:x1, c] = (1 - g) * img[y0:y1, x0:x1, c] + g * color[c]

    # 落水人员:很小的暖色点(6-12px)——最难、recall 优先
    for _ in range(rng.randint(3, 6)):
        cx, cy = rng.randint(40, W - 40), rng.randint(40, H - 40)
        r = rng.uniform(3, 6)
        add_blob(cx, cy, r, r * 1.2, np.array([200, 150, 120]))
        boxes.append([0, cx / W, cy / H, (2 * r + 4) / W, (2.4 * r + 4) / H])

    # 船只:白色短矩形(较大)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.randint(80, W - 80), rng.randint(80, H - 80)
        rw, rh = rng.uniform(16, 30), rng.uniform(7, 12)
        add_blob(cx, cy, rw, rh, np.array([225, 228, 230]), soft=0.8)
        boxes.append([1, cx / W, cy / H, (2 * rw) / W, (2 * rh) / H])

    # 浮标:橙色圆点(中等小)
    for _ in range(rng.randint(2, 4)):
        cx, cy = rng.randint(50, W - 50), rng.randint(50, H - 50)
        r = rng.uniform(5, 9)
        add_blob(cx, cy, r, r, np.array([240, 130, 40]))
        boxes.append([2, cx / W, cy / H, (2 * r + 4) / W, (2 * r + 4) / H])

    return img.astype(np.uint8), boxes


def _draw_boxes(ax, boxes, W, H):
    """在 matplotlib 轴上画 GT 框 + 类别色。"""
    import matplotlib.patches as patches
    colors = ["#ff4d4d", "#4dd2ff", "#ffb84d"]
    for b in boxes:
        cls, cx, cy, w, h = b
        x = (cx - w / 2) * W; y = (cy - h / 2) * H
        ax.add_patch(patches.Rectangle((x, y), w * W, h * H, fill=False,
                                       edgecolor=colors[int(cls)], linewidth=1.4))


def render_pair(img_before, img_after, boxes, placed, out_path, title_suffix=""):
    """拼 before|after 两栏对比图(标注 GT 框 + glint 数 + SYNTHETIC 水印),存 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = img_before.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), dpi=110)

    axes[0].imshow(img_before)
    axes[0].set_title(f"BEFORE — 合成海面 + GT 框{title_suffix}", fontsize=11)
    _draw_boxes(axes[0], boxes, W, H)

    axes[1].imshow(img_after)
    axes[1].set_title(f"AFTER — GT-Anchored Glint 难负样本 (+{placed} 高光斑, 标签不变)", fontsize=11)
    _draw_boxes(axes[1], boxes, W, H)

    legend_el = [
        Line2D([0], [0], color="#ff4d4d", lw=2, label="落水人员 swimmer"),
        Line2D([0], [0], color="#4dd2ff", lw=2, label="船只 boat"),
        Line2D([0], [0], color="#ffb84d", lw=2, label="浮标 buoy"),
    ]
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        # SYNTHETIC 水印
        ax.text(0.99, 0.02, "SYNTHETIC · 方案配图 · 非训练数据", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="white", alpha=0.85,
                bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.4))
    axes[0].legend(handles=legend_el, loc="upper left", fontsize=8, framealpha=0.85)

    fig.suptitle("水面反光误报抑制链 · GT-Anchored Glint 增广 before/after 对比 "
                 "(高光只贴在远离 GT 的水面 → 背景难负样本,逼模型学'亮≠目标')",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _setup_cjk_font():
    """让 matplotlib 能渲染中文；按本机已安装字体选择，失败也不致命。"""
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


def generate(n=1, seed=0, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_cjk_font()
    made = []
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        img, boxes = synth_sea(rng=rng)
        # 真实算法:在非GT水面贴 glint(贴 4-7 个),标签不变
        after, placed = add_glint(img, boxes, n=rng.randint(4, 8), rng=rng)
        out_path = out_dir / f"augment_water_before_after_{i}.png"
        render_pair(img, after, boxes, placed, out_path, title_suffix=f"  (#{i})")
        made.append((out_path, len(boxes), placed))
    return made


def _selftest():
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  OK  " if c else "  XX  ") + m)
        ok = ok and c

    rng = np.random.RandomState(42)
    img, boxes = synth_sea(H=480, W=720, rng=rng)
    check(img.shape == (480, 720, 3) and img.dtype == np.uint8, "合成海面形状/类型正确")
    check(len(boxes) >= 5, f"撒下 {len(boxes)} 个目标(>=5)")
    check(all(0 <= b[1] <= 1 and 0 <= b[2] <= 1 for b in boxes), "所有 GT 中心在 [0,1] 归一化范围")
    classes = {int(b[0]) for b in boxes}
    check(classes.issubset({0, 1, 2}), f"类别 id 合法(出现 {sorted(classes)})")

    # 确定性:同 seed 像素一致
    img2, boxes2 = synth_sea(H=480, W=720, rng=np.random.RandomState(42))
    check(np.array_equal(img, img2) and boxes == boxes2, "同 seed 合成可复现")

    # add_glint 真实增广:after 更亮且形状不变
    after, placed = add_glint(img, boxes, n=5, rng=np.random.RandomState(1))
    check(after.shape == img.shape and after.dtype == np.uint8, "增广后形状/类型不变")
    check(placed >= 1 and after.max() >= img.max(), f"放置 {placed} 个高光斑且更亮")

    # GT 中心区域基本不被 glint 污染(取第一个目标)
    b = boxes[0]
    W, H = 720, 480
    cx, cy = int(b[1] * W), int(b[2] * H)
    win = 4
    delta = int(after[max(0, cy - win):cy + win, max(0, cx - win):cx + win].astype(int).max()
                - img[max(0, cy - win):cy + win, max(0, cx - win):cx + win].astype(int).max())
    check(delta <= 40, f"GT 中心几乎未被 glint 污染(Δmax={delta})")

    print("\n" + ("OK augment 合成+增广配图 自测通过" if ok else "XX 自测未通过"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    made = generate(a.n, a.seed, a.out)
    for p, nb, np_ in made:
        sz = os.path.getsize(p)
        print(f"[OK] 生成 {p}  ({nb} 个GT, +{np_} glint, {sz} bytes)")
    print(f"\n共 {len(made)} 张 before/after 对比图 → {a.out}")


if __name__ == "__main__":
    main()
