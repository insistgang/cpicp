#!/usr/bin/env python3
"""
synth_aoi.py · 程序化合成 AOI 工件图(华为赛题一,无真数据先端到端联调)

华为赛题是通用 AOI 组装质检(屏幕/电池/中框等),真数据在 chaspark 报名后才有。
本模块用 PIL 程序化生成"正常工件纹理 + 4 类缺陷"的合成图,让 feature_backend /
patchcore_lite / fewshot_protocol / anomaly_score 在真实图像(而非随机 numpy)上
端到端跑通,产出真实 AUC/F1。缺陷合成复用 augment_defect 的 add_defect(划痕/斑点/缺件/色变)。

正常工件纹理:仿"组装件"——基板底色 + 规则栅格(模组/焊盘) + 若干元件方块 + 轻噪声,
带可复现 seed,同一产品的正常件之间有自然的小抖动(位置/亮度微扰),刻画真实产线波动。

`python synth_aoi.py --selftest`  自测(纯内存,不落盘)
`python synth_aoi.py --gen-dir <out> --n-normal 60 --n-defect 40`  生成数据集到磁盘
"""
import argparse
import os

import numpy as np
from PIL import Image

from augment_defect import add_defect, KINDS

DEFECT_KINDS = KINDS   # ("scratch", "spot", "missing", "discolor")


def make_normal(size=256, seed=0):
    """生成一张正常工件图 (H,W,3) uint8。仿组装件:基板 + 规则栅格 + 元件方块 + 噪声。

    seed 控制可复现;同产品不同件之间用 seed 派生的小抖动模拟产线自然波动。
    """
    rng = np.random.RandomState(seed)
    H = W = size
    # 基板底色(深灰偏蓝的 PCB/中框感),件间亮度微抖
    base = np.array([60, 68, 78], np.float32) + rng.uniform(-6, 6, 3)
    img = np.tile(base, (H, W, 1)).astype(np.float32)
    # 轻微基底噪声(纹理)
    img += rng.randn(H, W, 1) * 4.0

    # 规则栅格(模组/焊盘行列),线色略亮,位置随 seed 微抖
    step = size // 8
    off = rng.randint(0, step // 3 + 1)
    line_col = np.array([95, 105, 115], np.float32)
    for y in range(off, H, step):
        img[max(0, y-1):y+1, :] = line_col
    for x in range(off, W, step):
        img[:, max(0, x-1):x+1] = line_col

    # 若干元件方块(规则排布的"芯片/电容"),亮色,位置/亮度件间微抖
    comp_col = np.array([140, 150, 160], np.float32)
    n_rows, n_cols = 3, 3
    cell = size // (max(n_rows, n_cols) + 1)
    for r in range(n_rows):
        for c in range(n_cols):
            cy = int((r + 1) * size / (n_rows + 1)) + rng.randint(-2, 3)
            cx = int((c + 1) * size / (n_cols + 1)) + rng.randint(-2, 3)
            hw = cell // 3
            jitter = rng.uniform(-8, 8, 3)
            img[max(0, cy-hw):cy+hw, max(0, cx-hw):cx+hw] = comp_col + jitter

    return np.clip(img, 0, 255).astype(np.uint8)


def make_defect(size=256, seed=0, kind=None):
    """在一张正常件上贴 1 个合成缺陷。返回 (img, bbox, kind)。"""
    rng = np.random.RandomState(seed)
    base = make_normal(size, seed=seed)
    if kind is None:
        kind = DEFECT_KINDS[rng.randint(len(DEFECT_KINDS))]
    aug, bbox, k = add_defect(base, kind, rng)
    return aug, bbox, k


def gen_dataset(n_normal=60, n_defect=40, size=256, seed=0):
    """生成内存数据集。返回 (images list[(H,W,3)], labels list[int], metas list[dict])。
    label: 0=正常, 1=缺陷;meta 含 kind / bbox(缺陷件)。

    每张图的 seed 由顶层 seed 派生且分段隔离:不同的顶层 seed 产出**互不重叠**的件
    (训练/测试用不同 seed 调本函数即可保证无泄漏)。同一 seed 可完全复现。
    """
    # 顶层 seed 派生出正常/缺陷两条互不相交的件级 seed 序列;
    # 不同顶层 seed 的件级 seed 段不重叠(每段预留 1e6 容量,远大于任何数据集规模)。
    normal_base = 1_000_000 + seed * 1_000_000
    defect_base = 500_000_000 + seed * 1_000_000
    images, labels, metas = [], [], []
    for i in range(n_normal):
        images.append(make_normal(size, seed=normal_base + i))
        labels.append(0)
        metas.append({"kind": "normal", "bbox": None})
    for j in range(n_defect):
        kind = DEFECT_KINDS[j % len(DEFECT_KINDS)]   # 4 类均匀覆盖
        img, bbox, k = make_defect(size, seed=defect_base + j, kind=kind)
        images.append(img)
        labels.append(1)
        metas.append({"kind": k, "bbox": bbox})
    return images, labels, metas


def save_dataset(out_dir, n_normal=60, n_defect=40, size=256, seed=0):
    """生成并落盘:<out>/normal/*.png, <out>/defect/*.png, <out>/manifest.csv。"""
    import csv
    images, labels, metas = gen_dataset(n_normal, n_defect, size, seed)
    os.makedirs(os.path.join(out_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "defect"), exist_ok=True)
    rows = []
    ni = di = 0
    for img, lab, meta in zip(images, labels, metas):
        if lab == 0:
            rel = f"normal/n_{ni:04d}.png"
            ni += 1
        else:
            rel = f"defect/d_{di:04d}_{meta['kind']}.png"
            di += 1
        Image.fromarray(img).save(os.path.join(out_dir, rel))
        bb = meta["bbox"]
        rows.append({"image_path": rel, "label": lab, "defect_type": meta["kind"],
                     "bbox": "" if bb is None else ",".join(map(str, bb))})
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "defect_type", "bbox"])
        w.writeheader()
        w.writerows(rows)
    return out_dir, len(rows)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    # 正常件:形状/类型,且件间有自然抖动(不完全相同)但整体相似
    a = make_normal(256, seed=1)
    b = make_normal(256, seed=2)
    check(a.shape == (256, 256, 3) and a.dtype == np.uint8, "正常件形状/类型正确")
    check(not np.array_equal(a, b), "不同 seed 正常件有产线自然抖动(不相同)")
    diff = np.abs(a.astype(int) - b.astype(int)).mean()
    check(diff < 30, f"同产品正常件整体相似(均值差={diff:.1f}<30)")
    check(np.array_equal(make_normal(256, seed=1), make_normal(256, seed=1)), "同 seed 可复现")

    # 4 类缺陷都能生成且确实改变图像、bbox 在界内
    for kind in DEFECT_KINDS:
        img, bbox, k = make_defect(256, seed=42, kind=kind)
        clean = make_normal(256, seed=42)
        x1, y1, x2, y2 = bbox
        check(k == kind and not np.array_equal(img, clean), f"{kind}: 缺陷已注入")
        check(0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256, f"{kind}: bbox 在界内 {bbox}")

    # 数据集生成:数量/标签正确,4 类均覆盖
    imgs, labs, metas = gen_dataset(n_normal=12, n_defect=8)
    check(len(imgs) == 20 and sum(labs) == 8, "数据集数量/标签正确(12正+8缺)")
    kinds = {m["kind"] for m, l in zip(metas, labs) if l == 1}
    check(kinds == set(DEFECT_KINDS), f"缺陷件覆盖全部 4 类:{kinds}")

    # 关键:不同顶层 seed 的数据集**逐张互不重叠**(防训练/测试泄漏回归;
    # 之前 gen_dataset 忽略 seed 导致 train(seed=0)与 test(seed=777)件完全相同)
    a_imgs, a_labs, _ = gen_dataset(n_normal=20, n_defect=20, size=64, seed=0)
    b_imgs, b_labs, _ = gen_dataset(n_normal=20, n_defect=20, size=64, seed=777)
    a_set = {im.tobytes() for im in a_imgs}
    overlap = sum(1 for im in b_imgs if im.tobytes() in a_set)
    check(overlap == 0, f"不同 seed 数据集逐张无重叠(seed0 vs seed777 重叠={overlap})")
    # 同 seed 必须完全复现
    c_imgs, _, _ = gen_dataset(n_normal=5, n_defect=5, size=64, seed=0)
    d_imgs, _, _ = gen_dataset(n_normal=5, n_defect=5, size=64, seed=0)
    check(all(np.array_equal(x, y) for x, y in zip(c_imgs, d_imgs)), "同 seed 数据集可复现")

    print("\n" + ("✅ synth_aoi 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen-dir")
    ap.add_argument("--n-normal", type=int, default=60)
    ap.add_argument("--n-defect", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.gen_dir:
        out, n = save_dataset(a.gen_dir, a.n_normal, a.n_defect, a.size, a.seed)
        print(f"✓ 合成 AOI 数据集 {n} 张 → {out}(normal/ + defect/ + manifest.csv)")
    else:
        ap.error("用 --selftest 或 --gen-dir <out>")


if __name__ == "__main__":
    main()
