#!/usr/bin/env python3
"""
augment_defect.py · 合成缺陷增广(华为赛题一,解"异常样本稀缺")

迁移 03/augment_water.py 的 GT-Anchored 思路:在正常图上贴 划痕/污点/缺件 等合成缺陷,
生成"难样本"补充 30 张真实缺陷,并产出缺陷 bbox/mask 供定位评测。纯 numpy,`python augment_defect.py --selftest`。
"""
import argparse

KINDS = ("scratch", "spot", "missing", "discolor")


def add_defect(img, kind="scratch", rng=None):
    """img: (H,W,3) uint8。返回 (aug_img, bbox=(x1,y1,x2,y2), kind)。"""
    import numpy as np
    rng = rng if rng is not None else np.random
    out = img.astype(np.float32).copy()
    H, W = img.shape[:2]
    if kind == "scratch":                       # 细长划痕(暗线)
        x1, y1 = rng.randint(0, W // 2), rng.randint(0, H)
        # 端点夹在 [0,W-1]/[0,H-1] 内,避免出界(y 的随机增量可能为负把 y2 推到 <0)
        x2 = min(W - 1, x1 + rng.randint(W // 6, W // 2))
        y2 = min(H - 1, max(0, y1 + rng.randint(-H // 8, H // 8)))
        n = max(2, abs(x2 - x1))
        xs = np.linspace(x1, x2, n).astype(int); ys = np.linspace(y1, y2, n).astype(int)
        for x, y in zip(xs, ys):
            out[max(0, y-1):y+2, max(0, x-1):x+2] *= 0.3
        # 保证 bbox 非退化(x 方向天然 x2>x1;y 方向同值时给 1px 高度)
        bx1, bx2 = min(x1, x2), max(x1, x2)
        by1, by2 = min(y1, y2), max(y1, y2)
        if by2 == by1:
            by2 = min(H - 1, by1 + 1) if by1 + 1 <= H - 1 else by1
            by1 = by2 - 1 if by2 > 0 else by1
        bbox = (bx1, by1, bx2, by2)
    elif kind == "spot":                         # 圆形污点(暗高斯)
        cx, cy = rng.randint(0, W), rng.randint(0, H); r = rng.randint(max(3, W//40), max(6, W//15))
        yy, xx = np.mgrid[0:H, 0:W]
        mask = np.exp(-(((xx-cx)**2 + (yy-cy)**2) / (2.0*r*r)))
        out *= (1 - 0.7 * mask[..., None])
        bbox = (max(0, cx-r), max(0, cy-r), min(W-1, cx+r), min(H-1, cy+r))
    elif kind == "missing":                      # 缺件(填充均值块)
        bw, bh = rng.randint(W//12, W//5), rng.randint(H//12, H//5)
        x1, y1 = rng.randint(0, W-bw), rng.randint(0, H-bh)
        # 确保均值与当前块不同,否则图像无变化
        block_mean = out[y1:y1+bh, x1:x1+bw].mean()
        fill_val = block_mean + 20.0 if block_mean < 128 else block_mean - 20.0
        out[y1:y1+bh, x1:x1+bw] = fill_val
        bbox = (x1, y1, x1+bw, y1+bh)
    else:                                        # discolor 色变(局部偏色)
        bw, bh = rng.randint(W//10, W//4), rng.randint(H//10, H//4)
        x1, y1 = rng.randint(0, W-bw), rng.randint(0, H-bh)
        out[y1:y1+bh, x1:x1+bw, 0] = np.clip(out[y1:y1+bh, x1:x1+bw, 0] + 80, 0, 255)
        bbox = (x1, y1, x1+bw, y1+bh)
    return np.clip(out, 0, 255).astype(np.uint8), bbox, kind


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    H, W = 256, 256
    base = np.full((H, W, 3), 160, np.uint8)
    for kind in KINDS:
        aug, bbox, k = add_defect(base, kind, rng)
        x1, y1, x2, y2 = bbox
        check(aug.shape == base.shape and aug.dtype == np.uint8, f"{kind}: 形状/类型不变")
        check(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, f"{kind}: bbox 在界内 {bbox}")
        check(not np.array_equal(aug, base), f"{kind}: 图像确实被改动")

    # 确定性
    a1, b1, _ = add_defect(base, "spot", np.random.RandomState(7))
    a2, b2, _ = add_defect(base, "spot", np.random.RandomState(7))
    check(np.array_equal(a1, a2) and b1 == b2, "同 seed 可复现")

    # 鲁棒性:多 seed × 多尺寸下 bbox 必须始终在界内且非退化、图像必变。
    # (scratch 的 y 随机增量可能为负曾把 y2 推到 <0 或与 y1 相等 → 退化/出界 bbox)
    bad = 0
    for hw in (256, 128, 64):
        b2_ = np.full((hw, hw, 3), 160, np.uint8)
        for s in range(300):
            for kk in KINDS:
                aug2, bb, _ = add_defect(b2_, kk, np.random.RandomState(s))
                bx1, by1, bx2, by2 = bb
                if not (0 <= bx1 < bx2 <= hw and 0 <= by1 < by2 <= hw):
                    bad += 1
                if np.array_equal(aug2, b2_):
                    bad += 1
    check(bad == 0, f"多 seed×尺寸 bbox 全部在界内非退化且图像必变(异常={bad})")

    print("\n" + ("✅ augment_defect 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.error("用 --selftest(批处理接真实正常图目录,真数据到位再用)")


if __name__ == "__main__":
    main()
