#!/usr/bin/env python3
"""
augment_water.py · 水面场景专用增广:GT-Anchored Glint 难负样本 + 物理增广封装

核心创新(直击"水面光照反射/波纹干扰强"):
  GT-Anchored Glint —— 在**非GT区域**的水面贴高斯高光斑(模拟太阳反光/碎浪亮点),
  作为"难负样本"喂给模型:逼它学会"亮 ≠ 目标",显著抑制反光误检(false positive)。
  关键:高光只贴在远离 GT 框的地方,**不改标签**(它就是背景难负样本)。

两种用法:
  ① 训练期在线增广:在 dataloader/transform 里调 add_glint(img, boxes)。
  ② 离线批量增广(CLI):对 images/labels 目录批量生成 _glint 副本,扩充难负样本。
     python augment_water.py --img-dir datasets/searescue/images/train \
                             --lbl-dir datasets/searescue/labels/train --n 2

依赖 numpy(必需);可选 Albumentations(--use-albu 时叠 RandomSunFlare/Shadow/Fog/CLAHE)。
`python augment_water.py --selftest` 跑自测(纯 numpy,无需图片)。
"""
import argparse
import math


def _rng(seed):
    import numpy as np
    return np.random.RandomState(seed)


def _yolo_to_xyxy(box, W, H):
    _, cx, cy, w, h = box[:5]
    return [(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H]


def add_glint(img, boxes_yolo=None, n=3, max_radius_frac=0.04,
              intensity=(120, 255), margin_frac=0.06, rng=None):
    """在非GT区域贴 n 个高斯高光斑(uint8 HxWx3,BGR/RGB 均可)。
    boxes_yolo: [[cls,cx,cy,w,h],...] 归一化;高光中心避开这些框(含 margin)。返回新图,标签不变。"""
    import numpy as np
    rng = rng if rng is not None else np.random
    H, W = img.shape[:2]
    out = img.astype(np.float32).copy()
    gt = [_yolo_to_xyxy(b, W, H) for b in (boxes_yolo or [])]
    mx, my = margin_frac * W, margin_frac * H
    placed = 0
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n * 8):
        if placed >= n:
            break
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        # 避开 GT(含 margin)
        if any((g[0]-mx) <= cx <= (g[2]+mx) and (g[1]-my) <= cy <= (g[3]+my) for g in gt):
            continue
        r = max(2.0, rng.uniform(0.005, max_radius_frac) * max(W, H))
        amp = rng.uniform(intensity[0], intensity[1])
        g = amp * np.exp(-(((xx-cx)**2 + (yy-cy)**2) / (2.0 * r * r)))
        out += g[..., None]                 # 加性高光(屏幕感),三通道一起提亮
        placed += 1
    return np.clip(out, 0, 255).astype(np.uint8), placed


def apply_albu(img, boxes_yolo):
    """可选:Albumentations 物理增广(太阳耀斑/阴影/雾/CLAHE)。未安装则原样返回。"""
    try:
        import albumentations as A
    except Exception:
        return img, boxes_yolo
    tf = A.Compose([
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=120, p=0.4),
        A.RandomShadow(p=0.3),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"]))
    cls = [int(b[0]) for b in boxes_yolo]
    bb = [[b[1], b[2], b[3], b[4]] for b in boxes_yolo]
    r = tf(image=img, bboxes=bb, cls=cls)
    out_boxes = [[c, *b] for c, b in zip(r["cls"], r["bboxes"])]
    return r["image"], out_boxes


def _read_labels(path):
    boxes = []
    try:
        for ln in open(path, encoding="utf-8"):
            p = ln.split()
            if len(p) >= 5:
                boxes.append([int(float(p[0]))] + [float(x) for x in p[1:5]])
    except FileNotFoundError:
        pass
    return boxes


def batch_offline(img_dir, lbl_dir, n_per_img, use_albu, seed=0):
    import numpy as np, cv2, os
    rng = _rng(seed)
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    made = 0
    for f in imgs:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue
        boxes = _read_labels(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
        for k in range(n_per_img):
            aug, _ = add_glint(img, boxes, n=rng.randint(2, 6), rng=rng)
            if use_albu:
                aug, _ = apply_albu(aug, boxes)
            stem = os.path.splitext(f)[0] + f"_glint{k}"
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), aug)
            # 标签不变(glint 是背景难负样本);若该图本无目标则写空 txt(负样本)
            src_lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
            dst_lbl = os.path.join(lbl_dir, stem + ".txt")
            open(dst_lbl, "w", encoding="utf-8").write(
                open(src_lbl, encoding="utf-8").read() if os.path.exists(src_lbl) else "")
            made += 1
    print(f"✓ 离线增广生成 {made} 张 _glint 副本(标签随原图,glint 为背景难负样本)")


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    H, W = 360, 640
    img = np.full((H, W, 3), 60, np.uint8)          # 暗灰"水面"
    boxes = [[0, 0.5, 0.5, 0.1, 0.1]]               # 中心一个 GT(落水人员)
    out, placed = add_glint(img, boxes, n=4, rng=_rng(42))

    check(out.shape == img.shape and out.dtype == np.uint8, "输出形状/类型不变")
    check(placed >= 1, f"成功放置 {placed} 个高光斑")
    check(out.max() > img.max(), "图像出现更亮高光(max 提升)")

    # GT 框中心区域应基本未被高光污染(高光避开GT)
    gx1, gy1 = int(0.45*W), int(0.45*H)
    gx2, gy2 = int(0.55*W), int(0.55*H)
    gt_region_delta = int(out[gy1:gy2, gx1:gx2].astype(int).max() - 60)
    check(gt_region_delta <= 30, f"GT 框中心几乎未被高光污染(Δ={gt_region_delta})")

    # 确定性:同 seed 结果一致
    out2, _ = add_glint(img, boxes, n=4, rng=_rng(42))
    check(np.array_equal(out, out2), "同 seed 可复现")

    print("\n" + ("✅ augment_water 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--img-dir"); ap.add_argument("--lbl-dir")
    ap.add_argument("--n", type=int, default=2, help="每张图生成几张 glint 副本")
    ap.add_argument("--use-albu", action="store_true", help="叠加 Albumentations 物理增广")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if not (a.img_dir and a.lbl_dir):
        ap.error("批处理需 --img-dir 与 --lbl-dir(或用 --selftest)")
    batch_offline(a.img_dir, a.lbl_dir, a.n, a.use_albu, a.seed)


if __name__ == "__main__":
    main()
