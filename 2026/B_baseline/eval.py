#!/usr/bin/env python3
"""
eval.py · mAP@0.5 / mAP@0.5:0.95 + 按目标尺寸分桶的小目标召回

为什么要分桶召回:
  低空水上救援的核心目标(落水人员)在 30-80m 航高下仅数十像素，属 COCO 定义的 small。
  整体 mAP 会被大目标(船)拉高，掩盖小目标漏检 —— 必须单独看 small 桶召回。

桶定义(COCO 面积口径，按像素面积):
  small:  area <  32^2 (1024)
  medium: 32^2 <= area < 96^2 (9216)
  large:  area >= 96^2

用法:
  python eval.py --weights runs/detect/train/weights/best.pt --data configs/searescue.yaml
"""
import argparse
from pathlib import Path


def overall_map(weights, data, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=data, imgsz=imgsz, verbose=False)
    print("\n=== 整体指标 ===")
    print(f"  mAP@0.5      = {r.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 = {r.box.map:.4f}")
    return m


def bucketed_recall(model, data, imgsz, iou=0.5):
    """对 val 集逐图推理，按 GT 框面积分桶统计召回(命中=与某预测框 IoU>=iou)。"""
    import yaml, numpy as np, cv2
    from pathlib import Path as P
    cfg = yaml.safe_load(P(data).read_text(encoding="utf-8"))
    root = (P(data).parent / cfg["path"]).resolve()
    val_img_dir = root / cfg["val"]
    val_lbl_dir = P(str(val_img_dir).replace("/images/", "/labels/"))
    buckets = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [命中, 总数]

    def iou_xyxy(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    for img in sorted(val_img_dir.glob("*")):
        lbl = val_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        preds = model.predict(str(img), imgsz=imgsz, verbose=False)[0]
        pboxes = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else []
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = map(float, line.split()[:5])
            bw, bh = w * W, h * H
            area = bw * bh
            gt = [(cx*W-bw/2), (cy*H-bh/2), (cx*W+bw/2), (cy*H+bh/2)]
            bkt = "small" if area < 32**2 else ("medium" if area < 96**2 else "large")
            buckets[bkt][1] += 1
            if any(iou_xyxy(gt, pb) >= iou for pb in pboxes):
                buckets[bkt][0] += 1

    print(f"\n=== 分桶召回 (IoU>={iou}) ===")
    for k, (hit, tot) in buckets.items():
        rec = hit / tot if tot else 0.0
        print(f"  {k:6s}: recall={rec:.4f}  ({hit}/{tot})")
    print("  ↑ 重点看 small 桶：落水人员漏检直接关系救援，是本赛题最该优化的指标")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    m = overall_map(a.weights, a.data, a.imgsz)
    try:
        bucketed_recall(m, a.data, a.imgsz, a.iou)
    except Exception as e:
        print(f"[分桶召回] 跳过(数据路径/格式待对齐): {e}")
    # TODO[登录核对]: 若赛题以官方测试集线上评测，需按其提交格式导出预测，再对齐其主指标。


if __name__ == "__main__":
    main()
