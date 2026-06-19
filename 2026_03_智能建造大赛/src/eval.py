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


def bucketed_recall(model, data, imgsz, iou=0.5, conf=0.25):
    """对 val 集逐图推理，按 GT 框面积分桶统计召回。
    命中判定：存在同类别、未被占用、IoU>=iou 的预测框（一对一贪心匹配，避免跨类虚高/重复命中）。"""
    import yaml, numpy as np, cv2
    from pathlib import Path as P
    cfg = yaml.safe_load(P(data).read_text(encoding="utf-8"))
    root = (P(data).parent / cfg["path"]).resolve()
    val_img_dir = root / cfg["val"]
    # YOLO 约定: 只把路径中**最后一段** images 换成 labels。
    # 用 str.replace 会替换首个/全部 images，若项目路径任何上层目录名含 images
    # (如 /data/images/proj/.../images/val)就会得到错误的 labels 目录 → 静默找不到标签。
    s = str(val_img_dir)
    idx = s.rfind("/images/")
    val_lbl_dir = P(s[:idx] + "/labels/" + s[idx + len("/images/"):]) if idx >= 0 \
        else P(s.replace("images", "labels"))
    if not val_img_dir.exists() or not val_lbl_dir.exists():
        raise FileNotFoundError(
            f"分桶召回需要 images/labels 配对目录，但未找到：\n"
            f"  images: {val_img_dir} (exists={val_img_dir.exists()})\n"
            f"  labels: {val_lbl_dir} (exists={val_lbl_dir.exists()})\n"
            f"请确认 {data} 的 path/val 指向 .../images/val 且存在同构 .../labels/val。")
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
        preds = model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False)[0]
        if preds.boxes is not None and len(preds.boxes):
            pboxes = preds.boxes.xyxy.cpu().numpy()
            pcls = preds.boxes.cls.cpu().numpy().astype(int)
        else:
            pboxes, pcls = np.empty((0, 4)), np.empty((0,), dtype=int)
        used = [False] * len(pboxes)
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cid = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            bw, bh = w * W, h * H
            area = bw * bh
            gt = [(cx*W-bw/2), (cy*H-bh/2), (cx*W+bw/2), (cy*H+bh/2)]
            bkt = "small" if area < 32**2 else ("medium" if area < 96**2 else "large")
            buckets[bkt][1] += 1
            # 同类、未占用、IoU 最大且 >= 阈值 的预测才算命中（一对一）
            best_j, best_iou = -1, iou
            for j, pb in enumerate(pboxes):
                if used[j] or pcls[j] != cid:
                    continue
                v = iou_xyxy(gt, pb)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                buckets[bkt][0] += 1
                used[best_j] = True

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
    ap.add_argument("--conf", type=float, default=0.25, help="分桶召回的预测置信度阈值")
    a = ap.parse_args()
    m = overall_map(a.weights, a.data, a.imgsz)
    try:
        bucketed_recall(m, a.data, a.imgsz, a.iou, a.conf)
    except Exception as e:
        import traceback
        print(f"[分桶召回] 失败：{e}")
        traceback.print_exc()
    # 注：评分以现场答辩/性能报告为准（官方无线上测试集自动评测，见参赛指南 P81 评分标准）。


if __name__ == "__main__":
    main()
