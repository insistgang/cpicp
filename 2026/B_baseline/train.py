#!/usr/bin/env python3
"""
train.py · Ultralytics YOLO11(+P2) 训练入口

用法:
  python train.py --data configs/searescue.yaml --p2 --epochs 100 --imgsz 1024
  python train.py --data configs/searescue.yaml --model yolo11s.pt   # 不用 P2，用预训练权重

要点:
  - --p2 加载 configs/yolo11-p2.yaml（P2 小目标头）；若该 yaml 解析失败，自动回退 yolo11n.yaml。
  - 小目标建议 imgsz>=1024、关闭过强的 mosaic 尾段（close_mosaic）、可加 copy_paste 增广。
"""
import argparse
from pathlib import Path

HERE = Path(__file__).parent


def build_model(use_p2: bool, model_arg: str):
    from ultralytics import YOLO
    if model_arg:                      # 显式指定（.pt 预训练 或 .yaml）
        return YOLO(model_arg)
    if use_p2:
        p2 = HERE / "configs" / "yolo11-p2.yaml"
        try:
            m = YOLO(str(p2))
            print(f"[model] 已加载 P2 结构: {p2}")
            return m
        except Exception as e:         # 版本不兼容 → 回退，保证流程可跑
            print(f"[model][WARN] 加载 {p2} 失败({e})，回退 yolo11n.yaml。"
                  f" 请按 ultralytics 版本核对 yaml（见 README）。")
            return YOLO("yolo11n.yaml")
    return YOLO("yolo11n.pt")          # 默认：预训练 n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据集 yaml（如 configs/searescue.yaml）")
    ap.add_argument("--p2", action="store_true", help="启用 P2 小目标检测头")
    ap.add_argument("--model", default="", help="覆盖模型(.pt 或 .yaml)，留空按 --p2 逻辑")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)      # 小目标建议 >=1024
    ap.add_argument("--batch", type=int, default=8)         # 4070S 12-16G，1024 下 batch 不宜大
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = build_model(args.p2, args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        close_mosaic=10,        # 尾段关 mosaic，利于小目标定位
        # 小目标增广（按需开）：copy_paste / scale 调小，避免目标被裁掉
        copy_paste=0.1,
        scale=0.5,
        project="runs/detect",
        name="train",
        exist_ok=True,
    )
    print("✓ 训练完成。best 权重在 runs/detect/train/weights/best.pt")
    print("  下一步: python eval.py --weights runs/detect/train/weights/best.pt --data", args.data)
    # TODO[登录核对]: 主指标(mAP) 与 是否硬卡 FPS/功耗，待赛题指标权重确认后定。


if __name__ == "__main__":
    main()
