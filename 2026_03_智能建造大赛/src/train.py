#!/usr/bin/env python3
"""
train.py · Ultralytics YOLOv12(+P2) 训练入口（官方赛题7 推荐 YOLOv12 等前沿架构）

用法:
  python train.py --data configs/searescue.yaml --p2 --epochs 100 --imgsz 1024
  python train.py --data configs/searescue.yaml --model yolo12s.pt   # 不用 P2，用预训练权重

要点:
  - --p2 优先加载 configs/yolo12-p2.yaml（YOLOv12 + P2 小目标头）；逐级回退并响亮告知实际加载的结构，
    避免"以为在训 yolo12-p2、实则静默退化"。回退链: yolo12-p2 → yolo11-p2 → yolo12n → yolo11n。
  - 小目标建议 imgsz>=1024、关闭过强的 mosaic 尾段（close_mosaic）、可加 copy_paste 增广。
"""
import argparse
from pathlib import Path

HERE = Path(__file__).parent


def build_model(use_p2: bool, model_arg: str):
    from ultralytics import YOLO
    if model_arg:                      # 显式指定（.pt 预训练 或 .yaml）
        print(f"[model] 使用显式指定模型: {model_arg}")
        return YOLO(model_arg)
    if use_p2:
        # 官方推荐 YOLOv12 + P2 优先；逐级回退，每一步响亮打印实际加载的是什么
        candidates = [
            ("configs/yolo12-p2.yaml", "YOLOv12 + P2（官方推荐骨干 + 小目标头）", False),
            ("configs/yolo11-p2.yaml", "YOLO11 + P2（回退：yolo12-p2 解析失败）", True),
            ("yolo12n.yaml",           "YOLOv12n 标准结构（回退：自定义 P2 yaml 均失败）", True),
            ("yolo11n.yaml",           "YOLO11n 标准结构（最终兜底）", True),
        ]
        for rel, desc, is_fallback in candidates:
            path = str(HERE / rel) if rel.startswith("configs/") else rel
            try:
                m = YOLO(path)
                print(f"[model] ✓ 实际加载: {desc}  <-- {path}")
                if is_fallback:
                    print(f"[model][WARN] 未加载到 yolo12-p2，已回退到「{desc}」。"
                          f" 请核对 ultralytics 版本与 yaml（见 README）后再正式训练。")
                return m
            except Exception as e:
                print(f"[model][skip] {rel} 加载失败: {e}")
        raise RuntimeError("所有候选模型均加载失败，请检查 ultralytics 安装与 yaml。")
    print("[model] 使用默认 yolo12n.pt（官方推荐骨干，预训练权重）")
    return YOLO("yolo12n.pt")          # 默认：YOLOv12n 预训练


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
    # 官方指标(已核实 P81)：各类 PR 曲线 + 必须在边缘端实测 ≥30FPS；评分以现场答辩+性能报告为准。


if __name__ == "__main__":
    main()
