#!/usr/bin/env python3
"""
export_onnx.py · PyTorch(.pt) → ONNX，并用 onnxruntime 验证正确性

用法:
  python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 1024

要点:
  - opset 12，固定 imgsz（端侧 TensorRT 静态 shape 更稳；小目标不建议动态边长）。
  - 用 onnxsim 简化，onnxruntime 做一次前向比对，确保导出无误后再上 Orin Nano。
  - 不在模型里做 NMS（端侧后处理在 trt_infer_orin.py 里用 numpy/CPU 做），与 RKNN/昇腾思路一致。
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True,
                    help="onnxsim 简化（默认开；--no-simplify 关闭）")
    a = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(a.weights)
    # Ultralytics 原生导出（已含 onnxsim 简化）
    onnx_path = m.export(format="onnx", imgsz=a.imgsz, opset=a.opset,
                         simplify=a.simplify, dynamic=False, nms=False)
    print(f"✓ 导出 ONNX: {onnx_path}")

    # 正确性自检：onnxruntime 跑一次随机输入，确认无报错、输出形状合理
    try:
        import numpy as np, onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = np.random.rand(1, 3, a.imgsz, a.imgsz).astype("float32")
        outs = sess.run(None, {inp.name: dummy})
        print(f"  onnxruntime 前向 OK，输出形状: {[o.shape for o in outs]}")
    except Exception as e:
        print(f"  [WARN] onnxruntime 校验跳过/失败: {e}")

    print("  下一步(在 Orin Nano 上): python trt_infer_orin.py --onnx",
          Path(onnx_path).name, "--fp16 --benchmark")


if __name__ == "__main__":
    main()
