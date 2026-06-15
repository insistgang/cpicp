# B_baseline · 低空水上救援视觉识别 baseline 脚手架

> 目标：用公开数据 SeaDronesSee + AFO 在 RTX4070S 上跑通 YOLO11+P2 小目标检测 baseline，并打通 Orin Nano 端侧 ONNX→TensorRT FP16 测速；为跨域迁移评价留接口。
> 硬件：RTX4070S(训练) / Jetson Orin Nano(端侧)。
> ⚠️ 凡 `TODO[登录核对]` 的项（类别定义、指标权重等）等赛题7 PDF 核对后再定，不阻塞本 baseline。

## 运行顺序
```bash
# 0) 环境（建议独立 conda 环境）
conda create -n searescue python=3.10 -y && conda activate searescue
pip install -r requirements.txt

# 1) 数据准备：下载 SeaDronesSee + AFO，转 YOLO 格式，划分 陆/海 域
#    （脚本只给"下载指引+转换+划分"，大数据需手动按提示下载）
python prepare_data.py --root ./datasets --step convert

# 2) 训练 YOLO11 + P2 小目标头（默认 yolo11n，--p2 启用 P2 头）
python train.py --data configs/searescue.yaml --p2 --epochs 100 --imgsz 1024

# 3) 评测：mAP@0.5 / 0.5:0.95 + 按目标尺寸分桶的小目标召回
python eval.py --weights runs/detect/train/weights/best.pt --data configs/searescue.yaml

# 4) 端侧导出 + Orin Nano 测速（步骤 4a 在 PC，4b 在 Orin Nano 上跑）
python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 1024
#   将 best.onnx 拷到 Orin Nano 后：
python trt_infer_orin.py --onnx best.onnx --fp16 --imgsz 1024 --benchmark

# 5) 跨域迁移评价（陆→海 域差实验，模板/接口，含占位 TODO）
python crossdomain_eval.py --source-weights <陆域权重> --target-data configs/searescue.yaml
```

## 文件
| 文件 | 作用 |
|---|---|
| `requirements.txt` | 依赖声明 |
| `prepare_data.py` | SeaDronesSee+AFO 下载指引 + 转 YOLO + 陆/海域划分 |
| `configs/yolo11-p2.yaml` | YOLO11 + P2(stride4) 小目标检测头模型结构 |
| `configs/searescue.yaml` | 数据集与类别（类别名 `TODO[登录核对]`） |
| `train.py` | Ultralytics 训练入口（`--p2` 切换 P2 头，加载失败自动回退标准 yolo11n） |
| `eval.py` | mAP + 按 COCO 尺寸定义(small/medium/large)分桶召回 |
| `export_onnx.py` | PyTorch→ONNX(opset12, 动态/静态可选) |
| `trt_infer_orin.py` | ONNX→TensorRT FP16 + Orin Nano FPS/显存测速骨架 |
| `crossdomain_eval.py` | 陆→海 域差评估模板（MMD/特征分布占位接口） |

## 数据来源
- SeaDronesSee：https://github.com/Ben93kie/SeaDronesSee ｜ https://seadronessee.cs.uni-tuebingen.de/
- AFO：https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects ｜ Roboflow 可直接导出 YOLO 格式：https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object

## 注意
- `configs/yolo11-p2.yaml` 的 head 结构按 ultralytics YOLO11 + P2(参考官方 yolov8-p2) 改写；若你安装的 ultralytics 版本解析报错，`train.py` 会自动回退到内置 `yolo11n.yaml` 并打印告警，先保证流程跑通，再排查 yaml。
- 小目标强烈建议 `--imgsz 1024` 起步并启用 P2 头；INT8 对小目标敏感，端侧默认 **FP16**。
