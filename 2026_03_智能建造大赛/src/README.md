# 智能建造赛题7 · 低空水上救援高精度视觉识别 — 代码 (src)

> 赛题：**基于边缘计算的低空智能水上救援装备高精度视觉识别技术**（揭榜赛道 #7，命题方 深圳大江智造）
> 目标：公开数据 SeaDronesSee + AFO 在 RTX4070S 上训 **YOLOv12 + P2** 小目标检测，并在 **Jetson Orin Nano 8G** 打通 ONNX→TensorRT 测速；为跨域迁移评价留接口。
> 硬件：RTX4070S(训练) / Jetson Orin Nano 8G(端侧)。

## ✅ 官方赛题事实（已据《参赛指南》P80-82 核实，原 `TODO[登录核对]` 全部落定）

| 项 | 官方要求 |
|---|---|
| **识别类别** | **3 类：落水人员 / 船只 / 浮标**（已写入 `configs/searescue.yaml` 与 `prepare_data.py`）|
| **硬指标** | 必须提供指定边缘端真实 FPS，**动态视频 ≥30FPS**（达不到出局）|
| **推荐模型** | 推荐但不限于 **YOLOv12** 等前沿架构（本 src 默认即 YOLOv12）|
| **官方数据** | 命题方提供基于 SeaDronesSee 二次开发数据集，**需联系李老师 13714638358 申请** |
| **提交物** | 技术方案 + 源码(Python) + 性能报告(PR曲线 + 边缘端FPS) + QGroundControl demo(加分) |
| **评分** | 功能实现/完成度/技术创新/综合表现力（无线上自动评测，**现场路演答辩**定）|
| **时间线** | 报名截止 **8/20** ｜ 作品提交截止 **8/31** ｜ 初赛(线上) 9/1–10/15 ｜ 决赛 11月 |

> ⚠️ **8/20 必须先完成报名**（硬性，先于交稿）；产学研用企业数据的成果 IP 归企业与参赛队共有。

## 🖥️ 在你的硬件上怎么跑（runbook）

```bash
# 0) 环境（建议独立 conda 环境）— numpy 已钉 <2.0，避免与 torch ABI 冲突
conda create -n searescue python=3.10 -y && conda activate searescue
pip install -r requirements.txt
#    torch 按你的 CUDA 版本装对应 wheel（见 pytorch.org）

# 1) 数据准备：按 GUIDE 手动下载 SeaDronesSee + AFO 后，转 YOLO 格式 + 合并
#    AFO 请连同 Roboflow 导出的 data.yaml 一起放好（脚本据它把类别重映射到官方3类）
python prepare_data.py --root ./datasets --step guide      # 先看下载指引
python prepare_data.py --root ./datasets --step convert    # 下载完再转换

# 2) 训练 YOLOv12 + P2 小目标头（默认即 yolo12；--p2 启用 P2 头）
python train.py --data configs/searescue.yaml --p2 --epochs 100 --imgsz 1024
#    注意看启动日志 [model] ✓ 实际加载: ... 确认真的是 "YOLOv12 + P2"，而非回退结构

# 3) 评测：整体 mAP + 按尺寸分桶的小目标召回（同类一对一匹配）
python eval.py --weights runs/detect/train/weights/best.pt --data configs/searescue.yaml

# 4) 端侧:导出 ONNX(PC) → 拷到 Orin Nano → 实测 FPS（这步是 30FPS 硬门槛的证据）
python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 1024
#   在 Orin Nano 上：
python trt_infer_orin.py --onnx best.onnx --fp16 --imgsz 1024 --benchmark

# 5) 跨域迁移评价（陆→海 域差，模板/接口）
python crossdomain_eval.py --source-weights <陆域权重> --target-data configs/searescue.yaml
```

**🔑 第一刀（决定整盘方案生死，越早越好）**：跳过精调，直接拿一版小模型在 **Orin Nano 上实测 FP16 FPS**——先确认 30FPS 门槛能不能过。过不了就趁早换更轻骨干/降分辨率/上 INT8，不要等训练收敛后才发现端侧跑不动。

## 文件
| 文件 | 作用 |
|---|---|
| `requirements.txt` | 依赖声明（numpy 已钉 <2.0）|
| `prepare_data.py` | SeaDronesSee+AFO 下载指引 + 转 YOLO + **按 data.yaml 重映射到官方3类** + 陆/海域划分 + `add_negatives`(难负样本/空标签负图,`--neg-dir`) |
| `configs/yolo12-p2.yaml` | **YOLOv12 + P2(stride4)** 小目标头（默认，官方推荐骨干，逐层索引已校验）|
| `configs/yolo11-p2.yaml` | YOLO11 + P2 小目标头（回退备选，结构已校验）|
| `configs/searescue.yaml` | 数据集与类别（官方3类：落水人员/船只/浮标）|
| `train.py` | 训练入口（`--p2`；回退链 yolo12-p2→yolo11-p2→yolo12n→yolo11n，**响亮打印实际加载**）|
| `eval.py` | mAP + 按 COCO 尺寸分桶召回（同类一对一匹配，路径错误不静默）|
| `export_onnx.py` | PyTorch→ONNX(opset12)，onnxruntime 自检（`--no-simplify` 可关简化）|
| `trt_infer_orin.py` | Orin 端**端到端三档计时**(裸推理/含后处理/含编码)+ TRT8/10 兼容 + INT8校准器 + **30FPS红线判定** |
| `geolocate.py` | **检测框→GPS救援航点**(针孔+海平面求交;创新点①)。`python geolocate.py` 跑几何自测,8项全过 |
| `track_filter.py` | IoU 时序滤波(连续≥k帧才确认,滤反光闪点+EMA稳框);`python track_filter.py` 自测 |
| `augment_water.py` | **GT-Anchored Glint** 难负样本(非GT水面贴高光,抑反光误检)+物理增广;在线/离线CLI;`--selftest` |
| `losses_smalltarget.py` | **NWD**(归一化Wasserstein)+Wise-IoU+Inner-IoU + ultralytics 接入说明;`python losses_smalltarget.py` 自测 |
| `stream_qgc.py` | 实时闭环→QGC 融合 demo 骨架(推理→滤波→画框+GPS+遥测→H265软编→RTSP);`--selftest` |
| `crossdomain_eval.py` | 陆→海 域差评估模板（MMD/特征分布占位接口）|
| `run_all_selftests.py` | **一键本地自测总控**：运行所有 Mac 可测模块 + YAML 语法检查，返回汇总报告 |

## 数据来源
- SeaDronesSee：https://github.com/Ben93kie/SeaDronesSee ｜ https://seadronessee.cs.uni-tuebingen.de/
- AFO：https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object （选 YOLOv8 格式导出，含 data.yaml）

## 快速自测（Mac 本地，无 GPU/数据）

```bash
cd src
python3 run_all_selftests.py
```

覆盖：geolocate(8项) / track_filter(5项) / augment_water(5项) / losses_smalltarget(9项) / stream_qgc(接线检查) / crossdomain_eval(域差流程) / prepare_data(guide) / configs YAML 语法 —— **10 项全通过**即本地可推进部分就绪。

## 注意事项
- **模型回退要看日志**：`train.py` 启动会打印 `[model] ✓ 实际加载: ...`。若不是 "YOLOv12 + P2" 而是回退项，说明你的 ultralytics 版本解析自定义 yaml 失败，先排查再正式训练（别用回退结果当成绩）。
- **Orin 端 TRT API**：`trt_infer_orin.py` 已**同时兼容 TensorRT 8.x 与 10.x**（自动探测 `num_io_tensors`/`execute_async_v3`，回退 `num_bindings`/`execute_async_v2`），解码+NMS 后处理已补齐，端到端三档计时。测速务必带 `--source 真实视频`（合成帧后处理不真实）；报告以「③ 含编码」为准——Orin Nano **无硬件 NVENC**，实战软编码更重,含编码端到端请最终用 `stream_qgc.py` 实测 + `tegrastats` 记功耗。
- 小目标强烈建议 `--imgsz 1024` 起步并启用 P2 头；INT8 对小目标敏感，端侧默认 **FP16**，FPS 不够再评估 INT8（需做精度回归）。
