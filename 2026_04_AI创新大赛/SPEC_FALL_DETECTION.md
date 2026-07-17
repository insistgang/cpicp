# Spec: 企业赛题七 - 低算力端侧视觉跌倒检测

## Objective

面向第八届中国研究生人工智能创新大赛华为企业赛题七,构建一个端到端纯视觉跌倒检测方案。系统输入普通/红外视频片段,输出跌倒事件分数、报警片段和评测指标,目标是在低算力端侧设备上保持高召回、低误报和低延时。

当前阶段目标是先形成可运行 baseline:

- 用合成视频片段验证完整流程:数据 -> 人体区域/姿态代理特征 -> 时序判别 -> 指标 -> 报告。
- 预留公开数据入口,后续接 OmniFall 与 Kaggle Fall Video Dataset。
- 预留端侧部署入口,后续导出 ONNX/RKNN/NCNN 并测试海思/瑞芯微类 NPU。

## Tech Stack

- Language: Python 3.9+
- Runtime dependencies: standard library, numpy, PIL/Pillow
- Optional later dependencies: torch/timm 或 ultralytics/onnxruntime/rknn-toolkit2, 仅在真模型或端侧部署阶段引入
- Data format: clip-level manifest and frame sequence directories

## Commands

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src

# 全部自测
MPLCONFIGDIR=/private/tmp/ai04_mpl bash run_all_selftests.sh

# 跑跌倒检测 baseline
/usr/bin/python3 run_fall_pipeline.py

# 跑跌倒检测自测
/usr/bin/python3 fall_metrics.py --selftest
/usr/bin/python3 fall_synth.py --selftest
/usr/bin/python3 fall_features.py --selftest
/usr/bin/python3 fall_detector.py --selftest
/usr/bin/python3 fall_tiny_model.py --selftest
/usr/bin/python3 fall_video_io.py --selftest
/usr/bin/python3 fall_public_datasets.py --selftest
/usr/bin/python3 train_fall_model.py --selftest
/usr/bin/python3 run_fall_pipeline.py --selftest

# 训练/评测 tiny temporal model
/usr/bin/python3 train_fall_model.py
/usr/bin/python3 run_fall_pipeline.py --model ../output/fall_tiny_model.json --params-m 0.001
/usr/bin/python3 viz_fall_report.py --model ../output/fall_tiny_model.json

# 公开视频 manifest 评测
/usr/bin/python3 fall_public_datasets.py \
  --omnifall /path/to/omnifall \
  --kaggle /path/to/kaggle_fall_video \
  --out ../output/fall_public_manifest.csv
/usr/bin/python3 fall_data_audit.py \
  --manifest ../output/fall_public_manifest.csv \
  --sample-readable 20 \
  --out ../output/fall_public_audit.json
/usr/bin/python3 run_fall_pipeline.py --manifest ../output/fall_public_manifest.csv --skip-errors
```

## Project Structure

```text
2026_04_AI创新大赛/
├── SPEC_FALL_DETECTION.md          # 本规格
├── README.md                       # 04 主线说明,应指向赛题七
├── PROGRESS_04.md                  # 进度与卡点
├── docs/附件3-华为赛题.txt          # 官方赛题原文
├── src/
│   ├── fall_metrics.py             # 赛题七指标与竞赛分近似计算
│   ├── fall_synth.py               # 合成视频片段,用于无数据阶段端到端联调
│   ├── fall_features.py            # 纯视觉人体区域/姿态代理特征
│   ├── fall_detector.py            # 轻量时序跌倒判别与报警后处理
│   ├── fall_tiny_model.py          # 9 参数 z-score logistic 时序头
│   ├── fall_video_io.py            # 帧目录/视频文件读取与均匀采样
│   ├── fall_public_datasets.py     # OmniFall/Kaggle 本地视频目录 manifest 适配
│   ├── fall_data_audit.py          # 公开视频 manifest 体检与可读性抽查
│   ├── train_fall_model.py         # tiny model 训练入口
│   ├── run_fall_pipeline.py        # 跌倒检测端到端运行入口
│   ├── viz_fall_report.py          # 可视化报告图片生成
│   └── run_all_selftests.sh        # 一键回归,包含跌倒主线
└── output/
    ├── fall_pipeline_report.json   # baseline 指标报告
    └── fall_scores.npz             # 分数缓存,不入库
```

AOI 相关文件保留为历史实验,不再作为 04 当前主线。

## Code Style

```python
def precision_at_recall(scores, labels, target_recall=0.9):
    """返回达到指定召回率时的最佳 precision 与阈值。"""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    # 纯函数优先,显式输入输出,不依赖全局状态。
```

Conventions:

- 新主线文件统一使用 `fall_` 前缀。
- 自测函数使用 `_selftest()` 并支持 `--selftest`。
- JSON 报告字段使用英文 snake_case,文档说明用中文。
- 不把真实视频、隐私数据或大体积缓存提交进仓库。

## Testing Strategy

- Unit tests: 每个 `fall_*.py` 文件内置 `--selftest`,验证纯函数和小型合成数据。
- Integration test: `run_fall_pipeline.py --selftest` 生成小规模合成数据并验证 AUC/precision/recall/latency。
- Regression: `run_all_selftests.sh` 必须包含跌倒主线所有自测。
- Manual later: 在真实 OmniFall/Kaggle/华为数据上生成报告并人工抽查报警片段。

## Boundaries

- Always:
  - 遵守赛题七纯视觉约束,不依赖深度图、穿戴设备或云端实时推理。
  - 保持参数量、延时、内存字段在报告中可追踪。
  - 所有新增逻辑带自测并纳入一键回归。
  - 数据下载、真实数据路径、模型权重路径不硬编码进代码。
- Ask first:
  - 安装大型依赖或下载外部数据集/权重。
  - 改动或删除 AOI 历史文件。
  - 生成正式提交版 Word/PDF/视频。
- Never:
  - 提交真实隐私视频、人脸数据或平台受限数据。
  - 把学校、学院、导师等匿名化风险信息写入提交材料。
  - 用非视觉传感器结果冒充纯视觉。

## Success Criteria

First milestone:

- `src/run_fall_pipeline.py` 能在无外部数据情况下跑通合成跌倒/非跌倒视频片段。
- `src/train_fall_model.py` 能训练一个可保存/加载的极小参数时序模型。
- 输出 `output/fall_pipeline_report.json`,包含 precision@recall90、precision@recall95、参数量估计、单帧延时和总竞赛分近似。
- 输出分数分布、PR 曲线和报警示例拼图,可直接进入项目文档/演示视频。
- `run_all_selftests.sh` 包含并通过所有跌倒主线自测。
- README/PROGRESS 明确 04 当前主线是赛题七跌倒检测,AOI 为历史实验。

Competition-ready later:

- 接入 OmniFall 与 Kaggle Fall Video Dataset。
- 引入轻量检测/姿态模型并导出端侧格式。
- 在公开视频测试集和 chaspark 数据上产出指标报告。
- 完成匿名化项目文档、300字简介、演示视频和使用说明。

## Open Questions

- 队伍最终端侧平台优先选 RK3588/RKNN、海思、还是仅提交 ONNX + 延时证明。
- 是否能获取华为 chaspark 少量参考数据和 baseline。
- 是否允许在评测脚本中提交传统视觉后处理与轻量神经网络混合方案。
