# 企业赛题七跌倒检测推进记录

## 当前定位

04 当前主线为 **企业赛题七:面向低算力端侧平台基于视觉的实时跌倒检测**。

目标是提交一个端到端纯视觉、低算力端侧友好的跌倒检测作品,能够区分真实跌倒与坐下、弯腰、躺下等日常动作,并在参数量、推理耗时和高召回精准率上满足赛题要求。

## 已完成

| 模块 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 项目规格 | `SPEC_FALL_DETECTION.md` | 完成 | 明确赛题七目标、命令、结构、测试和边界 |
| 数据落地手册 | `DATA_FALL_DETECTION.md` | 完成 | 公开/官方数据目录、manifest、体检、评测和文档证据链 |
| 评分指标 | `src/fall_metrics.py` | 自测通过 | precision@recall90/95 + 参数量/延时得分 |
| 合成片段 | `src/fall_synth.py` | 自测通过 | fall/walk/sit/bend/lie,含红外/遮挡/噪声代理 |
| 特征提取 | `src/fall_features.py` | 自测通过 | 前景人体框 + 中心/宽高比/面积/速度 |
| 检测后处理 | `src/fall_detector.py` | 自测通过 | 快速下坠 + 横向姿态 + 报警窗口 |
| Tiny 时序模型 | `src/fall_tiny_model.py` | 自测通过 | 8维特征 + 9参数 logistic 时序头,可保存/加载 |
| 视频/帧读取 | `src/fall_video_io.py` | 自测通过 | 抽帧目录读取;imageio 失败时自动用 ffmpeg 兜底 |
| 公开视频清单 | `src/fall_public_datasets.py` | 自测通过 | OmniFall/Kaggle 本地视频目录 -> clip manifest |
| 数据体检 | `src/fall_data_audit.py` | 自测通过 | 检查 manifest 分布、路径、可读性和评测准备状态 |
| 训练入口 | `src/train_fall_model.py` | 自测通过 | 支持合成/manifest 训练 tiny model |
| 端到端 baseline | `src/run_fall_pipeline.py` | 自测通过 | 生成 synthetic_proxy 报告;支持 `--manifest` 评测 |
| 可视化 | `src/viz_fall_report.py` | 自测通过 | 分数分布、PR 曲线、报警示例拼图 |
| 一键回归 | `src/run_all_selftests.sh` | 已接入 | 优先跑跌倒主线,再跑 AOI 历史线 |
| 提交材料母稿 | `submission_fall/` | 初稿完成 | 作品简介、项目文档、视频脚本、运行说明、匿名化清单 |

## 公开数据落地状态

OmniFall 已开始落地:

- HuggingFace 轻量文件下载完成:292/292,约 46MB。
- OF-Syn 视频归档已下载:`data/omnifall/data_files/omnifall-synthetic_av1.tar`,大小 `9716480000` 字节。
- 已从归档抽取小样本:`data/omnifall/of_syn_sample/`,共 72 个 MP4。
- 小样本类别:fall 12、fallen 12、其余 walking/standing/sitting/sit_down/lie_down/lying/stand_up/other 各 6。
- 小样本 manifest:`output/fall_omnifall_ofsyn_sample_manifest.csv`。
- 小样本审计:`output/fall_omnifall_ofsyn_sample_audit.json`,路径完整、抽样可读、`ready_for_eval=true`。

Kaggle Fall Video Dataset 暂未落地:本机没有 Kaggle token/CLI,且当前公开 URL 返回 404,需要后续补有效账号或更新 slug。

## 当前 baseline 结果

运行:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src
/usr/bin/python3 run_fall_pipeline.py
```

产物:

- `output/fall_pipeline_report.json`
- `output/fall_scores.npz`

当前合成代理结果:

- 100 个合成片段:20 fall / 80 non-fall
- P@R90 = 1.0
- P@R95 = 1.0
- 规则代理单帧平均耗时约 0.17ms
- tiny model 路径单帧平均耗时约 0.21ms
- tiny model 为 9 参数时序头;完整系统参数量还需叠加检测/姿态模型

注意:这是 synthetic_proxy,不能作为公开/官方数据成绩;它证明的是工程闭环和指标计算可跑。

Tiny model 训练/评测:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src
/usr/bin/python3 train_fall_model.py
/usr/bin/python3 run_fall_pipeline.py --model ../output/fall_tiny_model.json --params-m 0.001
```

产物:

- `output/fall_tiny_model.json`
- `output/fall_tiny_model_report.json`
- `output/fall_score_distribution.png`
- `output/fall_pr_curve.png`
- `output/fall_demo_contact_sheet.png`

## 公开视频评测入口

当前已跑通 OmniFall OF-Syn 小样本真实视频评测:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛
/usr/bin/python3 src/run_fall_pipeline.py \
  --manifest output/fall_omnifall_ofsyn_sample_manifest.csv \
  --frames 32 --width 160 --height 120 --skip-errors
```

结果:

- 72 个真实 MP4:24 fall / 48 non-fall,失败 0。
- P@R90 = 0.342857。
- P@R95 = 0.342857。
- 规则后处理平均耗时约 0.23ms/frame。
- 产物:`output/fall_pipeline_report.manifest.json`,`output/fall_scores.manifest.npz`。
- tiny model 产物:`output/fall_tiny_model.manifest.json`,`output/fall_tiny_model_report.manifest.json`,`output/fall_pipeline_report.manifest.tiny_model.json`。
- tiny model 小样本评测 P@R90/P@R95 = 0.338028/0.338028,没有优于规则 baseline。

注意:这是公开视频接入 baseline,不是最终竞赛成绩;当前分数偏低,说明下一步需要训练真实轻量模型,不能只靠几何规则。

下载并解压 OmniFall/Kaggle 后:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src
/usr/bin/python3 fall_public_datasets.py \
  --omnifall /path/to/omnifall \
  --kaggle /path/to/kaggle_fall_video \
  --out ../output/fall_public_manifest.csv
/usr/bin/python3 fall_data_audit.py \
  --manifest ../output/fall_public_manifest.csv \
  --sample-readable 20 \
  --out ../output/fall_public_audit.json
/usr/bin/python3 run_fall_pipeline.py \
  --manifest ../output/fall_public_manifest.csv \
  --frames 32 --width 160 --height 120 --skip-errors

# 用公开数据训练 tiny model 后再评测
/usr/bin/python3 train_fall_model.py \
  --manifest ../output/fall_public_manifest.csv \
  --frames 32 --width 160 --height 120 --skip-errors
/usr/bin/python3 run_fall_pipeline.py \
  --manifest ../output/fall_public_manifest.csv \
  --model ../output/fall_tiny_model.manifest.json \
  --params-m 0.001 \
  --frames 32 --width 160 --height 120 --skip-errors
```

产物:

- `output/fall_pipeline_report.manifest.json`
- `output/fall_scores.manifest.npz`
- `output/fall_public_audit.json`

## 下一步

1. 数据扩大:基于 OmniFall splits 解压更多 train/val/test 视频,形成正式公开实验划分。
2. 真模型:接入轻量人体检测或姿态估计模型,替换当前前景阈值人体框。
3. 时序模型:在 OF-Syn 真实视频上训练 tiny model/TCN/GRU/轻量 Transformer head,重点压低躺下、坐下误报。
4. 端侧部署:导出 ONNX,评估 RKNN/NCNN/海思路径,补参数量、fp32大小、NPU内存和真实延时。
5. 数据补齐:继续申请官方参考数据;Kaggle 需补有效账号或更新 dataset slug。
6. 文档材料:基于 `submission_fall/` 母稿生成正式 PDF/DOCX,确保匿名化且完全对齐赛题七。

## 风险

- 公开视频数据多为受限下载,需要账号/协议;当前 Kaggle 数据未拿到。
- OF-Syn 小样本上规则 baseline 精准率偏低,必须训练真模型并做误报分析。
- 红外、遮挡、夜间场景必须靠增强或华为数据验证。
- 当前代理延时不包含神经网络检测器,正式报告必须补真实模型端到端延时。
- 赛题评分以视频片段为单位,后续评估不能只做单帧分类。
