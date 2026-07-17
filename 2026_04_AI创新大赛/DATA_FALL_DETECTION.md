# 跌倒检测公开/官方数据落地手册

当前项目已经完成 OmniFall 公开数据的本地落地和小样本真实视频冒烟评测。公开视频原始数据通常受平台协议限制,因此 `data/` 已加入 `.gitignore`,不提交原始视频。

当前已落地:

- OmniFall HuggingFace 轻量文件:292/292 个文件下载完成,约 46MB。
- OmniFall OF-Syn 视频归档:`data/omnifall/data_files/omnifall-synthetic_av1.tar`,大小 `9716480000` 字节。
- 已抽取 OF-Syn 小样本:`data/omnifall/of_syn_sample/`,共 72 个 MP4。
- 小样本 manifest:`output/fall_omnifall_ofsyn_sample_manifest.csv`。
- 小样本体检报告:`output/fall_omnifall_ofsyn_sample_audit.json`,`ready_for_eval=true`。
- 小样本 baseline 报告:`output/fall_pipeline_report.manifest.json`。
- 小样本 tiny model 报告:`output/fall_pipeline_report.manifest.tiny_model.json`。

Kaggle Fall Video Dataset 当前未下载:本机没有 Kaggle token/CLI,且赛题给出的公开 URL 当前访问返回 404。后续需要用有效 Kaggle 账号或更新后的 dataset slug 补齐。

## 1 目录约定

建议把数据放在项目下的忽略目录:

```text
2026_04_AI创新大赛/
├── data/
│   ├── omnifall/
│   ├── kaggle_fall_video/
│   └── chaspark_reference/
└── output/
```

`data/`、`datasets/`、`raw_data/` 已加入 `.gitignore`。如果数据来自平台协议或含隐私视频,不要提交到公开仓库。

## 2 下载来源

赛题附件给出的参考数据:

- OmniFall: https://huggingface.co/datasets/simplexsigil2/omnifall
- Kaggle Fall Video Dataset: https://www.kaggle.com/datasets/payutch/fall-video-dataset

官方/参考数据:

- 赛题七 chaspark 页面: https://www.chaspark.com/#/races/competitions/1264757525464276992
- 需要队长/队伍账号登录并报名后获取平台数据、baseline 或提交说明。

## 3 生成统一 manifest

已抽取的 OmniFall OF-Syn 小样本可直接使用:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛
/usr/bin/python3 src/fall_data_audit.py \
  --manifest output/fall_omnifall_ofsyn_sample_manifest.csv \
  --sample-readable 12 \
  --frames 8 \
  --width 96 \
  --height 72 \
  --out output/fall_omnifall_ofsyn_sample_audit.json
```

如果后续解压更多真实视频,可按目录重新生成 manifest:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src

/usr/bin/python3 fall_public_datasets.py \
  --omnifall ../data/omnifall \
  --kaggle ../data/kaggle_fall_video \
  --out ../output/fall_public_manifest.csv
```

manifest 字段:

```csv
clip_path,label,source,scenario
```

- `clip_path`: 视频文件或已抽帧目录。
- `label`: `1` 表示跌倒,`0` 表示非跌倒。
- `source`: 数据来源,如 `omnifall`、`kaggle_fall_video`。
- `scenario`: 场景或动作子类,用于后续误报/漏报分析。

## 4 评测前数据体检

```bash
/usr/bin/python3 fall_data_audit.py \
  --manifest ../output/fall_public_manifest.csv \
  --sample-readable 20 \
  --out ../output/fall_public_audit.json
```

重点看:

- `summary.ready_for_eval`: 是否可以进入评测。
- `summary.n_fall` / `summary.n_nonfall`: 是否同时包含正负样本。
- `path_checks.n_missing`: 是否存在缺失路径。
- `path_checks.n_unsupported`: 是否有不支持文件或无图片帧目录。
- `readability.failed`: 抽样视频/帧目录是否可读。

若 `readability.failed > 0`,优先确认本机 `ffmpeg` 是否可用;当前 `fall_video_io.py` 已在 `imageio` 失败时自动用 `ffmpeg` 抽帧兜底。若仍失败,再统一抽帧为图片目录并重新生成 manifest。

## 5 跑公开视频 baseline 评测

```bash
/usr/bin/python3 src/run_fall_pipeline.py \
  --manifest output/fall_omnifall_ofsyn_sample_manifest.csv \
  --frames 32 \
  --width 160 \
  --height 120 \
  --skip-errors
```

输出:

- `output/fall_pipeline_report.manifest.json`
- `output/fall_scores.manifest.npz`
- 如果使用 `--model`,输出 `output/fall_pipeline_report.manifest.tiny_model.json` 和 `output/fall_scores.manifest.tiny_model.npz`

当前 OF-Syn 小样本 baseline:

- 72 个真实 MP4:24 fall / 48 non-fall。
- 读取失败:0。
- P@R90 = 0.342857。
- P@R95 = 0.342857。
- 后处理平均耗时约 0.23ms/frame,不含视频解码和神经网络检测器。

当前 tiny model 小样本结果:

- 训练/验证报告:`output/fall_tiny_model_report.manifest.json`。
- 9 参数 logistic 时序头。
- 同一小样本 manifest 评测 P@R90 = 0.338028,P@R95 = 0.338028。
- 结果没有优于规则 baseline,说明下一步要换更强的视觉/时序特征,不能只依赖当前几何代理。

这个结果可以作为“公开视频接入 baseline”写入项目文档,但必须同时说明模型版本、输入尺寸、是否跳过失败样本、失败样本数量和端侧耗时口径。

## 6 用公开视频训练 tiny head

```bash
/usr/bin/python3 train_fall_model.py \
  --manifest ../output/fall_public_manifest.csv \
  --frames 32 \
  --width 160 \
  --height 120 \
  --skip-errors

/usr/bin/python3 run_fall_pipeline.py \
  --manifest ../output/fall_public_manifest.csv \
  --model ../output/fall_tiny_model.manifest.json \
  --params-m 0.001 \
  --frames 32 \
  --width 160 \
  --height 120 \
  --skip-errors
```

输出:

- `output/fall_tiny_model.manifest.json`
- `output/fall_tiny_model_report.manifest.json`
- `output/fall_pipeline_report.manifest.json`

注意: 如果训练和评测使用同一个 manifest,只能作为管线验证或初步 baseline。正式文档应划分 train/val/test,或至少在报告中明确数据划分策略。

## 7 生成公开实验图

```bash
/usr/bin/python3 viz_fall_report.py \
  --scores ../output/fall_scores.manifest.npz \
  --model ../output/fall_tiny_model.manifest.json
```

可用于文档/视频的图:

- `output/fall_score_distribution.png`
- `output/fall_pr_curve.png`
- `output/fall_demo_contact_sheet.png`

正式报告应把 synthetic proxy 图和 public manifest 图分开标注,避免把合成结果误写成公开视频结果。

## 8 写入项目文档的最小证据

拿到真实数据后,至少补齐:

- 数据来源、下载日期、许可/使用限制。
- 样本数量: 总 clip、fall、nonfall、不同 source/scenario 分布。
- 可读性体检: `fall_public_audit.json` 的 `ready_for_eval`、失败样本数。
- 评测指标: P@R90、P@R95、阈值、TP/FP/FN 或误报/漏报案例。
- 性能: 模型参数量、fp32 大小、端侧平均延时和 95 分位延时。
- 失败分析: 坐下/弯腰/躺下误报,遮挡/弱光/红外漏报。
