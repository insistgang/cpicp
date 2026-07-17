# 当前进度快照:企业赛题七跌倒检测

更新时间:2026-06-27

## 一句话状态

当前项目已经从“合成代理验证”推进到“公开视频真实样本可读、可审计、可评测”的阶段。OmniFall OF-Syn 数据已经下载并抽取小样本跑通端到端 pipeline,但当前规则/tiny model baseline 精度偏低,下一步重点是扩大真实数据划分并接入更强的轻量视觉/时序模型。

## 已经完成

- 赛题方向已确定:第八届中国研究生人工智能创新大赛,企业赛题“面向低算力端侧平台基于视觉的实时跌倒检测”。
- 已建立项目规格文档:`SPEC_FALL_DETECTION.md`。
- 已建立数据落地手册:`DATA_FALL_DETECTION.md`。
- 已建立完整推进记录:`PROGRESS_FALL_DETECTION.md`。
- 已完成合成数据 baseline、评分指标、视频读取、manifest、数据审计、训练、评测、可视化等脚本。
- 已完成提交材料母稿目录:`submission_fall/`。
- 已将 `data/`、`datasets/`、`raw_data/` 加入 `.gitignore`,避免原始数据误提交。

## 数据状态

OmniFall 已落地:

- HuggingFace 轻量文件:292/292 个文件下载完成,约 46MB。
- OF-Syn 视频归档:`data/omnifall/data_files/omnifall-synthetic_av1.tar`。
- 视频归档大小:`9716480000` 字节。
- 已抽取小样本:`data/omnifall/of_syn_sample/`。
- 小样本规模:72 个 MP4,其中 24 个 fall/fallen 正类,48 个日常动作负类。
- 覆盖动作:fall、fallen、walking、standing、sitting、sit_down、lie_down、lying、stand_up、other。
- 小样本 manifest:`output/fall_omnifall_ofsyn_sample_manifest.csv`。
- 小样本审计:`output/fall_omnifall_ofsyn_sample_audit.json`,结果为 `ready_for_eval=true`。

Kaggle Fall Video Dataset 暂未落地:

- 本机没有 Kaggle token/CLI。
- 当前赛题给出的公开 URL 访问返回 404。
- 后续需要有效 Kaggle 账号或更新后的 dataset slug。

## 当前实验结果

合成代理 baseline:

- 100 个合成片段:20 fall / 80 non-fall。
- P@R90 = 1.0。
- P@R95 = 1.0。
- 说明:只证明工程链路和指标计算可跑,不能当作公开或官方成绩。

OmniFall OF-Syn 小样本规则 baseline:

- 72 个真实 MP4:24 fall / 48 non-fall。
- 读取失败:0。
- P@R90 = 0.342857。
- P@R95 = 0.342857。
- 平均规则后处理耗时约 0.23ms/frame。
- 报告:`output/fall_pipeline_report.manifest.json`。

OmniFall OF-Syn 小样本 tiny model:

- 模型:`output/fall_tiny_model.manifest.json`。
- 训练报告:`output/fall_tiny_model_report.manifest.json`。
- 评测报告:`output/fall_pipeline_report.manifest.tiny_model.json`。
- P@R90 = 0.338028。
- P@R95 = 0.338028。
- 结论:tiny model 没有优于规则 baseline,说明当前几何代理特征表达力不足。

## 关键代码产物

- `src/fall_metrics.py`:赛题评分指标。
- `src/fall_synth.py`:合成跌倒/日常动作片段。
- `src/fall_features.py`:人体框和几何时序特征。
- `src/fall_detector.py`:规则后处理 baseline。
- `src/fall_video_io.py`:视频/帧读取,已支持 imageio 失败后用 ffmpeg 兜底。
- `src/fall_public_datasets.py`:公开视频目录生成 manifest。
- `src/fall_data_audit.py`:manifest 数据体检。
- `src/train_fall_model.py`:tiny model 训练入口。
- `src/run_fall_pipeline.py`:端到端评测入口。
- `src/viz_fall_report.py`:分数分布、PR 曲线和示例拼图。

## 验证情况

已运行:

```bash
cd /Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛/src
MPLCONFIGDIR=/private/tmp/ai04_mpl bash run_all_selftests.sh
```

结果:

- `04 全部自测通过`。
- 新增 ffmpeg fallback 后再次回归通过。

## 下一步建议

1. 基于 OmniFall splits 扩大 train/val/test,不要只停留在 72 个小样本。
2. 接入轻量人体检测或姿态估计模型,替换当前前景阈值代理。
3. 加入真正时序模型,如 tiny TCN、GRU 或轻量 Transformer head。
4. 重点分析 `lie_down`、`lying`、`sit_down`、`sitting` 误报,这是赛题核心痛点。
5. 补官方参考数据或 Kaggle 数据,尤其是红外、遮挡、弱光、视角变化场景。
6. 后续导出 ONNX/RKNN/NCNN,补真实端侧参数量、fp32 大小、NPU 内存和延时。

## 当前风险

- 规则 baseline 在真实视频上精度偏低,不能作为最终方案。
- tiny model 当前只用 8 个几何特征,表达力不够。
- Kaggle 数据还未拿到。
- 端侧部署尚未验证,当前延时不包含神经网络检测器和 NPU 全链路。
- 官方非公开测试集会包含长尾场景,需要数据增强和真实模型支撑。
