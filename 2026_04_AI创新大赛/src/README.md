# AI创新(华为杯)src · 两条线:华为AOI质检 + 开放AI+X违建

> ⚠️ 据官方纠偏:华为赛题一是**通用 AOI 组装质检**(few-shot 100正+30负→测1000+,**非**自监督LED芯片);
> 硬指标 **单图<200ms@2060 / CPU<2s,检测时间占竞赛分30%** → 用 **PatchCore/PaDiM 小模型(特征memory bank+最近邻)**,不堆大检测器。作品截止 **9/1**。

## 线A · 华为 AOI 异常检测(冲华为专项奖10万池)
| 文件 | 作用 | 复用 |
|---|---|---|
| `anomaly_score.py` | memory bank 最近邻异常打分(image=patch距离取max) | 02/similarity 思想 |
| `patchcore_lite.py` | memory bank + greedy coreset 子采样(降内存/加速,利于<200ms) | — |
| `aoi_metrics.py` | 阈值P/R/F1/AUC + **官方竞赛分加权**(方案50%+准确率20%+检测时间30%) | 02/metrics |
| `fewshot_protocol.py` | 严格复刻官方 **100正+30缺→测1000+** 评测协议 + per-image延时 | 02/metrics+prepare 思路 |
| `aoi_prepare.py` | manifest→few-shot切分(100正+30缺/测试1000+)+缺陷分布+合成清单 | 02/prepare 思路 |
| `augment_defect.py` | 合成缺陷增广(划痕/污点/缺件/色变)+bbox,解异常样本稀缺 | 03/augment_water 思想 |
| `latency_bench.py` | 端到端分档计时,<200ms@GPU / CPU<2s 红线判定(检测时间30%) | 03/trt 计时框架 |
| `synth_aoi.py` | **PIL 程序化合成 AOI 工件图**(正常组装件纹理 + 4 类缺陷,复用 augment_defect),无真数据先端到端联调 | — |
| `feature_backend.py` | **特征后端**:真特征 `TimmBackend`(torch/timm 中层 feature,接口保留)+ **经典 CPU 回退** `ClassicBackend`(分块颜色 mean/std + 梯度方向直方图,L2 归一化) | — |
| `run_real_pipeline.py` | **真实合成图上的端到端 few-shot 流水线**:synth→特征→coreset 建库→阈值校准→测试,产出**真实 AUC/F1/per-class 检出/延时/竞赛分** + 落盘 report/scores | 串起全部上游模块 |
| `viz_heatmap.py` | **matplotlib 异常可视化**:正常 vs 4 类缺陷的 patch 异常**热力叠加图** + 异常分分布直方图 + ROC/PR 曲线 | — |
| `online_learning.py` | **用户反馈驱动在线/主动学习闭环**(官方明列三大问题之一):误检→正常 patch 增量并入 memory bank、漏检→阈值在"历史校准+反馈"上重标定,产出"反馈次数→F1"提升曲线。纯 numpy,无需重训 | — |

### 真实闭环(本机即可跑出交付物,非随机特征)
之前 anomaly_score/patchcore_lite 跑在随机 numpy 特征上;现已用 **synth_aoi 合成真实工件图 +
feature_backend 经典 CPU 特征** 让 `feature_backend → patchcore_lite(coreset 建库)→ anomaly_score(最近邻打分)
→ fewshot_protocol/aoi_metrics(评测)→ viz_heatmap(定位可视化)` 全链路在真实图像上端到端跑通。

**本机实测(标准规模 100正+30缺→测600,经典 CPU 特征 baseline,训练/测试件无泄漏):**
- AUC = **0.9967**,F1 = **0.9471**,Recall = **0.985**,Precision 0.912,Acc 0.9633
- per-class 检出:scratch 1.0 / spot 1.0 / missing 0.94 / discolor 1.0
- 打分延时 ~0.1ms/图(160px,仅最近邻打分阶段);report 里的 `est_latency_2500px_cpu_ms` 是把这一亚毫秒打分延时按面积(2500/160)²≈244 线性外推,故被计时噪声放大、run 间在 ~0.6–1.4s 间抖动,**不是真实端到端延时**,仅供量级参考。**真实 2500px 单图端到端延时以 `bench_latency_gpu.py --size 2500` 直接实测为准 ≈245ms < 2s(见下文 CPU 档实测)。**
- `--full` 模式测试集放大到 1020 张(贴合官方 1000+ 口径):AUC 0.9968 / F1 0.9431 / Recall 0.9844
- 产物:`output/heatmap_overlay.png`(正常 vs 4 缺陷热力叠加,GT 框对齐缺陷热点)、
  `output/score_distribution.png`、`output/roc_pr_curves.png`、`output/pipeline_report.json`、`output/pipeline_scores.npz`
> 注:此前数据因 `gen_dataset` 忽略 seed 致训练/测试件完全相同(泄漏)已修复;后又发现件级 seed
> 在大顶层 seed 下仍可能跨流 alias(正常/缺陷同源),已用奇偶编码彻底隔离 → 上为隔离后的诚实指标。

> torch/timm 到位后:`get_backend(prefer_real=True)` 自动切 `TimmBackend`(真特征),下游代码零改动。
> 当前经典 CPU 特征明确标注为 **baseline**,验证的是流水线与协议正确性,真特征只换 backend。

## GPU 上机三步走(小队照着跑,启用真特征 + 测 <200ms 红线)
`feature_backend.py` 的 `TimmBackend` **已是完整可跑实现**(不再占位):加载 timm 预训练
WideResNet/ResNet → `features_only` 取 layer2+layer3 中层特征图 → 双线性对齐+通道拼接 →
重采样到 grid×grid 网格 → 逐 patch L2 归一化,返回签名 `(grid,grid,D)` / `(P,D)` 与
`ClassicBackend` **严格一致**,下游零改动。本机无 torch 时 `__init__` 内 `import torch`
抛 `ImportError` → `get_backend` 回退经典后端(本机自测即走此路径)。GPU 机上:

| 文件 | 作用 |
|---|---|
| `requirements-gpu.txt` | 真特征档依赖清单(torch/timm/版本建议 + CUDA 轮子说明 + Jetson 提示) |
| `setup_gpu.sh` | **一键**:装 torch+timm+依赖 → 预缓存权重 → 跑 feature_backend 自测 → 真特征端到端 → 2500px 延时实测 |
| `bench_latency_gpu.py` | 2500×2500 **真实流水线**单图延时分档实测,自动选档(GPU<200ms / CPU<2s)给 PASS/FAIL |

```bash
# ① 一键装环境 + 自测 + 出真特征指标 + 测延时(2060+ x86 GPU,自动探测 CUDA 轮子)
bash setup_gpu.sh
#    Jetson Orin Nano:JETSON=1 bash setup_gpu.sh(先按 NVIDIA 官方 Jetson wheel 装 torch)
#    指定 CUDA / 换小 backbone:CUDA=cu118 bash setup_gpu.sh   |   MODEL=resnet18 bash setup_gpu.sh

# ② 单独看真特征档端到端真实指标(is_real_feature 应为 true)
python3 run_real_pipeline.py            # get_backend 自动切 TimmBackend

# ③ 单独测 2500×2500 单图延时 vs <200ms@2060 红线
python3 bench_latency_gpu.py --size 2500 --runs 20
```

**本机 CPU 档实测(无 GPU,ClassicBackend baseline,`bench_latency_gpu.py --size 2500`):**
单图总延时 mean≈**245ms** / p95≈251ms,**< 2s CPU 红线 PASS**(特征 245ms + 打分 0.05ms)。
GPU 档 `<200ms@2060` 须在 2060 级显卡上跑 `bench_latency_gpu.py` 复测(脚本自动切真特征档判 PASS/FAIL)。

## 线B · 开放 AI+X 违建(冲大赛主奖,低成本第二投)
| 文件 | 作用 | 复用 |
|---|---|---|
| `illegal_build_pipeline.py` | 检测→时序滤波→像素GPS→治理决策(派巡查航点)demo | **直接复用 03 track_filter+geolocate(已自测)** |

`run_all_selftests.sh` 一键回归(14 模块,本机 numpy/PIL/sklearn/matplotlib 全可跑)。

## 运行
```bash
bash run_all_selftests.sh                 # 14 模块一键自测(含合成图真实端到端 + 2500px CPU 档延时)
python3 run_real_pipeline.py              # 真实合成图端到端,打印真实 AUC/F1 + 落盘 report
python3 run_real_pipeline.py --full       # 测试集放大到 1000+(贴合官方口径)
python3 viz_heatmap.py                    # 生成异常热力图/分布图/ROC-PR 曲线到 output/
python3 synth_aoi.py --gen-dir out --n-normal 60 --n-defect 40   # 合成数据集落盘(normal/defect/manifest.csv)
# 真数据(报名后从 chaspark 领):aoi_prepare 切分 → get_backend(真特征 timm)提特征 → run_real_pipeline 评测
```

## 阻塞(你来/等官方)
- 华为真实数据/baseline/提交入口/精确评测脚本在 **chaspark**,需队长华为账号报名后下载;当前用公开 DAGM2007/MVTec AD 顶替(需注册/协议)。
- `<200ms@2060` 本机无 NVIDIA GPU 无法复现:CPU 档已实测(`bench_latency_gpu.py --size 2500` ≈245ms<2s PASS);GPU 档在 2060 级显卡上 `bash setup_gpu.sh` 一键装 torch/timm 后**重跑 `bench_latency_gpu.py`** 即出真特征档 PASS/FAIL。真特征需 torch+timm 权重+算力(上机包已备:`setup_gpu.sh`/`requirements-gpu.txt`)。
- 赛道定夺(华为专项奖 vs 开放主奖)+报名(cpipc,≤3人,8/25)需你拍板。
