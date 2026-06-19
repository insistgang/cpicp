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

## 线B · 开放 AI+X 违建(冲大赛主奖,低成本第二投)
| 文件 | 作用 | 复用 |
|---|---|---|
| `illegal_build_pipeline.py` | 检测→时序滤波→像素GPS→治理决策(派巡查航点)demo | **直接复用 03 track_filter+geolocate(已自测)** |

`run_all_selftests.sh` 一键回归(本机 numpy/标准库全可跑)。

## 运行
```bash
bash run_all_selftests.sh
# 真数据(报名后从 chaspark 领):aoi_prepare 切分 → timm/CLIP 提特征 → fewshot_protocol 评测
```

## 阻塞(你来/等官方)
- 华为真实数据/baseline/提交入口/精确评测脚本在 **chaspark**,需队长华为账号报名后下载;当前用公开 DAGM2007/MVTec AD 顶替(需注册/协议)。
- `<200ms@2060` 本机无 NVIDIA GPU 无法复现,CPU 自测 + 异地 GPU 补测;真特征需 torch+timm/CLIP 权重+算力。
- 赛道定夺(华为专项奖 vs 开放主奖)+报名(cpipc,≤3人,8/25)需你拍板。
