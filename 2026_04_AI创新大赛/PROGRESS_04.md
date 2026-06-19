# 04 AI 创新大赛 · 项目进展报告

> 报告时间：2026-06-19
> 项目路径：`/Users/insistgang/Downloads/cpicp/2026_04_AI创新大赛`

## 一、项目概况

**赛题**："华为杯"第八届中国研究生人工智能创新大赛（2026）
**双赛道并投**：
- **线A（华为企业赛题一）**：可自学习的 AOI 实时在线 AI 质检 — 冲华为专项奖 10 万池
- **线B（开放赛题 AI+X）**：基于无人机航拍的城市违建智能识别与治理决策系统 — 冲大赛主奖

**核心约束**：
- 报名截止：8/25，作品提交截止：9/1 23:59
- 队伍上限 3 人
- 华为赛题硬指标：单图 <200ms@2060GPU / CPU<2s，检测时间占竞赛分 30%
- 少样本启动：100 正 + 30 负 → 测 1000+

## 二、当前完成度：约 75%

### 已完成（本地可跑通）

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 异常打分核 | `anomaly_score.py` | 自测通过 | PatchCore 最近邻距离打分，纯 numpy |
| Memory Bank 压缩 | `patchcore_lite.py` | 自测通过 | greedy coreset 子采样，降内存加速 |
| 评测指标 | `aoi_metrics.py` | 自测通过 | P/R/F1/AUC + 官方竞赛分加权（方案50%+准确率20%+检测时间30%） |
| 少样本评测协议 | `fewshot_protocol.py` | 自测通过 | 严格复刻 100正+30缺→测1000+ 协议，含 per-image 延时 |
| 数据准备 | `aoi_prepare.py` | 自测通过 | manifest 切分 + 缺陷分布统计 + 合成清单生成 |
| 合成缺陷增广 | `augment_defect.py` | 自测通过 | 划痕/污点/缺件/色变 四种合成缺陷 + bbox |
| 延时基准 | `latency_bench.py` | 自测通过 | 分档计时，CPU<2s 通过，GPU 红线待复测 |
| 违建检测管线 | `illegal_build_pipeline.py` | 自测通过 | 复用 03 的 track_filter + geolocate，时序滤波→GPS 航点 |
| 一键回归 | `run_all_selftests.sh` | 全部通过 | 8 个模块一键跑通 |

### 待完成（需外部条件）

| 事项 | 卡点 | 优先级 |
|------|------|--------|
| 真特征提取（timm ResNet/WideResNet 或 CLIP） | 需 torch + timm 权重 | 高 |
| 在 DAGM2007 / MVTec AD 上跑通端到端 | 需下载数据集（需注册/协议） | 高 |
| GPU <200ms 红线验证 | 本机无 NVIDIA GPU | 中 |
| 华为 chaspark 真实数据/baseline | 需队长华为账号报名后下载 | 高 |
| 在线学习闭环（误检样本入库/阈值重标定） | 需真数据验证 | 中 |
| 报名（cpipc + chaspark） | 需用户拍板队伍/赛道 | 高 |
| 作品文档/PPT/演示视频 | 9/1 提交前完成 | 中 |

## 三、本次修改（2026-06-19）

### 修复 1：`augment_defect.py` — missing 缺陷无变化
**问题**：`missing` 合成缺陷用 `out.mean()` 填充块，当原图是纯色（均值=块均值）时图像无变化，导致自测失败。
**修复**：计算块均值后，用 `block_mean ± 20` 确保填充值与当前块不同，保证图像确实被改动。

### 修复 2：`anomaly_score.py` + `latency_bench.py` — numpy 2.0 false positive 警告
**问题**：numpy 2.0+ 在 `matmul` 中触发 `divide by zero / overflow / invalid` 警告，但计算结果完全正确（无 inf/nan）。
**修复**：用 `np.errstate(divide='ignore', over='ignore', invalid='ignore')` 上下文抑制 false positive，保持代码简洁。

### 修改文件清单
- `src/augment_defect.py` — 修复 missing 缺陷合成逻辑
- `src/anomaly_score.py` — 抑制 numpy 2.0 false positive 警告
- `src/latency_bench.py` — 同上

## 四、运行结果

```bash
$ cd src && bash run_all_selftests.sh
=== anomaly_score === ✅ 自测通过
=== patchcore_lite === ✅ 自测通过
=== aoi_metrics === ✅ 自测通过
=== fewshot_protocol === ✅ 自测通过 (AUC=1.000, recall=1.000, F1=0.985)
=== latency_bench === ✅ 自测通过 (CPU total=21.4ms < 2000ms)
=== illegal_build_pipeline === ✅ 自测通过 (复用 03)
=== aoi_prepare --selftest === ✅ 自测通过
=== augment_defect --selftest === ✅ 自测通过
✅ 04 全部自测通过
```

## 五、仍然的卡点

1. **真数据/真特征**：当前用随机 numpy 特征跑通逻辑，真特征需 `torch + timm/CLIP` 权重 + DAGM2007/MVTec AD 数据集。
2. **GPU 延时验证**：本机无 NVIDIA GPU，`<200ms@2060` 需在异地 GPU 补测。
3. **华为 chaspark 数据**：需队长华为账号登录 https://www.chaspark.com 报名后下载。
4. **报名决策**：需确认 3 人队伍、赛道定夺（华为专项 vs 开放主奖 vs 双报）。
5. **在线学习闭环**：误检样本入库/主动学习阈值调整，需真数据验证效果。

## 六、下一步（需要用户/硬件/数据才能继续）

1. **【用户决策】确认队伍成员 + 报名策略**（华为专项奖 / 开放主奖 / 双报）
2. **【数据】下载 DAGM2007 + MVTec AD**（需注册/同意协议）
3. **【环境】安装 torch + timm，跑通真特征提取**（本机 CPU 可跑，慢但可验证逻辑）
4. **【硬件】在 2060 级 GPU 上验证 <200ms 红线**
5. **【报名】队长华为账号登录 chaspark，下载真实数据/baseline**
6. **【文档】开始撰写项目文档/方案说明**（9/1 提交 deadline）
