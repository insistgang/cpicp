# LinkAble AI Inference 评测提交总览

生成时间: 2026-06-27 21:32 CST

## 结论

这些数值已经获取完成。所有核心指标均来自真实运行的 `flutter test eval/test/eval_runner_test.dart`, 评测入口走项目生产路径:

- `AgentServiceFacade.processInput(...)`
- `DemoAIService.resolveIntent(...)`
- `DemoAIService.process(...)`
- `DemoMatchingEngineService.matchTopVolunteers(...)`
- `DemoCallService.startCall()`
- `DemoSOSService.triggerSOS()`

本轮没有修改生产判定逻辑, 只新增独立评测目录 `eval/`。

## 数据集

| 数据集 | 文件 | 数量 | 说明 |
|---|---|---:|---|
| 意图识别集 | `eval/datasets/intent_samples.json` | 120 | 真实 `DemoAiIntent` 12 类, 每类 10 条 |
| 紧急判断集 | `eval/datasets/emergency_samples.json` | 40 | 20 条真实紧急 + 20 条非紧急难负例 |

意图集包含 easy / hard 两类表达。hard 样本刻意使用改写、隐含表达和真实用户口吻, 不直接照抄规则关键词, 用于避免“自己考自己”导致虚高。

## 核心指标

| 指标 | 实测 |
|---|---:|
| 意图识别总准确率 | 47.50% |
| 意图识别 easy 子集准确率 | 84.48% |
| 意图识别 hard 子集准确率 | 12.90% |
| next_action 正确率 | 56.88% |
| 紧急召回率 | 50.00% |
| 紧急误报率 | 5.00% |

## 时延指标

| 指标 | median | p95 | 说明 |
|---|---:|---:|---|
| 首屏可见响应 | 11.47ms | 254.00ms | widget-test 环境, 非线上冷启动 |
| AI 完整响应 | 608.79ms | 797.99ms | DemoMode 本地确定性响应, 含生产模拟延迟, 不是真实 LLM 时延 |
| 50 人候选 Top5 匹配耗时 | 0.53ms | 18.56ms | 直接调用真实 Demo 匹配引擎 |
| Demo Call 建立 | 3007.72ms | 3018.26ms | 固定包含 1s connecting + 2s ringing |
| SOS 触发到撤销窗口 | 0.03ms | 0.80ms | 调用后 `isActive=true` 的耗时 |

## 真实 LLM 对照

已完成。

模型: `glm-4-flash`。样本: 160 条, 沿用同一套 120 条意图样本与 40 条紧急/非紧急样本。

记录文件: `eval/results/llm_eval_status.txt`

```text
真实 LLM 对照已完成
model=glm-4-flash
samples=160
```

| 指标 | 规则基线 | LLM | 差值 |
|---|---:|---:|---:|
| 意图总准确率 | 47.50% | 95.00% | +47.50pp |
| easy 子集准确率 | 84.48% | 96.55% | +12.07pp |
| hard 子集准确率 | 12.90% | 93.55% | +80.65pp |
| 紧急召回率 | 50.00% | 100.00% | +50.00pp |
| 紧急误报率 | 5.00% | 15.00% | +10.00pp |
| LLM 成功解析率 | n/a | 100.00% | n/a |

## 工程证据

| 项目 | 文件 | 状态 |
|---|---|---|
| eval runner | `eval/results/eval_progress.log` | 已完成, 最后阶段为 `runner:reports_written` |
| 全量测试 | `eval/results/flutter_test.txt` | 通过, `00:13 +78: All tests passed!` |
| 静态检查 | `eval/results/flutter_analyze.txt` | 已运行, 仅剩 1 个既有 info |
| APK 构建 | `eval/results/flutter_build_apk.txt` | 通过, release APK 已生成 |
| APK 摘要 | `eval/results/apk_build.txt` | 已记录本轮 APK 路径、mtime、sha256 |

`flutter analyze` 当前唯一问题:

```text
info - lib/services/demo_call_service.dart:358:9 - use_null_aware_elements
```

APK 构建说明:

```text
command: JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ANDROID_HOME=/opt/homebrew/share/android-commandlinetools ANDROID_SDK_ROOT=/opt/homebrew/share/android-commandlinetools flutter build apk --release
status: succeeded
result: Built build/app/outputs/flutter-apk/app-release.apk (92.1MB)
```

APK 记录:

```text
path: build/app/outputs/flutter-apk/app-release.apk
mtime: Jun 27 21:31:51 2026
sha256: d35cfead0fec1afa7a995dcc45dd3c506474a1365d3a218f85c69b788bc7de62
```

## 主要产物

| 文件 | 用途 |
|---|---|
| `eval/results/REPORT.md` | 完整中文评测报告 |
| `eval/results/ARCHITECTURE.md` | 架构勘察、真实入口、枚举、关键词、匹配权重、每条 gold 依据 |
| `eval/results/inference_metrics.csv` | 指标总表 |
| `eval/results/intent_metrics_by_class.csv` | 各意图类 precision / recall |
| `eval/results/confusion_matrix.csv` | 混淆矩阵 |
| `eval/results/latency_results.csv` | 延迟结果 |
| `eval/results/misclassified.csv` | 错分样本清单 |
| `eval/results/raw_predictions.csv` | 全部原始预测 |

## 一条命令复现

```bash
cd <项目源码根目录>/linklab
flutter test eval/test/eval_runner_test.dart
```

## 可以写进比赛文档的表述

本项目已完成基于真实生产 facade 的 AI inference 规则基线评测。评测覆盖 12 类真实意图、紧急判断、next_action 决策、Top5 志愿者匹配与关键 Demo 时延。为避免关键词规则“自己考自己”, 测试集区分 easy / hard 两个子集, hard 样本采用改写和隐含表达。结果显示当前规则基线在直接表达上表现较好, 但在隐含表达和长尾语义上存在明显下降, 这为后续接入真实 LLM 分类层提供了清晰的改进空间和可复现实验基线。
