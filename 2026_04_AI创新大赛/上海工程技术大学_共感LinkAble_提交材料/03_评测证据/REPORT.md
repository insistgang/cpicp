# LinkAble AI Inference 评测报告

生成时间:2026-06-27T10:14:57.290765

## 结论摘要

本报告保留上一轮 DemoMode 规则基线结论，并新增本轮真实 LLM 对照 runner。规则基线仍为可离线复现的确定性结果；真实 LLM 对照状态为 `completed`，原因/摘要为：真实 LLM 对照已完成，模型 glm-4-flash，样本 160 条。所有 LLM 数字只在真实 `glm-4-flash` 调用完成后写入，缺配置或请求失败时不估算、不回退 Demo。

## 架构事实摘要

- 统一生产入口: `AgentServiceFacade.processInput({String? text, String? imagePath, String inputType = "text"}) -> Future<AgentResult>`。
- 规则意图入口: `DemoAIService.resolveIntent(String input, {String? imagePath, List<Map<String,String>>? history}) -> DemoAiIntent`。
- 规则综合处理入口: `DemoAIService.process(String input, {String? imagePath, List<Map<String,String>>? history}) -> Future<AIResult>`。
- 实际意图枚举共 12 类: color_recognition, emergency, environment_description, fallback, medication_check, money_recognition, navigation, need_human, object_identify, ocr_text, scene_description, translation。
- 实际 `urgency` 输出: `normal`, `elevated`, `emergency`; 实际 `next_action` 输出包含 `answer`, `ask_followup`, `match_volunteer`, `trigger_sos`, 错误时可能为 `show_fallback`。
- 志愿者 Top5 匹配入口: `DemoMatchingEngineService.matchTopVolunteers(...)`; 权重为 availability 0.30, distance 0.25, skill 0.20, trust 0.15, reputation 0.10。
- 更完整的入口、关键词表、匹配权重和每条 gold 判定依据见 `eval/results/ARCHITECTURE.md`。

## 测试集设计与防循环说明

- 意图集: `eval/datasets/intent_samples.json`, 共 120 条, 覆盖真实 12 个 `DemoAiIntent` 类, 每类 10 条。
- 意图 easy/hard 分布: easy 58 条, hard 62 条; 每类 hard 至少 4 条。
- 紧急集: `eval/datasets/emergency_samples.json`, 共 40 条, 其中真实紧急 20 条, 非紧急难负例 20 条。
- 防“自己考自己”: hard 样本刻意使用真实口吻、改写和隐含表达, 不照抄代码关键词; easy 样本保留直接表达, 用于形成 easy/hard 对照。
- 所有预测均调用生产 facade, 没有在评测脚本中重写关键词规则或另造分类器。
- 本环境缺少 `openpyxl/xlsxwriter` 等 xlsx 写入库, 指标按用户允许的 CSV 形式输出。

## 指标总表

| 指标 | 目标 | 实测 | 备注 |
|---|---|---|---|
| 意图识别总准确率 | 越高越好；按真实 DemoAiIntent 12 类 | 47.50% | n=120, DemoMode 规则基线 |
| 意图识别 easy 子集准确率 | 越高越好 | 84.48% | 直接/显式表达样本 |
| 意图识别 hard 子集准确率 | 越高越好 | 12.90% | 改写/隐含表达样本；用于防止关键词自测虚高 |
| next_action 正确率 | 越高越好 | 56.88% | 覆盖意图集+紧急集, 对比 trigger_sos/match_volunteer/answer/ask_followup |
| 紧急召回率 | 目标高召回；不得漏 SOS | 50.00% | 10/20 真实紧急被命中 |
| 紧急误报率 | 越低越好 | 5.00% | 1/20 非紧急被误判紧急 |
| 首屏可见响应 | 每项>=20次；报 median/p95 | median=11.47ms; p95=254.00ms | 包含 initializeCompetitionDemoApp + pumpWidget 到首页主文案可见；非线上冷启动。 |
| AI 完整响应 | 每项>=20次；报 median/p95 | median=608.79ms; p95=797.99ms | 走 AgentServiceFacade -> DemoAIService；含生产模拟延迟，不是真实 LLM 时延。 |
| 50 人候选 Top5 匹配耗时 | 每项>=20次；报 median/p95 | median=0.53ms; p95=18.56ms | 直接调用 DemoMatchingEngineService.matchTopVolunteers, volunteerPool=50。 |
| Demo Call 建立 | 每项>=20次；报 median/p95 | median=3007.72ms; p95=3018.26ms | DemoCallService.startCall 固定包含 1s connecting + 2s ringing。 |
| SOS 触发到撤销窗口 | 每项>=20次；报 median/p95 | median=0.03ms; p95=0.80ms | 测量 DemoSOSService.triggerSOS 调用后 isActive=true；撤销窗口 UI 文案在页面层展示。 |

## 各类 precision / recall

| class | precision | recall | tp | fp | fn | support |
|---|---:|---:|---:|---:|---:|---:|
| color_recognition | 1.000 | 0.400 | 4 | 0 | 6 | 10 |
| emergency | 1.000 | 0.600 | 6 | 0 | 4 | 10 |
| environment_description | 0.714 | 0.500 | 5 | 2 | 5 | 10 |
| fallback | 0.169 | 1.000 | 10 | 49 | 0 | 10 |
| medication_check | 1.000 | 0.500 | 5 | 0 | 5 | 10 |
| money_recognition | 1.000 | 0.500 | 5 | 0 | 5 | 10 |
| navigation | 1.000 | 0.400 | 4 | 0 | 6 | 10 |
| need_human | 1.000 | 0.500 | 5 | 0 | 5 | 10 |
| object_identify | 0.000 | 0.000 | 0 | 0 | 10 | 10 |
| ocr_text | 0.667 | 0.400 | 4 | 2 | 6 | 10 |
| scene_description | 0.333 | 0.500 | 5 | 10 | 5 | 10 |
| translation | 1.000 | 0.400 | 4 | 0 | 6 | 10 |

## 混淆矩阵文件

- `eval/results/confusion_matrix.csv`
- 标签顺序:color_recognition, emergency, environment_description, fallback, medication_check, money_recognition, navigation, need_human, object_identify, ocr_text, scene_description, translation

## 时延说明

- 首屏可见响应: median=11.47ms, p95=254.00ms。包含 initializeCompetitionDemoApp + pumpWidget 到首页主文案可见；非线上冷启动。
- AI 完整响应: median=608.79ms, p95=797.99ms。走 AgentServiceFacade -> DemoAIService；含生产模拟延迟，不是真实 LLM 时延。
- 50 人候选 Top5 匹配耗时: median=0.53ms, p95=18.56ms。直接调用 DemoMatchingEngineService.matchTopVolunteers, volunteerPool=50。
- Demo Call 建立: median=3007.72ms, p95=3018.26ms。DemoCallService.startCall 固定包含 1s connecting + 2s ringing。
- SOS 触发到撤销窗口: median=0.03ms, p95=0.80ms。测量 DemoSOSService.triggerSOS 调用后 isActive=true；撤销窗口 UI 文案在页面层展示。

## LLM 对照

- 本轮新增 eval-only 真实 LLM runner: `flutter test eval/test/llm_eval_runner_test.dart`。
- 状态: completed。真实 LLM 对照已完成，模型 glm-4-flash，样本 160 条。
- 真实 LLM 路径不使用 `AgentServiceFacade._chatWithLLM()` 的回答型结果作为分类器；本 runner 调用智谱 `glm-4-flash` Chat Completions，并强制输出 12 类同口径 JSON。

## 错分样本与 limitation

完整错分清单见 `eval/results/misclassified.csv`。前 20 条如下:

| id | text | gold | pred | gold_action | pred_action | note |
|---|---|---|---|---|---|---|
| intent_ocr_text_003 | 路边牌子写了什么,念给我听 | ocr_text/normal | fallback/elevated | answer | ask_followup | 真实用户表达为念出牌子内容,语义上是读文字。 |
| intent_ocr_text_004 | 包装背面的那几行小字太密了,帮我确认一下 | ocr_text/normal | fallback/elevated | answer | ask_followup | 隐含文字识别,没有直接照抄规则中的关键词。 |
| intent_ocr_text_005 | 请把收据上的项目和金额念出来 | ocr_text/normal | fallback/elevated | answer | ask_followup | 收据内容读取属于 OCR,即使没有出现 OCR 字样。 |
| intent_ocr_text_006 | 我拍了个快递单,上面单号是多少 | ocr_text/normal | fallback/elevated | answer | ask_followup | 快递单号读取是文字识别任务。 |
| intent_ocr_text_008 | 这个票据上的抬头和日期是什么 | ocr_text/normal | fallback/elevated | answer | ask_followup | 票据字段读取是 OCR 类任务。 |
| intent_scene_description_003 | 这张照片大概是在什么地方 | scene_description/normal | fallback/elevated | answer | ask_followup | 隐含对画面整体语义的描述。 |
| intent_scene_description_004 | 帮我概括一下镜头里都有啥 | scene_description/normal | fallback/elevated | answer | ask_followup | 口语化画面概括请求,应归为场景描述。 |
| intent_scene_description_005 | 我想知道图片里是室内还是街上 | scene_description/normal | fallback/elevated | answer | ask_followup | 判断图片环境类别,属于场景描述。 |
| intent_scene_description_008 | 这是不是在公园附近,周围有些什么 | scene_description/normal | environment_description/normal | answer | match_volunteer | 图片/周边整体理解,人工标为场景描述。 |
| intent_object_identify_001 | 这个物体是什么 | object_identify/normal | scene_description/normal | answer | answer | 带图片路径且询问物体,应走物体识别入口。 |
| intent_object_identify_002 | 帮我看看这个东西是啥 | object_identify/normal | scene_description/normal | answer | answer | 图片中的东西识别,对应 object_identify。 |
| intent_object_identify_003 | 桌上这个圆圆的能不能告诉我是什么 | object_identify/normal | scene_description/normal | answer | answer | 隐含物体识别,没有直接使用物体关键词。 |
| intent_object_identify_004 | 我摸到一个盒子,拍给你看,它可能是什么 | object_identify/normal | scene_description/normal | answer | answer | 拍图确认物件身份,属于物体识别。 |
| intent_object_identify_005 | 这个商品包装是哪一类东西 | object_identify/normal | scene_description/normal | answer | answer | 商品/包装识别在图片路径下应归为物体识别。 |
| intent_object_identify_006 | 请判断照片里的用品是不是杯子 | object_identify/normal | scene_description/normal | answer | answer | 判断具体用品类别,属于 object_identify。 |
| intent_object_identify_007 | 产品外观看起来像什么 | object_identify/normal | scene_description/normal | answer | answer | 产品识别是物体识别分支关键词覆盖范围。 |
| intent_object_identify_008 | 这件小工具我分不出来,帮忙辨认 | object_identify/normal | scene_description/normal | answer | answer | 辨认小工具身份,语义属于物体识别。 |
| intent_color_recognition_003 | 我想确认这条线偏深还是偏浅 | color_recognition/normal | fallback/elevated | answer | ask_followup | 隐含色彩/明暗判断。 |
| intent_color_recognition_004 | 这个标签看起来是哪种色调 | color_recognition/normal | fallback/elevated | answer | ask_followup | 询问色调属于颜色识别。 |
| intent_color_recognition_005 | 帮我分辨两瓶药外包装的色差 | color_recognition/normal | fallback/elevated | answer | ask_followup | 色差辨别是颜色识别。 |

主要限制:规则引擎对 hard 子集的改写/隐含表达天然更弱;物体识别等图片类任务依赖图片路径和视觉适配层;DemoMode AI 响应时延包含模拟延迟,不代表真实大模型服务耗时。

## 工程证据材料

- `eval/results/flutter_analyze.txt`: `flutter analyze` 输出文件。当前已知仅剩 1 个既有 info: `lib/services/demo_call_service.dart:358 use_null_aware_elements`。
- `eval/results/flutter_test.txt`: 全量 `flutter test` 输出文件。最近一次通过时结尾为 `All tests passed!`。
- `eval/results/flutter_build_apk.txt`: `flutter build apk --release` 输出文件。当前 release 构建已成功, 产物为 `build/app/outputs/flutter-apk/app-release.apk`。
- `eval/results/apk_build.txt`: 记录本轮 APK 的路径、mtime、sha256, 提交包内 APK 位于 `04_APK/app-release.apk`。

## 一条命令复现

```bash
cd <项目源码根目录>/linklab
flutter test eval/test/eval_runner_test.dart
```

## 真实 LLM 对照（本轮增量）

生成时间:2026-06-27T11:54:56.983322

- 状态: completed。真实 LLM 对照已完成，模型 glm-4-flash，样本 160 条。
- 模型路径: eval-only runner -> 智谱 Chat Completions `glm-4-flash`。
- 样本: 沿用 `eval/datasets/intent_samples.json` 120 条与 `eval/datasets/emergency_samples.json` 40 条，未改动 gold 标注。
- 图片样本: prompt 只传 `has_image=true` 与用户文本，不发送 dummy 图片字节，避免把占位图当真实视觉证据。

### LLM 指标摘要

- 意图总准确率: 95.00%
- easy / hard: 96.55% / 93.55%
- 紧急召回率: 100.00%
- 紧急误报率: 15.00%
- next_action 正确率: 51.25%
- 成功解析率: 100.00% (160/160)

### 规则 vs LLM 对照

| metric | 规则基线 | LLM | 差值 | note |
|---|---:|---:|---:|---|
| intent_accuracy | 47.50% | 95.00% | +47.50pp | 意图集总准确率 |
| easy_intent_accuracy | 84.48% | 96.55% | +12.07pp | 意图 easy 子集 |
| hard_intent_accuracy | 12.90% | 93.55% | +80.65pp | 意图 hard 子集，最能体现改写/隐含表达差距 |
| next_action_accuracy | 56.88% | 51.25% | -5.63pp | 意图集+紧急集 |
| emergency_recall | 50.00% | 100.00% | +50.00pp | 真实紧急样本命中率 |
| emergency_false_positive_rate | 5.00% | 15.00% | +10.00pp | 真实非紧急被误判紧急比例 |
| llm_success_rate | n/a | 100.00% | n/a | 160/160 条成功解析 |
| llm_latency_median_ms | n/a | 3697.28 | n/a | 真实 LLM HTTP 调用中位耗时 |
| llm_latency_p95_ms | n/a | 6234.10 | n/a | 真实 LLM HTTP 调用 p95 耗时 |

完整文件: `eval/results/rule_vs_llm.csv`、`eval/results/llm_raw_predictions.csv`、`eval/results/llm_inference_metrics.csv`。

### object_identify 诊断结论

`object_identify` 0% 是生产规则 limitation: 评测脚本已传 `imagePath` 与 `inputType=mixed`，但 `DemoAIService._detectDemoIntent()` 图片分支没有 object 判断，默认落到 `scene_description`。详见 `eval/results/ARCHITECTURE.md` 的同名小节。

### 紧急召回 gap 分析

规则基线真实紧急漏报样本数: 10。逐条分析见 `eval/results/emergency_gap_analysis.csv`。
- LLM 接住: 10 条；两者都漏: 0 条；LLM 无输出: 0 条。

后续改进方向: 规则层可补组合风险特征（夜间+迷路+低电量、被跟随、被困、呼吸困难、他人倒地无回应），并把 LLM 作为 hard 子集和低置信样本的补充分流层；任何规则改动都需要用同一测试集前后复测。

### 复现命令

```bash
cd <项目源码根目录>/linklab
flutter test eval/test/llm_eval_runner_test.dart
# 规则基线刷新: flutter test eval/test/eval_runner_test.dart
```
