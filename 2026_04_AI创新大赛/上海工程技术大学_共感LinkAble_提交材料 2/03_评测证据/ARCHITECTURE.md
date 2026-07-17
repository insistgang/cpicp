# LinkAble AI Inference 架构勘察

生成时间:2026-06-27T10:14:57.289314

## 真实入口

- 规则意图入口:`DemoAIService.resolveIntent(String input, {String? imagePath, List<Map<String,String>>? history}) -> DemoAiIntent`。
- 规则综合处理入口:`DemoAIService.process(String input, {String? imagePath, List<Map<String,String>>? history}) -> Future<AIResult>`。
- 统一生产 facade:`AgentServiceFacade.processInput({String? text, String? imagePath, String inputType = "text"}) -> Future<AgentResult>`。本评测使用该入口输出 intent/urgency/next_action。
- 匹配入口:`DemoMatchingEngineService.matchTopVolunteers(DemoMatchRequest request, {List<DemoVolunteer>? volunteerPool}) -> Future<DemoMatchResponse>`。

## 意图类别枚举

- `ocr_text`:OCR / 读文字
- `scene_description`:场景描述
- `object_identify`:物体识别
- `color_recognition`:颜色识别
- `money_recognition`:钞票 / 面额识别
- `translation`:翻译 / 转译
- `environment_description`:环境描述
- `navigation`:导航 / 找路
- `medication_check`:药品确认
- `emergency`:紧急词检测
- `need_human`:转人工
- `fallback`:兜底回答

## urgency 与 next_action

- urgency 实际输出来自 `AgentResult.urgency`: `normal`, `elevated`, `emergency`。
- next_action 实际输出来自 `AgentResult.nextAction`: `answer`, `ask_followup`, `match_volunteer`, `trigger_sos`, 错误时可能为 `show_fallback`。
- 映射规则:紧急 -> `trigger_sos`; 需要人工/nextStatus=matching -> `match_volunteer`; confidence < 0.65 -> `ask_followup`; 其他 -> `answer`。

## 紧急关键词表

- DemoAIService 内置:`救命`, `晕倒`, `摔倒`, `胸口痛`, `迷路了`, `我很害怕`, `紧急`。
- DemoDataLoader 资源表:`救命`, `着火了`, `有人抢劫`, `心脏病`, `流血不止`, `摔倒了`, `找不到路`, `迷路`, `头晕`, `不舒服`, `快来人`, `紧急求助`, `SOS` 等。

## 匹配评分函数

- 真实 Demo Top5 评分函数:`DemoMatchingEngineService._scoreVolunteer`。
- 权重:availability 0.30, distance 0.25, skill 0.20, trust 0.15, reputation 0.10。
- 排序 tie-break:score 降序 -> distance 升序 -> reputation 降序 -> estimatedResponseSeconds 升序 -> volunteer id。

## 测试调用方法

```bash
cd <项目源码根目录>/linklab
flutter test eval/test/eval_runner_test.dart
```

## 每条 gold 标注依据

| id | gold_intent | gold_urgency | gold_next_action | subset | 判定依据 |
|---|---|---|---|---|---|
| intent_ocr_text_001 | ocr_text | normal | answer | easy | 用户明确要求读取文字/药盒,对应 OCR 读文字能力。 |
| intent_ocr_text_002 | ocr_text | normal | answer | easy | 含通知单和文字读取需求,应走 OCR。 |
| intent_ocr_text_003 | ocr_text | normal | answer | hard | 真实用户表达为念出牌子内容,语义上是读文字。 |
| intent_ocr_text_004 | ocr_text | normal | answer | hard | 隐含文字识别,没有直接照抄规则中的关键词。 |
| intent_ocr_text_005 | ocr_text | normal | answer | hard | 收据内容读取属于 OCR,即使没有出现 OCR 字样。 |
| intent_ocr_text_006 | ocr_text | normal | answer | hard | 快递单号读取是文字识别任务。 |
| intent_ocr_text_007 | ocr_text | normal | answer | easy | 直接出现读一下和告示文字场景。 |
| intent_ocr_text_008 | ocr_text | normal | answer | hard | 票据字段读取是 OCR 类任务。 |
| intent_scene_description_001 | scene_description | normal | answer | easy | 直接询问面前场景,对应场景描述。 |
| intent_scene_description_002 | scene_description | normal | answer | easy | 含前方画面理解需求,对应场景描述。 |
| intent_scene_description_003 | scene_description | normal | answer | hard | 隐含对画面整体语义的描述。 |
| intent_scene_description_004 | scene_description | normal | answer | hard | 口语化画面概括请求,应归为场景描述。 |
| intent_scene_description_005 | scene_description | normal | answer | hard | 判断图片环境类别,属于场景描述。 |
| intent_scene_description_006 | scene_description | normal | answer | easy | 直接出现画面和环境描述需求。 |
| intent_scene_description_007 | scene_description | normal | answer | easy | 直接出现场景,应描述整体布局。 |
| intent_scene_description_008 | scene_description | normal | answer | hard | 图片/周边整体理解,人工标为场景描述。 |
| intent_object_identify_001 | object_identify | normal | answer | easy | 带图片路径且询问物体,应走物体识别入口。 |
| intent_object_identify_002 | object_identify | normal | answer | easy | 图片中的东西识别,对应 object_identify。 |
| intent_object_identify_003 | object_identify | normal | answer | hard | 隐含物体识别,没有直接使用物体关键词。 |
| intent_object_identify_004 | object_identify | normal | answer | hard | 拍图确认物件身份,属于物体识别。 |
| intent_object_identify_005 | object_identify | normal | answer | easy | 商品/包装识别在图片路径下应归为物体识别。 |
| intent_object_identify_006 | object_identify | normal | answer | hard | 判断具体用品类别,属于 object_identify。 |
| intent_object_identify_007 | object_identify | normal | answer | easy | 产品识别是物体识别分支关键词覆盖范围。 |
| intent_object_identify_008 | object_identify | normal | answer | hard | 辨认小工具身份,语义属于物体识别。 |
| intent_color_recognition_001 | color_recognition | normal | answer | easy | 直接询问颜色。 |
| intent_color_recognition_002 | color_recognition | normal | answer | easy | 颜色确认任务。 |
| intent_color_recognition_003 | color_recognition | normal | answer | hard | 隐含色彩/明暗判断。 |
| intent_color_recognition_004 | color_recognition | normal | answer | hard | 询问色调属于颜色识别。 |
| intent_color_recognition_005 | color_recognition | normal | answer | hard | 色差辨别是颜色识别。 |
| intent_color_recognition_006 | color_recognition | normal | answer | hard | 主色判断属于颜色识别。 |
| intent_color_recognition_007 | color_recognition | normal | answer | easy | 直接颜色识别需求。 |
| intent_color_recognition_008 | color_recognition | normal | answer | hard | 颜色确认问句,人工标为 color_recognition。 |
| intent_money_recognition_001 | money_recognition | normal | answer | easy | 直接询问人民币金额。 |
| intent_money_recognition_002 | money_recognition | normal | answer | easy | 含钞票/面额关键词。 |
| intent_money_recognition_003 | money_recognition | normal | answer | easy | 纸币面额识别。 |
| intent_money_recognition_004 | money_recognition | normal | answer | hard | 真实付款场景中的金额确认。 |
| intent_money_recognition_005 | money_recognition | normal | answer | hard | 票子金额是面额识别。 |
| intent_money_recognition_006 | money_recognition | normal | answer | easy | 硬币金额识别。 |
| intent_money_recognition_007 | money_recognition | normal | answer | hard | 隐含现金面额确认。 |
| intent_money_recognition_008 | money_recognition | normal | answer | hard | 现金分辨任务属于 money_recognition。 |
| intent_translation_001 | translation | normal | answer | easy | 含听不清/转译/外卖电话,对应听障沟通转译。 |
| intent_translation_002 | translation | normal | answer | easy | 快递电话沟通协助。 |
| intent_translation_003 | translation | normal | answer | hard | 隐含沟通转写/转译需求。 |
| intent_translation_004 | translation | normal | answer | hard | 听障场景中的转文字沟通。 |
| intent_translation_005 | translation | normal | answer | easy | 含帮我说,对应转译。 |
| intent_translation_006 | translation | normal | answer | hard | 电话沟通表达辅助。 |
| intent_translation_007 | translation | normal | answer | easy | 听障沟通是 translation 分支。 |
| intent_translation_008 | translation | normal | answer | hard | 语义转写/表达重组,人工标为 translation。 |
| intent_environment_description_001 | environment_description | normal | match_volunteer | easy | 环境安全/障碍物提示需要真人确认,生产映射应转志愿者。 |
| intent_environment_description_002 | environment_description | normal | match_volunteer | easy | 路况和人流判断属于环境描述。 |
| intent_environment_description_003 | environment_description | normal | match_volunteer | hard | 隐含移动安全检查。 |
| intent_environment_description_004 | environment_description | normal | match_volunteer | hard | 通行安全判断,应为环境描述并建议人工确认。 |
| intent_environment_description_005 | environment_description | normal | match_volunteer | hard | 周边动态风险提示。 |
| intent_environment_description_006 | environment_description | normal | match_volunteer | easy | 直接命中环境/障碍物。 |
| intent_environment_description_007 | environment_description | normal | match_volunteer | hard | 环境拥挤程度判断。 |
| intent_environment_description_008 | environment_description | normal | match_volunteer | easy | 安全环境判断。 |
| intent_navigation_001 | navigation | normal | match_volunteer | easy | 科室/电梯/怎么走属于导航,生产要求人工兜底。 |
| intent_navigation_002 | navigation | normal | match_volunteer | easy | 找不到出口是导航找路。 |
| intent_navigation_003 | navigation | normal | match_volunteer | hard | 医院动线引导,属于 navigation。 |
| intent_navigation_004 | navigation | normal | match_volunteer | hard | 目的地导向属于找路。 |
| intent_navigation_005 | navigation | normal | match_volunteer | easy | 门诊/取药窗口位置属于导航。 |
| intent_navigation_006 | navigation | normal | match_volunteer | hard | 入口方向引导,人工标为 navigation。 |
| intent_navigation_007 | navigation | normal | match_volunteer | easy | 电梯位置查询属于导航。 |
| intent_navigation_008 | navigation | normal | match_volunteer | hard | 路线规划与安全通行,标为 navigation。 |
| intent_medication_check_001 | medication_check | normal | match_volunteer | easy | 剂量/用法/禁忌需要药品确认并人工兜底。 |
| intent_medication_check_002 | medication_check | normal | answer | easy | 简单药品问答,生产路径可直接回答并提示安全边界。 |
| intent_medication_check_003 | medication_check | normal | match_volunteer | hard | 隐含用药确认,应药品检查并人工确认。 |
| intent_medication_check_004 | medication_check | normal | answer | hard | 药物合用风险,当前 simpleMedicineQa 可直接安全说明。 |
| intent_medication_check_005 | medication_check | normal | answer | easy | 感冒药成分问答属于药品确认。 |
| intent_medication_check_006 | medication_check | normal | answer | easy | 有效期/过期判断属于药品确认。 |
| intent_medication_check_007 | medication_check | normal | match_volunteer | hard | 隐含药品注意事项确认。 |
| intent_medication_check_008 | medication_check | normal | match_volunteer | hard | 药名核对属于 medication_check。 |
| intent_emergency_001 | emergency | emergency | trigger_sos | easy | 显式救命/胸口痛/摔倒,应触发 SOS。 |
| intent_emergency_002 | emergency | emergency | trigger_sos | easy | 直接紧急关键词。 |
| intent_emergency_003 | emergency | emergency | trigger_sos | hard | 隐含跌倒/行动受限,人工判为紧急。 |
| intent_emergency_004 | emergency | emergency | trigger_sos | hard | 人身安全风险,应紧急。 |
| intent_emergency_005 | emergency | emergency | trigger_sos | hard | 迷路+夜间+低电量组合风险,标为紧急。 |
| intent_emergency_006 | emergency | emergency | trigger_sos | easy | DemoDataLoader 紧急词应触发。 |
| intent_emergency_007 | emergency | emergency | trigger_sos | hard | 呼吸困难是医疗紧急。 |
| intent_emergency_008 | emergency | emergency | trigger_sos | easy | 着火和无法撤离,应触发 SOS。 |
| intent_need_human_001 | need_human | normal | match_volunteer | easy | 显式转人工。 |
| intent_need_human_002 | need_human | normal | match_volunteer | easy | 直接找人确认。 |
| intent_need_human_003 | need_human | normal | match_volunteer | easy | 真人协助意图。 |
| intent_need_human_004 | need_human | normal | match_volunteer | easy | 显式志愿者兜底。 |
| intent_need_human_005 | need_human | normal | match_volunteer | easy | 陪同/人工帮助。 |
| intent_need_human_006 | need_human | normal | match_volunteer | hard | 隐含真人确认。 |
| intent_need_human_007 | need_human | normal | match_volunteer | hard | 用户明确拒绝 AI 独立处理。 |
| intent_need_human_008 | need_human | normal | match_volunteer | hard | 隐含转人工/专家协助。 |
| intent_fallback_001 | fallback | normal | answer | easy | 问候语在当前代码中归 fallback 但可直接回答。 |
| intent_fallback_002 | fallback | normal | answer | easy | 问候/能力询问,代码归 fallback 小聊。 |
| intent_fallback_003 | fallback | elevated | ask_followup | hard | 没有明确服务意图,应低置信追问。 |
| intent_fallback_004 | fallback | elevated | ask_followup | hard | 模糊表达,应兜底追问。 |
| intent_fallback_005 | fallback | elevated | ask_followup | hard | 情绪陪伴尚未单独枚举,按当前代码应 fallback。 |
| intent_fallback_006 | fallback | elevated | ask_followup | hard | 礼貌语不属于服务意图,当前规则会低置信兜底。 |
| intent_fallback_007 | fallback | normal | answer | easy | 英文问候被 greeting 规则覆盖但 intent 仍为 fallback。 |
| intent_fallback_008 | fallback | elevated | ask_followup | hard | 无法归入真实服务类,标为 fallback。 |
| intent_ocr_text_009 | ocr_text | normal | answer | hard | 读取缴费单字段属于 OCR,没有直接使用规则关键词。 |
| intent_ocr_text_010 | ocr_text | normal | answer | easy | 直接读标签文字,应归为 OCR。 |
| intent_scene_description_009 | scene_description | normal | answer | easy | 直接请求画面整体理解,对应场景描述。 |
| intent_scene_description_010 | scene_description | normal | answer | hard | 判断整体场景语义,不是单一物体识别。 |
| intent_object_identify_009 | object_identify | normal | answer | easy | 带图片路径并要求识别具体配件,应为物体识别。 |
| intent_object_identify_010 | object_identify | normal | answer | hard | 隐含物体身份辨认,人工标为 object_identify。 |
| intent_color_recognition_009 | color_recognition | normal | answer | easy | 直接询问颜色。 |
| intent_color_recognition_010 | color_recognition | normal | answer | hard | 比较配色差异,属于颜色识别。 |
| intent_money_recognition_009 | money_recognition | normal | answer | hard | 现金金额确认属于 money_recognition。 |
| intent_money_recognition_010 | money_recognition | normal | answer | easy | 直接询问钞票面额。 |
| intent_translation_009 | translation | normal | answer | hard | 把用户意图转成可给他人看的表达,属于转译沟通。 |
| intent_translation_010 | translation | normal | answer | easy | 直接翻译/转述给他人。 |
| intent_environment_description_009 | environment_description | normal | match_volunteer | easy | 周围障碍物和通行风险属于环境描述。 |
| intent_environment_description_010 | environment_description | normal | match_volunteer | hard | 隐含移动路径环境风险检查。 |
| intent_navigation_009 | navigation | normal | match_volunteer | easy | 寻找服务台位置属于导航。 |
| intent_navigation_010 | navigation | normal | match_volunteer | hard | 隐含室内路线指引,应归为 navigation。 |
| intent_medication_check_009 | medication_check | normal | match_volunteer | easy | 药品注意事项需要药品确认并人工兜底。 |
| intent_medication_check_010 | medication_check | normal | match_volunteer | hard | 隐含用法时机确认,属于 medication_check。 |
| intent_emergency_009 | emergency | emergency | trigger_sos | easy | SOS 与紧急帮助应触发紧急分流。 |
| intent_emergency_010 | emergency | emergency | trigger_sos | hard | 被困且无法离开,人工判为紧急。 |
| intent_need_human_009 | need_human | normal | match_volunteer | hard | 隐含但明确要求真人介入。 |
| intent_need_human_010 | need_human | normal | match_volunteer | easy | 直接转人工请求。 |
| intent_fallback_009 | fallback | elevated | ask_followup | hard | 无明确任务意图,按当前代码应低置信追问。 |
| intent_fallback_010 | fallback | normal | answer | easy | 问候语在当前真实枚举中仍归 fallback 小聊。 |
| emergency_pos_001 | emergency | emergency | trigger_sos | easy | 显式救命/危险,应触发 SOS。 |
| emergency_pos_002 | emergency | emergency | trigger_sos | easy | 摔倒且无法起身,应紧急。 |
| emergency_pos_003 | emergency | emergency | trigger_sos | easy | 胸口痛是内置紧急关键词。 |
| emergency_pos_004 | emergency | emergency | trigger_sos | easy | 着火和无法撤离。 |
| emergency_pos_005 | emergency | emergency | trigger_sos | easy | 抢劫/快来人来自 demo 紧急词。 |
| emergency_pos_006 | emergency | emergency | trigger_sos | easy | 晕倒触发紧急。 |
| emergency_pos_007 | emergency | emergency | trigger_sos | hard | 组合风险:迷路、天黑、低电量。 |
| emergency_pos_008 | emergency | emergency | trigger_sos | hard | 人身安全威胁,隐式紧急。 |
| emergency_pos_009 | emergency | emergency | trigger_sos | hard | 地下车库迷失方向,风险较高。 |
| emergency_pos_010 | emergency | emergency | trigger_sos | hard | 呼吸困难和肢体异常,医疗紧急。 |
| emergency_pos_011 | emergency | emergency | trigger_sos | hard | 被困且无法联系,应紧急。 |
| emergency_pos_012 | emergency | emergency | trigger_sos | hard | 火灾烟雾风险。 |
| emergency_pos_013 | emergency | emergency | trigger_sos | easy | 含我很害怕且有人身安全风险。 |
| emergency_pos_014 | emergency | emergency | trigger_sos | hard | 交通环境下突然失去导航能力。 |
| emergency_pos_015 | emergency | emergency | trigger_sos | hard | 他人倒地无回应,应紧急。 |
| emergency_pos_016 | emergency | emergency | trigger_sos | hard | 人身控制风险。 |
| emergency_pos_017 | emergency | emergency | trigger_sos | hard | 走散加电量即将耗尽。 |
| emergency_pos_018 | emergency | emergency | trigger_sos | easy | 流血不止来自紧急词。 |
| emergency_pos_019 | emergency | emergency | trigger_sos | easy | 心脏病来自紧急词。 |
| emergency_pos_020 | emergency | emergency | trigger_sos | easy | SOS 自动触发词。 |
| emergency_neg_001 | medication_check | normal | match_volunteer | hard | 有担心情绪但不是即时紧急,应药品确认。 |
| emergency_neg_002 | navigation | normal | match_volunteer | hard | 担心但目标是导航。 |
| emergency_neg_003 | money_recognition | normal | answer | hard | 不敢确定但只是面额识别。 |
| emergency_neg_004 | ocr_text | normal | answer | hard | 看错药盒文字,非紧急。 |
| emergency_neg_005 | navigation | normal | match_volunteer | easy | 导航担忧,非 SOS。 |
| emergency_neg_006 | translation | normal | answer | easy | 沟通转译,非紧急。 |
| emergency_neg_007 | need_human | normal | match_volunteer | easy | 害怕但主动要志愿者陪同,非 SOS。 |
| emergency_neg_008 | medication_check | normal | answer | easy | 普通用药问答,非紧急。 |
| emergency_neg_009 | color_recognition | normal | answer | hard | 颜色确认,非紧急。 |
| emergency_neg_010 | environment_description | normal | match_volunteer | hard | 环境判断,非紧急。 |
| emergency_neg_011 | need_human | normal | match_volunteer | easy | 转人工,非紧急。 |
| emergency_neg_012 | need_human | normal | match_volunteer | easy | 陪同协助,非 SOS。 |
| emergency_neg_013 | ocr_text | normal | answer | easy | 读通知,非紧急。 |
| emergency_neg_014 | ocr_text | normal | answer | hard | 菜单文字和价格读取,非紧急。 |
| emergency_neg_015 | fallback | elevated | ask_followup | hard | 情绪词但不是具体紧急事件。 |
| emergency_neg_016 | medication_check | normal | answer | easy | 药品有效期查询,非紧急。 |
| emergency_neg_017 | translation | normal | answer | hard | 电话转译,非紧急。 |
| emergency_neg_018 | environment_description | normal | match_volunteer | easy | 环境确认,非 SOS。 |
| emergency_neg_019 | need_human | normal | match_volunteer | hard | 人工确认票据,非紧急。 |
| emergency_neg_020 | translation | normal | answer | hard | 沟通表达辅助,非紧急。 |

## object_identify 路由诊断

生成时间:2026-06-27T11:43:36.919354

结论: `object_identify` 规则基线 0% 不是评测脚本漏传图片导致的，而是 DemoMode 生产规则缺少图片物体识别分支。本阶段只诊断，不修改生产逻辑。

证据:
- `eval/test/eval_runner_test.dart:353-357` 调用 `AgentServiceFacade.processInput(...)` 时已经传入 `imagePath: sample.imagePath`，且 `inputType` 在有图时为 `mixed`。
- `lib/services/demo/demo_ai_service.dart:274-285` 的 `imagePath != null` 分支只判断 color / ocr / environment，未判断 object / product / 商品 / 物体语义，最后默认返回 `DemoAiIntent.sceneDescription`。
- `lib/services/facades/agent_service_facade.dart:68-89` 在 `FeatureFlags.enableRealAI=false` 时优先走 `_processDemoInput(...)`，因此规则基线先进入 `DemoAIService`。
- `lib/services/facades/agent_service_facade.dart:235-258` 的 image fallback 层有 object keyword 分支，但 DemoMode 规则基线未触达该分支。

建议修法: 后续若进入规则改进阶段，可在 `DemoAIService._detectDemoIntent()` 的图片分支中加入 object/product/商品/物体/东西/用品/工具等语义判断，或调整 facade 的图片 fallback 路由顺序。不要在本轮 LLM 对照阶段为测试集逐条硬编码。
