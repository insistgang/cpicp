# 网安赛题#4 · 通用 AI Agent 安全检测与防御闭环 — 代码 (src)

> ⚠️ **官方第五届赛题/日期截至 2026-06-18 未发布**;本 src 依据 docs 内 **第四届(2024)problem4 占位文本** + 官方评分5维构建,
> 是"官方靶场未到手前的离线检测内核",第五届题库一公布即按其指标校准。**纯标准库/numpy,本机可全部自测。**

## 真任务(problem4 原文)
对官方 Agent 靶场,构建"发现-分析-监控-阻断"闭环,两大功能:
1. **资产风险静态识别**:盘点 AI Agent/框架/模型/Skills/MCP → 资产图谱 → 配置不当 + 恶意 Skills/MCP 供应链隐患。(指标:盘点准确率≥99%、风险识别≥95%、漏报<5%)
2. **运行时异常监控与阻断**:检测 漏洞利用异常 + 提示词攻击(越权/外泄/恶意执行)并阻断,保业务可用。(指标:检出≥95%、误报<5%、响应<1s、CPU<5%、0中断)

评分5维:① 供应链/异常发现 ② 关键证据定位(可审计) ③ 检测效率/资源 ④ 报告质量(代码可运行) ⑤ 创新性(长链路行为分析)。

## 文件(每个都 `python xxx.py` 自测)
| 文件 | 作用 | 对应 |
|---|---|---|
| `audit_schema.py` | 统一审计事件/trace + JSON round-trip | 证据链基础(评分②) |
| `asset_scanner.py` | 扫描目录识别 Agent/框架/模型/Skills/MCP 资产 + 图谱 | 任务1 资产识别 |
| `config_risk_scanner.py` | 扫描配置文件识别 硬编码凭据/危险命令/过宽路径/未加密传输/代码执行启用 | 任务1 配置风险(新增) |
| `malicious_skill_detector.py` | 恶意 Skill/MCP 静态检测(隐藏指令/危险能力/外联/base64),带证据 span | 任务1 供应链(评分①②) |
| `prompt_injection_detector.py` | 直接+间接提示注入检测,区分来源 | 任务2(评分①) |
| `chain_anomaly.py` | 工具调用链路时序异常(频率+转移基线+去抖告警) | **创新点(评分⑤)**,复用 03 track_filter 思想 |
| `metrics_eval.py` | 检出率/误报率/F1/AUC + 在误报<5%约束下选阈值 | 对齐官方硬指标,复用 02 metrics |
| `interceptor.py` | tool_call 拦截器:策略+注入+敏感行为检测→allow/alert/block,测<1s | 任务2 阻断(评分①②③) |
| `defense_policy.py` | 自适应防御策略(屏蔽工具/拦外联/系统约束),从检测结果加规则 | 闭环防御侧 |
| `fake_agent.py` | **最小靶场+端到端闭环演示**:良性放行/攻击阻断/链路告警/加规则复测被挡 | 官方靶场未到手前的近似 |
| `run_all_selftests.py` | 一键运行全部 10 模块自测并汇总 | 质量门禁 |

## 运行
```bash
# 一键全测
python3 run_all_selftests.py

# 或逐个运行
for f in audit_schema asset_scanner config_risk_scanner malicious_skill_detector \
         prompt_injection_detector chain_anomaly metrics_eval interceptor \
         defense_policy fake_agent; do python3 $f.py; done
```

## 阻塞(只能你来/等官方)
- 第五届官方赛题清单、Agent 靶场镜像、提交模板、报名时间 **均未发布**;Chaspark 数据/答疑需登录。
- "去年 70-100% 可复用"需找回去年作品自证;真实指标须在官方靶场+真 Agent 上复测(本地靶场仅预验证)。
