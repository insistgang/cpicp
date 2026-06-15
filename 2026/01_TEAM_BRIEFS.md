# 01 · 各队执行清单（可直接转发队长）

> 生成 2026-06-15。牵头分工：**A 我牵头 / B 我牵头 / C 挂名 / D 挂名 / E 挂名**。
> 硬件基线：RTX4070S(训练) + Jetson Orin Nano + 8G 边缘设备，**无国产 NPU/RISC-V 板卡**。
> 标"未获取"的项需登录 cpipc 下载赛题书/附件核实，**不要按推断当事实开工**。

---

## 🟢 B 队 · 智能建造揭榜7（低空智能水上救援高精度视觉识别）— 我牵头【主攻】
**一句话目标**：用 YOLO11+P2 小目标检测，把无人机航拍水面落水目标做到"高精度+Jetson 边缘实时"，并用我的跨域迁移评价方法处理陆→海域差，冲揭榜奖+一篇论文。

**本周必做 3 件事**
1. **登录 cpipc 报名并下载赛题7参赛指南 PDF**，核对 5 个未获取项：出题企业 / 识别目标(人?船?救生具?) / **是否提供数据集** / 评测指标(mAP·FPS·功耗权重) / 决赛形式。把结果回填到 `B_*.md`。
2. **不等企业数据**：下载 **SeaDronesSee** 与 **AFO**，用 Roboflow 导出/转 YOLO 格式，合并成"水面目标域"训练集。
3. 在 RTX4070S 上跑通 **YOLOv11 + P2 检测头** baseline，记录 mAP@0.5/0.5:0.95，建好评测脚本。

**牵头人关键能力(配人)**：YOLO 训练调参 + 小目标检测经验；会 Jetson/TensorRT 部署；懂域适应/迁移评价(我本人) → 另配 1 人做数据工程(下载/清洗/增广)。

**直接可用资源**
- 数据：SeaDronesSee [代码](https://github.com/Ben93kie/SeaDronesSee)/[主页](https://seadronessee.cs.uni-tuebingen.de/)；AFO [Kaggle](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects)/[Roboflow](https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object)
- SOTA：[SODet-YOLO](https://doi.org/10.3390/rs18111714)、[SeaLSOD-YOLO](https://doi.org/10.3390/sens26072017)、小目标切片 [SAHI](https://github.com/obss/sahi)
- 部署：[YOLOv8 on Jetson Orin 基准](https://www.mdpi.com/2073-431X/15/2/74)

**第一个里程碑**：**6/29 前**——SeaDronesSee+AFO 上 YOLO11+P2 baseline 跑通并出 mAP，能在 Orin Nano 上 FP16 推理出帧率。

---

## 🟢 A 队 · 人工智能华为7（低算力端侧视觉跌倒检测）+ 开放3 — 我牵头
**一句话目标**：轻量 YOLO 检测 + 姿态/时序状态机降误报(区分跌倒vs主动躺蹲) + 端侧部署，做"低算力实时跌倒检测"，兼报 AI+X(智慧养老)。

**本周必做 3 件事**
1. **补赛题书**：把华为赛题7原文放到 `~/Downloads/inputs/`(或给我路径)——确认确切参数量/推理耗时/**部署平台(是否强制海思/昇腾)**/评分权重。**这决定要不要采购国产板**(见 `00_OVERVIEW` §2)。
2. 复现 **LFD-YOLO / BMR-YOLO** baseline(YOLOv8n/s 改注意力+轻量卷积)，在 RTX4070S 跑通。
3. 下载 **OmniFall**(统一基准，含 In-the-Wild 真实跌倒)整理训练/验证集；确认 YOLO-fall 引用的是轻量化版(Oxford)还是开放空间版。

**牵头人关键能力(配人)**：YOLO+知识蒸馏(我)；**姿态估计+时序动作识别(YOLOv8-pose+TCN/GRU)需有人专攻**(降误报核心)；端侧部署(若需国产 NPU，配懂 RKNN/昇腾的人或我自学)。

**直接可用资源**
- 论文：[LFD-YOLO](https://www.nature.com/articles/s41598-025-89214-7)、[BMR-YOLO](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335992)、[YOLO-fall 轻量版](https://academic.oup.com/comjnl/advance-article-abstract/doi/10.1093/comjnl/bxaf005/7994663)
- 工程参考：[Human-Falling-Detect-Tracks(YOLO+姿态+ST-GCN，含动作分类)](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)、[YOLOv8-pose](https://docs.ultralytics.com/tasks/pose)
- 数据：[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall)、[Kaggle Thermal Fall](https://www.kaggle.com/datasets/boeychunhong/thermal-fall-detection-and-activity-dataset)

**第一个里程碑**：**6/29 前**——LFD/BMR baseline 复现出 mAP + OmniFall 训练跑通 + 赛题平台是否需国产板已确认。

---

## ⚪ C 队 · 金融科技揭榜7（港股IPO招股书+多智能体风险预警，东吴证券）— 我挂名
**一句话目标(给队长)**：用 Agentic AI(法务合规/财务穿透/市场分析多角色)解析港股招股书暗雷 + 融合发行期市场情绪，**重点识别"上市后5个交易日内显著下跌"风险**，交付可运行 API。赛题书硬指标：风险要素抽取≥80%、证据召回≥85%、推理可追踪率100%。

**本周必做 3 件事**
1. **组队**：务必拉 **1 名金融/财会背景队友**(港股IPO/未盈利生物科技估值/对赌赎回是最大认知缺口)。
2. **领取东吴提供的数据集**(赛题书明确提供：3-5年港股招股书PDF + 港股基本信息 + 历史行情)——登录 cpipc 揭榜流程领取；若需对接命题方，联系人 任川 rench@dwzq.com.cn。
3. 跑通最小链路：**MinerU 解析一篇招股书 → LightRAG 建图 → 一个"风险问答+证据span回链"** demo(对齐"可追踪率100%")。

**牵头人关键能力(配人)**：LangGraph/多智能体编排 + FastAPI(可由我支援)；**金融领域知识(关键，需金融队友)**；文档解析/RAG 工程。

**直接可用资源**
- 工程脚手架：**[TradingAgents-CN(港股+多Agent+多供应商+FastAPI)](https://github.com/hsliuping/TradingAgents-CN)**、[FinRobot](https://github.com/ai4finance-foundation/finrobot)、[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
- 解析+检索：**[MinerU](https://github.com/opendatalab/MinerU)** + **[LightRAG(原生集成MinerU)](https://github.com/hkuds/lightrag)**、[PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.md)
- 编排：[LangGraph](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)

**第一个里程碑**：**6/29 前**——MinerU 解析东吴样例招股书 + LightRAG 问答 + 证据回链跑通(单篇端到端)。

---

## ⚪ E 队 · 操作系统开源（openKylin 应用创新，荐赛题3 个性化智能助手）— 我挂名【最急 7/30】
**一句话目标(给队长)**：在 openKylin 上做"本地+云"混合个性化智能助手(多供应商 LLM 路由 + 端侧视觉)，打包成 .deb，GitLink 持续提交。**赛题任务书未核实，先下载附件2确认。**

**本周必做 3 件事**
1. **登录 cpipc 下载附件2**，核对应用创新赛道赛题3(及2/4)的任务书、硬性指标、是否强制 RISC-V、评分权重——**别按推断开工**。
2. 在 RTX4070S 主机装 **openKylin x86 虚拟机/双系统**，跑通 "Qt(或Electron) hello-world → `dpkg-deb` 打成 .deb → openKylin 里安装运行" 最小闭环。
3. 建 **GitLink** 仓库、组队(3 人+1 导师)、**从今天起规律 commit**(评审看开发过程，忌临交一次性 push)。

**牵头人关键能力(配人)**：Linux/Qt 或 Electron 桌面开发 + 打包(deb)；LLM 应用集成(多供应商路由可由我支援)；GitLink 协作。

**直接可用资源**
- 打包：[openKylin SDK 指南](https://docs.openkylin.top/zh/04_%E7%A4%BE%E5%8C%BA%E8%B4%A1%E7%8C%AE/%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97/openKylin+SDK%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97)、[Kylin Qt 打 deb](https://blog.csdn.net/baidu_39064543/article/details/131477278)
- 本地 LLM：[Ollama](https://github.com/ollama/ollama)、[llama.cpp](https://github.com/ggml-org/llama.cpp)
- 往届范本：[GitLink OSKYCX 组织](https://www.gitlink.org.cn/OSKYCX/)

**第一个里程碑**：**6/22 前**(工期极短)——openKylin 虚拟机内一个能打成 .deb 并安装运行的最小 GUI 应用 + 赛题3任务书已核实。

---

## ⚪ D 队 · 数学建模（研赛固定命题）— 我挂名
**一句话目标(给队长)**：固定 3 人赛，赛时(9/19报名、约9/27-28交论文)选含图像/深度学习/数据预测的题，靠"能跑出有效 AI 模型 + 严谨实验 + 清晰论文"拿奖。

**本周(或赛前 2-4 周)必做 3 件事**
1. 三人定分工：**建模手(假设/公式) + 编程手=我(AI/数据/出图) + 写作手(LaTeX/逻辑)**。
2. 跑通 1-2 道往年 **AI/数据类题**(如 2025C 地质裂隙 U-Net、2023F 降水临近预报)完整流程。
3. 固化代码模板(数据加载/交叉验证/画图/导出)与论文 **LaTeX(ctex)** 骨架；装好 Gurobi(学生免费)。

**牵头人关键能力(配人)**：数学建模功底(建模手) + Python/ML(我) + 学术写作(写作手)，三者缺一不可。

**直接可用资源**：[2025 题目(官方)](https://www.cmathc.org.cn/mcm/st/325.html)、[历年题型 D_数学建模.md](./D_数学建模.md)、方法库(sklearn/xgboost/lightgbm/PyTorch/Gurobi/CVXPY)。

**第一个里程碑**：**赛前约 9/5**——往年一道 AI/数据题端到端跑通 + 代码/LaTeX 模板就绪。

---

### 配人建议汇总
| 队 | 我的角色 | 最需补的队友 |
|---|---|---|
| B | 牵头(CV/迁移/部署) | 数据工程 1 人 |
| A | 牵头(YOLO/蒸馏) | **姿态+时序动作识别 1 人**；(若需国产板)端侧部署 1 人 |
| C | 挂名(支援多智能体/FastAPI) | **金融/财会背景队长(关键)** + 文档解析/RAG 1 人 |
| E | 挂名(支援 LLM 集成) | **Linux/Qt 桌面+deb 打包队长** |
| D | 挂名(编程手) | 建模手 + 写作手 |
