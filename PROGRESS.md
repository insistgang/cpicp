# 2026 六赛构建进展看板

> 配套: `TIMELINE.md` (日期/推进顺序)、`battle_plan.md` (总战略)、各赛 `README.md` + `src/`。
> 状态更新于 2026-06-19 (第二轮深度推进后)。原则: Mac 上 (无 GPU/数据/硬件) 能建能验的全部做掉, 剩下标注卡在何处。
> 第二轮关键突破: 用本地可用库 (python-pptx / python-docx / matplotlib / scikit-learn / PIL / networkx) 把各赛从"代码骨架"推到"**实际交付物草稿**"——可提交级 PPTX/DOCX、攻防演示 HTML、真实合成图像端到端指标 (非随机向量)。

## 总览

| 序 | 比赛 | 真实赛题 (已核实) | 完成度 | 代码状态 | 本地交付物草稿 | 卡在哪 |
|---|---|---|---|---|---|---|
| 1 | **01 网安** | AI Agent 安全闭环 (第五届) | ~85% | 13/13 模块自测通过 | 攻防闭环 demo.html (10攻击场景10/10闭环) + ROC/PR图 (AUC 0.998) + 攻击轨迹JSON | 官方赛题/靶场/报名均未发布; 去年作品需找回; Chaspark 答疑 |
| 2 | **02 金融科技** | 多模态影像分类+相似度去重 (无锡农商行命题) | ~72% | 11/11 模块自测通过; 真实像素端到端 (AUC 0.99/Top-1 1.0) | 合成票据/证照影像 + 去重可视化热图 + manifest | 签 NDA 领数据 (苏老师); 报名截止 8/17 (最早红线); CLIP真特征需算力机 |
| 3 | **03 智能建造** | 低空水上救援视觉 (分类3类+边缘30FPS) | ~65% | 15/15 项自测通过; train/eval/export 骨架待 GPU | 技术方案+性能报告 双 Word初稿(617+239KB,内嵌7图) + 海面增广 before/after | 数据 (李老师); Jetson Orin 硬件; 报名截止 8/20 |
| 4 | **04 AI 创新** | AOI few-shot 异常检测 + 违建 AI+X 线 | ~85% | 14/14 模块自测通过; 真实合成图端到端 (AUC 0.99/F1 0.94) | 合成AOI工件图 + 缺陷定位热力图 + ROC/PR + 异常分布图 | 真数据 (DAGM2007/MVTec); GPU <200ms 红线; 华为 chaspark; 报名策略 |
| 5 | **05 未来飞行器** | 低空未来飞行器应用场景设计 (纯方案/报告书) | ~60% | 3 模块自测通过; 报告书初稿 7 章+附录 | 正文+附件 双 Word 报告书(640+41KB,正文4图嵌入§2.1/2.4/4.2/5) + 4张架构/流程/商业/矩阵图 | 队员招募; docx→PDF需Office机; 真实成本调研 |
| 6 | **06 文化中国** | 数字人文可视化 (赛题2: 古籍互文+传播) | ~55% | 6/6 模块自测通过 | **8页16:9 PPTX草稿(逐字稿写入备注) + 6张配图嵌入** + 静态HTML | 组队报名 (9/14 截止); 《颜氏家训》全文史料+书影; 录语音 |

---

## 01 网络安全大赛 — 13/13 模块自测通过 + 攻防演示 Demo

- `src/`: 审计 schema / 资产扫描 / **配置风险静态扫描** / 恶意技能检测 / 提示注入检测 / 链式异常 / 指标评估 / 拦截器 / 防御策略 / 伪 Agent 模拟器 (10 检测内核)。
- **第二轮新增 (方向C 答辩杀手锏 + 方向E benchmark)**: `attack_demo.py` (合成攻击轨迹→检测命中→自动生成防御策略→拦截器阻断的完整闭环) / `build_demo_html.py` (单文件静态 `output/demo.html`, 内嵌数据+图, 双击即开, 已 headless Chrome 渲染核验) / `benchmark_plot.py` (检测内核在 400 条合成正负轨迹上 **AUC=0.998, 检出 1.0/误报 0.035/F1 0.983, 达官方双线**)。
- **第三轮 (扩攻击面)**: 攻击场景库 **4→10 条**, 新增 A5 越权提权 / A6 数据投毒 / A7 MCP 协议层攻击 / A8 敏感信息外泄 / A9 工具滥用(反弹shell) / A10 RAG 检索投毒, 每条≥2 检测器命中; 实跑 **10/10 闭环, 攻击得手 3→0, 拦截 9→20, 良性零误报**; demo.html 重生含 A1-A10 锚点 (156KB)。
- 新增 `run_all_selftests.py` 一键总控, **13 模块全部通过**。
- 文档: `README.md` 已更新为 13 模块完整文档。
- 卡点: ① 第五届官方赛题/靶场/报名/提交模板均未发布 (等 cpipc 官网); ② 去年作品需找回自证可复用性; ③ 真实指标需在官方靶场+真 Agent 上复测; ④ Chaspark 答疑平台需注册登录。
- 下一步 (需用户/硬件/数据): 找回去年作品; 注册 Chaspark; 官方靶场到手后开发 GUI Demo (方向C) 和多 Agent 协同防御 (方向D); 拿到官方数据集后做量化评估 Benchmark (方向E)。

## 02 金融科技大赛 — 11/11 模块自测通过 + 真实像素端到端

- 已自测: `prepare_data` / `metrics` (阈值扫描 P/R/F1+AUC+Top-k) / `similarity` (余弦去重+跨客户套用) / `pipeline` (端到端) / `app_demo` (Gradio 汇总视图)。
- **第二轮新增 (从随机向量→真实图像像素)**: `features.py` (经典 CPU 特征后端: PIL 颜色直方图+灰度梯度分块+sklearn PCA, 作 CLIP 不可用时 baseline; 关键发现: 结构化非负特征不应做 StandardScaler, 否则 AUC 0.99→0.69) / `synth_images.py` (PIL 程序化合成票据/证照 4 模板, 注入重复+跨客户套用) / `viz_dedup.py` (相似度矩阵热图+套用对拼图)。`classify`/`embed` 在无 torch 时**优雅回退经典特征**并实跑。
- **真实像素端到端实测**: 121 张合成影像/33 面签, **AUC=0.992, 最优F1阈值 P0.889/R1.0/F0.941, Top-1 检索 1.0**; 多 seed 复测 AUC 0.88–1.0 稳定。
- 新增: `run_all_selftests.py` (**11 模块全通过, 零警告**) / `Dockerfile` / `pr_curve.png` / `.gitignore` (排除 121 张可重生合成图, 保留 dedup_viz.png+manifest)。
- 修复: `similarity.py` 浮点溢出警告 → float64 + nan_to_num + np.errstate。
- 卡点: ① torch/open_clip/timm/faiss 未安装 (Mac 无 GPU); ② 真实合成数据未领取 (需签 NDA 联系苏老师 jrkjds@nju.edu.cn); ③ CLIP 实际精度未知; ④ 度量学习微调需算力机 (4090/4070S)。
- 下一步 (需用户/硬件/数据): 报名截止 8/17 (本周即办); 签 NDA 领数据; 算力机装环境跑 classify/embed; 拿到真实数据跑完整流水线、调优阈值、做消融实验。

## 03 智能建造大赛 — 15/15 项自测通过 + 技术方案 Word 初稿

- 已自测: `geolocate` (8项几何) / `track_filter` (5项时序) / `augment_water` (5项增广) / `losses_smalltarget` (9项损失) / `stream_qgc` (接线检查) / `crossdomain_eval` (域差流程) / `prepare_data` (guide 输出) / 3 个 YAML 配置语法合法。
- **第二轮新增 (本地交付物草稿, src/tools/)**: `gen_water_scene.py` (PIL 合成俯拍海面+太阳反光带+小目标, 调真实 augment_water 做 GT-Glint 增广, 出 before/after 对比图) / `gen_report_figs.py` (分桶召回柱状图/各类PR曲线/FPS-精度散点含30FPS红线, 数据函数化待真值替换) / `gen_tech_plan_docx.py` (python-docx 把技术方案大纲渲染成结构化 Word: 封面+硬指标对照表+§0-§8章节+消融表+内嵌4图) / `build_all_deliverables.py` (一键产出)。
- **第三轮新增**: `gen_perf_report_docx.py` 把 性能报告_模板.md 渲成结构化 Word (封面+6章+6表+内嵌3张性能图), 接入 build_all_deliverables (4/4步) 与自测。
- **交付物**: `output/docx/技术方案_初稿.docx` (617KB OOXML, 内嵌4图) + `output/docx/性能报告_初稿.docx` (239KB OOXML, 内嵌3图) + 5张 PNG。官方三大交付物中两份 Word 草稿齐全。
- 新增 `run_all_selftests.py` (**15 项全通过**); 修复 PR 曲线反向 bug (改 COCO 风格 maximum.accumulate 精度包络) + np.trapz→trapezoid。
- 修复: `prepare_data.py` GUIDE 字符串 format 误解析; `crossdomain_eval.py` mmd_rbf 负零/溢出。
- 文档: `BATTLEPLAN.md` (冲国一打法)、`技术方案_大纲.md`、`性能报告_模板.md`。
- 骨架待验证: `train.py` / `eval.py` / `export_onnx.py` (需 RTX4070S+数据集); `trt_infer_orin.py` / `stream_qgc.py` (需 Jetson Orin Nano 8G+TensorRT+摄像头/MAVLink)。
- 卡点: ① 无 GPU (Mac); ② 无 SeaDronesSee/AFO 数据集; ③ 无 Jetson Orin Nano; ④ 官方命题方数据未申请 (李老师 13714638358); ⑤ 报名截止 8/20。
- 下一步 (需用户/硬件/数据): 联系李老师申请数据; RTX4070S 装环境跑 `make build && make train`; Orin Nano 验证 30FPS 硬门槛 (决定方案生死); 拿到数据后做消融实验 (P2头/NWD/Wise-IoU/GT-Glint/INT8)。

## 04 AI 创新大赛 — 14/14 模块自测通过 + 真实合成图端到端

- 已自测: `anomaly_score` / `patchcore_lite` (coreset 压缩 ~20%) / `aoi_metrics` / `fewshot_protocol` / `latency_bench` (CPU 21.4ms<2000ms) / `illegal_build_pipeline` (复用 03 track_filter+geolocate 产 GPS 航点) / `aoi_prepare` / `augment_defect` (4 种缺陷+可复现)。
- **第二轮新增 (从随机特征→真实合成图像)**: `synth_aoi.py` (PIL 合成 PCB 工件图: 基板+栅格+元件方块+产线抖动, 复用 augment_defect 贴 4 类缺陷) / `feature_backend.py` (TimmBackend 真特征接口保留 + ClassicBackend 经典 CPU 特征, 自动回退) / `run_real_pipeline.py` (真实合成图 few-shot 端到端) / `viz_heatmap.py` (缺陷定位热力图)。
- **真实合成图端到端实测**: 标准 100正+30缺→测600, **AUC=0.993, F1=0.942, Recall=0.975, Precision=0.911**; per-class scratch 1.0/spot 0.96/missing 0.94/discolor 1.0; `--full` 测1020张 AUC=0.994。
- **交付物**: `output/heatmap_overlay.png` (缺陷件热点对齐 GT 框) + `score_distribution.png` + `roc_pr_curves.png` + sample_aoi 样例集。
- 新增 `run_all_selftests.sh` (**14 模块全通过**) + `.gitignore` (排除可重生 npz/批量合成图)。
- 修复: `augment_defect.py` missing 纯色图无变化 → block_mean ± 20; `anomaly_score.py`/`latency_bench.py` numpy 2.0 matmul false positive → np.errstate。
- 卡点: ① 真数据/真特征 (当前用随机 numpy 特征跑逻辑, 需 torch+timm/CLIP + DAGM2007/MVTec AD); ② GPU <200ms 红线验证 (本机无 NVIDIA GPU, 需 2060 级显卡复测); ③ 华为 chaspark 数据 (需队长华为账号报名后下载); ④ 报名决策 (3 人队伍、赛道定夺: 华为专项 vs 开放主奖 vs 双报); ⑤ 在线学习闭环 (需真数据验证)。
- 下一步 (需用户/硬件/数据): 确认队伍+报名策略; 下载 DAGM2007+MVTec AD; 安装 torch+timm 跑真特征; 2060 级 GPU 验证 <200ms; 队长华为账号登录 chaspark 下载真实数据; 开始撰写项目文档 (9/1 提交 deadline)。

## 05 未来飞行器大赛 — 正式 Word 报告书 + 4 图嵌入

- 性质: 纯方案/报告书赛, 无 src 代码要求。赛道: 交叉赛道 3.1 — 低空未来飞行器应用场景设计竞赛。主题: 基于低空智能感知的城市违建巡查系统。提交截止: 2026-09-01。
- 已完成: 官方资料研读 (参赛指南 PDF 20页精读) / 赛题理解 (6维评分标准对齐) / 项目报告书初稿 (7章+附录) / 技术成熟度报告 (TRL 评估, 附件1) / 附件清单与任务追踪。
- markdown 源文件: `项目报告书_初稿.md` (15.5KB) / `附件清单_任务追踪.md` / `技术成熟度报告_附件1.md` (已移出 docs/ 避免被 .gitignore 拦截)。
- **第二轮新增 (src/, markdown→可提交级 Word)**: `gen_figures.py` (matplotlib 出 4 图: 端-边-云架构图/实施流程图/商业模式成本收益图/6学科交叉矩阵, 中文用 Arial Unicode MS) / `build_report_docx.py` (解析初稿 MD 按官方 5 段骨架渲染 docx: 封面+TOC域+标题层级+原生表格+插图+A4页边距) / `run_all_selftests.py`。
- **第三轮新增**: `build_attachment_docx.py` (薄封装复用 build_report_docx 通用 MD 解析管线) 把 技术成熟度报告_附件1.md 也渲成 docx。
- **交付物**: `output/项目报告书_低空违建巡查系统.docx` (640KB OOXML, 段落155/表格8/内嵌图4, 4图精确命中 §2.1/§2.4/§4.2/§5) + `output/技术成熟度报告_附件1.docx` (41KB OOXML, 段落76/标题9/表格4) + 4 张 PNG。正文+附件双 docx 齐全。
- TRL 评估: 综合 TRL 6-7, 复用 03/04 技术底座 (YOLOv12n+P2 / geolocate / track_filter)。
- 卡点: ① 队员招募 (需补招城乡规划/公共管理/GIS, 吃满 8 人上限); ② 数据佐证 (违建统计、巡查成本); ③ 可视化图表 (系统架构图、实施流程图、商业模式画布); ④ 真实成本数据 (无人机租赁/服务市场价格); ⑤ 演示视频 (决赛阶段再制作); ⑥ 数字孪生原型 (视时间而定)。
- 下一步 (按时间线): 本周制作系统架构图+实施流程图+调研飞行器参数+补充成本数据; 7 月招募跨学科队员+制作商业模式画布+补充违建统计数据; 8 月报告书定稿+图表美化+排版+附件打包; 9/1 前最终检查提交。

## 06 文化中国大赛 — 6/6 模块自测通过 + 8页 PPTX 草稿(重头戏)

- 性质: 数字人文可视化, 赛题 2 (古籍互文+传播)。提交: 带语音讲解 PPT (≤5分钟)。
- 已自测: `text_tools` (古文分句/TF-IDF 关键词/词云 JSON) / `similarity_search` (互文片段检索/余弦相似度排序) / `generate_visuals` (时间轴/传播路径/词云/互文网络 JSON + 单文件静态 HTML)。
- **第二轮新增 (从"指南"→实际可打开的 PPTX 成品草稿)**: `render_figures.py` (matplotlib/networkx 把 output 数据渲染成 6 张 PNG: 时间轴/传播金字塔/词云/互文网络/封面/结尾, 古风配色; 修了首版词云碰撞检测把词挤掉只剩1字的 bug, 现放置 30/30 词) / `build_pptx.py` (python-pptx 生成 8 页 16:9 PPTX: 标题+要点+配图区, **逐字稿写入 speaker notes 作旁白脚本**, 封面/结尾排版)。
- **第三轮 (扩讲稿)**: 逐字稿从 412 字/125s 扩写到 **924 字/280s** (贴合 ≤5 分钟预算上沿), 每页紧扣书籍生命史/教化传播/数字人文方法, 用 output 数据 (时间轴7节点/传播4层/互文网络) 支撑; 未编造原文史实, 待补项如实标注。
- **交付物**: `output/作品_初稿.pptx` (400KB OOXML, 读回校验: **页数=8, 16:9 画布, 每页备注非空 8/8, 含配图页 6, 讲解稿估时 280s≤300s, 匿名敏感词扫描通过**; 已用 LibreOffice 转 PDF 逐页肉眼核验中文/版式正常) + 6 张 PNG。
- 产出: 5 个可视化数据文件 + `output/visual.html` (浏览器打开即可截图)。
- 卡点: ① 组队报名 (官网 cpipc.acge.org.cn, 9/14 截止); ② 《颜氏家训》全文史料 (需找原文 20 篇、版本书影、颜之推传记 — 决定文化内涵分); ③ PPT 视觉设计 (古风封面底图、古籍书影、雕版印刷图等素材); ④ 讲解语音录制; ⑤ 院校审核盖章 (留 1 周缓冲)。
- 下一步 (按时间线): 立即注册官网确认报名流程; 本周找《颜氏家训》全文 (中华经典古籍库/汉典/维基文库); 下周按分镜骨架制作 PPT (用 visual.html 截图嵌入数据页); 9 月初录制语音+合成 PPT+走院校审核。

---

## 前置事项清单 (需用户亲自处理)

| 优先级 | 事项 | 比赛 | 类型 | 截止时间 | 联系方式/动作 |
|---|---|---|---|---|---|
| P0 | 签 NDA 联系苏老师领取真实合成数据 | 02 金融科技 | 数据 | 报名 8/17 | jrkjds@nju.edu.cn |
| P0 | 报名 02 金融科技大赛 | 02 金融科技 | 报名 | 2026-08-17 | 官网报名 |
| P0 | 联系李老师申请命题方数据集 | 03 智能建造 | 数据 | 报名 8/20 | 李老师 13714638358 |
| P0 | 报名 03 智能建造大赛 | 03 智能建造 | 报名 | 2026-08-20 | 官网报名 |
| P0 | 找回去年 01 网安作品 (代码/PPT/答辩录像/专家反馈) | 01 网安 | 资料 | — | 用户自行查找 |
| P0 | 注册 Chaspark 答疑平台 | 01 网安 | 平台 | — | 用户注册登录 |
| P1 | 确认 04 AI 创新队伍成员 + 报名策略 (华为专项/开放主奖/双报) | 04 AI 创新 | 报名 | — | 队长华为账号 |
| P1 | 下载 DAGM2007 + MVTec AD 数据集 | 04 AI 创新 | 数据 | — | 注册/同意协议 |
| P1 | 队长华为账号登录 chaspark 下载真实数据/baseline | 04 AI 创新 | 数据 | — | 华为账号 |
| P1 | 报名 06 文化中国大赛 (官网 cpipc.acge.org.cn) | 06 文化中国 | 报名 | 2026-09-14 | 官网注册 |
| P1 | 找《颜氏家训》全文史料 (20篇+版本书影+颜之推传记) | 06 文化中国 | 数据 | — | 中华经典古籍库/汉典/维基文库 |
| P1 | 招募 05 飞行器跨学科队员 (城乡规划/公共管理/GIS) | 05 飞行器 | 组队 | — | 校内招募 |
| P2 | 算力机安装 torch/open_clip/timm/faiss-gpu | 02 金融科技 | 硬件 | — | RTX4070S/4090 |
| P2 | RTX4070S 装环境跑 `make build && make train` | 03 智能建造 | 硬件 | — | RTX4070S |
| P2 | Jetson Orin Nano 验证 30FPS 硬门槛 | 03 智能建造 | 硬件 | — | Jetson Orin Nano 8G |
| P2 | 2060 级 GPU 验证 04 AI 创新 <200ms 红线 | 04 AI 创新 | 硬件 | — | RTX2060 级 |
| P2 | 安装 torch + timm 跑通真特征提取 | 04 AI 创新 | 环境 | — | CPU 慢验证 / GPU 快验证 |
| P2 | 05 报告书 docx→PDF (官方要求 PDF 提交) | 05 飞行器 | 格式 | 提交前 | 装 Word/WPS 的机器: 更新目录域→删提示文字→另存PDF |
| P2 | 06 PPTX 录语音讲解 + 补真实书影/人物像/地图 | 06 文化中国 | 素材 | 提交前 | PowerPoint/Keynote 录音; 国图古籍影像/维基共享 |

> **第二+三轮小结**: 本地代码与可生成交付物已全部做掉 (6 赛自测全绿: 01=13 / 02=11 / 03=15 / 04=14 / 06=6 模块 + 05=3 渲染模块; 实际交付物草稿: 攻防 demo.html(10场景闭环) / 真实像素端到端指标 / 03 技术方案+性能报告双Word / 05 正文+附件双Word / 06 8页 PPTX(讲稿280s))。**剩余全部为上表与下表中需用户/硬件/数据/外部素材的前置事项**——Mac 本地已推到边界。
