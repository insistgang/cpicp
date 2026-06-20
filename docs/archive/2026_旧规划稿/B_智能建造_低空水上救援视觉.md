# 赛事 B · 第二届中国研究生智能建造创新大赛
## 揭榜赛道 赛题7：基于边缘计算的低空智能水上救援装备高精度视觉识别

> 调研日期 2026-06-15｜报名截止 8/20｜作品截止 8/31
> 纪律说明：赛题①出题企业、②识别目标、③数据集、④评测指标、⑤决赛形式的精确定义，均写在**登录态"参赛指南/赛题指南 PDF"**中，公开网页未披露。凡公开渠道无法证实者一律标"未获取(在登录态参赛指南PDF)"，不臆测企业名与指标。

---

## 1. 赛题要点

| 项目 | 内容 | 来源 |
|---|---|---|
| 赛题名称(已确认) | "基于边缘计算的低空智能水上救援装备高精度视觉识别技术"，属**揭榜赛道**，第二届邀请函原列 11 项揭榜赛题之一 | 邀请函 |
| 赛事归属 | 第二届中国研究生智能建造创新大赛，平台 cpipc.acge.org.cn | 大赛主页 |
| 时间(已确认) | 报名 2026-05-18 至 **08-20**；作品提交至 **08-31** | 邀请函 |
| 赛道近况 | 2026-06-12《揭榜赛道赛题增补通知》新增"空天地海一体化机器人协同运维监测"，揭榜赛题由 11 增至 12 项；**以登录后最新赛题树为准** | 增补通知 |
| ① 出题企业/发榜单位 | **未获取(在登录态参赛指南PDF)**。公开邀请函/增补通知均未列各赛题对应企业。不臆测 | — |
| ② 识别目标 | **未获取(在登录态参赛指南PDF)**。赛题名仅含"水上救援装备"+"高精度视觉识别"，未公开界定是落水人员/遇险船只/救生设备中哪类。注：同类公开基准(SeaDronesSee/AFO)目标类为 swimmer/floater/boat/buoy/life jacket，仅供合理范围参考，**非本赛题官方定义** | — |
| ③ 是否提供数据集 | **未获取(在登录态参赛指南PDF)**。**本方案最大不确定项(见第6节)** | — |
| ④ 评测指标 | **未获取(在登录态参赛指南PDF)**。mAP/精度/帧率/功耗权重未公开。"高精度"+"边缘计算"强烈暗示同时考核**精度与端侧实时/能耗**，但官方权重未知 | — |
| ⑤ 决赛形式 | **部分获取**：挑战赛道为"现场比拼"；**揭榜赛道决赛形式"将根据参赛队伍数量适当调整"**——是否现场实测/答辩未最终确定。首届决赛 2025-11 江苏南通现场举办，可作参考非本届承诺 | 邀请函 |

## 2. 评分/评审标准

**未获取(在登录态参赛指南PDF)**。仅见 2026-06-10《推荐评审专家入库通知》说明采专家评议制；具体维度/分值未公开。合理推断(非官方)：揭榜赛通常由发榜企业按"是否达成揭榜技术指标"判定 + 创新性/可落地性。**务必登录核对赛题PDF的"考核指标/验收标准"段落**——往往直接给 mAP/FPS/功耗硬门槛。

## 3. 技术路线：低空/水面小目标检测

### 3.1 核心难点
低空无人机视角水面救援目标：**目标极小**(30–80m 航高下落水人员仅数十像素)、**动态背景**(波浪)、**强反光/眩光**(sun glint)、**密集遮挡**。

### 3.2 关键数据集（公开，可预训练/迁移）
- **SeaDronesSee**（海上搜救基准，**最贴合目标域，首选**）：54k+ 无人机开阔水域影像，类别 swimmer/floater/boat/buoy/life jacket，含检测/单目标/多目标跟踪。论文 [WACV 2022 arXiv 2105.01922](https://arxiv.org/abs/2105.01922)｜[主页](https://seadronessee.cs.uni-tuebingen.de/)｜[代码](https://github.com/Ben93kie/SeaDronesSee)。
- **AFO (Aerial Floating Objects)**：首个免费海上搜救航拍集，3647 帧/39991 标注(human/board/boat/buoy/sailboat/kayak)，航高 30–80m。[Kaggle](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects)｜[Roboflow(可导 YOLO 格式)](https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object)。
- **WSODD**(水面小目标船/浮标)、**AI-TOD**(极小目标航拍基准)做域增广/评测协议。

### 3.3 综述与 SOTA
- 综述：[Aerial Person Detection for SAR (Journal of Remote Sensing)](https://spj.science.org/doi/10.34133/remotesensing.0474)、[Maritime SAR with Aerial Images Survey](https://arxiv.org/html/2411.07649v1)、[Maritime UAV Detection Review](https://arxiv.org/pdf/2311.07955)、[Small Object Detection Survey 2025](https://arxiv.org/pdf/2503.20516)。
- SOTA(小目标/水面 YOLO 改进)：[SODet-YOLO (YOLO11n+P2+渐进FPN)](https://doi.org/10.3390/rs18111714)、[SeaLSOD-YOLO (YOLOv11 轻量海上)](https://doi.org/10.3390/sens26072017)、[SMA-YOLO (UAV 多尺度小目标)](https://www.nature.com/articles/s41598-025-92344-7)、抗天气增广 [Augmented Aerial for Maritime SAR](https://arxiv.org/pdf/2408.13766)。

### 3.4 推荐技术栈（对齐"高精度+边缘计算"）
- 检测主干：YOLOv8/YOLO11 + **P2 小目标检测头**(必加)；叠加切片推理 **SAHI** 提小目标召回。
- 反光/波浪：增广模拟 sun glint + 偏振/HSV 扰动；可加通道注意力([CAFE-YOLO](https://www.nature.com/articles/s41598-025-18881-3))。
- 边缘端：PyTorch FP32 训练 → ONNX → **TensorRT FP16 引擎**；Orin Nano 8G 上 YOLOv8n FP16 约 50+ FPS、INT8 约 60+ FPS，但**小目标对 INT8 敏感，精度若掉建议 FP16**。基准 [YOLOv8 on Jetson Orin](https://www.mdpi.com/2073-431X/15/2/74)。

> **🔧 硬件可行性（本轮新增）**：赛题名仅要求"边缘计算"，**是否指定国产平台未获取**(参赛指南 PDF 未提供)。你已有的 **Jetson Orin Nano 本就是主流边缘 AI 平台**，TensorRT FP16/INT8 即可满足"边缘+高精度+实时"，**本赛事大概率无需额外采购国产 NPU 板卡**——是五个赛事里硬件最契合的一个。仅当赛题书/决赛明确要求国产信创平台时，才按 A 篇🔧 b) 路径采购(RK3588 整机 ≈ ¥800–1200)。

## 4. 对照我的背景（契合度很高）

**可直接复用（几乎零增量）**：
- **无人机航拍视角 + 小目标检测**：违建与水面救援同为低空俯视、目标占比小，检测头/切片/训练 trick 可直接迁移。
- **跨域迁移评价方法论（学位论文核心）**：正好对应"陆域公开数据→海域目标域"的迁移问题——**这是你相对其他参赛者最大的差异化优势**。可迁移到本赛题（参考 [跨域人检测+实例分割+MMD](https://www.mdpi.com/2072-4292/15/11/2928)）。
- **Jetson Orin Nano + TensorRT 部署管线**：ONNX→TRT FP16/INT8、端侧测帧率/显存整套可原样复用；8G 显存约束的坑你已踩过。
- **知识蒸馏**：RTX4070S 训教师→Orin Nano 跑学生，直接服务"边缘+高精度"双目标。
- **FastAPI**：边缘端推理服务/可视化/答辩演示接口。

**需新做的增量**：① 水面目标域适配（接 SeaDronesSee+AFO 重训/微调）；② 反光与波浪干扰抑制（陆域无此问题，新增 sun glint/波浪增广与鲁棒性实验）；③ 目标类别重定义（违建单类大目标→救援多类极小目标，调检测头+加 P2/SAHI）；④ 数据缺口（若企业不给数据需自建/合成）；⑤ 指标对齐（补端侧 FPS/功耗 benchmark，Orin Nano 功耗需自测）。

## 5. 工作量粗估（单人，人/周）

| 阶段 | 估时 |
|---|---|
| 赛题确认+数据准备(登录核对PDF、下载 SeaDronesSee+AFO、转YOLO、划域) | 1.0–1.5 |
| 基线复现(YOLO11/v8+P2 跑通水面 baseline+评测脚本) | 1.0 |
| 跨域迁移与增广(陆/海适配、sun glint+波浪、跨域评价复用论文) | 2.0 |
| 精度优化(SAHI、注意力/FPN 改进、消融) | 2.0 |
| 边缘部署(ONNX→TRT FP16/INT8、Orin Nano FPS/功耗实测、蒸馏) | 1.5–2.0 |
| 系统集成与答辩(FastAPI demo、报告 PPT、现场预演) | 1.5 |
| **合计** | **约 9–10 人·周** |

**按 3-4 人团队换算**：9–10 人·周 ÷ 3 人 ≈ **3–3.5 个日历周**，÷ 4 人 ≈ **2.5 周**（数据/检测/部署并行）。距 8/31 约 11 周 → **3-4 人非常宽裕**，可做充分的跨域迁移消融 + 端侧 FPS/功耗实验冲奖。**最大变量仍是数据集是否由出题方提供（未获取，需登录核对）**，若需自建则数据线吃掉约 2 人·周。

## 6. 推荐方案与主要风险

**推荐(一句话)**：YOLO11+P2 小目标头为主干，SeaDronesSee+AFO 做水面域预训练/迁移，复用你的**跨域迁移评价框架**处理"陆→海"域差，反光波浪用增广抑制，端侧 ONNX→TensorRT FP16 在 Orin Nano 8G 做高精度+实时，知识蒸馏压模型，FastAPI 做演示。差异化亮点 = 跨域迁移评价(你的论文强项)+端侧能耗实测。

**主要风险**：
1. **数据缺失(最大)**：是否提供企业数据、是否允许公开数据均未获取。**应对：第一时间登录确认；同步立刻启动 SeaDronesSee+AFO+合成数据 fallback，不等数据。**
2. **评测指标未知**：可能做错优化方向。应对：登录核对"考核指标"；未知前精度与端侧 FPS 双线并进，FP16 优先保小目标精度。
3. **决赛形式未定**：若现场部署实测，需 Orin Nano 现场可复现部署包+备用硬件。
4. **出题企业未知**：无法预研其方案偏好。应对：登录查赛题PDF 发榜单位。
5. **小目标+INT8 掉精度**：默认 FP16，INT8 仅在 FPS 不达标时谨慎启用并做精度回归。

> 行动优先级：**立即登录 cpipc 报名并下载赛题7参赛指南PDF，核对所有"未获取"项**——尤其③数据集与④评测指标，它们决定整个方案走向。

## 7. 信息来源链接

**官方**：[智能建造大赛主页](https://cpipc.acge.org.cn/cw/hp/2c90801795a92a850195d03b537b1bac)｜[第二届邀请函](https://cpipc.acge.org.cn/cw/contestNews/detail/2c90801795a92a850195d03b537b1bac/2c9080189dcfa24e019e39fefa5515d2?page=0)｜[揭榜赛题增补通知 06-12](https://cpipc.acge.org.cn/cw/contestNews/detail/2c90801795a92a850195d03b537b1bac/2c9080189eab1072019ebbd173b91403?page=0)｜[评审专家入库通知 06-10](https://cpipc.acge.org.cn/cw/contestNews/detail/2c90801795a92a850195d03b537b1bac/2c9080189eab1072019eaf9332ae2138?page=0)
**数据集**：[SeaDronesSee 论文](https://arxiv.org/abs/2105.01922)/[主页](https://seadronessee.cs.uni-tuebingen.de/)/[代码](https://github.com/Ben93kie/SeaDronesSee)｜[AFO Kaggle](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects)/[Roboflow](https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object)
**综述/SOTA**：[SAR Survey](https://spj.science.org/doi/10.34133/remotesensing.0474)｜[Maritime SAR Survey](https://arxiv.org/html/2411.07649v1)｜[Maritime UAV Review](https://arxiv.org/pdf/2311.07955)｜[Small Object Survey](https://arxiv.org/pdf/2503.20516)｜[SODet-YOLO](https://doi.org/10.3390/rs18111714)｜[SeaLSOD-YOLO](https://doi.org/10.3390/sens26072017)｜[SMA-YOLO](https://www.nature.com/articles/s41598-025-92344-7)｜[跨域+MMD](https://www.mdpi.com/2072-4292/15/11/2928)｜[海上天气增广](https://arxiv.org/pdf/2408.13766)
**边缘部署**：[YOLOv8 on Jetson Orin Benchmark](https://www.mdpi.com/2073-431X/15/2/74)

---
**核心结论**：赛题名/赛道(揭榜)/时间(8/20、8/31)已从官方邀请函确认，揭榜赛题已增至 12 项。但**出题企业、识别目标、是否提供数据、评测指标、揭榜决赛形式五项公开网页一概未披露**，只能登录态 PDF 获取——已逐项标"未获取"。技术侧路线与你背景高度契合(SeaDronesSee/AFO+YOLO11-P2+跨域迁移评价+Orin Nano TensorRT)，最大风险是数据集缺失，**立即登录核对 PDF 并同步启动公开数据 fallback**。
