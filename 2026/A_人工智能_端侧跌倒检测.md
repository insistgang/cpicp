# 赛事 A · 华为杯第八届中国研究生人工智能创新大赛
## 华为赛题7【低算力端侧视觉跌倒检测】+ 兼报开放赛题3【AI+X】

> 调研日期 2026-06-15｜报名截止 8/25｜作品截止 9/1
> 纪律说明：关键事实均附 URL；论文查不到确切出处的标"未获取"；华为方具体指标/数据/算力约束一律不编造。
> ⚠️ 赛题全文原应在 `./inputs`，但该目录在本机不存在、未提供 → 赛题确切要求标"未获取"。

---

## 1. 赛题要点

- **华为赛题7 确切要求（赛题全文/确切指标/算力上限/评测平台/数据集）**：**未获取（原在 ./inputs，未提供）**。在官方平台 cpipc.acge.org.cn 检索第八届赛题全文未果（仅返回第六、七届）。来源：[cpipc 第七届主页](https://cpipc.acge.org.cn/cw/hp/2c9088a5696cbf370169a3f8101510bd)。
- 本简报**仅基于"低算力端侧视觉跌倒检测"这一公开已知方向**做技术调研，不替华为方设定任何 mAP/FPS/功耗/算力门槛。
- 公开方向可确定的工程内核（行业共识，非赛题方指标）：
  - **端侧实时**：模型须在低算力边缘 NPU/嵌入式实时运行（非云端）。
  - **轻量化**：参数量/FLOPs/内存受限，需剪枝、蒸馏、量化（INT8）。
  - **降误报是关键难点**：把"真实跌倒"与"主动下蹲/坐下/躺下/弯腰"区分开（见第3节②）。
  - **泛化**：夜间/红外、遮挡、长尾（跌倒样本天然稀少）。
- **开放赛题3【AI+X】**：建议把端侧跌倒检测包装成"AI+养老/AI+医疗健康"应用创意，复用同一技术栈，补"产品化/落地/多智能体告警编排"叙事。第八届开放赛题细则**未获取**。

## 2. 评分/评审标准

- **未获取（原在 ./inputs，未提供）**。第八届评分维度/权重未在公开渠道检索到。
- 经验提示（非官方）：此类赛历届通常综合"技术创新性 + 完成度/实测指标 + 端侧落地可行性 + 答辩"。以 ./inputs 原文为准。

## 3. 技术路线（本简报重点）

### ① 三篇参考论文逐篇核心做法

**LFD-YOLO**（置信度高）
- *LFD-YOLO: a lightweight fall detection network with enhanced feature extraction and fusion*，广东工业大学，2025-02，*Scientific Reports*。
- [Nature 全文](https://www.nature.com/articles/s41598-025-89214-7)｜DOI 10.1038/s41598-025-89214-7｜[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11814241/)
- 核心（基线 YOLOv8s）：**CSRG**(Cross Split RepGhost) + **EMA** 注意力 + **WFPN** 加权融合金字塔 + **GSConv** + **Inner-WIoU** 损失。指标：参数 −48.6%、计算量 −56.1%，mAP@0.5 微升约 0.3%。

**BMR-YOLO**（置信度高）
- *BMR-YOLO: A deep learning approach for fall detection in complex environments*，2025-11，*PLOS ONE*。
- [PLOS 全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335992)｜DOI 10.1371/journal.pone.0335992｜[PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594333/)
- 核心（基线 YOLOv8n，名=**B**iFormer+**M**ultiSEAM+**R**VB）：主干末端 **BiFormer**(动态稀疏注意力) + **C2f_RVB** + 检测头 **MultiSEAM**(抗遮挡) + **SIoU**。自建 BMR-fall（11000+ 图，70% 遮挡 + 30% 低光），URFD/Le2i 交叉验证。指标：mAP@0.5 0.852→**0.899**，**6.5 GFLOPs**。"复杂环境/遮挡/低光"导向与本赛题夜间+遮挡难点高度对口。

**YOLO-fall**（置信度中-高，存在同名两篇，需确认你引用哪篇）
- 轻量化版：*YOLO-fall: ... high precision, shrunk size, and low latency*，2025-07，*The Computer Journal*(Oxford)。[Oxford](https://academic.oup.com/comjnl/advance-article-abstract/doi/10.1093/comjnl/bxaf005/7994663)｜DOI 10.1093/comjnl/bxaf005。轻量主干+重参数化卷积+改进C3+5×5深度卷积；E-FPDS；mAP 78.4%，参数 2.45M，12.2 GFLOPs。
- 另一同名（开放空间，非轻量）：[ResearchGate](https://www.researchgate.net/publication/378019448)。**请确认你指哪篇。**

> 三篇共同范式：都在 YOLOv8n/s 上做"注意力+轻量卷积+IoU损失+特征融合"，把跌倒当**单帧检测**。局限：单帧分不清"已躺地"与"主动躺下"，需 ② 时序/姿态补强。

### ② 轻量 YOLO 上接动作/姿态识别，区分跌倒 vs 日常动作

核心结论：**单帧规则**便宜但分不清主动/被动；降误报必须引入**时序**（下落速度、质心高度突变、落地静止时长）。
- 单帧规则法综述：[PMC8321307](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321307/)。
- 关键帧姿态接法：YOLOv8s+AlphaPose [PMC11751301](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751301/)；YOLOv7-W6-Pose 一体化+20帧投票 [MDPI](https://www.mdpi.com/1999-5903/16/12/472)；CPU 实时姿态 [arXiv 2503.19501](https://arxiv.org/pdf/2503.19501)。
- 轻量姿态模型：[YOLOv8/11-pose](https://docs.ultralytics.com/tasks/pose)（推荐，检测+关键点合一）、[MoveNet](https://www.tensorflow.org/hub/tutorials/movenet)、[BlazePose/MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)、[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)（支持 ncnn 端侧）。
- 骨架时序 GCN：[ST-GCN](https://arxiv.org/abs/1801.07455)([代码](https://github.com/yysijie/st-gcn))、[2s-AGCN](https://arxiv.org/abs/1805.07694)、[PoseConv3D](https://arxiv.org/abs/2104.13586)。**强推工程参考**：[Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)（Tiny-YOLO+AlphaPose+ST-GCN+SORT，动作类含 Standing/Sitting/Lying Down/Sit down/Fall Down）。
- 端侧最现实的轻量时序：关节点序列接 **TCN/GRU**（最轻可量化）；端侧排序 **TCN/GRU > LSTM > ST-GCN > PoseC3D**。
- **降误报状态机（核心）**：质心高度快速突降（速度阈值滤掉慢速主动躺/蹲）+ 落地后静止 ≥ T 秒（T≈1–3s）+ 期间无主动起身。

### ③ 端侧 NPU 量化部署链路（与 Jetson+TensorRT 的映射）

> ATC≈trtexec、.om≈.engine、AscendCL≈Runtime、AMCT≈量化校准器；RKNN 的 convert.py≈builder、.rknn≈.engine。四家都把 Detect 解码+NMS 放 host CPU。

**昇腾 Ascend（最接近 TensorRT 心智）**：CANN→**ATC**(ONNX→.om)→**AMCT**(量化)→**AscendCL/pyACL**。opset 11/12；动态 shape 用 `--dynamic_dims`；输出节点 `onnx::` 前缀会让 ATC 失败需改名；`soc_version` 须严格匹配(310/310B/310P3 不通用)。来源：[AMCT 文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/amct/atlasamct_16_0131.html)、[yolov5-ascend](https://github.com/jackhanyuan/yolov5-ascend)。

**海思 Hi35xx NNIE（工程量最重、SDK 闭源需授权，版本/下载链接未获取）**：**nnie_mapper**(转 .wk，RuyiStudio)，链路常为 PyTorch→ONNX→**Caffe**→.wk。算子坑：Focus→stride2 卷积；**SiLU 不支持→ReLU/LeakyReLU**(掉点)；Upsample→ConvTranspose；只支持 4D reshape→去 Detect 的 permute。来源：[NNIE 算子坑](https://blog.csdn.net/weixin_41765899/article/details/125514791)、[华为云博客](https://bbs.huaweicloud.com/blogs/395170)。

**瑞芯微 RK3588 / RKNN（约 6 TOPS INT8，INT8 近乎刚需）**：仓库迁到 [airockchip](https://github.com/airockchip/rknn-toolkit2)（v2.3.2）。三件套 rknn-toolkit2(PC转换)/lite2(板上py)/RKNPU2(板上C)。坑：toolkit 锁 `onnx==1.14.1` 需独立 venv；**ONNX 导出必须用 [airockchip ultralytics fork](https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md)**(移除后处理+DFL+加 score-sum)；必须 x86 PC 转换；固定 640×640；混合量化两步；**PC toolkit / 板上 librknnrt.so / 内核 NPU driver 三处版本必须对齐**；DFL/NMS 必须移出模型。来源：[Ultralytics RKNN 文档](https://docs.ultralytics.com/integrations/rockchip-rknn/)、[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/README.md)。

> **🔧 硬件可行性（本轮新增 · 我当前只有 RTX4070S + Jetson Orin Nano + 8G 边缘设备，无海思/瑞芯微国产 NPU 板卡）**
> 赛题7 是否强制华为端侧平台(海思/昇腾)**未获取**(赛题书未提供)。两条路径：
> **a) 仅用 Jetson + 仿真/ONNX**：算法研发、轻量化(蒸馏/剪枝/INT8)、精度评估、以及"在 Orin Nano 上端侧 FPS/功耗实测"全部可做并写进论文/答辩；ONNX 导出后用 onnxruntime/onnxsim 验证算子兼容性。**足以证明端侧可行性 + 拿到完整实验**，决赛若不强制特定国产板即可冲奖。
> **b) 若评分强制特定国产平台才达标**：需采购开发板——瑞芯微 **RK3588**(约6 TOPS)首选，整机 **香橙派 Orange Pi 5 Plus 16G ≈ ¥800–1200**、**Firefly ROC-RK3588 ≈ ¥1500–2500**；昇腾 **Atlas 200I DK A2 ≈ ¥1000–1500**；**海思 Hi35xx 板卡一般不零售、SDK 需授权，不建议自购**(渠道/价位未获取)。价位为大致市场价，以实时电商为准。
> **结论**：先按 a) 用 Jetson 跑通全流程；仅当赛题书明确要求海思/昇腾时再按 b) 采购(优先 RK3588 整机 ¥1k 内)。

### ④ 红外/夜间 + 长尾泛化

- 红外/热成像：[红外+ST-GCN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11280873/)、[红外相机昼夜检测](https://dl.acm.org/doi/abs/10.3233/AIS-210605)、[暗环境改进 YOLOv5s](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11175020/)。
- 域适应（RGB→热/昼→夜）：[对抗域适应到热成像](https://arxiv.org/pdf/2106.07165)、[渐进式昼夜域适应](https://arxiv.org/pdf/2407.19430)。
- 合成数据：[CycleGAN 合成热成像](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11175020/)、[多级 GAN 低光热成像](https://arxiv.org/pdf/2209.09808)。
- 长尾：[Equalized Focal Loss (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Equalized_Focal_Loss_for_Dense_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf)、[Awesome-Long-Tailed](https://github.com/GZWQ/Awesome-Long-Tailed)。注意：重采样在跨域时会加剧域漂移。

### ⑤ 开源数据集

| 数据集 | 模态/规模 | 链接 |
|---|---|---|
| **OmniFall**（2025 统一基准，强推） | RGB，约 80h/约 15000 段，16 类；含 OF-Staged/OF-Synthetic/**OF-In-the-Wild(真实意外跌倒)** | [arXiv](https://arxiv.org/abs/2505.19889)｜[HF](https://huggingface.co/datasets/simplexsigil2/omnifall) |
| Kaggle Fall (10000 videos) | 约 10000 段 RGB | [Kaggle](https://www.kaggle.com/datasets/unidpro/fall-detection) |
| Kaggle Thermal Fall（红外，重点） | 热成像跌倒+活动 | [Kaggle](https://www.kaggle.com/datasets/boeychunhong/thermal-fall-detection-and-activity-dataset) |
| URFD | RGB+深度+IMU+骨架，70序列 | [官网](https://fenix.ur.edu.pl/mkepski/ds/uf.html) |
| Le2i (IMVIA) | RGB，191段，帧级标注 | [Kaggle镜像](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) |
| UP-Fall | 多模态，17人/11类 | [数据页](https://sites.google.com/up.edu.mx/challenge-up-2019/data) |
| SisFall（含真实老人） | 可穿戴 IMU，含15名60-75岁老人 | [论文](https://www.mdpi.com/1424-8220/17/1/198) |
| eHomeSeniors（红外热阵列，隐私友好） | Omron/Melexis 热阵列，448次跌倒 | [论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6832422/) |

> 关键提示：**真实老人数据极少**，多为成人模拟；含真实老人主要是 SisFall、eHomeSeniors，OmniFall In-the-Wild 含真实事故。许可多为学术用途，商用需逐一确认。

## 4. 对照我的背景

**可直接复用**：YOLO 系+实例分割（三篇论文都基于 YOLOv8 改）；知识蒸馏（端侧轻量化核心，可作论文创新点）；TensorRT+Jetson Orin Nano（开发期 baseline / 高算力对照，昇腾链路最接近）；FastAPI（告警服务/可视化）；多智能体（"检测→姿态→时序→投票"降误报编排，AI+X 差异化叙事）。

**需新做的增量**：① **国产 NPU 链路**（海思 NNIE 或 RK3588，与 Jetson 不同：PC 端转换、INT8 校准、DFL/后处理移出、算子替换、版本对齐）；② **动作/时序判别**（你偏检测/分割，需新增"关键点序列+TCN/GRU+状态机"）；③ **红外/夜间数据与域适应**。

## 5. 工作量粗估（人/周，单人折算）

| 模块 | 估时 |
|---|---|
| 复现 LFD/BMR/YOLO-fall + 选 baseline | 1.5 |
| 数据整合(OmniFall+Le2i+URFD)+标注统一 | 1.5 |
| 轻量化(蒸馏/剪枝/结构改进)训练 | 2 |
| 姿态+时序降误报(YOLOv8-pose+TCN+状态机) | 2 |
| 红外/夜间数据+域适应/合成 | 1.5–2 |
| **国产 NPU 部署(RK3588 优先;NNIE 更久)** | **2–3** |
| FastAPI 告警+多智能体编排+Demo | 1 |
| 论文/文档/答辩 | 1.5 |
| **合计** | **约 13–15 人·周** |

**按 3-4 人团队换算**：13–15 人·周 ÷ 3 人 ≈ **4.5–5 个日历周**，÷ 4 人 ≈ **3.5–4 周**（检测/部署/数据三线并行）。距 9/1 约 11 周 → **3-4 人节奏宽裕**；真正瓶颈不在人力，而在**国产 NPU 硬件(见上🔧)与赛题指标未知**。红外可"合成+少量真实"控成本。

## 6. 推荐方案与主要风险

**推荐主线**：YOLOv8n/11n + 借鉴 LFD 的 GSConv/EMA、BMR 的抗遮挡头做轻量化 + 知识蒸馏 → **降误报核心**(YOLOv8n-pose 关键点 + TCN/GRU 动作分类 + 质心突降速度阈值+落地静止时长状态机) → **端侧优先 RK3588+RKNN**，Jetson+TensorRT 作高算力对照 → 夜间用热成像+CycleGAN合成+域适应+Equalized Focal Loss → FastAPI+多智能体编排，包装"AI+智慧养老"。

**主要风险**：
1. **赛题确切指标未知（最大）**：mAP/FPS/算力/评测平台全未获取，选型可能返工。**尽快补 ./inputs 原文**。
2. **国产 NPU 掉点与算子坑**：SiLU→ReLU、DFL 移除、INT8 后可能掉 1–3% mAP；NNIE SDK 需授权。**你硬件清单是 Jetson+8G 边缘设备，是否有 RK3588/海思板卡未明确——若无需购板或仿真。**
3. **真实老人/夜间数据稀缺**：泛化论证不足，合成域差距是答辩易被质疑点。
4. **时间紧**：单人 13–15 周 vs 剩约 11 周。
5. **YOLO-fall 同名歧义**：确认引用哪篇。

## 7. 信息来源链接

- 官方平台(仅检索到六/七届)：https://cpipc.acge.org.cn/cw/hp/2c9088a5696cbf370169a3f8101510bd
- 论文：[LFD-YOLO](https://www.nature.com/articles/s41598-025-89214-7)｜[BMR-YOLO](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335992)｜[YOLO-fall 轻量版](https://academic.oup.com/comjnl/advance-article-abstract/doi/10.1093/comjnl/bxaf005/7994663)｜[YOLO-Fall 开放空间版](https://www.researchgate.net/publication/378019448)
- 姿态/时序：[ST-GCN](https://arxiv.org/abs/1801.07455)｜[PoseConv3D](https://arxiv.org/abs/2104.13586)｜[Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)｜[YOLOv8-pose](https://docs.ultralytics.com/tasks/pose)｜[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)｜[YOLOv7-W6-Pose 跌倒](https://www.mdpi.com/1999-5903/16/12/472)｜[姿态规则综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321307/)
- NPU 部署：[airockchip rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)｜[RKNN model zoo YOLOv8](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/README.md)｜[airockchip ultralytics fork](https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md)｜[Ultralytics RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/)｜[昇腾 AMCT](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/amct/atlasamct_16_0131.html)｜[NNIE 算子坑](https://blog.csdn.net/weixin_41765899/article/details/125514791)
- 数据集：[OmniFall](https://arxiv.org/abs/2505.19889)/[HF](https://huggingface.co/datasets/simplexsigil2/omnifall)｜[URFD](https://fenix.ur.edu.pl/mkepski/ds/uf.html)｜[Le2i](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)｜[UP-Fall](https://sites.google.com/up.edu.mx/challenge-up-2019/data)｜[SisFall](https://www.mdpi.com/1424-8220/17/1/198)｜[eHomeSeniors](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6832422/)｜[Kaggle 10000](https://www.kaggle.com/datasets/unidpro/fall-detection)｜[Kaggle Thermal](https://www.kaggle.com/datasets/boeychunhong/thermal-fall-detection-and-activity-dataset)
- 长尾：[Equalized Focal Loss](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Equalized_Focal_Loss_for_Dense_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf)

---
**需你决定/补充**：① 补 ./inputs 华为赛题7 原文（确切指标关系选型）；② 确认手头是否有 RK3588 或海思板卡（清单里只有 Jetson+8G 边缘设备）。
