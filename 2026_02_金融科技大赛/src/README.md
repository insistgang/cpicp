# 金融科技赛题#23 · 金融影像智能分类 + 相似度检测 — 代码 (src)

> 赛题:**基于多模态大模型的金融影像智能相似度检测模型**（揭榜挂帅 #23，命题方 **无锡农村商业银行**）
> 本 src 按**官方赛题原文**(docs 内 23-…docx)构建。早期 README 写的"Qwen2.5-VL+DINOv3 篡改定位/真伪鉴别"是**读错赛题**,已废弃——真任务见下。

## 官方赛题事实(据 23-…docx 原文核实)

**任务本质 = 影像分类 + 相似度去重检测,不是篡改/真伪鉴别。**
信贷面签环节采集大量多类型影像(面签合影/证件/权属证明/合同文档…)。要做两步端到端:
1. **影像分类**:从海量多类型影像中准确**筛出"面签照片"**;
2. **相似度检测**:对筛出的面签照片做**向量化 + 相似度比对**,发现**重复使用 / 跨客户套用**(同一张面签照被反复提交、或套用到不同客户)。

| 项 | 官方要求 |
|---|---|
| 命题方 | 无锡农村商业银行(技术联系 胡陆欢 huluhuan@wrcb.com.cn) |
| 数据领取 | 签《数据资料保密承诺书》后承办方发指引;联系 南京大学 苏老师 **jrkjds@nju.edu.cn** |
| 数据 | **合成数据集**:多类型影像,每张带 ①类型标签(分类用) ②相似度标注(检测用,已构造相似/重复对作正样本) ③业务类型标签 |
| 官方现有基础 | 生产系统用 **CLIP** 做特征提取→分类过滤面签→向量化→相似度计算(商户易贷已落地,检测>4万张) |
| 量化/交付 | 分类准确率 + 相似度检测精度;**阈值选取策略 + 不同阈值对 P/R 的影响分析**;检测汇总视图(统计+查看高相似可疑交易及关联业务数据);完整代码+环境说明+技术报告(方法/实验设计/结果/**消融**)+可运行 demo |
| 时间线 | **报名截止 8/17**(最早红线,之后不得改题换人) ｜ 作品提交 **8/31 15:00**(精确到分) ｜ 决赛 10月中下旬现场答辩 |
| 硬性出局 | 统一**可运行性测试**(跑不起来直接扣分)→ 必须 Docker 一键可跑;**原创性声明**未交无初赛资格;**匿名**(不得出现校名/logo) |

## 技术路线
```
多类型影像
   │  ① 分类(筛面签)：CLIP/ViT 零样本或微调分类器  → 只保留"面签照片"
   ▼
面签照片
   │  ② 度量学习嵌入：backbone(CLIP/DINOv2) + ArcFace/对比/三元组 → 判别性向量
   ▼
向量库(FAISS/余弦)
   │  ③ 相似度检索 + 去重判定：找高相似对,区分"同客户重复" vs "跨客户套用(可疑)"
   ▼
④ 阈值策略：扫阈值出 P/R/F1 曲线,给推荐阈值与权衡分析(官方明确要求)
   ▼
⑤ 检测汇总视图 demo：统计 + 可疑对影像并排 + 关联业务数据
```

## 双特征后端:CLIP(真特征)+ 经典 CPU 特征(无 GPU baseline)

官方现有基础是 **CLIP**,但 CLIP 需 torch/open_clip + GPU。本 src 提供**双后端 + 优雅回退**:
torch 可用 → 走 CLIP/timm 真特征;**torch 不可用(Mac 无 GPU/离线)→ 自动回退到
`features.py` 的经典 CPU 特征**(PIL 颜色直方图 + 灰度梯度/分块统计 + sklearn PCA),
使整条流水线 **prepare_data→特征→similarity 去重→metrics** 能在**真实图像像素**上
端到端跑通(baseline),输出**真实** AUC / Top-k / 去重检出,而非随机向量。
真特征接口(`extract_embeddings_clip` / `zero_shot_classify`)保留,算力机上无缝切回 CLIP。

> 经典特征注意:这些是结构化非负直方图特征,默认**不做 StandardScaler**(保守工程默认,
> 避免低方差噪声维被放大到与判别维同权),仅 PCA 去相关降噪——在本合成集上该选择的增益落在
> 噪声裕度内(见下方消融,随 seed 翻转),真实数据上需重跑消融再定。其余弦量纲与 CLIP 不同
> (最优阈值远低于 0.85),故去重阈值**从标注数据自动选最优 F1**(对齐官方"阈值选取策略"要求),不套用固定 0.85。

## 文件清单

| 文件 | 作用 | 自测方式 | 状态 |
|---|---|---|---|
| `prepare_data.py` | 数据清单加载 + 分类集/相似度对构建 + 合成清单生成器 | `python prepare_data.py --selftest` | ✅ 通过 |
| `metrics.py` | **阈值扫描 P/R/F1 + 最优阈值选取 + ROC-AUC + Top-k 检索准确率**(官方核心交付) | `python metrics.py` | ✅ 通过 |
| `similarity.py` | 嵌入相似度索引 + **去重检测**(高相似对,标记跨客户套用 vs 同客户重复) | `python similarity.py` | ✅ 通过 |
| `features.py` | 🆕 **经典 CPU 特征提取器**(CLIP 回退后端):PIL 颜色直方图+梯度/分块+sklearn PCA | `python features.py --selftest` | ✅ 通过 |
| `synth_images.py` | 🆕 **程序化生成合成金融票据/证照影像**(PIL):多模板+重复/跨客户套用样本+manifest | `python synth_images.py --selftest` | ✅ 通过 |
| `classify.py` | CLIP 零样本/线性探针筛面签;**无 torch 时回退经典特征线性探针**(真实像素有监督筛) | `python classify.py --selftest` | ✅ 通过 |
| `embed.py` | 嵌入提取(CLIP/DINOv2);**无 torch 时回退经典特征**(`extract_embeddings` 自动切换) | `python embed.py --selftest` | ✅ 通过 |
| `app_demo.py` | Gradio 检测汇总视图(统计+可疑对+业务数据) | `python app_demo.py --demo` | ✅ 通过 |
| `pipeline.py` | **端到端流水线**:模拟向量路径 + 🆕 **真实像素路径**(经典特征跑在真实 PNG 上,出真 AUC/Top-k) | `python pipeline.py --selftest` | ✅ 通过 |
| `viz_dedup.py` | 🆕 **去重结果可视化**(matplotlib):相似度矩阵热图 + 跨客户套用可疑对拼图 | `python viz_dedup.py --selftest` | ✅ 通过 |
| `ablation_study.py` | 🆕 **消融实验框架**(官方明确要求):特征组成(色/梯度/块/全拼接)+ PCA 维度扫描 + 标准化 on/off,出 CSV 对照表 | `python ablation_study.py --selftest` | ✅ 通过 |
| `run_all_selftests.py` | **一键全模块自测(11 模块)** + HTML 报告生成 | `python run_all_selftests.py --verbose --html` | ✅ 通过 |
| `setup_gpu.sh` | 🆕 **算力机一键上机包**:装 torch+open_clip → classify/embed 真特征 → pipeline `--backend clip` 出真指标(与 04 风格一致) | `bash setup_gpu.sh`(本机 baseline 自检)/ `INSTALL=1 bash setup_gpu.sh`(算力机装真栈) | bash -n ✅ / 本机 baseline 分支已跑通 |
| `Dockerfile` | 初赛"统一可运行性测试"硬门槛:一键 Docker 构建 | `docker build -t fintech23 .` | 就绪 |
| `requirements.txt` | 依赖清单(**两档分明**:CPU baseline 区块 + GPU 真特征追加区块) | — | 就绪 |

## 快速开始

### 无 GPU/无数据环境验证(本地 Mac 即可,跑在真实合成图像像素上)
```bash
# 一键全模块自测(11 个模块全部通过)
python run_all_selftests.py --verbose --html

# 端到端流水线自测(模拟向量 + 真实像素两条路径)
python pipeline.py --selftest

# ⭐ 一键:程序化生成合成金融影像 → 经典特征端到端去重 → 可视化(全程真实像素)
python pipeline.py --gen-and-run --out-dir output/synth_demo --n-groups 30 --reuse-frac 0.35
#   产出: output/synth_demo/*.png(合成影像) + manifest.csv + dedup_viz.png(去重可视化)

# 单独跑各环节
python synth_images.py --out output/synth --n-groups 30          # 只生成合成影像
python pipeline.py --real-images output/synth --plot              # 对已有影像跑端到端+可视化
python features.py --images output/synth/img_*.png                # 看经典特征提取

# 生成阈值-P/R 曲线图(模拟向量)
python pipeline.py --plot --n-groups 40 --reuse-frac 0.3

# ⭐ 消融实验(官方技术文档明确要求):特征组成 + PCA 维度 + 标准化 on/off → CSV 对照表
python ablation_study.py --gen --n-groups 40 --reuse-frac 0.3 --csv output/ablation.csv
```

> **消融结论(经典特征 baseline,真实合成像素,上方文档命令 `--gen --n-groups 40 --reuse-frac 0.3`,默认 seed=0,实测可复现):**
> - **特征组成**:色直方图 AUC=**0.918** 明显 < 梯度(HOG)=0.999 ≈ 块统计=0.997 ≈ 全拼接=0.993;
>   稳健结论(跨 seed 一致)是**色调最弱、梯度/块统计最判别**——版式/笔画的梯度与空间结构比整体色调更能判别"同一张/套用"。
> - **PCA 维度**:0(不降维,176 维)/16/32/64/128 维 AUC 均≈0.99,16–32 维即足够(降维去相关、压缩检索代价),高维无明显增益。
> - **标准化**:本数据上 `standardize` 的影响落在**噪声裕度内、且方向随 seed 翻转**——seed=0 上 on 反而略**升**(AUC 0.993→1.0、F1 0.937→1.0),seed=2 上 on 略降(1.0→0.9998),seed=3/4 上 on/off 持平。
>   因此**不能据本合成集断言"标准化拉低 AUC"**;`features.py` 默认 `standardize=False` 是针对**结构化非负直方图特征**的保守工程默认(避免低方差噪声维被放大),而非本集实测优势——真实数据/CLIP 特征上需重跑本消融再定。
>
> 注:合成数据本身可分性极高(多数配置已触顶 AUC≈1.0),消融差异裕度小、对随机种子敏感;**上表数值均取自文档命令的默认 seed=0,可逐位复现**。换 CLIP 真特征 / 真实数据后裕度增大、结论更稳,
> 此框架原样复用(把 backbone 嵌入喂入同一套 `_auc_for_embeddings` 即可)。

### 拿到合成数据后
```bash
pip install -r requirements.txt
# 0) 先用合成数据验证流水线
python prepare_data.py --selftest
# 1) 真实数据:按官方清单格式放好,构建分类集与相似度对
python prepare_data.py --manifest data/manifest.csv --out data/prepared
# 2) 分类筛面签(算力机上)
python classify.py --mode zeroshot --images data/faces/*.jpg
# 3) 嵌入提取(算力机上)
python embed.py --images data/faces/*.jpg --backbone clip --out embeddings
# 4) 相似度去重 + 阈值分析 + demo
python similarity.py
python metrics.py
python app_demo.py --embeddings embeddings.npy --ids embeddings.json
```

### ⭐ GPU 上机(算力机切 CLIP 真特征,小队照着跑)

本 src **双特征后端**:本机 Mac 跑经典 CPU baseline,算力机装 torch+open_clip 后切 **CLIP 真特征**。
一键脚本 `setup_gpu.sh`(与 04 风格一致)把"装栈 → 真特征跑 classify/embed → pipeline 出真指标"串成一条命令。

```bash
# 0) 本机 Mac(无 torch):只做 baseline 自检,确认逻辑通(本脚本 baseline 分支已实测跑通)
bash setup_gpu.sh

# 1) 算力机:一键装 torch+open_clip+timm+faiss,再用 CLIP 真特征跑全链路
INSTALL=1 bash setup_gpu.sh
INSTALL=1 CUDA=cu121 bash setup_gpu.sh        # CUDA wheel 标签按机器改(cu118/cu121/cu124)
INSTALL=1 NG=60 OUT=output/gpu_run bash setup_gpu.sh   # 自定组数/输出目录
```

脚本做 4 步:① `INSTALL=1` 时按 CUDA wheel 装 torch + 从 requirements GPU 区块装 open_clip/timm/faiss;
② 探测 `torch.cuda.is_available()` 打印 GPU 名;③ 生成合成影像 → `classify.py`(CLIP 零样本筛面签)+ `embed.py --backend clip`(CLIP 真特征嵌入);
④ `pipeline.py --real-images <out> --backend clip --plot` 端到端出**真 AUC / 最优阈值 / Top-k**。
无 torch 时每步自动退到经典 baseline(便于本机先验证逻辑)。

**关键:`pipeline.py` 新增 `--backend` 切真实像素特征后端**(默认 `classic` 保持本机零依赖):

| `--backend` | 特征来源 | 何时用 |
|---|---|---|
| `classic`(默认) | `features.py` 经典 CPU 特征(PIL+sklearn) | 本机 Mac / 无 GPU / Docker 自检 |
| `clip` | `embed.extract_embeddings` → open_clip CLIP 真特征 | 算力机(无 torch 直接报错,便于上机自检) |
| `auto` | 有 torch 走 CLIP,否则回退经典 | 不确定环境时的稳妥默认 |

```bash
# 算力机手动单跑(不走 setup_gpu.sh 时)
python pipeline.py --real-images output/synth --backend clip --clip-backbone clip --plot
python pipeline.py --real-images output/synth --backend auto   # 有 torch 用 CLIP,否则经典
```

> 之前 `run_pipeline_real_images` 硬编码经典特征,即便算力机装了 torch 也切不到 CLIP;现已接 `embed.extract_embeddings`,
> README 承诺的"算力机无缝切回 CLIP 真特征"对 pipeline 路径也真正生效。

### Docker 构建(可运行性测试)
```bash
docker build -t fintech23 .
# 自检(纯逻辑,无 GPU/无数据)
docker run --rm fintech23
# 全流程(需 GPU + 真实数据)
docker run --gpus all -v $PWD/data:/app/data fintech23 python embed.py --images ...
```

## 注意事项
- **官方现有基础就是 CLIP**,优先沿 CLIP 路线(分类用零样本/线性探针,相似度用微调嵌入),别盲目上 7B VLM——任务是检索/去重,不是生成/问答。
- 数据是**合成**的且有现成相似/重复标注 → 度量学习(对比/ArcFace)有监督信号,重点打磨**阈值选取**(官方明确考)。
- `2026/C_baseline`(招股书多智能体/东吴证券)是**另一道题**,与本赛题无关,勿复用。
- `classify.py` / `embed.py` 优先走 CLIP/timm(需 torch),但**已实现经典 CPU 特征回退**:
  在 Mac 无 GPU 环境下也能用 `features.py` 的经典特征在**真实合成影像**上跑通分类与嵌入
  (baseline),无需 torch 即可 `--selftest` 验证。算力机装好 torch+open_clip 后自动切回 CLIP 真特征。
- 真实流水线性能(本地实测,经典特征 baseline,121 张合成影像 / 33 张面签):
  **AUC ≈ 0.99,Top-1 同组检索 = 1.0,最优 F1 阈值 ≈ 0.25(F1 ≈ 0.94)**——证明流水线跑在
  真实像素而非随机向量上。换 CLIP 真特征后预期进一步提升(尤其细微"套用"对的判别)。
