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

## 文件清单

| 文件 | 作用 | 自测方式 | 状态 |
|---|---|---|---|
| `prepare_data.py` | 数据清单加载 + 分类集/相似度对构建 + **合成数据生成器**(无真实数据也能跑通流水线) | `python prepare_data.py --selftest` | 通过 |
| `metrics.py` | **阈值扫描 P/R/F1 + 最优阈值选取 + ROC-AUC + Top-k 检索准确率**(官方核心交付) | `python metrics.py` | 通过 |
| `similarity.py` | 嵌入相似度索引 + **去重检测**(高相似对,标记跨客户套用 vs 同客户重复) | `python similarity.py` | 通过 |
| `classify.py` | CLIP 零样本/线性探针 分类器筛面签照片(需 torch/open_clip,在算力机上跑) | `py_compile` 通过 | 骨架就绪 |
| `embed.py` | 度量学习嵌入训练/提取(CLIP/DINOv2 + 投影头) | `py_compile` 通过 | 骨架就绪 |
| `app_demo.py` | Gradio 检测汇总视图(统计+可疑对+业务数据) | `python app_demo.py --demo` | 通过 |
| `pipeline.py` | **端到端流水线**: 合成数据模拟完整流程(分类→嵌入→去重→阈值分析→汇总) | `python pipeline.py --selftest` | 通过 |
| `run_all_selftests.py` | **一键全模块自测** + HTML 报告生成 | `python run_all_selftests.py --verbose --html` | 通过 |
| `Dockerfile` | 初赛"统一可运行性测试"硬门槛:一键 Docker 构建 | `docker build -t fintech23 .` | 就绪 |
| `requirements.txt` | 依赖清单(torch/open_clip/faiss/sklearn/pandas/matplotlib/gradio) | — | 就绪 |

## 快速开始

### 无 GPU/无数据环境验证(纯逻辑)
```bash
# 一键全模块自测(4 个模块全部通过)
python run_all_selftests.py --verbose --html

# 端到端流水线演示(合成数据)
python pipeline.py --selftest

# 生成阈值-P/R 曲线图
python pipeline.py --plot --n-groups 40 --reuse-frac 0.3
```

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
- `classify.py` 和 `embed.py` 依赖 torch/open_clip/timm,在 Mac 无 GPU 环境下无法运行,骨架已就绪,待算力机上填充。
