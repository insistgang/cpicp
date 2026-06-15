# C_baseline · 港股IPO招股书解析 + 多智能体风险预警 最小链路脚手架

> 目标：跑通 MVP —— **解析 → 证据可追踪问答 → 财务暗雷抽取**，再扩 市场情绪共振 + 5 日下跌预警。
> 硬件：RTX4070S(本地 OCR/向量/时序) + 云端大模型 API（DeepSeek/通义，OpenAI 兼容）。
> 对齐赛题书硬指标：抽取准确率≥80% / 证据召回≥85% / **推理可追踪率 100%** / 重点"上市后5日内显著下跌"。

## 运行顺序
```bash
# 0) 环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 LLM_API_KEY / LLM_BASE_URL / EMBED_* （DeepSeek/通义 等 OpenAI 兼容端点）

# 1) 解析：招股书 PDF → 结构化 JSON（表格/数字走确定性抽取，不让 LLM 编数）
python parse_prospectus.py --pdf samples/demo_prospectus.pdf --out out/demo.json

# 2) 建索引：LightRAG 建知识图谱 + 向量库
python build_index.py --doc out/demo.json --workdir out/rag_demo

# 3) 风险问答 + 证据 span 回链（对齐"可追踪率100%"）
python risk_qa.py --workdir out/rag_demo --q "该公司是否存在对赌/赎回条款？现金消耗压力如何？"

# 4) 多智能体编排（法务合规/财务穿透/市场分析 + Critic）——先空实现可跑，再接真实逻辑
python agents_graph.py --doc out/demo.json --demo

# 5) 评测（按赛题书口径）：抽取准确率 / 证据召回 / 可追踪率
python eval_metrics.py --pred out/pred.json --gold samples/gold.json

# 6) API：输入 公司名/代码/招股书 → 输出 预警+报告
uvicorn api_app:app --reload --port 8000
# POST /predict  {"company":"XX","stock_code":"xxxxx.HK","prospectus_path":"samples/demo_prospectus.pdf"}
```

## 文件
| 文件 | 作用 | 关键库 |
|---|---|---|
| `requirements.txt` / `.env.example` | 依赖 / 模型端点配置 | — |
| `parse_prospectus.py` | MinerU 跑招股书 PDF → 结构化 JSON（确定性表格/数字） | MinerU(mineru) |
| `build_index.py` | LightRAG 建图 + 向量库 | LightRAG |
| `risk_qa.py` | 风险问答 + **证据 span 回链**（只取 retrieved context 作证据） | LightRAG |
| `agents_graph.py` | LangGraph 状态机：法务合规/财务穿透/市场分析 + Critic（空实现+接口） | LangGraph |
| `eval_metrics.py` | 抽取准确率 / 证据召回 / 可追踪率 评测模板 | — |
| `api_app.py` | FastAPI：公司名/代码/招股书 → 预警+报告 | FastAPI |

## 设计纪律（对齐赛题书）
- **防幻觉**：表格/财务数字由 MinerU 确定性抽取，**LLM 只做语义判断、不重写数字**；每个风险结论强制回链到原文 span（满足可追踪率 100%）。
- **可运行性**：交付前用 `docker build` + 一键 `make run` 自检（赛题初赛会统一跑可运行性测试）。
- **MVP 先行**：先把 1→2→3（解析→问答→抽取）跑通拿到 ≥80%/≥85%，再上 4/市场情绪/5 日预警。

## TODO
- `TODO[数据]` 领取东吴数据集后，按其字段对齐 `parse_prospectus.py` 输出 schema 与 `samples/gold.json` 标注。
- `TODO[登录核对]` 评分分项数值权重未获取。
