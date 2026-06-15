#!/usr/bin/env python3
"""
api_app.py · FastAPI 服务：输入 公司名/股票代码/招股书 → 输出 预警 + 报告

对齐赛题书"成果交付：可运行的风险预警原型系统或 API 服务，支持输入公司名称、股票代码
或招股书文件后，输出预警提示和报告，支持投研人员人机协同复核"。

运行:
  uvicorn api_app:app --reload --port 8000
  curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"company":"XX生物","stock_code":"xxxxx.HK","prospectus_path":"samples/demo.pdf"}'

说明: /predict 串起 解析→索引→多智能体→预警 的占位管线；接真实实现时把 TODO 替换。
"""
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="港股IPO风险预警 API", version="0.1.0")


class PredictRequest(BaseModel):
    company: Optional[str] = None
    stock_code: Optional[str] = None
    prospectus_path: Optional[str] = None   # 招股书 PDF 路径（领取东吴数据集后填）


class EvidenceSpan(BaseModel):
    page: int
    block: int
    text: str


class RiskItem(BaseModel):
    risk_type: str
    conclusion: str
    score: float
    evidence_spans: List[EvidenceSpan]      # 可追踪：每条结论挂证据


class PredictResponse(BaseModel):
    company: Optional[str]
    stock_code: Optional[str]
    risks: List[RiskItem]
    five_day_drop_prob: Optional[float]     # 重点指标：上市后5日内显著下跌概率
    report_markdown: str
    traceable: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """占位管线。真实实现见各 TODO。"""
    # TODO[1] 解析: parse_prospectus.run_mineru(req.prospectus_path) → 结构化 JSON
    # TODO[2] 索引: build_index.build(doc) → LightRAG workdir
    # TODO[3] 多智能体: agents_graph.build_graph().invoke(...) → legal/finance/market/critic
    # TODO[4] 行情时序: 拉 req.stock_code 发行期行情 → 市场情绪共振 → 5日下跌概率
    demo_risk = RiskItem(
        risk_type="对赌赎回(占位)", conclusion="[占位] 待接管线", score=0.5,
        evidence_spans=[EvidenceSpan(page=0, block=0, text="[占位证据]")],
    )
    return PredictResponse(
        company=req.company, stock_code=req.stock_code,
        risks=[demo_risk], five_day_drop_prob=None,
        report_markdown="# 风险预警报告(占位)\n\nTODO: 接 解析→多智能体→共振 管线后生成。",
        traceable=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
