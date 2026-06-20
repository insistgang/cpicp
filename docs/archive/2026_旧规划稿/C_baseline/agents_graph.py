#!/usr/bin/env python3
"""
agents_graph.py · LangGraph 多智能体状态机骨架
  角色: 法务合规 Agent / 财务穿透 Agent / 市场分析 Agent → Critic(辩论+证据查证)
  对齐赛题书"多专家协作 + 自主规划/辩论/查证 + 可追踪"。

  本文件先给【空实现 + 接口】，--demo 模式无需 LLM 即可跑通整张图(返回占位结论)，
  便于先验证编排与状态流转；接真实 LLM 时把各节点的 TODO 换成 LLM 调用即可。

用法:
  python agents_graph.py --doc out/demo.json --demo        # 占位逻辑，离线可跑
  python agents_graph.py --doc out/demo.json               # 接真实 LLM（需 .env）
"""
import argparse
import json
import os
from pathlib import Path
from typing import TypedDict, List, Dict, Annotated
import operator


class RiskState(TypedDict):
    doc_path: str
    legal: Dict          # 法务合规结论
    finance: Dict        # 财务穿透结论
    market: Dict         # 市场分析结论
    critic: Dict         # Critic 交叉验证结论
    evidence: Annotated[List[Dict], operator.add]   # 累积证据 span（可追踪）
    demo: bool


def _agent_stub(role: str, focus: str):
    """生成一个角色节点。demo=True 返回占位；否则 TODO 接 LLM。"""
    def node(state: RiskState) -> Dict:
        if state["demo"]:
            concl = {"role": role, "risk": f"[占位] {focus} 待接 LLM", "score": 0.5}
            ev = [{"page": 0, "block": 0, "text": f"[占位证据-{role}]"}]
            return {role.split()[0].lower() if False else _key(role): concl, "evidence": ev}
        # TODO[接 LLM]: 用 LangChain ChatOpenAI(base_url=LLM_BASE_URL) 对 doc 做 role 视角分析，
        #   prompt 要求"结论必须引用 [page|block] 证据"，返回 {risk, score, evidence_spans}
        raise NotImplementedError(f"{role} 真实实现 TODO")
    return node


def _key(role: str) -> str:
    return {"法务合规": "legal", "财务穿透": "finance", "市场分析": "market"}[role]


def legal_node(s):   return {"legal":   _stub_concl("法务合规", "对赌/赎回/关联交易合规", s), "evidence": _stub_ev("法务合规")}
def finance_node(s): return {"finance": _stub_concl("财务穿透", "现金消耗率/毛利/客户集中度", s), "evidence": _stub_ev("财务穿透")}
def market_node(s):  return {"market":  _stub_concl("市场分析", "发行期大盘/板块流动性/破发风险", s), "evidence": _stub_ev("市场分析")}


def _stub_concl(role, focus, s):
    if s["demo"]:
        return {"role": role, "focus": focus, "risk_level": "中(占位)", "score": 0.5}
    raise NotImplementedError(f"{role} TODO 接 LLM")   # TODO


def _stub_ev(role):
    return [{"page": 0, "block": 0, "text": f"[占位证据-{role}]"}]


def critic_node(s: RiskState) -> Dict:
    """Critic: 汇总三方结论，冲突时辩论/查证（占位：取均分 + 列冲突）。"""
    if s["demo"]:
        scores = [s.get(k, {}).get("score", 0) for k in ("legal", "finance", "market")]
        fused = sum(scores) / max(len(scores), 1)
        return {"critic": {
            "fused_score": round(fused, 3),
            "five_day_drop_risk": "TODO[接行情时序]：W8 跨模态共振后给出 5 日下跌概率",
            "conflicts": "[占位] 无冲突；真实实现需对比三方结论并触发查证链路",
            "traceable": True,   # 每条结论挂证据 → 可追踪率目标 100%
        }}
    raise NotImplementedError("Critic TODO 接 LLM 辩论/查证")


def build_graph():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(RiskState)
    g.add_node("legal", legal_node)
    g.add_node("finance", finance_node)
    g.add_node("market", market_node)
    g.add_node("critic", critic_node)
    # 三专家并行 → Critic 汇总
    g.add_edge(START, "legal")
    g.add_edge(START, "finance")
    g.add_edge(START, "market")
    g.add_edge("legal", "critic")
    g.add_edge("finance", "critic")
    g.add_edge("market", "critic")
    g.add_edge("critic", END)
    return g.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--demo", action="store_true", help="占位逻辑，无需 LLM")
    a = ap.parse_args()
    app = build_graph()
    init: RiskState = {"doc_path": a.doc, "legal": {}, "finance": {}, "market": {},
                       "critic": {}, "evidence": [], "demo": a.demo}
    out = app.invoke(init)
    print(json.dumps({k: out[k] for k in ("legal", "finance", "market", "critic")},
                     ensure_ascii=False, indent=2))
    print(f"\n累计证据 span 数: {len(out['evidence'])}（真实实现每条结论都应挂证据→可追踪率100%）")


if __name__ == "__main__":
    main()
