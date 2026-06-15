#!/usr/bin/env python3
"""
eval_metrics.py · 按赛题书口径评测：抽取准确率 / 证据召回 / 可追踪率

赛题书硬指标:
  - 风险要素抽取准确率 ≥ 80%
  - 关键证据片段召回率 ≥ 85%
  - Agent 推理链路/角色/工具/证据来源 可追踪率 = 100%

输入(JSON):
  pred.json: [{"risk_type","conclusion","evidence_spans":[{"page","block","text"}],
               "agent_trace":{"roles":[...],"tools":[...]}}]
  gold.json: [{"risk_type","gold_evidence":[{"page","block"}]}]   # 人工标注

用法:
  python eval_metrics.py --pred out/pred.json --gold samples/gold.json
  python eval_metrics.py --selftest     # 用内置玩具样例跑通流程
"""
import argparse
import json
from pathlib import Path


def extraction_accuracy(pred, gold):
    """抽取准确率：预测的风险类型命中 gold 类型集合的比例。"""
    gold_types = {g["risk_type"] for g in gold}
    if not pred:
        return 0.0
    hit = sum(1 for p in pred if p.get("risk_type") in gold_types)
    return hit / len(pred)


def evidence_recall(pred, gold):
    """证据召回：gold 标注的关键证据(page,block)被某预测的 evidence_spans 覆盖的比例。"""
    pred_spans = {(s.get("page"), s.get("block")) for p in pred for s in p.get("evidence_spans", [])}
    gold_spans = [(s.get("page"), s.get("block")) for g in gold for s in g.get("gold_evidence", [])]
    if not gold_spans:
        return 0.0
    hit = sum(1 for gs in gold_spans if gs in pred_spans)
    return hit / len(gold_spans)


def traceability(pred):
    """可追踪率：每条结论是否都带 非空 evidence_spans 且有 agent_trace（角色+工具）。要求=100%。"""
    if not pred:
        return 0.0
    ok = 0
    for p in pred:
        has_ev = bool(p.get("evidence_spans"))
        tr = p.get("agent_trace", {})
        has_trace = bool(tr.get("roles")) and ("tools" in tr)
        if has_ev and has_trace:
            ok += 1
    return ok / len(pred)


def report(pred, gold):
    acc = extraction_accuracy(pred, gold)
    rec = evidence_recall(pred, gold)
    tr = traceability(pred)
    print("\n=== 赛题书口径评测 ===")
    print(f"  抽取准确率 = {acc:.3f}   目标 ≥0.80   {'✅' if acc>=0.80 else '❌'}")
    print(f"  证据召回率 = {rec:.3f}   目标 ≥0.85   {'✅' if rec>=0.85 else '❌'}")
    print(f"  可追踪率   = {tr:.3f}    目标 =1.00   {'✅' if tr>=1.0 else '❌'}")


SELFTEST_PRED = [
    {"risk_type": "对赌赎回", "conclusion": "存在赎回权",
     "evidence_spans": [{"page": 12, "block": 3, "text": "..."}],
     "agent_trace": {"roles": ["法务合规"], "tools": ["retriever"]}},
    {"risk_type": "现金消耗", "conclusion": "现金跑道<18个月",
     "evidence_spans": [{"page": 88, "block": 1, "text": "..."}],
     "agent_trace": {"roles": ["财务穿透"], "tools": ["table_parser"]}},
]
SELFTEST_GOLD = [
    {"risk_type": "对赌赎回", "gold_evidence": [{"page": 12, "block": 3}]},
    {"risk_type": "现金消耗", "gold_evidence": [{"page": 88, "block": 1}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("[selftest] 内置玩具样例：")
        report(SELFTEST_PRED, SELFTEST_GOLD)
        return
    if not (a.pred and a.gold):
        ap.error("需 --pred 和 --gold，或用 --selftest")
    report(json.loads(a.pred.read_text(encoding="utf-8")),
           json.loads(a.gold.read_text(encoding="utf-8")))
    # TODO[数据]: gold.json 需领取东吴数据集后人工标注关键风险与证据 page/block。


if __name__ == "__main__":
    main()
