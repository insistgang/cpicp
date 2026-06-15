#!/usr/bin/env python3
"""
risk_qa.py · 风险问答 + 证据 span 回链（对齐赛题书"可追踪率 100%"）

做法:
  1) 用 LightRAG 检索 → 先取 retrieved context（这些就是"证据 span"，带 page/block 标注）;
  2) 再生成答案；
  3) 同时返回 {答案, 证据列表(原文片段)} —— 每条风险结论都能回链到原文，满足可追踪。

用法:
  python risk_qa.py --workdir out/rag_demo --q "是否存在对赌/赎回条款？现金消耗压力如何？"
"""
import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from build_index import _llm_func, _embed_func  # 复用同一 LLM/Embed 配置

load_dotenv()


async def ask(workdir: Path, question: str):
    from lightrag import LightRAG, QueryParam
    rag = LightRAG(working_dir=str(workdir), llm_model_func=_llm_func(),
                   embedding_func=_embed_func())
    await rag.initialize_storages()

    # (1) 只取检索上下文作为"证据 span"
    evidence = await rag.aquery(
        question, param=QueryParam(mode="hybrid", only_need_context=True))
    # (2) 生成带答案
    answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))

    print("\n=== 答案 ===\n", answer)
    print("\n=== 证据 span（回链原文，含 [page|block] 标注）===")
    print(evidence[:3000] if isinstance(evidence, str) else evidence)
    print("\n[可追踪] 每条风险结论应引用上述证据中的 [page x | block y] 标注；"
          "前端/报告里把结论与证据 span 一一挂钩 → 满足赛题'可追踪率100%'。")
    # TODO: 结构化输出 {risk_type, conclusion, evidence_spans:[{page,block,text}]} 写入 out/pred.json，供 eval_metrics
    return {"answer": answer, "evidence": evidence}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="out/rag_demo", type=Path)
    ap.add_argument("--q", required=True, help="风险问题")
    a = ap.parse_args()
    asyncio.run(ask(a.workdir, a.q))


if __name__ == "__main__":
    main()
