#!/usr/bin/env python3
"""
build_index.py · 用 LightRAG 对解析后的招股书建知识图谱 + 向量库

用法:
  python build_index.py --doc out/demo.json --workdir out/rag_demo

依赖: lightrag-hku；LLM/Embed 端点见 .env（OpenAI 兼容，DeepSeek/通义）。
说明: 把每个 block 的原文按 page 标注后插入 LightRAG；检索时即可回链到 page/原文(证据 span)。
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _llm_func():
    """OpenAI 兼容 LLM 函数（供 LightRAG 抽实体/关系）。"""
    from lightrag.llm.openai import openai_complete_if_cache

    async def llm(prompt, system_prompt=None, history_messages=[], **kw):
        return await openai_complete_if_cache(
            os.environ.get("LLM_MODEL", "deepseek-chat"),
            prompt, system_prompt=system_prompt, history_messages=history_messages,
            api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"], **kw,
        )
    return llm


def _embed_func():
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    async def embed(texts):
        return await openai_embed(
            texts, model=os.environ.get("EMBED_MODEL", "text-embedding-3-large"),
            api_key=os.environ["EMBED_API_KEY"], base_url=os.environ["EMBED_BASE_URL"],
        )
    return EmbeddingFunc(embedding_dim=int(os.environ.get("EMBED_DIM", 1024)),
                         max_token_size=8192, func=embed)


async def build(doc_path: Path, workdir: Path):
    from lightrag import LightRAG
    workdir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(working_dir=str(workdir), llm_model_func=_llm_func(),
                   embedding_func=_embed_func())
    await rag.initialize_storages()

    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    # 把每块标注 page 后插入，便于检索回链证据
    chunks = []
    for i, b in enumerate(doc["blocks"]):
        if b["type"] == "text" and b.get("text"):
            chunks.append(f"[page {b.get('page')} | block {i}] {b['text']}")
        elif b["type"] == "table":
            chunks.append(f"[page {b.get('page')} | table {i}] {b.get('caption','')}\n{b.get('table_html','')}")
    await rag.ainsert(chunks)   # 批量插入
    print(f"✓ 已建索引到 {workdir}（插入 {len(chunks)} 块）")
    print("  下一步: python risk_qa.py --workdir", workdir, '--q "..."')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, type=Path)
    ap.add_argument("--workdir", default="out/rag_demo", type=Path)
    a = ap.parse_args()
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "EMBED_API_KEY", "EMBED_BASE_URL"):
        if not os.environ.get(k):
            raise SystemExit(f"[缺环境变量] {k} —— 请先 cp .env.example .env 并填写")
    asyncio.run(build(a.doc, a.workdir))


if __name__ == "__main__":
    main()
