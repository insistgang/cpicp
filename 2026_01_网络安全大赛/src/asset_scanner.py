#!/usr/bin/env python3
"""
asset_scanner.py · AI Agent 资产静态识别 + 资产图谱(赛题#4 任务1)

遍历目标目录,识别 AI Agent 相关资产指纹:
  - MCP/Agent 配置:claude_desktop_config.json / mcp.json / *.mcp / config 里的 mcpServers
  - Skills:SKILL.md / skills/ 目录
  - 框架/依赖:requirements*.txt / pyproject 里的 langchain|anthropic|openai|mcp|autogen|crewai
  - Agent 代码:*.py 里的 Tool(...) / @tool / AgentExecutor / mcp.server
  - 模型权重:*.pt/.gguf/.safetensors/.onnx
产出资产清单 + 简单邻接表(资产图谱),供后续配置风险/供应链检测。
纯标准库,`python asset_scanner.py` 用内置 fixtures 自测。
"""
import os
import re
import json
import tempfile

FRAMEWORK_RE = re.compile(r"(?i)\b(langchain|langgraph|anthropic|openai|mcp|autogen|crewai|llama_index|llamaindex|transformers)\b")
TOOL_RE = re.compile(r"(?i)(@tool\b|Tool\s*\(|AgentExecutor|FunctionTool|mcp\.server|@mcp\.tool)")
MODEL_EXT = (".pt", ".pth", ".gguf", ".safetensors", ".onnx", ".bin")


def scan(root: str):
    assets = {"agents": [], "frameworks": set(), "models": [], "skills": [], "mcp_servers": [], "configs": []}
    edges = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            low = fn.lower()
            try:
                if low in ("claude_desktop_config.json", "mcp.json") or low.endswith(".mcp"):
                    assets["configs"].append(rel)
                    for s in _parse_mcp_servers(_read(p)):
                        assets["mcp_servers"].append(s); edges.append((rel, s))
                elif low == "skill.md" or "/skills/" in p.replace("\\", "/").lower():
                    assets["skills"].append(rel)
                elif low.startswith("requirements") and low.endswith(".txt") or low == "pyproject.toml":
                    for m in FRAMEWORK_RE.finditer(_read(p)):
                        assets["frameworks"].add(m.group(1).lower())
                elif low.endswith(".py"):
                    txt = _read(p)
                    if TOOL_RE.search(txt):
                        assets["agents"].append(rel); edges.append((rel, "tool_def"))
                    for m in FRAMEWORK_RE.finditer(txt):
                        assets["frameworks"].add(m.group(1).lower())
                elif fn.endswith(MODEL_EXT):
                    assets["models"].append(rel)
            except Exception:
                continue
    assets["frameworks"] = sorted(assets["frameworks"])
    return {"assets": assets, "edges": edges,
            "counts": {k: len(v) for k, v in assets.items()}}


def _parse_mcp_servers(txt):
    """从配置文本取 MCP server 名:优先 JSON 的 mcpServers 键;回退到最内层含 command 的对象。"""
    try:
        d = json.loads(txt)
        node = d.get("mcpServers") or d.get("mcp_servers") or {}
        if isinstance(node, dict):
            return list(node.keys())
    except Exception:
        pass
    return re.findall(r'"([\w.-]+)"\s*:\s*\{[^{}]*"command"', txt)  # 收紧到最内层对象


def _read(p, limit=200_000):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "skills"), exist_ok=True)
    open(os.path.join(d, "mcp.json"), "w").write(
        '{"mcpServers": {"filesystem": {"command": "npx", "args": []}, "evil": {"command": "sh"}}}')
    open(os.path.join(d, "skills", "SKILL.md"), "w").write("# a skill")
    open(os.path.join(d, "requirements.txt"), "w").write("langchain==0.2\nanthropic\nmcp\nnumpy")
    open(os.path.join(d, "agent.py"), "w").write("from langchain.agents import Tool\nt = Tool(name='x')")
    open(os.path.join(d, "model.safetensors"), "wb").write(b"\x00\x01")

    r = scan(d)
    a = r["assets"]
    check("mcp.json" in a["configs"], "识别 mcp.json 配置")
    check(set(a["mcp_servers"]) == {"filesystem", "evil"}, f"解析 MCP servers({a['mcp_servers']})")
    check(any("SKILL.md" in s for s in a["skills"]), "识别 SKILL.md")
    check({"langchain", "anthropic", "mcp"} <= set(a["frameworks"]), f"识别框架({a['frameworks']})")
    check(any("agent.py" in x for x in a["agents"]), "识别含 Tool() 的 agent 代码")
    check(any("model.safetensors" in x for x in a["models"]), "识别模型权重")
    check(len(r["edges"]) >= 2, f"产出资产图谱边({len(r['edges'])}条)")

    import shutil; shutil.rmtree(d)
    print("\n" + ("✅ asset_scanner 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
