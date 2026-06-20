#!/usr/bin/env python3
"""
parse_prospectus.py · 招股书 PDF → 结构化 JSON（MinerU，表格/数字走确定性抽取）

防幻觉原则: 表格与财务数字由 MinerU 版面/表格识别得到，原样保留(HTML/markdown)，
            下游 LLM 只做语义判断，绝不重写数字。每个块带 page/类型/原文，供证据回链。

用法:
  python parse_prospectus.py --pdf samples/demo_prospectus.pdf --out out/demo.json

依赖: MinerU(命令行 `mineru`)。安装: pip install -U mineru  (或 magic-pdf)
"""
import argparse
import json
import subprocess
from pathlib import Path


def run_mineru(pdf: Path, work: Path) -> Path:
    """调用 MinerU CLI 解析 PDF；返回输出目录。"""
    work.mkdir(parents=True, exist_ok=True)
    # MinerU 2.x: mineru -p <pdf> -o <outdir>   （旧版 magic-pdf 用法见 README）
    cmd = ["mineru", "-p", str(pdf), "-o", str(work)]
    print("  运行:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            f"[MinerU 未就绪] {e}\n  请先 `pip install -U mineru` 并确保 `mineru` 在 PATH；"
            f"或改用 magic-pdf（见 README）。")
    return work


def collect_blocks(work: Path) -> list:
    """从 MinerU 输出的 content_list.json 收集结构化块（text/table/image），保留 page 与原文。"""
    cl = list(work.rglob("*content_list.json"))
    if not cl:
        raise SystemExit(f"[未找到 content_list.json] 检查 MinerU 输出目录 {work}")
    items = json.loads(cl[0].read_text(encoding="utf-8"))
    blocks = []
    for it in items:
        t = it.get("type")
        blk = {"type": t, "page": it.get("page_idx")}
        if t == "text":
            blk["text"] = it.get("text", "")
        elif t == "table":
            # 表格保留 HTML/markdown 原文（确定性，不交给 LLM 重写）
            blk["table_html"] = it.get("table_body") or it.get("html") or ""
            blk["caption"] = it.get("table_caption", "")
        elif t == "image":
            blk["img_path"] = it.get("img_path", "")
            blk["caption"] = it.get("img_caption", "")
        blocks.append(blk)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workdir", default="out/mineru", type=Path)
    a = ap.parse_args()
    if not a.pdf.exists():
        raise SystemExit(f"[招股书 PDF 不存在] {a.pdf}（领取东吴数据集后置于 samples/）")
    work = run_mineru(a.pdf, a.workdir)
    blocks = collect_blocks(work)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "source_pdf": str(a.pdf),
        "n_blocks": len(blocks),
        "blocks": blocks,   # 每块含 page+原文，供 risk_qa 证据回链
        # TODO[数据]: 领取东吴数据集后，补 公司名/股票代码/上市日期 等结构化字段
    }
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 解析完成: {a.out}（{len(blocks)} 块）")
    print("  下一步: python build_index.py --doc", a.out, "--workdir out/rag_demo")


if __name__ == "__main__":
    main()
