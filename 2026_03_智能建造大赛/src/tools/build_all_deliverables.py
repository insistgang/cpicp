#!/usr/bin/env python3
"""
build_all_deliverables.py · 一键产出本地可生成的交付物草稿(配图 + 报告图表 + 技术方案 docx)

顺序固定(docx 会内嵌前两步的图):
  1) gen_water_scene     → output/figs/augment_water_before_after_*.png   (增广 before/after 配图)
  2) gen_report_figs     → output/figs/report_fig{1,2,3}_*.png            (分桶召回/PR/FPS-精度)
  3) gen_tech_plan_docx  → output/docx/技术方案_初稿.docx                 (内嵌上面 4 张图)

用法: python3 tools/build_all_deliverables.py
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import gen_water_scene
import gen_report_figs
import gen_tech_plan_docx


def main():
    print("=" * 60)
    print("03 智能建造 · 本地交付物草稿一键生成")
    print("=" * 60)

    print("\n[1/3] 合成海面 + GT-Anchored Glint 增广对比图 ...")
    for p, nb, npx in gen_water_scene.generate(n=2, seed=7):
        print(f"  ✓ {p.name}  ({nb} GT, +{npx} glint, {p.stat().st_size}B)")

    print("\n[2/3] 性能报告图表(分桶召回 / PR 曲线 / FPS-精度)...")
    for k, p in gen_report_figs.generate().items():
        print(f"  ✓ {k}: {p.name}  ({p.stat().st_size}B)")

    print("\n[3/3] 技术方案 Word 初稿(内嵌配图)...")
    out = gen_tech_plan_docx.generate()
    print(f"  ✓ {out.name}  ({out.stat().st_size}B)")

    print("\n" + "=" * 60)
    print("全部交付物草稿已生成 → output/figs (PNG) + output/docx (Word)")
    print("=" * 60)


if __name__ == "__main__":
    main()
