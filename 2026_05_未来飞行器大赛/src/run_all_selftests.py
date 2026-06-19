# -*- coding: utf-8 -*-
"""
05 未来飞行器大赛 · 全模块自测入口
依次跑: gen_figures(出4图) -> build_report_docx(渲染正文docx并读回校验)
        -> build_attachment_docx(渲染附件1 TRL docx并读回校验)
全部 PASS 才 exit 0。离线可跑, 无需 GPU。

运行: python3 run_all_selftests.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_figures
import build_report_docx
import build_attachment_docx


def main():
    results = []

    print("########## [1/3] 图表生成自测 ##########")
    fig_dir = os.path.join(HERE, "..", "output", "figures")
    fig_dir = os.path.abspath(fig_dir)
    gen_figures.setup_cjk_font()
    ok_fig = gen_figures.selftest(fig_dir)
    results.append(("gen_figures", ok_fig))

    print("\n########## [2/3] 正文 Word 报告书渲染自测 ##########")
    ok_doc = build_report_docx.selftest()
    results.append(("build_report_docx", ok_doc))

    print("\n########## [3/3] 附件1 TRL Word 渲染自测 ##########")
    ok_att = build_attachment_docx.selftest()
    results.append(("build_attachment_docx", ok_att))

    print("\n========== 汇总 ==========")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("========== 全部自测", "PASS ==========" if all_ok else "FAIL ==========")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
