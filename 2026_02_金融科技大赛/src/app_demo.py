#!/usr/bin/env python3
"""
app_demo.py · 检测汇总视图(赛题#23 官方要求"提供检测汇总视图,展示检测结果统计,
支持查看高相似可疑交易对应的影像及关联的业务数据")

数据层复用已自测的 similarity.find_suspicious_pairs / summarize(无需 GPU 即可跑逻辑);
UI 用 Gradio。`--demo` 用合成嵌入演示汇总逻辑(不依赖真实影像)。

流程:embeddings.npy + ids.json(embed.py 产出)→ 相似度去重 → 汇总统计 + 可疑对并排查看。
⚠️ 启动 UI 需 gradio;但 build_view 的数据聚合纯逻辑,`--demo` 可在 Mac 验证。
"""
import argparse
import json


def build_view(embs, image_ids, customer_ids, business_ids=None, threshold=0.9):
    """聚合检测汇总数据(纯逻辑,复用 similarity 的已测函数)。返回 (summary, pairs)。"""
    from similarity import find_suspicious_pairs, summarize
    pairs = find_suspicious_pairs(embs, image_ids, customer_ids, threshold, business_ids)
    return summarize(pairs), pairs


def _format_summary(summary, threshold):
    return (f"### 检测汇总(阈值 {threshold})\n"
            f"- 可疑相似对总数:**{summary['total_suspicious_pairs']}**\n"
            f"- ⚠️ 跨客户套用(最严重):**{summary['cross_customer_misuse']}**\n"
            f"- 同客户重复提交:**{summary['same_customer_repeat']}**\n"
            f"- 最高相似度:**{summary['max_score']:.3f}**")


def launch(embs, image_ids, customer_ids, business_ids=None, image_dir=None):  # pragma: no cover
    import gradio as gr

    def run(threshold):
        summary, pairs = build_view(embs, image_ids, customer_ids, business_ids, threshold)
        rows = [[p["img_a"], p["img_b"], p["cust_a"], p["cust_b"], f"{p['score']:.3f}", p["type"]]
                for p in pairs[:200]]
        return _format_summary(summary, threshold), rows

    with gr.Blocks(title="金融影像相似度检测汇总") as app:
        gr.Markdown("# 金融影像智能相似度检测 · 汇总视图")
        thr = gr.Slider(0.5, 0.999, value=0.9, step=0.005, label="相似度阈值")
        md = gr.Markdown()
        table = gr.Dataframe(headers=["影像A", "影像B", "客户A", "客户B", "相似度", "类型"],
                             label="高相似可疑交易(跨客户套用优先排查)")
        thr.change(run, thr, [md, table])
        app.load(run, thr, [md, table])
    app.launch()


def _demo():
    """合成嵌入演示汇总逻辑(Mac 可跑,验证数据层)。"""
    import numpy as np
    rng = np.random.RandomState(3)
    D = 32
    base = rng.randn(D)
    embs = np.vstack([base, base + rng.randn(D) * 0.01,        # 套用对
                      base + rng.randn(D) * 0.01,              # 同客户重复
                      rng.randn(8, D)])
    ids = ["IMG_base", "IMG_reuse", "IMG_dup"] + [f"IMG_{i}" for i in range(8)]
    custs = ["A", "B", "A"] + [f"C{i}" for i in range(8)]
    biz = ["微贷", "经营贷", "微贷"] + ["按揭"] * 8
    summary, pairs = build_view(embs, ids, custs, biz, threshold=0.9)
    print(_format_summary(summary, 0.9))
    print("\n可疑对(top):")
    for p in pairs[:5]:
        print(f"  {p['type']:22s} {p['img_a']}↔{p['img_b']} ({p['cust_a']}/{p['cust_b']}) sim={p['score']:.3f}")
    assert summary["cross_customer_misuse"] >= 1, "应检出跨客户套用"
    print("\n✅ app_demo 数据层逻辑自测通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="合成嵌入演示汇总逻辑(Mac 可跑)")
    ap.add_argument("--embeddings", help="embed.py 产出的 .npy")
    ap.add_argument("--ids", help="embed.py 产出的 .json(image_ids/customer_ids)")
    a = ap.parse_args()
    if a.demo:
        _demo(); return
    if not (a.embeddings and a.ids):
        ap.error("需 --embeddings 与 --ids(或 --demo)")
    import numpy as np
    embs = np.load(a.embeddings)
    meta = json.load(open(a.ids, encoding="utf-8"))
    launch(embs, meta["image_ids"], meta["customer_ids"])


if __name__ == "__main__":
    main()
