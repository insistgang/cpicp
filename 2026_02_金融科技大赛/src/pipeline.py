#!/usr/bin/env python3
"""
pipeline.py · 赛题#23 端到端流水线: 分类 → 嵌入 → 相似度去重 → 阈值分析 → 汇总视图

本脚本用合成数据模拟完整流程(无需 GPU/真实影像),验证各模块衔接:
  1. 生成合成清单(prepare_data.synth_manifest)
  2. 构建分类集 + 相似度对
  3. 模拟分类结果(用类型标签代替 CLIP 分类器)
  4. 模拟嵌入(用 group_id 构造相似/不相似向量)
  5. 相似度去重 + 阈值分析 + 汇总视图

用法:
  python pipeline.py --selftest          # 端到端自测
  python pipeline.py --plot              # 生成阈值-P/R 曲线图
"""
import argparse
import sys
import os

# 把 src/ 加入路径以 import 同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import synth_manifest, build_classification, build_similarity_pairs
from metrics import threshold_sweep, best_threshold, roc_auc, pr_at_threshold
from similarity import find_suspicious_pairs, summarize
from app_demo import build_view, _format_summary


def simulate_embeddings(records, dim=128, seed=42):
    """用 group_id 构造模拟嵌入:同 group 向量相近,不同 group 向量远。输出已 L2 归一化,避免 matmul 溢出。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    groups = {r["group_id"] for r in records}
    group_centers = {g: rng.randn(dim) for g in groups}
    embs = []
    for r in records:
        base = group_centers[r["group_id"]]
        noise = rng.randn(dim) * 0.05  # 同组内微小差异
        embs.append(base + noise)
    embs = np.array(embs, dtype=np.float64)
    # 显式 L2 归一化:避免后续 cosine_sim_matrix 中极端范数
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, None)
    return embs


def simulate_classifier(records):
    """用类型标签模拟分类器结果:面签照片保留,其他筛掉。"""
    return [r for r in records if r["type_label"] == "面签照片"]


def run_pipeline(n_groups=30, reuse_frac=0.35, threshold=0.85, seed=42):
    """端到端流水线,返回 (summary_dict, pairs_list, metrics_dict)。"""
    # 1. 合成数据
    records = synth_manifest(n_groups=n_groups, reuse_frac=reuse_frac, seed=seed)
    print(f"[1/5] 合成数据: {len(records)} 条影像")

    # 2. 分类筛面签
    face_records = simulate_classifier(records)
    print(f"[2/5] 分类筛面签: {len(face_records)}/{len(records)} 张面签照片")
    if not face_records:
        print("  ⚠️ 无面签照片,流水线终止")
        return None, None, None

    # 3. 模拟嵌入(真实场景用 embed.py 的 CLIP 提取)
    embs = simulate_embeddings(face_records, dim=128, seed=seed)
    print(f"[3/5] 模拟嵌入: {embs.shape}")

    # 4. 相似度对(用于阈值分析)
    pairs = build_similarity_pairs(face_records, neg_per_pos=2, only_face=False, seed=seed)
    pos_pairs = [p for p in pairs if p["label"] == 1]
    neg_pairs = [p for p in pairs if p["label"] == 0]
    print(f"[4/5] 相似度对: 正 {len(pos_pairs)} / 负 {len(neg_pairs)}")

    # 5. 相似度去重检测
    image_ids = [r["image_path"] for r in face_records]
    customer_ids = [r["customer_id"] for r in face_records]
    business_ids = [r["business_label"] for r in face_records]
    suspicious = find_suspicious_pairs(embs, image_ids, customer_ids, threshold, business_ids)
    summary = summarize(suspicious)
    print(f"[5/5] 去重检测(阈值 {threshold}):")
    print(f"      可疑对 {summary['total_suspicious_pairs']} | 跨客户套用 {summary['cross_customer_misuse']} | 同客户重复 {summary['same_customer_repeat']}")

    # 6. 阈值分析(用 pair 的模拟相似度)
    sim_scores = []
    labels = []
    # 用嵌入余弦相似度作为 scores
    from similarity import cosine_sim_matrix
    S = cosine_sim_matrix(embs)
    img2idx = {img: i for i, img in enumerate(image_ids)}
    for p in pairs:
        ia = img2idx.get(p["img_a"])
        ib = img2idx.get(p["img_b"])
        if ia is not None and ib is not None:
            sim_scores.append(float(S[ia, ib]))
            labels.append(p["label"])

    auc = roc_auc(sim_scores, labels) if sim_scores else float("nan")
    best = best_threshold(sim_scores, labels, objective="f1") if sim_scores else None
    print(f"\n  📊 阈值分析:")
    print(f"     AUC = {auc:.4f}")
    if best:
        print(f"     最优 F1 阈值 = {best['thr']:.3f} (P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f})")

    # 7. 汇总视图
    view_summary, view_pairs = build_view(embs, image_ids, customer_ids, business_ids, threshold)
    print(f"\n{_format_summary(view_summary, threshold)}")

    metrics = {"auc": auc, "best_thr": best, "n_pairs": len(pairs),
               "n_suspicious": len(suspicious), "n_cross": summary["cross_customer_misuse"]}
    return summary, suspicious, metrics


def plot_pr_curve(scores, labels, out_path="pr_curve.png"):
    """绘制阈值-P/R 曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = threshold_sweep(scores, labels, n=100)
    thrs = [r["thr"] for r in rows]
    ps = [r["precision"] for r in rows]
    rs = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, ps, label="Precision", color="#e74c3c", linewidth=1.5)
    ax.plot(thrs, rs, label="Recall", color="#3498db", linewidth=1.5)
    ax.plot(thrs, f1s, label="F1", color="#2ecc71", linewidth=1.5)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Threshold vs Precision / Recall / F1", fontsize=13)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"📈 阈值-P/R 曲线已保存: {os.path.abspath(out_path)}")


def _selftest():
    print("=" * 60)
    print("赛题#23 端到端流水线自测(合成数据)")
    print("=" * 60)
    summary, pairs, metrics = run_pipeline(n_groups=30, reuse_frac=0.4, threshold=0.85, seed=7)

    ok = True
    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    check(summary is not None, "流水线完成")
    check(summary["cross_customer_misuse"] >= 1, "检出跨客户套用")
    check(metrics["auc"] > 0.8, f"AUC>0.8(={metrics['auc']:.3f})")
    check(metrics["best_thr"] is not None and metrics["best_thr"]["f1"] > 0.5, "最优阈值 F1>0.5")
    check(metrics["n_suspicious"] > 0, "存在可疑对")

    print("\n" + ("✅ pipeline 端到端自测通过" if ok else "❌ pipeline 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="端到端自测")
    ap.add_argument("--plot", action="store_true", help="生成阈值-P/R 曲线图")
    ap.add_argument("--n-groups", type=int, default=30)
    ap.add_argument("--reuse-frac", type=float, default=0.35)
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    summary, pairs, metrics = run_pipeline(n_groups=a.n_groups, reuse_frac=a.reuse_frac, threshold=a.threshold)

    if a.plot and metrics and metrics.get("best_thr"):
        # 重新计算 scores/labels 用于绘图
        import numpy as np
        records = synth_manifest(n_groups=a.n_groups, reuse_frac=a.reuse_frac, seed=42)
        face_records = simulate_classifier(records)
        embs = simulate_embeddings(face_records, dim=128, seed=42)
        pair_list = build_similarity_pairs(face_records, neg_per_pos=2, only_face=False, seed=42)
        image_ids = [r["image_path"] for r in face_records]
        img2idx = {img: i for i, img in enumerate(image_ids)}
        from similarity import cosine_sim_matrix
        S = cosine_sim_matrix(embs)
        sim_scores, labels = [], []
        for p in pair_list:
            ia, ib = img2idx.get(p["img_a"]), img2idx.get(p["img_b"])
            if ia is not None and ib is not None:
                sim_scores.append(float(S[ia, ib]))
                labels.append(p["label"])
        if sim_scores:
            plot_pr_curve(sim_scores, labels)


if __name__ == "__main__":
    main()
