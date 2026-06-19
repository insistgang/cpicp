#!/usr/bin/env python3
"""
pipeline.py · 赛题#23 端到端流水线: 分类 → 嵌入 → 相似度去重 → 阈值分析 → 汇总视图

本脚本用合成数据模拟完整流程(无需 GPU/真实影像),验证各模块衔接:
  1. 生成合成清单(prepare_data.synth_manifest)
  2. 构建分类集 + 相似度对
  3. 模拟分类结果(用类型标签代替 CLIP 分类器)
  4. 模拟嵌入(用 group_id 构造相似/不相似向量)
  5. 相似度去重 + 阈值分析 + 汇总视图

本脚本另提供 **--real-images 真实像素端到端模式**:用 synth_images 程序化生成的
真实 PNG 影像,经 features.py 的经典 CPU 特征提取(CLIP 不可用时的 baseline 后端),
让 prepare_data→特征→similarity 去重→metrics 全程跑在**真实图像像素**上,
输出**真实** AUC / Top-k / 去重检出(而非模拟向量)。

用法:
  python pipeline.py --selftest                 # 端到端自测(含真实像素路径)
  python pipeline.py --plot                      # 生成阈值-P/R 曲线图(模拟向量)
  python pipeline.py --real-images output/synth  # 真实像素端到端(先 synth_images 生成)
  python pipeline.py --gen-and-run               # 一键:生成合成图 + 真实像素端到端 + 可视化
"""
import argparse
import sys
import os

# 把 src/ 加入路径以 import 同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import synth_manifest, build_classification, build_similarity_pairs
from metrics import threshold_sweep, best_threshold, roc_auc, pr_at_threshold, topk_retrieval_accuracy
from similarity import find_suspicious_pairs, summarize, cosine_sim_matrix
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


def run_pipeline_real_images(image_dir, threshold=None, pca_dim=32, verbose=True):
    """**真实像素端到端**:对 synth_images 生成的真实 PNG,经经典 CPU 特征 → 去重 → 阈值/AUC/Top-k。

    与 run_pipeline 的区别:嵌入不是用 group_id 模拟,而是 features.py 在**真实图像像素**上
    提取(CLIP 不可用时的 baseline)。manifest 来自磁盘上的真实文件。

    threshold=None 时**从标注数据自动选最优 F1 阈值**(对齐官方"阈值选取策略"要求)——
    经典特征的余弦量纲与 CLIP 不同(PCA 压缩后最优阈值远低于 0.85),故不能套用固定阈值。
    返回 dict(含 auc/best_thr/topk/去重统计 等真实指标)。
    """
    import numpy as np
    from prepare_data import load_manifest, build_classification, build_similarity_pairs
    from features import extract_embeddings_classic

    manifest = os.path.join(image_dir, "manifest.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"未找到 {manifest};先运行 synth_images.py 生成合成影像")
    records = load_manifest(manifest)
    if verbose:
        print(f"[1/6] 加载真实影像清单: {len(records)} 条 ({image_dir})")

    # 2. 分类筛面签:这里用真实标签筛(分类器单独在 classify.py 自测;此处聚焦相似度去重)
    face_records = [r for r in records if r["type_label"] == "面签照片"]
    if verbose:
        print(f"[2/6] 分类筛面签: {len(face_records)}/{len(records)} 张面签照片")
    if len(face_records) < 2:
        if verbose:
            print("  ⚠️ 面签照片不足,流水线终止")
        return None

    # 3. 经典 CPU 特征提取(真实像素!不是随机向量)
    face_paths = [os.path.join(image_dir, r["image_path"]) for r in face_records]
    embs = extract_embeddings_classic(face_paths, pca_dim=pca_dim)
    if verbose:
        print(f"[3/6] 经典特征提取(真实像素,PIL+sklearn baseline): {embs.shape}")

    # 4. 相似度对(基于 group_id 的真值标注)
    pairs = build_similarity_pairs(face_records, neg_per_pos=2, only_face=False, seed=0)
    pos_pairs = [p for p in pairs if p["label"] == 1]
    neg_pairs = [p for p in pairs if p["label"] == 0]
    if verbose:
        print(f"[4/6] 相似度对(group_id 真值): 正 {len(pos_pairs)} / 负 {len(neg_pairs)}")

    # 5. 真实指标:在真实特征余弦上算 AUC / 最优阈值 / Top-k
    image_ids = [r["image_path"] for r in face_records]
    customer_ids = [r["customer_id"] for r in face_records]
    business_ids = [r["business_label"] for r in face_records]
    S = cosine_sim_matrix(embs)
    img2idx = {img: i for i, img in enumerate(image_ids)}
    sim_scores, labels = [], []
    for p in pairs:
        ia, ib = img2idx.get(p["img_a"]), img2idx.get(p["img_b"])
        if ia is not None and ib is not None:
            sim_scores.append(float(S[ia, ib]))
            labels.append(p["label"])
    auc = roc_auc(sim_scores, labels) if sim_scores else float("nan")
    best = best_threshold(sim_scores, labels, objective="f1") if sim_scores else None
    topk = _real_topk(S, face_records, k=1)
    if verbose:
        print(f"[5/6] 阈值分析(真实特征余弦,非随机): AUC={auc:.4f}", end="")
        if best:
            print(f" | 最优 F1 阈值={best['thr']:.3f} "
                  f"(P={best['precision']:.3f},R={best['recall']:.3f},F1={best['f1']:.3f})", end="")
        print(f" | Top-1 同组检索={topk:.4f}")

    # 6. 去重检测:阈值=用户指定或自动选的最优 F1 阈值(经典特征量纲≠CLIP,必须数据驱动)
    det_thr = threshold if threshold is not None else (best["thr"] if best else 0.5)
    suspicious = find_suspicious_pairs(embs, image_ids, customer_ids, det_thr, business_ids)
    summary = summarize(suspicious)
    if verbose:
        auto = "(自动选)" if threshold is None else "(指定)"
        print(f"[6/6] 去重检测(阈值 {det_thr:.3f}{auto}): 可疑对 {summary['total_suspicious_pairs']} | "
              f"⚠️跨客户套用 {summary['cross_customer_misuse']} | 同客户重复 {summary['same_customer_repeat']}")

    return {
        "image_dir": image_dir, "n_records": len(records), "n_face": len(face_records),
        "emb_shape": tuple(embs.shape), "auc": auc, "best_thr": best, "topk1": topk,
        "det_threshold": det_thr,
        "n_suspicious": len(suspicious), "n_cross": summary["cross_customer_misuse"],
        "n_same": summary["same_customer_repeat"],
        "sim_scores": sim_scores, "labels": labels,
        "embs": embs, "image_ids": image_ids, "customer_ids": customer_ids,
        "suspicious": suspicious, "summary": summary, "S": S,
    }


def _real_topk(S, face_records, k=1):
    """对每个"有同组伙伴"的面签照,看其最相似的 k 张里是否含同组成员。"""
    import numpy as np
    from collections import defaultdict
    g2idx = defaultdict(list)
    for i, r in enumerate(face_records):
        g2idx[r["group_id"]].append(i)
    queries = [i for i, r in enumerate(face_records) if len(g2idx[r["group_id"]]) >= 2]
    if not queries:
        return float("nan")
    hits = 0
    for q in queries:
        row = S[q].copy()
        row[q] = -np.inf  # 排除自身
        topk_idx = np.argsort(row)[::-1][:k]
        same_group = set(g2idx[face_records[q]["group_id"]]) - {q}
        if same_group & set(topk_idx.tolist()):
            hits += 1
    return hits / len(queries)


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
    import tempfile
    print("=" * 60)
    print("赛题#23 端到端流水线自测")
    print("=" * 60)

    ok = True
    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    # ---- A. 模拟向量端到端(快速衔接性验证) ----
    print("\n[A] 模拟向量端到端:")
    summary, pairs, metrics = run_pipeline(n_groups=30, reuse_frac=0.4, threshold=0.85, seed=7)
    check(summary is not None, "模拟流水线完成")
    check(summary["cross_customer_misuse"] >= 1, "检出跨客户套用")
    check(metrics["auc"] > 0.8, f"AUC>0.8(={metrics['auc']:.3f})")
    check(metrics["best_thr"] is not None and metrics["best_thr"]["f1"] > 0.5, "最优阈值 F1>0.5")
    check(metrics["n_suspicious"] > 0, "存在可疑对")

    # ---- B. 真实像素端到端(核心:跑在真实合成图像上,非随机) ----
    print("\n[B] 真实像素端到端(经典 CPU 特征,真实 PNG):")
    with tempfile.TemporaryDirectory() as td:
        from synth_images import generate
        out = os.path.join(td, "synth")
        generate(out, n_groups=25, reuse_frac=0.4, seed=11, size=(128, 128))
        res = run_pipeline_real_images(out, threshold=None, pca_dim=32, verbose=True)
        check(res is not None, "真实像素流水线完成")
        check(res["emb_shape"][0] == res["n_face"], "嵌入行数=面签照片数(真实像素提取)")
        check(res["auc"] > 0.9, f"真实像素 AUC>0.9(={res['auc']:.3f})—特征非随机")
        check(res["best_thr"]["f1"] > 0.85,
              f"真实像素最优 F1>0.85(={res['best_thr']['f1']:.3f})")
        check(res["topk1"] > 0.85, f"真实像素 Top-1 同组检索>0.85(={res['topk1']:.3f})")
        check(res["n_cross"] >= 1, f"真实像素检出跨客户套用 {res['n_cross']} 对")

    print("\n" + ("✅ pipeline 端到端自测通过(模拟向量 + 真实像素)" if ok else "❌ pipeline 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="端到端自测(模拟向量+真实像素)")
    ap.add_argument("--plot", action="store_true", help="生成阈值-P/R 曲线图")
    ap.add_argument("--real-images", help="真实像素端到端:指定 synth_images 输出目录")
    ap.add_argument("--gen-and-run", action="store_true",
                    help="一键:生成合成影像 + 真实像素端到端 + 去重可视化")
    ap.add_argument("--out-dir", default="output/synth", help="合成影像输出目录(--gen-and-run)")
    ap.add_argument("--n-groups", type=int, default=30)
    ap.add_argument("--reuse-frac", type=float, default=0.35)
    ap.add_argument("--threshold", type=float, default=None,
                    help="去重判定阈值;不给则真实像素路径**自动选最优 F1 阈值**(经典特征量纲≠CLIP)")
    ap.add_argument("--pca-dim", type=int, default=32)
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    if a.gen_and_run:
        from synth_images import generate
        print(f"=== 生成合成影像 → {a.out_dir} ===")
        recs, _ = generate(a.out_dir, n_groups=a.n_groups, reuse_frac=a.reuse_frac, seed=0)
        print(f"  生成 {len(recs)} 张\n=== 真实像素端到端 ===")
        res = run_pipeline_real_images(a.out_dir, threshold=a.threshold, pca_dim=a.pca_dim)
        if res:
            from viz_dedup import visualize
            png = visualize(res, out_path=os.path.join(a.out_dir, "dedup_viz.png"))
            print(f"\n📊 去重可视化已保存: {png}")
        return

    if a.real_images:
        res = run_pipeline_real_images(a.real_images, threshold=a.threshold, pca_dim=a.pca_dim)
        if res and a.plot:
            from viz_dedup import visualize
            png = visualize(res, out_path=os.path.join(a.real_images, "dedup_viz.png"))
            print(f"\n📊 去重可视化已保存: {png}")
        return

    # 模拟向量路径:CLIP 量纲,沿用 0.85 固定阈值
    sim_thr = a.threshold if a.threshold is not None else 0.85
    summary, pairs, metrics = run_pipeline(n_groups=a.n_groups, reuse_frac=a.reuse_frac, threshold=sim_thr)

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
