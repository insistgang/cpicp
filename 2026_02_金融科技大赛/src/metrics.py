#!/usr/bin/env python3
"""
metrics.py · 相似度检测的阈值分析 + 检索指标(赛题#23 官方明确要求的核心交付)

官方原文要求:"对相似度判定阈值的选取策略进行说明,分析不同阈值对准确率与召回率的影响"。
本模块提供:阈值扫描(P/R/F1)、最优阈值选取(F1 或 目标精度)、ROC-AUC、Top-k 检索准确率。
纯 numpy,`python metrics.py` 跑自测。
"""
import math


def pr_at_threshold(scores, labels, thr):
    """scores: 相似度(越大越"是重复对");labels: 1=真重复/套用对,0=不同。pred = scores>=thr。"""
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"thr": float(thr), "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def threshold_sweep(scores, labels, n=50):
    """在 [min,max] 扫 n 个阈值,返回每个阈值的 P/R/F1。用于画"阈值-P/R"曲线。
    空输入返回 [](而非崩在 s.min() 的 zero-size reduction)。"""
    import numpy as np
    s = np.asarray(scores, float)
    if s.size == 0:
        return []
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        hi = lo + 1e-6
    return [pr_at_threshold(s, labels, t) for t in np.linspace(lo, hi, n)]


def best_threshold(scores, labels, objective="f1", min_precision=None):
    """选阈值:objective='f1' 取最大F1;若给 min_precision,则在 P>=min_precision 中取最大 recall。
    空输入(无样本)返回 None,由调用方决定回退阈值,避免 max() over empty sequence 崩溃。"""
    rows = threshold_sweep(scores, labels, n=200)
    if not rows:
        return None
    if min_precision is not None:
        cand = [r for r in rows if r["precision"] >= min_precision]
        if cand:
            return max(cand, key=lambda r: r["recall"])
    return max(rows, key=lambda r: r[objective])


def roc_auc(scores, labels):
    """ROC-AUC(Mann-Whitney U,含平均秩处理 ties)。无正或无负样本返回 nan。"""
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 平均秩(1-based)
        ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def topk_retrieval_accuracy(sim_to_gallery, correct_idx, k=1):
    """检索:sim_to_gallery (Nq x Ng) 每行查询对画廊的相似度;correct_idx[q]=正确匹配的画廊下标。
    返回 top-k 命中率(正确匹配是否在相似度最高的 k 个里)。"""
    import numpy as np
    S = np.asarray(sim_to_gallery, float)
    hits = 0
    for q in range(S.shape[0]):
        topk = np.argsort(S[q])[::-1][:k]
        if correct_idx[q] in topk:
            hits += 1
    return hits / S.shape[0] if S.shape[0] else 0.0


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    pos = rng.normal(0.85, 0.05, 200).clip(0, 1)   # 重复对:高相似
    neg = rng.normal(0.30, 0.10, 800).clip(0, 1)   # 不同:低相似
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(200), np.zeros(800)])

    auc = roc_auc(scores, labels)
    check(auc > 0.99, f"可分场景 AUC≈1(={auc:.4f})")

    best = best_threshold(scores, labels, objective="f1")
    check(best["f1"] > 0.9, f"最优F1阈值 F1>0.9(thr={best['thr']:.3f}, F1={best['f1']:.3f})")

    # 目标精度模式:要求 P>=0.99,recall 应略降但 precision 达标
    hp = best_threshold(scores, labels, min_precision=0.99)
    check(hp["precision"] >= 0.99, f"目标精度≥0.99 可满足(P={hp['precision']:.3f}, R={hp['recall']:.3f})")

    # 阈值越高 precision 不降(单调性 sanity)
    lowp = pr_at_threshold(scores, labels, 0.5)["precision"]
    highp = pr_at_threshold(scores, labels, 0.8)["precision"]
    check(highp >= lowp, f"阈值↑→precision↑({lowp:.3f}→{highp:.3f})")

    # 退化:全同分 AUC=0.5 量级
    check(abs(roc_auc(np.ones(10), [1,0]*5) - 0.5) < 1e-9, "全同分 AUC=0.5")

    # 边界:空输入不崩(threshold_sweep/best_threshold 优雅降级,非 zero-size reduction 崩溃)
    check(threshold_sweep([], []) == [], "空输入 threshold_sweep 返回 [](不崩)")
    check(best_threshold([], []) is None, "空输入 best_threshold 返回 None(不崩)")
    # 边界:无正样本或无负样本 AUC=nan(而非异常或错误数值)
    check(math.isnan(roc_auc([0.1, 0.2, 0.3], [1, 1, 1])), "全正样本 AUC=nan")
    check(math.isnan(roc_auc([0.1, 0.2, 0.3], [0, 0, 0])), "全负样本 AUC=nan")
    # 边界:无法满足的 min_precision 时回退到最大 F1(不返回空/不崩)
    fb = best_threshold(scores, labels, min_precision=2.0)
    check(fb is not None, "不可满足的 min_precision 回退到最大 F1(非 None)")

    # Top-k 检索:对角线最高 → top1=100%
    Ng = 20
    S = rng.rand(Ng, Ng) * 0.3
    for i in range(Ng):
        S[i, i] = 0.9
    acc1 = topk_retrieval_accuracy(S, list(range(Ng)), k=1)
    check(acc1 == 1.0, f"对角最高 → top1=100%(={acc1:.2f})")

    print("\n" + ("✅ metrics 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
