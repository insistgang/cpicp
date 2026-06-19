#!/usr/bin/env python3
"""
aoi_metrics.py · AOI 评测 + 官方竞赛分加权(华为赛题一)

复用 02 金融已验证的阈值分析(threshold_sweep/pr_at_threshold/roc_auc),
新增 compute_competition_score:按官方竞赛得分构成 方案完整度50% + 准确率20% + 检测时间30% 加权。
纯 numpy,`python aoi_metrics.py` 自测。
"""


def pr_at_threshold(scores, labels, thr):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / max(1, len(y))
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"thr": float(thr), "precision": p, "recall": r, "f1": f1, "accuracy": acc}


def threshold_sweep(scores, labels, n=200):
    import numpy as np
    s = np.asarray(scores, float); lo, hi = float(s.min()), float(s.max()) + 1e-6
    return [pr_at_threshold(s, labels, t) for t in np.linspace(lo, hi, n)]


def best_threshold(scores, labels, objective="f1"):
    return max(threshold_sweep(scores, labels), key=lambda r: r[objective])


def roc_auc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float); ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def latency_score(latency_ms, budget_ms=200.0):
    """检测时间得分(0-1):≤budget 满分,线性衰减到 2×budget 处为0。对应竞赛分'检测时间30%'。"""
    if latency_ms <= budget_ms:
        return 1.0
    return max(0.0, 1.0 - (latency_ms - budget_ms) / budget_ms)


def compute_competition_score(accuracy, latency_ms, plan_completeness=1.0, budget_ms=200.0):
    """官方竞赛得分(占总分60%)构成:方案完整度50% + 准确率20% + 检测时间30%。返回 0-1。"""
    lat = latency_score(latency_ms, budget_ms)
    return {
        "plan_completeness": plan_completeness, "accuracy": accuracy, "latency_score": lat,
        "competition_score": 0.5 * plan_completeness + 0.2 * accuracy + 0.3 * lat,
    }


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    defect = rng.normal(0.8, 0.07, 200).clip(0, 1)
    normal = rng.normal(0.25, 0.08, 800).clip(0, 1)
    scores = np.concatenate([defect, normal]); labels = np.concatenate([np.ones(200), np.zeros(800)])
    check(roc_auc(scores, labels) > 0.99, "可分场景 AUC≈1")
    check(best_threshold(scores, labels)["f1"] > 0.9, "最优F1>0.9")

    # 延时得分:满足/超时
    check(latency_score(150, 200) == 1.0, "150ms≤200ms预算→满分")
    check(latency_score(300, 200) == 0.5, "300ms→0.5分")
    check(latency_score(400, 200) == 0.0, "400ms(2×预算)→0分")

    # 竞赛分加权
    good = compute_competition_score(accuracy=0.96, latency_ms=120, plan_completeness=0.9)
    slow = compute_competition_score(accuracy=0.96, latency_ms=500, plan_completeness=0.9)
    check(abs(good["competition_score"] - (0.5*0.9 + 0.2*0.96 + 0.3*1.0)) < 1e-9, "竞赛分加权正确")
    check(good["competition_score"] > slow["competition_score"], "同精度下更快→竞赛分更高(检测时间占30%)")

    print("\n" + ("✅ aoi_metrics 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
