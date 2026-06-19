#!/usr/bin/env python3
"""
metrics_eval.py · 检测器评测(对齐官方硬指标:检出率≥95%、误报率<5%)

复用 02 金融已验证的阈值分析逻辑(threshold_sweep/pr_at_threshold/roc_auc),
新增 detection_report:在"误报率<阈值"约束下找工作点,报 检出率(recall)/误报率(FPR)/F1/AUC。
纯 numpy,`python metrics_eval.py` 自测。
"""


def pr_at_threshold(scores, labels, thr):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # 检出率
    fpr = fp / (fp + tn) if (fp + tn) else 0.0             # 误报率
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"thr": float(thr), "recall": recall, "fpr": fpr, "precision": precision, "f1": f1}


def roc_auc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def detection_report(scores, labels, max_fpr=0.05, target_recall=0.95):
    """在 误报率<max_fpr 约束下取最大检出率的阈值;并判定是否达官方双线。"""
    import numpy as np
    s = np.asarray(scores, float)
    lo, hi = float(s.min()), float(s.max()) + 1e-6
    rows = [pr_at_threshold(s, labels, t) for t in np.linspace(lo, hi, 300)]
    feasible = [r for r in rows if r["fpr"] < max_fpr]
    best = max(feasible, key=lambda r: r["recall"]) if feasible else max(rows, key=lambda r: r["f1"])
    return {
        "auc": roc_auc(s, labels),
        "chosen_thr": best["thr"],
        "recall": best["recall"], "fpr": best["fpr"], "f1": best["f1"],
        "meets_official": best["recall"] >= target_recall and best["fpr"] < max_fpr,
    }


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    attack = rng.normal(0.85, 0.06, 300).clip(0, 1)     # 攻击=高分
    normal = rng.normal(0.20, 0.08, 700).clip(0, 1)
    scores = np.concatenate([attack, normal]); labels = np.concatenate([np.ones(300), np.zeros(700)])

    rep = detection_report(scores, labels)
    check(rep["auc"] > 0.99, f"可分场景 AUC≈1(={rep['auc']:.4f})")
    check(rep["fpr"] < 0.05, f"工作点误报率<5%(={rep['fpr']:.3f})")
    check(rep["recall"] >= 0.95, f"工作点检出率≥95%(={rep['recall']:.3f})")
    check(rep["meets_official"], "达官方双线(检出≥95% 且 误报<5%)")

    # 不可分场景:不应虚报达标
    bad = np.concatenate([rng.rand(100), rng.rand(100)])
    badrep = detection_report(bad, np.concatenate([np.ones(100), np.zeros(100)]))
    check(not badrep["meets_official"], "不可分场景如实不达标")

    print("\n" + ("✅ metrics_eval 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
