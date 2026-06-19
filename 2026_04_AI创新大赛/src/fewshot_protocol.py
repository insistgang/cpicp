#!/usr/bin/env python3
"""
fewshot_protocol.py · 严格复刻官方少样本评测协议(华为赛题一)

官方:评测仅给 100 张无缺陷 + 30 张有缺陷做"迁移学习",再测 1000+ 张测试集。
本协议:用正常样本(留一部分做阈值校准)建 memory bank → 用 30 缺陷+留出正常 校准阈值
→ 在测试集上报 检出率/误报率/F1/AUC + per-image 延时。特征为 image-level 向量(占位,
真特征接 timm/CLIP)。纯 numpy,`python fewshot_protocol.py` 用合成特征端到端自测。
"""
import time

from patchcore_lite import build_memory_bank
from anomaly_score import nn_distance
from aoi_metrics import best_threshold, pr_at_threshold, roc_auc


def run_protocol(normal_train, defect_cal, test_feats, test_labels,
                 bank_frac=0.8, coreset_ratio=0.5):
    """normal_train (100,D): 拆 bank_frac 建库、其余做阈值校准的正常侧;
    defect_cal (30,D): 校准的缺陷侧;test_feats/test_labels: 1000+ 测试。"""
    import numpy as np
    nt = np.asarray(normal_train, float)
    k = int(len(nt) * bank_frac)
    bank = build_memory_bank(nt[:k], coreset_ratio)          # 仅用部分正常建库
    cal_normal = nt[k:]                                      # 留出正常用于校准(不在库内)

    cal_scores = np.concatenate([nn_distance(defect_cal, bank), nn_distance(cal_normal, bank)])
    cal_labels = np.concatenate([np.ones(len(defect_cal)), np.zeros(len(cal_normal))])
    thr = best_threshold(cal_scores, cal_labels, objective="f1")["thr"]

    # 测试 + per-image 延时
    lat = []
    tscores = np.empty(len(test_feats))
    for i, f in enumerate(np.asarray(test_feats, float)):
        t0 = time.perf_counter()
        tscores[i] = nn_distance(f[None], bank)[0]
        lat.append((time.perf_counter() - t0) * 1000)

    m = pr_at_threshold(tscores, test_labels, thr)
    return {
        "bank_size": len(bank), "threshold": thr,
        "auc": roc_auc(tscores, test_labels),
        "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
        "per_image_latency_ms_mean": float(np.mean(lat)),
        "per_image_latency_ms_p95": float(np.percentile(lat, 95)),
    }


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    D = 64
    center = rng.randn(D)
    normal_train = center + rng.randn(100, D) * 0.15          # 官方 100 正常
    defect_cal = center + rng.randn(30, D) * 0.15 + rng.randn(D) * 1.5   # 官方 30 缺陷
    test_normal = center + rng.randn(700, D) * 0.15
    test_defect = center + rng.randn(300, D) * 0.15 + rng.randn(D) * 1.5
    test = np.vstack([test_normal, test_defect])
    labels = np.concatenate([np.zeros(700), np.ones(300)])

    r = run_protocol(normal_train, defect_cal, test, labels)
    check(r["auc"] > 0.95, f"少样本协议 AUC>0.95(={r['auc']:.3f})")
    check(r["recall"] > 0.85, f"缺陷检出率>0.85(={r['recall']:.3f})")
    check(r["bank_size"] <= 80, f"memory bank 经 coreset 压缩(={r['bank_size']}≤80)")
    check(r["per_image_latency_ms_mean"] >= 0, f"记录 per-image 延时(mean={r['per_image_latency_ms_mean']:.3f}ms)")
    print(f"  协议输出:{ {k: (round(v,3) if isinstance(v,float) else v) for k,v in r.items()} }")

    print("\n" + ("✅ fewshot_protocol 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
