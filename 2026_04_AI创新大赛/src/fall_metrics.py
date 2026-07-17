#!/usr/bin/env python3
"""
fall_metrics.py · 赛题七跌倒检测指标与竞赛分近似

官方核心:在 Recall=90% / 95% 时的 Precision 均值占 50%,模型参数量占 25%,
推理耗时占 25%。原附件公式图片未转成文本,这里按文字约束实现一个透明的近似评分:
  - 参数≤20M:25分基线,越小最多+10;20M~40M:线性扣到0;>40M:0
  - 延时≤100ms:25分基线,越快最多+10;100ms~200ms:线性扣到0;>200ms:0
"""
import argparse
import math

import numpy as np


def precision_recall_at_threshold(scores, labels, threshold):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pred = scores >= float(threshold)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn}


def precision_at_recall(scores, labels, target_recall=0.9):
    """返回达到 target_recall 时 precision 最高的阈值点。"""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length")
    if int((labels == 1).sum()) == 0:
        return {"threshold": math.nan, "precision": 0.0, "recall": 0.0,
                "tp": 0, "fp": 0, "fn": 0}

    candidates = np.unique(scores)
    # 加一个略低于最小分的阈值,保证可达到 recall=1。
    candidates = np.concatenate(([float(scores.min()) - 1e-9], candidates))
    feasible = []
    for thr in candidates:
        m = precision_recall_at_threshold(scores, labels, thr)
        if m["recall"] >= target_recall:
            feasible.append(m)
    if not feasible:
        return {"threshold": math.nan, "precision": 0.0, "recall": 0.0,
                "tp": 0, "fp": 0, "fn": int((labels == 1).sum())}
    return max(feasible, key=lambda r: (r["precision"], r["threshold"]))


def param_score(params_m, baseline_m=20.0, hard_limit_m=40.0):
    """参数量得分:≤20M 有25基线+最多10奖励;20M~40M线性扣到0。"""
    p = float(params_m)
    if p <= 0:
        return 35.0
    if p <= baseline_m:
        return 25.0 + 10.0 * (1.0 - p / baseline_m)
    if p <= hard_limit_m:
        return 25.0 * (hard_limit_m - p) / (hard_limit_m - baseline_m)
    return 0.0


def latency_score(latency_ms, baseline_ms=100.0, hard_limit_ms=200.0):
    """推理耗时得分:≤100ms 有25基线+最多10奖励;100ms~200ms线性扣到0。"""
    t = float(latency_ms)
    if t <= 0:
        return 35.0
    if t <= baseline_ms:
        return 25.0 + 10.0 * (1.0 - t / baseline_ms)
    if t <= hard_limit_ms:
        return 25.0 * (hard_limit_ms - t) / (hard_limit_ms - baseline_ms)
    return 0.0


def competition_score(scores, labels, params_m, latency_ms):
    p90 = precision_at_recall(scores, labels, 0.90)
    p95 = precision_at_recall(scores, labels, 0.95)
    map_part = 50.0 * ((p90["precision"] + p95["precision"]) / 2.0)
    ps = param_score(params_m)
    ls = latency_score(latency_ms)
    return {
        "precision_at_recall90": p90,
        "precision_at_recall95": p95,
        "map_score": map_part,
        "param_score": ps,
        "latency_score": ls,
        "total_score": map_part + ps + ls,
    }


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    scores = np.array([0.99, 0.93, 0.91, 0.88, 0.12, 0.11, 0.10, 0.09])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p90 = precision_at_recall(scores, labels, 0.90)
    check(p90["precision"] == 1.0 and p90["recall"] == 1.0, "可分数据 precision@recall90=1")

    mixed_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.85, 0.2])
    mixed_labels = np.array([1, 1, 1, 1, 0, 0])
    p95 = precision_at_recall(mixed_scores, mixed_labels, 0.95)
    check(abs(p95["precision"] - 0.8) < 1e-9, "召回95时 precision 计入误报")

    check(param_score(10) > param_score(20) > param_score(30) > param_score(40) == 0, "参数量越小得分越高")
    check(latency_score(50) > latency_score(100) > latency_score(150) > latency_score(200) == 0, "延时越低得分越高")

    total = competition_score(scores, labels, params_m=5.0, latency_ms=20.0)
    check(total["total_score"] > 100.0, "低参数低延时可拿奖励分")
    check(total["map_score"] == 50.0, "满 precision 时 mAP 部分为50")

    print("\n" + ("✅ fall_metrics 自测通过" if ok else "❌ fall_metrics 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;指标由 run_fall_pipeline.py 调用")


if __name__ == "__main__":
    main()
