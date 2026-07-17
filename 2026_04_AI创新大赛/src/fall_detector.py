#!/usr/bin/env python3
"""
fall_detector.py · 低算力跌倒检测后处理 baseline

轻量规则基线:横向姿态 + 快速下坠 + 明显高度变化。它不是最终神经网络,
但能作为端侧后处理和可解释报警逻辑,后续可接轻量 YOLO/姿态/TCN 输出。
"""
import argparse

import numpy as np

from fall_features import FEATURE_NAMES, extract_clip_features, summarize_features


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def score_from_summary(summary):
    """summary -> [0,1] 跌倒分数。"""
    horizontal = _clip01((summary["max_aspect"] - 1.10) / 1.00)
    velocity = _clip01((summary["max_down_velocity"] - 0.008) / 0.016)
    shift = _clip01((summary["total_down_shift"] - 0.08) / 0.10)
    visible = _clip01((summary["visible_ratio"] - 0.50) / 0.50)
    score = (0.50 * horizontal + 0.35 * velocity + 0.15 * shift) * visible
    return float(score)


def find_alarm_window(features, score, threshold=0.70):
    """返回报警片段 [start,end],无报警则 (-1,-1)。"""
    if score < threshold:
        return (-1, -1)
    f = np.asarray(features)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    aspect = f[:, idx["aspect"]]
    dcy = f[:, idx["dcy"]]
    candidates = np.where((aspect > 1.15) | (dcy > 0.012))[0]
    if len(candidates) == 0:
        return (-1, -1)
    return (int(candidates[0]), int(candidates[-1]))


def detect_from_features(features, threshold=0.70):
    summary = summarize_features(features)
    score = score_from_summary(summary)
    start, end = find_alarm_window(features, score, threshold=threshold)
    return {
        "score": score,
        "label": int(score >= threshold),
        "alarm_start": start,
        "alarm_end": end,
        "summary": summary,
    }


def detect_clip(frames, threshold=0.70):
    feat = extract_clip_features(frames)
    out = detect_from_features(feat["features"], threshold=threshold)
    out["boxes"] = feat["boxes"]
    return out


def _selftest():
    import sys
    from fall_synth import make_clip

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    outputs = {}
    for kind in ("fall", "walk", "sit", "bend", "lie"):
        c = make_clip(kind, seed=11)
        outputs[kind] = detect_clip(c.frames)

    check(outputs["fall"]["score"] > 0.85, f"跌倒分数高({outputs['fall']['score']:.3f})")
    check(outputs["fall"]["label"] == 1 and outputs["fall"]["alarm_start"] >= 0, "跌倒触发报警窗口")
    for kind in ("walk", "sit", "bend", "lie"):
        check(outputs[kind]["label"] == 0, f"{kind} 不误报(score={outputs[kind]['score']:.3f})")
    check(outputs["lie"]["score"] < outputs["fall"]["score"], "慢慢躺下分数低于快速跌倒")

    print("\n" + ("✅ fall_detector 自测通过" if ok else "❌ fall_detector 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;检测由 run_fall_pipeline.py 调用")


if __name__ == "__main__":
    main()
