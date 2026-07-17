#!/usr/bin/env python3
"""
fall_tiny_model.py · 赛题七轻量时序模型

纯 numpy logistic baseline:
  frames -> fall_features.summarize_features -> 8维向量 -> z-score -> sigmoid。

这个模型的目的不是替代最终 YOLO/姿态/TCN,而是提供可训练、可保存、可复现的
极小参数量时序头。后续真实检测器输出同样的 summary 后,本模型可直接复用。
"""
import argparse
import json

import numpy as np

from fall_features import extract_clip_features, summarize_features


MODEL_FEATURES = (
    "max_aspect",
    "final_aspect",
    "max_down_velocity",
    "total_down_shift",
    "min_height",
    "final_height",
    "max_area",
    "visible_ratio",
)


def vector_from_summary(summary):
    return np.array([float(summary[name]) for name in MODEL_FEATURES], np.float32)


def vector_from_frames(frames):
    feat = extract_clip_features(frames)
    return vector_from_summary(summarize_features(feat["features"]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(X, y, lr=0.35, epochs=900, l2=1e-3):
    """训练一个很小的 logistic regression,返回模型 dict。"""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be (N,D) and y must be length N")
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    w = np.zeros(Z.shape[1], np.float32)
    b = np.float32(0.0)
    for _ in range(int(epochs)):
        p = sigmoid(Z @ w + b)
        err = p - y
        grad_w = (Z.T @ err) / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w.astype(np.float32)
        b -= np.float32(lr * grad_b)
    return {
        "model_type": "zscore_logistic",
        "features": list(MODEL_FEATURES),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "weights": w.astype(float).tolist(),
        "bias": float(b),
        "params": int(len(w) + 1),
    }


def predict_scores(model, X):
    X = np.asarray(X, np.float32)
    mean = np.asarray(model["mean"], np.float32)
    std = np.asarray(model["std"], np.float32)
    w = np.asarray(model["weights"], np.float32)
    b = float(model["bias"])
    if X.ndim == 1:
        X = X[None, :]
    Z = (X - mean) / std
    return sigmoid(Z @ w + b)


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _selftest():
    import os
    import sys
    import tempfile
    from fall_metrics import precision_at_recall
    from fall_synth import make_dataset

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    clips = make_dataset(n_per_kind=10, n_frames=24, size=(128, 96), seed=7)
    X = np.vstack([vector_from_frames(c.frames) for c in clips])
    y = np.array([c.label for c in clips], int)
    check(X.shape == (50, len(MODEL_FEATURES)), "合成片段转 8 维模型特征")
    model = fit_logistic(X, y)
    scores = predict_scores(model, X)
    p90 = precision_at_recall(scores, y, 0.90)
    check(p90["precision"] >= 0.95, f"训练后 precision@recall90 足够高(={p90['precision']:.3f})")
    check(model["params"] == len(MODEL_FEATURES) + 1, "模型参数量极小(D+1)")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fall_tiny_model.json")
        save_model(model, path)
        loaded = load_model(path)
        scores2 = predict_scores(loaded, X)
        check(np.allclose(scores, scores2), "模型保存/加载后预测一致")

    print("\n" + ("✅ fall_tiny_model 自测通过" if ok else "❌ fall_tiny_model 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;训练入口见 train_fall_model.py")


if __name__ == "__main__":
    main()
