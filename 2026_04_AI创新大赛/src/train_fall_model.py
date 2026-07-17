#!/usr/bin/env python3
"""
train_fall_model.py · 训练赛题七 tiny temporal model

支持两种输入:
  - 默认合成数据训练:无外部数据时产出可运行模型。
  - --manifest:读取 OmniFall/Kaggle/抽帧目录 manifest 后训练。
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fall_metrics import competition_score
from fall_synth import make_clip, make_dataset
from fall_tiny_model import fit_logistic, predict_scores, save_model, vector_from_frames
from fall_video_io import load_clip
from run_fall_pipeline import load_manifest


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def _split_indices(n, val_ratio=0.30, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    return idx[n_val:], idx[:n_val]


def _features_from_clips(clips):
    X = np.vstack([vector_from_frames(c.frames) for c in clips])
    y = np.array([c.label for c in clips], int)
    return X, y


def _features_from_manifest(manifest, n_frames=32, size=(160, 120), skip_errors=False):
    records = load_manifest(manifest)
    X, y, failures = [], [], []
    for rec in records:
        try:
            frames = load_clip(rec["clip_path"], n_frames=n_frames, size=size)
            X.append(vector_from_frames(frames))
            y.append(rec["label"])
        except Exception as exc:
            if not skip_errors:
                raise
            failures.append({"clip_path": rec["clip_path"], "error": str(exc)})
    if not X:
        raise ValueError("no readable clips for training")
    return np.vstack(X), np.asarray(y, int), failures


def train_from_arrays(X, y, seed=0, val_ratio=0.30, params_m=0.001):
    train_idx, val_idx = _split_indices(len(y), val_ratio=val_ratio, seed=seed)
    model = fit_logistic(X[train_idx], y[train_idx])
    train_scores = predict_scores(model, X[train_idx])
    val_scores = predict_scores(model, X[val_idx])
    train_comp = competition_score(train_scores, y[train_idx], params_m=params_m, latency_ms=1.0)
    val_comp = competition_score(val_scores, y[val_idx], params_m=params_m, latency_ms=1.0)
    return model, {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train": _compact_score(train_comp),
        "val": _compact_score(val_comp),
    }


def _compact_score(comp):
    return {
        "precision_at_recall90": round(float(comp["precision_at_recall90"]["precision"]), 6),
        "precision_at_recall95": round(float(comp["precision_at_recall95"]["precision"]), 6),
        "total_score_proxy": round(float(comp["total_score"]), 6),
    }


def train_synthetic(n_per_kind=24, n_frames=32, size=(160, 120), seed=0,
                    save=True, verbose=True):
    clips = make_dataset(n_per_kind=n_per_kind, n_frames=n_frames, size=size, seed=seed)
    X, y = _features_from_clips(clips)
    t0 = time.perf_counter()
    model, metrics = train_from_arrays(X, y, seed=seed)
    train_ms = (time.perf_counter() - t0) * 1000.0
    report = {
        "mode": "synthetic_train",
        "dataset": {"n_clips": int(len(y)), "n_fall": int(y.sum()), "n_nonfall": int((y == 0).sum())},
        "model": {"type": model["model_type"], "features": model["features"], "params": model["params"]},
        "metrics": metrics,
        "profile": {"train_ms": round(float(train_ms), 6)},
    }
    if save:
        _save_outputs(model, report)
    if verbose:
        _print_report(report)
    return model, report


def train_manifest(manifest, n_frames=32, size=(160, 120), seed=0, skip_errors=False,
                   save=True, verbose=True):
    X, y, failures = _features_from_manifest(manifest, n_frames=n_frames, size=size,
                                             skip_errors=skip_errors)
    t0 = time.perf_counter()
    model, metrics = train_from_arrays(X, y, seed=seed)
    train_ms = (time.perf_counter() - t0) * 1000.0
    report = {
        "mode": "manifest_train",
        "source_manifest": os.path.abspath(manifest),
        "dataset": {"n_clips": int(len(y)), "n_fall": int(y.sum()),
                    "n_nonfall": int((y == 0).sum()), "n_failed": len(failures)},
        "model": {"type": model["model_type"], "features": model["features"], "params": model["params"]},
        "metrics": metrics,
        "profile": {"train_ms": round(float(train_ms), 6)},
        "failures_preview": failures[:10],
    }
    if save:
        _save_outputs(model, report, suffix=".manifest")
    if verbose:
        _print_report(report)
    return model, report


def _save_outputs(model, report, suffix=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, f"fall_tiny_model{suffix}.json")
    report_path = os.path.join(OUT_DIR, f"fall_tiny_model_report{suffix}.json")
    save_model(model, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _print_report(report):
    ds = report["dataset"]
    val = report["metrics"]["val"]
    print(f"=== train_fall_model ({report['mode']}) ===")
    print(f"  clips          : {ds['n_clips']} ({ds['n_fall']} fall / {ds['n_nonfall']} non-fall)")
    print(f"  model params   : {report['model']['params']}")
    print(f"  val P@R90/R95  : {val['precision_at_recall90']} / {val['precision_at_recall95']}")
    print(f"  val score proxy: {val['total_score_proxy']}")


def _write_frames(frames, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"{i:04d}.png")


def _selftest():
    import csv
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    model, report = train_synthetic(n_per_kind=8, n_frames=20, size=(128, 96),
                                    seed=21, save=False, verbose=False)
    check(model["params"] == 9, "合成训练得到 9 参数 tiny model")
    check(report["metrics"]["val"]["precision_at_recall90"] >= 0.90, "合成训练验证 P@R90 可用")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        for kind, label in (("fall", 1), ("walk", 0), ("sit", 0), ("lie", 0)):
            for i in range(3):
                clip = make_clip(kind, n_frames=12, size=(96, 72), seed=200 + i)
                frame_dir = root / f"{kind}_{i}"
                _write_frames(clip.frames, frame_dir)
                rows.append({"clip_path": str(frame_dir), "label": str(label),
                             "source": "toy_frames", "scenario": kind})
        manifest = root / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip_path", "label", "source", "scenario"])
            w.writeheader()
            w.writerows(rows)
        model2, report2 = train_manifest(str(manifest), n_frames=10, size=(96, 72),
                                         seed=22, save=False, verbose=False)
        check(model2["params"] == 9, "manifest 训练得到 9 参数 tiny model")
        check(report2["dataset"]["n_clips"] == 12, "manifest 训练读取全部帧目录")

    print("\n" + ("✅ train_fall_model 自测通过" if ok else "❌ train_fall_model 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", help="clip-level manifest;不提供则训练合成数据")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-kind", type=int, default=24)
    ap.add_argument("--skip-errors", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.manifest:
        train_manifest(a.manifest, n_frames=a.frames, size=(a.width, a.height),
                       seed=a.seed, skip_errors=a.skip_errors)
    else:
        train_synthetic(n_per_kind=a.n_per_kind, n_frames=a.frames,
                        size=(a.width, a.height), seed=a.seed)


if __name__ == "__main__":
    main()
