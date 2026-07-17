#!/usr/bin/env python3
"""
run_fall_pipeline.py · 赛题七低算力视觉跌倒检测端到端 baseline

当前默认跑合成片段,用于在无外部数据时验证完整链路:
  合成视频 -> 人体框/姿态代理特征 -> 时序检测 -> precision@recall90/95 -> 延时/参数量评分。

报告明确标注为 synthetic_proxy,不能当作官方数据成绩。
"""
import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from fall_detector import detect_clip
from fall_metrics import competition_score
from fall_synth import make_dataset
from fall_tiny_model import load_model, predict_scores, vector_from_frames
from fall_video_io import load_clip


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def _resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_manifest(manifest_path):
    """读取 clip-level manifest,支持 clip_path 或 frames_dir 字段。"""
    manifest_path = os.path.abspath(manifest_path)
    base = os.path.dirname(manifest_path)
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_path = row.get("clip_path") or row.get("frames_dir") or row.get("path")
            if not clip_path:
                raise ValueError("manifest row must include clip_path or frames_dir")
            records.append({
                "clip_path": _resolve_path(clip_path, base),
                "label": int(row["label"]),
                "source": row.get("source", ""),
                "scenario": row.get("scenario", ""),
                "clip_id": row.get("clip_id") or Path(clip_path).stem,
            })
    return records


def score_clip(frames, model=None):
    """规则或 tiny model 打分。返回统一事件字段。"""
    if model is None:
        out = detect_clip(frames)
        out["scorer"] = "rule_baseline"
        return out
    rule_out = detect_clip(frames)
    x = vector_from_frames(frames)
    score = float(predict_scores(model, x)[0])
    return {
        "score": score,
        "label": int(score >= 0.5),
        "alarm_start": rule_out["alarm_start"],
        "alarm_end": rule_out["alarm_end"],
        "summary": rule_out["summary"],
        "scorer": "tiny_model",
    }


def run_synthetic(n_per_kind=20, n_frames=32, size=(160, 120), seed=0,
                  params_m=5.0, save=True, verbose=True, model_path=None):
    clips = make_dataset(n_per_kind=n_per_kind, n_frames=n_frames, size=size, seed=seed)
    model = load_model(model_path) if model_path else None
    scores = []
    labels = []
    kinds = []
    lat_clip = []
    events = []

    for clip in clips:
        t0 = time.perf_counter()
        out = score_clip(clip.frames, model=model)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scores.append(out["score"])
        labels.append(clip.label)
        kinds.append(clip.kind)
        lat_clip.append(elapsed_ms)
        events.append({
            "clip_id": clip.clip_id,
            "kind": clip.kind,
            "label": clip.label,
            "score": round(float(out["score"]), 6),
            "alarm_start": out["alarm_start"],
            "alarm_end": out["alarm_end"],
        })

    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    lat_clip = np.asarray(lat_clip, float)
    per_frame_ms = lat_clip / float(n_frames)
    comp = competition_score(scores, labels, params_m=params_m,
                             latency_ms=float(per_frame_ms.mean()))
    by_kind = {}
    for kind in sorted(set(kinds)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        by_kind[kind] = {
            "n": len(idx),
            "label": int(labels[idx[0]]),
            "mean_score": round(float(scores[idx].mean()), 6),
            "max_score": round(float(scores[idx].max()), 6),
        }

    report = {
        "task": "huawei_enterprise_7_low_power_visual_fall_detection",
        "mode": "synthetic_proxy",
        "scorer": "tiny_model" if model is not None else "rule_baseline",
        "note": "当前为合成片段+几何时序后处理 baseline;公开/官方数据成绩需接入真实视频后重跑。",
        "dataset": {
            "n_clips": len(clips),
            "n_fall": int(labels.sum()),
            "n_nonfall": int((labels == 0).sum()),
            "n_frames_per_clip": n_frames,
            "input_size": {"width": size[0], "height": size[1]},
            "kinds": by_kind,
        },
        "metrics": {
            "precision_at_recall90": round(float(comp["precision_at_recall90"]["precision"]), 6),
            "threshold_at_recall90": round(float(comp["precision_at_recall90"]["threshold"]), 6),
            "precision_at_recall95": round(float(comp["precision_at_recall95"]["precision"]), 6),
            "threshold_at_recall95": round(float(comp["precision_at_recall95"]["threshold"]), 6),
        },
        "profile": {
            "assumed_params_m": float(params_m),
            "mean_latency_ms_per_clip": round(float(lat_clip.mean()), 6),
            "p95_latency_ms_per_clip": round(float(np.percentile(lat_clip, 95)), 6),
            "mean_latency_ms_per_frame": round(float(per_frame_ms.mean()), 6),
            "p95_latency_ms_per_frame": round(float(np.percentile(per_frame_ms, 95)), 6),
            "latency_note": "当前为 CPU 上规则后处理耗时,最终端侧耗时需包含检测/姿态模型和 NPU 实测。",
        },
        "competition_score_proxy": {
            "map_score": round(float(comp["map_score"]), 6),
            "param_score": round(float(comp["param_score"]), 6),
            "latency_score": round(float(comp["latency_score"]), 6),
            "total_score": round(float(comp["total_score"]), 6),
        },
        "events_preview": events[:10],
    }

    if verbose:
        _print_report(report)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        report_path = os.path.join(OUT_DIR, "fall_pipeline_report.json")
        scores_path = os.path.join(OUT_DIR, "fall_scores.npz")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        np.savez(scores_path, scores=scores, labels=labels, kinds=np.array(kinds))
        if verbose:
            print(f"\n✓ 报告:{report_path}")
            print(f"✓ 分数:{scores_path}")
    return report


def _group_summary(keys, scores, labels):
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        out[key or "unknown"] = {
            "n": len(idx),
            "n_fall": int(labels[idx].sum()),
            "mean_score": round(float(scores[idx].mean()), 6),
            "max_score": round(float(scores[idx].max()), 6),
        }
    return out


def run_manifest(manifest, n_frames=32, size=(160, 120), params_m=5.0,
                 save=True, verbose=True, skip_errors=False, model_path=None):
    """用公开视频/抽帧目录 manifest 跑同一套跌倒检测 pipeline。"""
    records = load_manifest(manifest)
    if not records:
        raise ValueError("manifest is empty")
    model = load_model(model_path) if model_path else None

    scores, labels, sources, scenarios, lat_clip, events = [], [], [], [], [], []
    failures = []
    for rec in records:
        try:
            frames = load_clip(rec["clip_path"], n_frames=n_frames, size=size)
            t0 = time.perf_counter()
            out = score_clip(frames, model=model)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            if not skip_errors:
                raise
            failures.append({"clip_path": rec["clip_path"], "error": str(exc)})
            continue

        scores.append(out["score"])
        labels.append(rec["label"])
        sources.append(rec["source"])
        scenarios.append(rec["scenario"])
        lat_clip.append(elapsed_ms)
        events.append({
            "clip_id": rec["clip_id"],
            "source": rec["source"],
            "scenario": rec["scenario"],
            "label": rec["label"],
            "score": round(float(out["score"]), 6),
            "alarm_start": out["alarm_start"],
            "alarm_end": out["alarm_end"],
        })

    if not scores:
        raise ValueError("no readable clips after manifest loading")

    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    lat_clip = np.asarray(lat_clip, float)
    per_frame_ms = lat_clip / float(n_frames)
    comp = competition_score(scores, labels, params_m=params_m,
                             latency_ms=float(per_frame_ms.mean()))
    report = {
        "task": "huawei_enterprise_7_low_power_visual_fall_detection",
        "mode": "manifest_eval",
        "scorer": "tiny_model" if model is not None else "rule_baseline",
        "source_manifest": os.path.abspath(manifest),
        "note": "manifest_eval 使用本地公开视频或抽帧目录;若使用公开数据,可作为正式公开实验基础。",
        "dataset": {
            "n_clips": int(len(scores)),
            "n_fall": int(labels.sum()),
            "n_nonfall": int((labels == 0).sum()),
            "n_frames_per_clip": int(n_frames),
            "input_size": {"width": size[0], "height": size[1]},
            "by_source": _group_summary(sources, scores, labels),
            "by_scenario": _group_summary(scenarios, scores, labels),
            "n_failed": len(failures),
        },
        "metrics": {
            "precision_at_recall90": round(float(comp["precision_at_recall90"]["precision"]), 6),
            "threshold_at_recall90": round(float(comp["precision_at_recall90"]["threshold"]), 6),
            "precision_at_recall95": round(float(comp["precision_at_recall95"]["precision"]), 6),
            "threshold_at_recall95": round(float(comp["precision_at_recall95"]["threshold"]), 6),
        },
        "profile": {
            "assumed_params_m": float(params_m),
            "mean_latency_ms_per_clip": round(float(lat_clip.mean()), 6),
            "p95_latency_ms_per_clip": round(float(np.percentile(lat_clip, 95)), 6),
            "mean_latency_ms_per_frame": round(float(per_frame_ms.mean()), 6),
            "p95_latency_ms_per_frame": round(float(np.percentile(per_frame_ms, 95)), 6),
            "latency_note": "当前计时包含后处理,不包含视频解码;正式端侧需补模型+NPU全链路。",
        },
        "competition_score_proxy": {
            "map_score": round(float(comp["map_score"]), 6),
            "param_score": round(float(comp["param_score"]), 6),
            "latency_score": round(float(comp["latency_score"]), 6),
            "total_score": round(float(comp["total_score"]), 6),
        },
        "events_preview": events[:10],
        "failures_preview": failures[:10],
    }

    if verbose:
        _print_manifest_report(report)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        suffix = ".manifest.tiny_model" if model is not None else ".manifest"
        report_path = os.path.join(OUT_DIR, f"fall_pipeline_report{suffix}.json")
        scores_path = os.path.join(OUT_DIR, f"fall_scores{suffix}.npz")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        np.savez(scores_path, scores=scores, labels=labels,
                 sources=np.array(sources), scenarios=np.array(scenarios))
        if verbose:
            print(f"\n✓ 报告:{report_path}")
            print(f"✓ 分数:{scores_path}")
    return report


def _print_report(report):
    ds = report["dataset"]
    m = report["metrics"]
    p = report["profile"]
    s = report["competition_score_proxy"]
    print("=== Fall Detection Baseline (synthetic_proxy) ===")
    print(f"  clips          : {ds['n_clips']} ({ds['n_fall']} fall / {ds['n_nonfall']} non-fall)")
    print(f"  input          : {ds['input_size']['width']}x{ds['input_size']['height']} x {ds['n_frames_per_clip']} frames")
    print(f"  P@R90 / P@R95  : {m['precision_at_recall90']} / {m['precision_at_recall95']}")
    print(f"  latency/frame  : mean={p['mean_latency_ms_per_frame']}ms p95={p['p95_latency_ms_per_frame']}ms")
    print(f"  params assumed : {p['assumed_params_m']}M")
    print(f"  score proxy    : {s['total_score']} (mAP {s['map_score']} + param {s['param_score']} + latency {s['latency_score']})")


def _print_manifest_report(report):
    ds = report["dataset"]
    m = report["metrics"]
    p = report["profile"]
    s = report["competition_score_proxy"]
    print("=== Fall Detection Baseline (manifest_eval) ===")
    print(f"  clips          : {ds['n_clips']} ({ds['n_fall']} fall / {ds['n_nonfall']} non-fall, failed {ds['n_failed']})")
    print(f"  input          : {ds['input_size']['width']}x{ds['input_size']['height']} x {ds['n_frames_per_clip']} frames")
    print(f"  P@R90 / P@R95  : {m['precision_at_recall90']} / {m['precision_at_recall95']}")
    print(f"  latency/frame  : mean={p['mean_latency_ms_per_frame']}ms p95={p['p95_latency_ms_per_frame']}ms")
    print(f"  score proxy    : {s['total_score']} (mAP {s['map_score']} + param {s['param_score']} + latency {s['latency_score']})")


def _selftest():
    import sys
    from PIL import Image
    from fall_synth import make_clip
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    r = run_synthetic(n_per_kind=4, n_frames=24, size=(128, 96), seed=123,
                      params_m=5.0, save=False, verbose=False)
    check(r["dataset"]["n_clips"] == 20, "自测数据集 5 类×4 条")
    check(r["metrics"]["precision_at_recall90"] >= 0.95, "precision@recall90 足够高")
    check(r["metrics"]["precision_at_recall95"] >= 0.95, "precision@recall95 足够高")
    check(r["profile"]["mean_latency_ms_per_frame"] < 100.0, "单帧 CPU 代理耗时 <100ms")
    check(r["competition_score_proxy"]["total_score"] > 90.0, "代理竞赛分可用")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        for kind, label in (("fall", 1), ("walk", 0), ("sit", 0), ("lie", 0)):
            for i in range(2):
                clip = make_clip(kind, n_frames=10, size=(96, 72), seed=100 + i)
                frame_dir = root / f"{kind}_{i}"
                frame_dir.mkdir(parents=True)
                for j, frame in enumerate(clip.frames):
                    Image.fromarray(frame).save(frame_dir / f"{j:04d}.png")
                rows.append({"clip_path": str(frame_dir), "label": str(label),
                             "source": "toy_frames", "scenario": kind})
        manifest = root / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip_path", "label", "source", "scenario"])
            w.writeheader()
            w.writerows(rows)
        rm = run_manifest(str(manifest), n_frames=8, size=(96, 72), params_m=5.0,
                          save=False, verbose=False)
        check(rm["dataset"]["n_clips"] == 8, "manifest 入口读取 8 个帧目录片段")
        check(rm["dataset"]["n_fall"] == 2 and rm["dataset"]["n_nonfall"] == 6, "manifest 正负样本统计正确")
        check(rm["metrics"]["precision_at_recall90"] >= 0.95, "manifest 入口 precision@recall90 足够高")

        from fall_tiny_model import fit_logistic, save_model
        X_rows, y_rows = [], []
        for row in rows:
            frames = load_clip(row["clip_path"], n_frames=8, size=(96, 72))
            X_rows.append(vector_from_frames(frames))
            y_rows.append(int(row["label"]))
        model = fit_logistic(np.vstack(X_rows), np.array(y_rows))
        model_path = root / "tiny.json"
        save_model(model, model_path)
        rm_model = run_manifest(str(manifest), n_frames=8, size=(96, 72), params_m=0.001,
                                model_path=str(model_path), save=False, verbose=False)
        check(rm_model["scorer"] == "tiny_model", "manifest 入口可使用 tiny model 打分")
        check(rm_model["metrics"]["precision_at_recall90"] >= 0.95, "tiny model manifest 评测可用")
        fall_events = [e for e in rm_model["events_preview"] if e["label"] == 1]
        check(fall_events and fall_events[0]["alarm_start"] >= 0, "tiny model 评测保留报警窗口")

    print("\n" + ("✅ run_fall_pipeline 自测通过" if ok else "❌ run_fall_pipeline 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-per-kind", type=int, default=20)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", help="clip-level manifest: clip_path,label,source,scenario")
    ap.add_argument("--skip-errors", action="store_true", help="批量公开视频评测时跳过无法读取的片段")
    ap.add_argument("--model", help="fall_tiny_model JSON;不提供则使用规则 baseline")
    ap.add_argument("--params-m", type=float, default=5.0,
                    help="代理参数量估计:轻量检测/姿态模型+时序头,正式提交需用真实模型替换")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.manifest:
        run_manifest(a.manifest, n_frames=a.frames, size=(a.width, a.height),
                     params_m=a.params_m, skip_errors=a.skip_errors, model_path=a.model)
    else:
        run_synthetic(n_per_kind=a.n_per_kind, n_frames=a.frames, size=(a.width, a.height),
                      seed=a.seed, params_m=a.params_m, model_path=a.model)


if __name__ == "__main__":
    main()
