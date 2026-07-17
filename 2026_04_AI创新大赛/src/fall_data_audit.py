#!/usr/bin/env python3
"""
fall_data_audit.py · 跌倒检测公开数据 manifest 体检

在训练/评测前先检查 manifest 是否具备可用证据链:
  - clip_path/label 字段是否完整
  - 路径是否存在,标签是否同时包含 fall/nonfall
  - source/scenario 分布是否可追踪
  - 可选抽样读取片段,提前暴露视频解码或抽帧目录问题
"""
import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

from fall_synth import make_clip
from fall_video_io import IMAGE_EXTS, VIDEO_EXTS, load_clip
from run_fall_pipeline import load_manifest


def _path_kind(path):
    p = Path(path)
    if p.is_dir():
        has_images = any(c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in p.rglob("*"))
        return "frame_dir" if has_images else "dir_without_frames"
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return "video"
    if p.is_file():
        return "unsupported_file"
    return "missing"


def _counter_dict(values):
    return dict(sorted(Counter(v or "unknown" for v in values).items()))


def _readability_check(records, sample_readable=0, n_frames=8, size=(96, 72)):
    checked, readable, failures = 0, 0, []
    if sample_readable <= 0:
        return {"checked": 0, "readable": 0, "failed": 0, "failures_preview": []}
    for rec in records[:sample_readable]:
        checked += 1
        try:
            frames = load_clip(rec["clip_path"], n_frames=n_frames, size=size)
            if len(frames) != n_frames:
                raise ValueError(f"expected {n_frames} frames, got {len(frames)}")
            readable += 1
        except Exception as exc:
            failures.append({"clip_path": rec["clip_path"], "error": str(exc)})
    return {
        "checked": checked,
        "readable": readable,
        "failed": len(failures),
        "failures_preview": failures[:10],
    }


def audit_manifest(manifest_path, sample_readable=0, n_frames=8, size=(96, 72)):
    records = load_manifest(manifest_path)
    labels = [int(r["label"]) for r in records]
    kinds = [_path_kind(r["clip_path"]) for r in records]
    missing = [r["clip_path"] for r, k in zip(records, kinds) if k == "missing"]
    unsupported = [
        r["clip_path"]
        for r, k in zip(records, kinds)
        if k in {"unsupported_file", "dir_without_frames"}
    ]
    readability = _readability_check(
        records,
        sample_readable=sample_readable,
        n_frames=n_frames,
        size=size,
    )

    n_fall = int(sum(1 for y in labels if y == 1))
    n_nonfall = int(sum(1 for y in labels if y == 0))
    has_both_classes = n_fall > 0 and n_nonfall > 0
    paths_exist = len(missing) == 0 and len(unsupported) == 0
    readable_ok = readability["failed"] == 0
    ready_for_eval = len(records) > 0 and has_both_classes and paths_exist and readable_ok

    return {
        "manifest": os.path.abspath(manifest_path),
        "summary": {
            "n_clips": int(len(records)),
            "n_fall": n_fall,
            "n_nonfall": n_nonfall,
            "has_both_classes": has_both_classes,
            "ready_for_eval": ready_for_eval,
        },
        "distribution": {
            "by_source": _counter_dict(r["source"] for r in records),
            "by_scenario": _counter_dict(r["scenario"] for r in records),
            "by_path_kind": _counter_dict(kinds),
        },
        "path_checks": {
            "n_missing": len(missing),
            "n_unsupported": len(unsupported),
            "missing_preview": missing[:10],
            "unsupported_preview": unsupported[:10],
        },
        "readability": readability,
        "next_actions": _next_actions(len(records), has_both_classes, paths_exist, readability),
    }


def _next_actions(n_records, has_both_classes, paths_exist, readability):
    actions = []
    if n_records == 0:
        actions.append("manifest 为空:先用 fall_public_datasets.py 扫描 OmniFall/Kaggle 本地目录。")
    if not has_both_classes:
        actions.append("缺少正类或负类:检查目录命名和 label 字段,评测 P@R90/P@R95 必须同时包含跌倒/非跌倒。")
    if not paths_exist:
        actions.append("存在缺失或不支持路径:修正 clip_path,或先把视频抽帧为图片目录。")
    if readability["failed"] > 0:
        actions.append("存在无法读取片段:安装/修复视频解码器,或统一抽帧后再评测。")
    if not actions:
        actions.append("manifest 通过基础体检:可以运行 run_fall_pipeline.py --manifest 进行公开视频评测。")
    return actions


def save_audit(report, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _write_frames(frames, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"{i:04d}.png")


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        for kind, label in (("fall", 1), ("walk", 0), ("sit", 0)):
            clip = make_clip(kind, n_frames=7, size=(80, 60), seed=300 + label)
            frame_dir = root / kind / "clip001"
            _write_frames(clip.frames, frame_dir)
            rows.append({
                "clip_path": str(frame_dir),
                "label": str(label),
                "source": "toy_public",
                "scenario": kind,
            })
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip_path", "label", "source", "scenario"])
            w.writeheader()
            w.writerows(rows)

        report = audit_manifest(manifest, sample_readable=3, n_frames=5, size=(64, 48))
        check(report["summary"]["n_clips"] == 3, "统计 manifest clip 数")
        check(report["summary"]["n_fall"] == 1 and report["summary"]["n_nonfall"] == 2, "统计正负样本")
        check(report["distribution"]["by_path_kind"] == {"frame_dir": 3}, "识别抽帧目录")
        check(report["readability"]["checked"] == 3 and report["readability"]["failed"] == 0, "抽样可读性检查通过")
        check(report["summary"]["ready_for_eval"] is True, "正负样本齐全且路径可读时可进入评测")

        broken = root / "broken.csv"
        rows.append({
            "clip_path": str(root / "missing" / "clip.mp4"),
            "label": "1",
            "source": "toy_public",
            "scenario": "fall",
        })
        with broken.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip_path", "label", "source", "scenario"])
            w.writeheader()
            w.writerows(rows)
        broken_report = audit_manifest(broken, sample_readable=4, n_frames=5, size=(64, 48))
        check(broken_report["path_checks"]["n_missing"] == 1, "报告缺失路径")
        check(broken_report["summary"]["ready_for_eval"] is False, "有缺失路径时不能标记为可评测")

        out = root / "audit.json"
        save_audit(report, out)
        check(out.exists() and out.stat().st_size > 0, "审计报告可写出 JSON")

    print("\n" + ("✅ fall_data_audit 自测通过" if ok else "❌ fall_data_audit 自测未通过"))
    sys.exit(0 if ok else 1)


def _print_report(report):
    s = report["summary"]
    p = report["path_checks"]
    r = report["readability"]
    print("=== Fall Public Data Audit ===")
    print(f"  clips          : {s['n_clips']} ({s['n_fall']} fall / {s['n_nonfall']} non-fall)")
    print(f"  ready_for_eval : {s['ready_for_eval']}")
    print(f"  paths          : missing={p['n_missing']} unsupported={p['n_unsupported']}")
    print(f"  readability    : checked={r['checked']} readable={r['readable']} failed={r['failed']}")
    for action in report["next_actions"]:
        print(f"  next           : {action}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", help="clip-level manifest: clip_path,label,source,scenario")
    ap.add_argument("--out", default="../output/fall_public_audit.json")
    ap.add_argument("--sample-readable", type=int, default=0,
                    help="抽样读取前 N 个片段;0 表示只检查路径和分布")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if not a.manifest:
        ap.error("需提供 --manifest,或运行 --selftest")
    report = audit_manifest(
        a.manifest,
        sample_readable=a.sample_readable,
        n_frames=a.frames,
        size=(a.width, a.height),
    )
    save_audit(report, a.out)
    _print_report(report)
    print(f"\n✓ 审计报告:{os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
