#!/usr/bin/env python3
"""
fall_public_datasets.py · OmniFall/Kaggle 跌倒视频 manifest 适配器

本脚本不下载或解码视频,只扫描本地数据目录并生成 clip-level manifest:
  clip_path,label,source,scenario

标签按路径 token 推断:
  fall/falling/fallen -> 1
  nonfall/normal/adl/walk/sit/bend/lie/stand -> 0
负类 token 优先,避免 "nonfall" 被误判为 fall。
"""
import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FALL_TOKENS = {"fall", "falling", "fallen", "跌倒"}
NONFALL_TOKENS = {
    "nonfall", "non-fall", "notfall", "normal", "adl", "daily",
    "walk", "walking", "sit", "sitting", "bend", "bending", "lie",
    "lying", "stand", "standing", "no_fall", "nofall", "negative",
}


def _tokens(path):
    text = str(path).lower()
    raw = [t for t in re.split(r"[^a-z0-9_\-\u4e00-\u9fff]+", text) if t]
    split_more = []
    for token in raw:
        split_more.append(token)
        split_more.extend(t for t in re.split(r"[_\-]+", token) if t)
    return set(split_more)


def infer_label(path):
    toks = _tokens(path)
    if toks & NONFALL_TOKENS:
        return 0
    if toks & FALL_TOKENS:
        return 1
    return None


def scan_video_manifest(root, source="public"):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    records = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        label = infer_label(p.relative_to(root))
        if label is None:
            continue
        rel = p.relative_to(root)
        scenario = rel.parts[0] if len(rel.parts) > 1 else ""
        records.append({
            "clip_path": str(p),
            "label": str(label),
            "source": source,
            "scenario": scenario,
        })
    # 兼容已抽帧数据:每个含图片文件的目录视为一个 clip。
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        has_images = any(c.is_file() and c.suffix.lower() in IMAGE_EXTS for c in d.iterdir())
        if not has_images:
            continue
        label = infer_label(d.relative_to(root))
        if label is None:
            continue
        rel = d.relative_to(root)
        scenario = rel.parts[0] if len(rel.parts) > 1 else ""
        records.append({
            "clip_path": str(d),
            "label": str(label),
            "source": source,
            "scenario": scenario,
        })
    return records


def write_manifest(records, path):
    fields = ["clip_path", "label", "source", "scenario"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize(records):
    n = len(records)
    pos = sum(1 for r in records if str(r["label"]) == "1")
    return {"total": n, "fall": pos, "nonfall": n - pos}


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _touch(root / "Fall" / "clip001.mp4")
        _touch(root / "nonfall" / "walk001.avi")
        _touch(root / "ADL" / "sit001.mp4")
        _touch(root / "fall" / "seq001" / "0001.png")
        _touch(root / "fall" / "seq001" / "0002.png")
        _touch(root / "unknown" / "sample.mp4")
        recs = scan_video_manifest(root, source="toy")
        s = summarize(recs)
        check(s["total"] == 4, "跳过无法推断标签的视频/帧目录")
        check(s["fall"] == 2 and s["nonfall"] == 2, "fall/nonfall 标签推断正确")
        check(any(Path(r["clip_path"]).is_dir() for r in recs), "已抽帧目录可作为 clip")
        check(infer_label("nonfall/fall_like_name.mp4") == 0, "nonfall token 优先于 fall")
        out = root / "manifest.csv"
        write_manifest(recs, out)
        check(out.exists() and out.stat().st_size > 0, "manifest 可写出")

    print("\n" + ("✅ fall_public_datasets 自测通过" if ok else "❌ fall_public_datasets 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--omnifall", help="OmniFall 本地解压目录")
    ap.add_argument("--kaggle", help="Kaggle Fall Video Dataset 本地解压目录")
    ap.add_argument("--out", default="../output/fall_public_manifest.csv")
    a = ap.parse_args()

    if a.selftest:
        _selftest()

    records = []
    if a.omnifall:
        records.extend(scan_video_manifest(a.omnifall, source="omnifall"))
    if a.kaggle:
        records.extend(scan_video_manifest(a.kaggle, source="kaggle_fall_video"))
    if not records:
        ap.error("需提供 --omnifall 或 --kaggle,或运行 --selftest")

    write_manifest(records, a.out)
    s = summarize(records)
    print(f"✓ manifest -> {os.path.abspath(a.out)}")
    print(f"  total={s['total']} fall={s['fall']} nonfall={s['nonfall']}")


if __name__ == "__main__":
    main()
