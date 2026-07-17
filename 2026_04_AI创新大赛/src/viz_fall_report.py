#!/usr/bin/env python3
"""
viz_fall_report.py · 赛题七跌倒检测可视化产物

生成可直接放进项目文档/答辩视频的三类图:
  - fall_score_distribution.png:跌倒/非跌倒分数分布
  - fall_pr_curve.png:Precision-Recall 曲线
  - fall_demo_contact_sheet.png:五类动作示例 + 报警窗口
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fall_detector import detect_clip
from fall_metrics import precision_recall_at_threshold
from fall_synth import make_clip
from fall_tiny_model import load_model
from run_fall_pipeline import OUT_DIR, run_synthetic, score_clip


W, H = 960, 540


def _font(size=16):
    for p in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _canvas(title):
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.text((24, 18), title, fill=(20, 28, 38), font=_font(24))
    return img, d


def _load_scores(scores_path=None):
    path = scores_path or os.path.join(OUT_DIR, "fall_scores.npz")
    if not os.path.exists(path):
        run_synthetic(verbose=False)
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["scores"], float), np.asarray(data["labels"], int)


def pr_points(scores, labels):
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.min() - 1e-9], thresholds, [scores.max() + 1e-9]))
    pts = []
    for thr in thresholds:
        m = precision_recall_at_threshold(scores, labels, thr)
        pts.append((m["recall"], m["precision"], thr))
    pts.sort(key=lambda x: x[0])
    return pts


def draw_score_distribution(scores, labels, out_path):
    img, d = _canvas("Fall Detection Score Distribution")
    left, top, right, bottom = 80, 90, 900, 430
    d.rectangle((left, top, right, bottom), outline=(90, 96, 105), width=2)
    bins = np.linspace(0, 1, 21)
    fall_hist, _ = np.histogram(scores[labels == 1], bins=bins)
    non_hist, _ = np.histogram(scores[labels == 0], bins=bins)
    max_count = max(1, int(max(fall_hist.max(initial=0), non_hist.max(initial=0))))
    bar_w = (right - left) / (len(bins) - 1)
    for i in range(len(bins) - 1):
        x0 = left + i * bar_w
        x1 = left + (i + 1) * bar_w - 2
        nf_h = (non_hist[i] / max_count) * (bottom - top - 20)
        f_h = (fall_hist[i] / max_count) * (bottom - top - 20)
        d.rectangle((x0, bottom - nf_h, x1, bottom), fill=(80, 140, 210))
        d.rectangle((x0, bottom - nf_h - f_h, x1, bottom - nf_h), fill=(220, 82, 74))
    for x, label in ((left, "0.0"), ((left + right) / 2, "0.5"), (right - 28, "1.0")):
        d.text((x, bottom + 12), label, fill=(45, 52, 60), font=_font(14))
    d.text((left, bottom + 44), "score", fill=(45, 52, 60), font=_font(14))
    d.rectangle((690, 92, 710, 112), fill=(220, 82, 74))
    d.text((720, 88), "fall", fill=(30, 30, 30), font=_font(16))
    d.rectangle((690, 122, 710, 142), fill=(80, 140, 210))
    d.text((720, 118), "non-fall", fill=(30, 30, 30), font=_font(16))
    img.save(out_path)
    return out_path


def draw_pr_curve(scores, labels, out_path):
    pts = pr_points(scores, labels)
    img, d = _canvas("Precision-Recall Curve")
    left, top, right, bottom = 90, 90, 850, 440
    d.rectangle((left, top, right, bottom), outline=(90, 96, 105), width=2)
    d.line((left, bottom, right, bottom), fill=(90, 96, 105), width=2)
    d.line((left, top, left, bottom), fill=(90, 96, 105), width=2)
    coords = []
    for recall, precision, _ in pts:
        x = left + recall * (right - left)
        y = bottom - precision * (bottom - top)
        coords.append((x, y))
    if len(coords) > 1:
        d.line(coords, fill=(36, 150, 130), width=4)
    for val in (0.0, 0.5, 0.9, 0.95, 1.0):
        x = left + val * (right - left)
        d.line((x, bottom, x, bottom + 6), fill=(90, 96, 105), width=1)
        d.text((x - 16, bottom + 14), f"{val:g}", fill=(45, 52, 60), font=_font(13))
    for val in (0.0, 0.5, 1.0):
        y = bottom - val * (bottom - top)
        d.line((left - 6, y, left, y), fill=(90, 96, 105), width=1)
        d.text((left - 48, y - 8), f"{val:g}", fill=(45, 52, 60), font=_font(13))
    d.text((400, bottom + 44), "Recall", fill=(45, 52, 60), font=_font(16))
    d.text((20, 245), "Precision", fill=(45, 52, 60), font=_font(16))
    img.save(out_path)
    return out_path


def _draw_frame_with_box(frame, box, title, alarm=False):
    img = Image.fromarray(frame).resize((160, 120))
    d = ImageDraw.Draw(img)
    sx, sy = 160 / frame.shape[1], 120 / frame.shape[0]
    if box is not None:
        x1, y1, x2, y2 = box
        color = (230, 70, 65) if alarm else (70, 180, 110)
        d.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=color, width=3)
    d.rectangle((0, 0, 160, 22), fill=(0, 0, 0))
    d.text((5, 3), title, fill=(255, 255, 255), font=_font(12))
    return img


def draw_demo_contact_sheet(out_path, model_path=None):
    model = load_model(model_path) if model_path else None
    kinds = ("fall", "walk", "sit", "bend", "lie")
    cell_w, cell_h = 170, 160
    img = Image.new("RGB", (cell_w * 3 + 190, cell_h * len(kinds) + 70), (248, 248, 246))
    d = ImageDraw.Draw(img)
    d.text((24, 18), "Fall Detection Demo Frames", fill=(20, 28, 38), font=_font(24))
    for row, kind in enumerate(kinds):
        clip = make_clip(kind, n_frames=32, size=(160, 120), seed=20 + row)
        rule = detect_clip(clip.frames)
        scored = score_clip(clip.frames, model=model)
        y = 62 + row * cell_h
        label = "fall" if clip.label else "non-fall"
        d.text((20, y + 48), f"{kind}\nscore={scored['score']:.3f}\n{label}",
               fill=(25, 32, 42), font=_font(15), spacing=4)
        for col, idx in enumerate((0, 16, 31)):
            alarm = scored["alarm_start"] <= idx <= scored["alarm_end"] if scored["alarm_start"] >= 0 else False
            title = f"t={idx}" + (" alarm" if alarm else "")
            tile = _draw_frame_with_box(clip.frames[idx], rule["boxes"][idx], title, alarm=alarm)
            img.paste(tile, (170 + col * cell_w, y))
    img.save(out_path)
    return out_path


def generate_all(scores_path=None, model_path=None, out_dir=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    scores, labels = _load_scores(scores_path)
    paths = {
        "score_distribution": draw_score_distribution(scores, labels, os.path.join(out_dir, "fall_score_distribution.png")),
        "pr_curve": draw_pr_curve(scores, labels, os.path.join(out_dir, "fall_pr_curve.png")),
        "demo_contact_sheet": draw_demo_contact_sheet(os.path.join(out_dir, "fall_demo_contact_sheet.png"), model_path=model_path),
    }
    return paths


def _selftest():
    import sys
    import tempfile

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        scores = np.array([0.95, 0.9, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        np.savez(os.path.join(td, "scores.npz"), scores=scores, labels=labels)
        paths = generate_all(scores_path=os.path.join(td, "scores.npz"), out_dir=td)
        for name, path in paths.items():
            check(os.path.exists(path) and os.path.getsize(path) > 1000, f"{name} PNG 已生成")

    print("\n" + ("✅ viz_fall_report 自测通过" if ok else "❌ viz_fall_report 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores", default=None, help="fall_scores.npz 路径")
    ap.add_argument("--model", default=None, help="可选 tiny model JSON,用于 demo 分数")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    paths = generate_all(scores_path=a.scores, model_path=a.model)
    for k, p in paths.items():
        print(f"✓ {k}: {p}")


if __name__ == "__main__":
    main()
