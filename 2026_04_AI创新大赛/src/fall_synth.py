#!/usr/bin/env python3
"""
fall_synth.py · 跌倒检测合成视频片段

无公开数据下载前,用程序化片段覆盖赛题中的易混动作:
  - fall:快速下坠并变为横向姿态
  - walk/sit/bend/lie:日常动作,用于压低误报

帧是低分辨率代理输入,模拟 1080P 视频经过端侧预处理后的算法输入。
"""
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Clip:
    clip_id: str
    frames: list
    label: int
    kind: str
    bboxes: list
    event: tuple


def _bbox_from_center(cx, cy, w, h, width, height):
    x1 = int(max(0, round(cx - w / 2)))
    y1 = int(max(0, round(cy - h / 2)))
    x2 = int(min(width - 1, round(cx + w / 2)))
    y2 = int(min(height - 1, round(cy + h / 2)))
    return (x1, y1, x2, y2)


def render_frame(width, height, bbox, infrared=False, occlusion=0.0, noise=0.0, seed=0):
    """渲染单帧:深色背景 + 亮色人体框 + 可选遮挡/噪声。"""
    rng = np.random.RandomState(seed)
    bg = 18 if infrared else 28
    fg = 210 if infrared else 190
    arr = np.full((height, width, 3), bg, np.uint8)
    if noise > 0:
        jitter = rng.normal(0, noise, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    d.rounded_rectangle(bbox, radius=3, fill=(fg, fg, fg))
    head_r = max(2, min(x2 - x1, y2 - y1) // 7)
    d.ellipse((x1 + 2, y1 + 2, x1 + 2 + head_r * 2, y1 + 2 + head_r * 2),
              fill=(245, 245, 245))
    if occlusion > 0:
        occ_w = int((x2 - x1) * occlusion)
        if occ_w > 0:
            d.rectangle((x1, y1, x1 + occ_w, y2), fill=(bg, bg, bg))
    return np.asarray(img)


def _trajectory(kind, n_frames, width, height, rng):
    bboxes = []
    cx0 = width * rng.uniform(0.42, 0.58)
    ground = height * rng.uniform(0.72, 0.82)
    person_h = height * rng.uniform(0.36, 0.44)
    person_w = width * rng.uniform(0.10, 0.14)

    for t in range(n_frames):
        u = t / max(1, n_frames - 1)
        cx = cx0 + np.sin(u * np.pi * 2) * width * 0.04
        if kind == "fall":
            phase = min(1.0, max(0.0, (u - 0.35) / 0.30))
            cy = ground - person_h / 2 + phase * height * 0.20
            h = person_h * (1.0 - 0.58 * phase)
            w = person_w * (1.0 + 3.2 * phase)
        elif kind == "sit":
            phase = min(1.0, max(0.0, (u - 0.45) / 0.45))
            cy = ground - person_h / 2 + phase * height * 0.10
            h = person_h * (1.0 - 0.28 * phase)
            w = person_w * (1.0 + 0.50 * phase)
        elif kind == "bend":
            phase = np.sin(u * np.pi)
            cy = ground - person_h / 2 + phase * height * 0.04
            h = person_h * (1.0 - 0.20 * phase)
            w = person_w * (1.0 + 0.65 * phase)
        elif kind == "lie":
            phase = min(1.0, max(0.0, (u - 0.15) / 0.75))
            cy = ground - person_h / 2 + phase * height * 0.14
            h = person_h * (1.0 - 0.52 * phase)
            w = person_w * (1.0 + 2.4 * phase)
        else:  # walk
            cy = ground - person_h / 2 + np.sin(u * np.pi * 4) * height * 0.01
            h = person_h
            w = person_w
        bboxes.append(_bbox_from_center(cx, cy, w, h, width, height))
    return bboxes


def make_clip(kind, clip_id=None, n_frames=32, size=(160, 120), seed=0,
              infrared=False, occlusion=0.0, noise=0.0):
    """生成一个动作片段。label=1 仅对应 fall。"""
    width, height = size
    rng = np.random.RandomState(seed)
    bboxes = _trajectory(kind, n_frames, width, height, rng)
    frames = [render_frame(width, height, bb, infrared=infrared, occlusion=occlusion,
                           noise=noise, seed=(seed * 1000 + i) % (2**32 - 1))
              for i, bb in enumerate(bboxes)]
    event = (int(n_frames * 0.35), int(n_frames * 0.78)) if kind == "fall" else (-1, -1)
    return Clip(clip_id or f"{kind}_{seed}", frames, 1 if kind == "fall" else 0,
                kind, bboxes, event)


def make_dataset(n_per_kind=12, n_frames=32, size=(160, 120), seed=0):
    rng = np.random.RandomState(seed)
    clips = []
    kinds = ("fall", "walk", "sit", "bend", "lie")
    for kind in kinds:
        for i in range(n_per_kind):
            clips.append(make_clip(
                kind, clip_id=f"{kind}_{i:03d}", n_frames=n_frames, size=size,
                seed=int(rng.randint(0, 10_000_000)),
                infrared=(i % 5 == 0), occlusion=0.25 if i % 7 == 0 else 0.0,
                noise=5.0 if i % 6 == 0 else 0.0,
            ))
    rng.shuffle(clips)
    return clips


def _bbox_aspect(bb):
    x1, y1, x2, y2 = bb
    return (x2 - x1 + 1) / max(1, (y2 - y1 + 1))


def _center_y(bb):
    return (bb[1] + bb[3]) / 2.0


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    fall = make_clip("fall", seed=1)
    sit = make_clip("sit", seed=1)
    walk = make_clip("walk", seed=1)
    check(fall.label == 1 and sit.label == 0 and walk.label == 0, "只有 fall 标为正类")
    check(fall.frames[0].shape == (120, 160, 3), "帧尺寸为 H×W×3")
    check(_center_y(fall.bboxes[-1]) - _center_y(fall.bboxes[0]) > 15, "跌倒片段有明显下坠")
    check(_bbox_aspect(fall.bboxes[-1]) > 1.0, "跌倒末尾呈横向姿态")
    check(_center_y(sit.bboxes[-1]) - _center_y(sit.bboxes[0]) < 15, "坐下下坠幅度低于跌倒")
    ds = make_dataset(n_per_kind=3, seed=42)
    check(len(ds) == 15, "数据集覆盖 5 类动作")
    check(sum(c.label for c in ds) == 3, "合成数据正负类数量正确")
    ds2 = make_dataset(n_per_kind=3, seed=42)
    check(ds[0].clip_id == ds2[0].clip_id, "同 seed 数据顺序可复现")

    print("\n" + ("✅ fall_synth 自测通过" if ok else "❌ fall_synth 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;合成数据由 run_fall_pipeline.py 调用")


if __name__ == "__main__":
    main()
