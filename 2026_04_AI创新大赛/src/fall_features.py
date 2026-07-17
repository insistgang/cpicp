#!/usr/bin/env python3
"""
fall_features.py · 纯视觉跌倒检测轻量特征

当前 baseline 使用简单前景阈值提人体框,计算中心点、宽高比、面积和速度等时序特征。
后续真模型阶段可把 `foreground_bbox` 替换为 YOLO/姿态估计输出,下游检测器接口不变。
"""
import argparse

import numpy as np


FEATURE_NAMES = (
    "cx", "cy", "w", "h", "aspect", "area", "dcy", "daspect", "darea",
)


def foreground_bbox(frame, threshold=80, min_area=20):
    """从单帧中提取亮前景人体框。返回 (x1,y1,x2,y2),无目标则 None。"""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=2)
    else:
        gray = arr
    mask = gray > threshold
    ys, xs = np.where(mask)
    if len(xs) < min_area:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_feature(bbox, frame_shape):
    """人体框 -> 归一化几何特征。"""
    h_img, w_img = frame_shape[:2]
    if bbox is None:
        return np.zeros(6, np.float32)
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1 + 1))
    h = max(1.0, float(y2 - y1 + 1))
    cx = (x1 + x2) / 2.0 / max(1.0, w_img)
    cy = (y1 + y2) / 2.0 / max(1.0, h_img)
    nw = w / max(1.0, w_img)
    nh = h / max(1.0, h_img)
    aspect = w / h
    area = (w * h) / max(1.0, w_img * h_img)
    return np.array([cx, cy, nw, nh, aspect, area], np.float32)


def extract_clip_features(frames):
    """帧序列 -> dict(boxes, features, names)。features shape=(T,9)。"""
    boxes = [foreground_bbox(f) for f in frames]
    base = np.vstack([bbox_feature(bb, frames[i].shape) for i, bb in enumerate(boxes)])
    delta = np.zeros((len(frames), 3), np.float32)
    if len(frames) > 1:
        delta[1:, 0] = base[1:, 1] - base[:-1, 1]      # dcy
        delta[1:, 1] = base[1:, 4] - base[:-1, 4]      # daspect
        delta[1:, 2] = base[1:, 5] - base[:-1, 5]      # darea
    return {"boxes": boxes, "features": np.hstack([base, delta]), "names": FEATURE_NAMES}


def summarize_features(features):
    """把帧级特征压成检测器需要的轻量统计量。"""
    f = np.asarray(features, np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return {
        "max_aspect": float(f[:, idx["aspect"]].max()),
        "final_aspect": float(f[-1, idx["aspect"]]),
        "max_down_velocity": float(f[:, idx["dcy"]].max()),
        "total_down_shift": float(f[-1, idx["cy"]] - f[0, idx["cy"]]),
        "min_height": float(f[:, idx["h"]].min()),
        "final_height": float(f[-1, idx["h"]]),
        "max_area": float(f[:, idx["area"]].max()),
        "visible_ratio": float((f[:, idx["area"]] > 0).mean()),
    }


def _selftest():
    import sys
    from fall_synth import make_clip

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    fall = make_clip("fall", seed=3)
    walk = make_clip("walk", seed=3)
    ff = extract_clip_features(fall.frames)
    wf = extract_clip_features(walk.frames)
    fs = summarize_features(ff["features"])
    ws = summarize_features(wf["features"])
    check(ff["features"].shape == (32, len(FEATURE_NAMES)), "跌倒片段特征形状正确")
    check(all(bb is not None for bb in ff["boxes"]), "每帧都能提取人体框")
    check(fs["total_down_shift"] > ws["total_down_shift"] + 0.10, "跌倒总下坠显著大于行走")
    check(fs["max_aspect"] > 1.0 and ws["max_aspect"] < 1.0, "跌倒横向姿态显著")
    check(fs["visible_ratio"] == 1.0, "可见率统计正确")

    print("\n" + ("✅ fall_features 自测通过" if ok else "❌ fall_features 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;特征由 run_fall_pipeline.py 调用")


if __name__ == "__main__":
    main()
