#!/usr/bin/env python3
"""
illegal_build_pipeline.py · 开放赛题 AI+X(违建)线 demo 骨架

低成本第二投:复用 03 智能建造已自测通过的模块串成
  检测(占位/真权重接 ultralytics) → 时序滤波(track_filter) → 像素→GPS(geolocate) → 治理决策(派巡查航点)。
本机用合成检测框跑通逻辑自测;真数据/权重到位接 03 的 train/eval 即可。
`python illegal_build_pipeline.py` 自测(复用 03 src,纯 Python)。
"""
import os
import sys

# 复用 03 智能建造 src 里已自测的 track_filter / geolocate(均无第三方依赖)
_03_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "2026_03_智能建造大赛", "src")
sys.path.insert(0, os.path.abspath(_03_SRC))

from track_filter import TrackFilter           # noqa: E402
from geolocate import Intrinsics, Pose, geolocate_box  # noqa: E402


def process_frames(detections_per_frame, intr, pose, min_hits=3):
    """detections_per_frame: 每帧 [[x1,y1,x2,y2,score,cls],...]。
    返回确认违建目标的 GPS 航点列表(去抖后)。"""
    tf = TrackFilter(min_hits=min_hits, max_age=5, iou_thr=0.3)
    waypoints = []
    for dets in detections_per_frame:
        confirmed = tf.update(dets)
        for t in confirmed:
            g = geolocate_box(t.box, intr, pose)
            if g:
                waypoints.append({"track": t.tid, "lat": round(g["lat"], 6),
                                  "lon": round(g["lon"], 6), "dist_m": round(g["ground_dist_m"], 1)})
    return waypoints


def _selftest():
    import sys as _s
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    intr = Intrinsics.from_fov(1920, 1080, 84.0)
    pose = Pose(lat=31.23, lon=121.47, alt_agl=120, yaw_deg=0, gimbal_tilt_deg=15)

    # 一个持续出现的违建框(5帧)+ 一帧反光闪点
    box = [900, 500, 980, 580, 0.9, 0]
    frames = [[box], [[901, 501, 981, 581, 0.9, 0]], [[902, 502, 982, 582, 0.9, 0]],
              [[903, 503, 983, 583, 0.9, 0]], [[1500, 200, 1520, 220, 0.7, 0]]]  # 末帧是闪点
    wps = process_frames(frames, intr, pose, min_hits=3)

    check(len(wps) >= 1, f"持续违建目标产出 GPS 航点({len(wps)}个)")
    check(all("lat" in w and "lon" in w for w in wps), "每个航点含经纬度(可派执法巡查)")
    check(all(w["track"] == wps[0]["track"] for w in wps), "闪点未生成新航点(时序滤波生效)")
    print(f"  示例航点:{wps[0]}")

    print("\n" + ("✅ illegal_build_pipeline 自测通过(复用 03 track_filter+geolocate)" if ok else "❌ 自测未通过"))
    _s.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
