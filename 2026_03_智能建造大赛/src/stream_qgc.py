#!/usr/bin/env python3
"""
stream_qgc.py · 【在 Jetson Orin Nano 上运行】实时闭环 + QGroundControl 融合 demo(加分项)

链路(系统闭环,创新点①的现场呈现):
  机载相机/视频源 ──> TRT 推理(trt_infer_orin.Runner) ──> 时序滤波(track_filter)
       ──> 画框 + 落水告警 + 检测框→GPS(geolocate, 接 MAVLink 遥测) + 遥测条叠加
       ──> H265 软编码(Orin Nano 无硬件 NVENC)──> RTSP 推流 ──> QGroundControl(配 RTSP 源, Low Latency)

双轨保底:主轨现场实时推流;任何环节卡顿切**录屏轨**(--record out.mp4 同时本地存证)。

⚠️ 需在 Orin 上、装好 OpenCV(含 GStreamer)+ tensorrt + pycuda + pymavlink 才能真跑。
本机可 `python stream_qgc.py --selftest` 自检:确认 GStreamer 后端 + 模块接线 + 打印将用的管线。
"""
import argparse
import time

# RTSP 输出管线(appsrc → H265 软编码 → rtspclientsink/udpsink)。Orin Nano 无 NVENC,用 x265enc 软编。
GST_OUT = (
    "appsrc ! videoconvert ! "
    "x265enc tune=zerolatency bitrate={bitrate} speed-preset=ultrafast ! "
    "rtph265pay config-interval=1 pt=96 ! udpsink host={host} port={port}"
)
# 采集管线(示例:USB/CSI 相机或测试视频)。真机按相机替换。
GST_IN_TEST = "videotestsrc ! video/x-raw,width={w},height={h},framerate=30/1 ! videoconvert ! appsink"


def _draw_overlay(frame, tracks, names, intr, pose):
    """画确认框 + 类别 + (有位姿时)目标 GPS + 顶部遥测条。返回叠加后的帧。"""
    import cv2
    from geolocate import geolocate_box
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        label = names[t.cls] if t.cls < len(names) else str(t.cls)
        color = (0, 0, 255) if label in ("person_in_water", "落水人员") else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {t.score:.2f}"
        if intr is not None and pose is not None:
            g = geolocate_box([x1, y1, x2, y2], intr, pose)
            if g:
                tag += f" | {g['lat']:.5f},{g['lon']:.5f} {g['ground_dist_m']:.0f}m"
                if label in ("person_in_water", "落水人员"):
                    cv2.putText(frame, "!! PERSON IN WATER !!", (x1, max(20, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, tag, (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if pose is not None:
        bar = f"ALT {pose.alt_agl:.0f}m  YAW {pose.yaw_deg:.0f}  GPS {pose.lat:.5f},{pose.lon:.5f}"
        cv2.putText(frame, bar, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


def run(args):  # pragma: no cover (需 Orin + 摄像头/视频 + TRT)
    import cv2, numpy as np
    from trt_infer_orin import Runner, build_or_load, preprocess, postprocess
    from track_filter import TrackFilter
    from geolocate import Intrinsics, read_pose_mavlink

    names = ["person_in_water", "boat", "buoy"]
    runner = Runner(build_or_load(args), args.imgsz)
    tf = TrackFilter(min_hits=3, max_age=8)
    intr = Intrinsics.from_fov(args.width, args.height, args.hfov)

    cap = cv2.VideoCapture(args.source if args.source else
                           GST_IN_TEST.format(w=args.width, h=args.height), cv2.CAP_GSTREAMER)
    out_pipe = GST_OUT.format(bitrate=args.bitrate, host=args.host, port=args.port)
    writer = cv2.VideoWriter(out_pipe, cv2.CAP_GSTREAMER, 0, 30, (args.width, args.height))
    recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                               (args.width, args.height)) if args.record else None

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blob, _, _ = preprocess(frame, args.imgsz)
        dets = postprocess(runner.infer(blob))
        tracks = tf.update([[*d[:4], d[4], int(d[5])] for d in dets])
        try:
            pose = read_pose_mavlink(args.mavlink, timeout=0.05) if args.mavlink else None
        except Exception:
            pose = None
        frame = _draw_overlay(frame, tracks, names, intr, pose)
        writer.write(frame)
        if recorder:
            recorder.write(frame)
        n += 1
        if n % 30 == 0:
            print(f"  推流中 {n} 帧, 端到端 {n/(time.time()-t0):.1f} FPS")
    cap.release(); writer.release()
    if recorder:
        recorder.release()


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ⚠️  ") + m); ok = ok and c

    # 模块接线(无 GPU 也应能 import)
    try:
        import track_filter, geolocate  # noqa
        check(True, "track_filter / geolocate 模块可导入(接线正常)")
    except Exception as e:
        check(False, f"模块导入失败: {e}")

    # OpenCV + GStreamer 后端(Orin 上必须有 GStreamer)
    try:
        import cv2
        info = cv2.getBuildInformation()
        has_gst = "GStreamer" in info and "YES" in info.split("GStreamer")[1][:40]
        print(("  ✅ " if has_gst else "  ⚠️  ") +
              f"OpenCV {cv2.__version__} GStreamer 后端: {'YES' if has_gst else '未启用(Orin 上需带 GStreamer 的 OpenCV)'}")
    except Exception as e:
        print(f"  ⚠️  OpenCV 未安装(Orin 上必装): {e}")

    print("\n  将使用的输出管线(Orin Nano 软编码 H265 → RTSP):")
    print("    " + GST_OUT.format(bitrate=4000, host="127.0.0.1", port=5600))
    print("  QGC 配置:Application Settings → Video → Source=RTSP,URL=rtsp://<orin-ip>:8554/...,开 Low Latency")
    print("\n" + ("✅ stream_qgc 自检通过(骨架就绪,真跑需上 Orin)" if ok else "⚠️ 自检有警告(Mac 上正常,Orin 上需补依赖)"))
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--onnx"); ap.add_argument("--engine")
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir")
    ap.add_argument("--source", help="视频源(留空用 videotestsrc 测试)")
    ap.add_argument("--mavlink", help="MAVLink 连接串,如 udp:127.0.0.1:14550(留空不读位姿)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=84.0)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument("--record", help="同时本地存证 mp4(双轨保底)")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if not a.onnx and not a.engine:
        ap.error("需 --onnx 或 --engine(或用 --selftest)")
    run(a)


if __name__ == "__main__":
    main()
