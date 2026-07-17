#!/usr/bin/env python3
"""
fall_video_io.py · 跌倒检测公开视频/帧目录读取

真实数据接入分两层:
  - 稳定路径:先把视频抽帧到目录,本模块读取 png/jpg 序列。
  - 可选路径:本机装有 imageio 时直接读取 mp4/avi 等视频文件。

输出统一为 list[np.ndarray(H,W,3)],供 fall_detector.detect_clip 直接使用。
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _find_ffmpeg():
    return shutil.which("ffmpeg") or (
        "/opt/homebrew/bin/ffmpeg"
        if Path("/opt/homebrew/bin/ffmpeg").exists()
        else None
    )


def sample_indices(n_items, n_samples):
    """从 n_items 中均匀采样 n_samples 个索引;不足时重复边界帧。"""
    if n_items <= 0:
        raise ValueError("cannot sample from empty sequence")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_items == 1:
        return [0] * n_samples
    return [int(round(x)) for x in np.linspace(0, n_items - 1, n_samples)]


def _resize_rgb(arr, size):
    img = Image.fromarray(np.asarray(arr).astype(np.uint8))
    img = img.convert("RGB")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size))
    return np.asarray(img)


def load_frame_dir(path, n_frames=32, size=(160, 120)):
    """读取帧目录,按文件名排序并均匀采样。"""
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise ValueError(f"no image frames found in {root}")
    frames = []
    for idx in sample_indices(len(files), n_frames):
        with Image.open(files[idx]) as im:
            frames.append(_resize_rgb(np.asarray(im.convert("RGB")), size))
    return frames


def _load_video_file_ffmpeg(path, n_frames=32, size=(160, 120)):
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; extract frames first")

    with tempfile.TemporaryDirectory() as td:
        out_pattern = str(Path(td) / "%06d.png")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path)]
        if size is not None:
            width, height = tuple(size)
            cmd.extend(["-vf", f"scale={int(width)}:{int(height)}:flags=bicubic"])
        cmd.extend(["-vsync", "0", out_pattern])
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"failed to decode video via ffmpeg: {path}; {detail}") from exc
        return load_frame_dir(td, n_frames=n_frames, size=size)


def load_video_file(path, n_frames=32, size=(160, 120)):
    """优先用 imageio 读取视频;失败时用 ffmpeg 抽帧兜底。"""
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(p)
    try:
        import imageio.v3 as iio
    except Exception as exc:
        return _load_video_file_ffmpeg(p, n_frames=n_frames, size=size)

    try:
        arr = iio.imread(p)
    except Exception as exc:
        return _load_video_file_ffmpeg(p, n_frames=n_frames, size=size)
    if arr.ndim < 4 or arr.shape[0] == 0:
        return _load_video_file_ffmpeg(p, n_frames=n_frames, size=size)
    return [_resize_rgb(arr[idx, ..., :3], size) for idx in sample_indices(arr.shape[0], n_frames)]


def load_clip(path, n_frames=32, size=(160, 120)):
    """读取帧目录或视频文件。"""
    p = Path(path).expanduser()
    if p.is_dir():
        return load_frame_dir(p, n_frames=n_frames, size=size)
    if p.suffix.lower() in VIDEO_EXTS:
        return load_video_file(p, n_frames=n_frames, size=size)
    raise ValueError(f"unsupported clip path: {path}")


def _write_frames(frames, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"{i:04d}.png")


def _selftest():
    import sys
    from fall_synth import make_clip

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    check(sample_indices(4, 4) == [0, 1, 2, 3], "等长采样保持原索引")
    check(sample_indices(2, 5) == [0, 0, 0, 1, 1], "短序列采样可重复边界")

    with tempfile.TemporaryDirectory() as td:
        clip = make_clip("fall", n_frames=9, size=(96, 72), seed=9)
        frame_dir = Path(td) / "fall_frames"
        _write_frames(clip.frames, frame_dir)
        frames = load_frame_dir(frame_dir, n_frames=5, size=(64, 48))
        check(len(frames) == 5, "帧目录按目标帧数采样")
        check(frames[0].shape == (48, 64, 3), "帧目录读取后缩放为 H×W×3")
        frames2 = load_clip(frame_dir, n_frames=5, size=(64, 48))
        check(np.array_equal(frames[0], frames2[0]), "load_clip 自动识别帧目录")
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            video = Path(td) / "toy.mp4"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-framerate", "5", "-i", str(frame_dir / "%04d.png"),
                "-pix_fmt", "yuv420p", str(video),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frames3 = load_video_file(video, n_frames=4, size=(64, 48))
            check(len(frames3) == 4, "视频文件可通过 imageio/ffmpeg 读取")
            check(frames3[0].shape == (48, 64, 3), "视频读取后缩放为 H×W×3")
        else:
            print("  ⚠️ 未找到 ffmpeg,跳过视频文件读取自测")

    print("\n" + ("✅ fall_video_io 自测通过" if ok else "❌ fall_video_io 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    ap.error("当前仅支持 --selftest;读取函数由 run_fall_pipeline.py 调用")


if __name__ == "__main__":
    main()
