#!/usr/bin/env python3
"""
trt_infer_orin.py · 【在 Jetson Orin Nano 上运行】TensorRT 引擎端到端测速 + 30FPS 红线判定

为什么重写:赛题硬指标是"动态视频处理 ≥30FPS",且必须是**真实端到端**帧率。
只测裸 GPU 推理会虚高,现场含 解码+NMS+画框+编码 的整链一定更慢 → 必须分档计时:
  ① 裸推理       = H2D + GPU推理 + D2H + sync
  ② 含后处理     = ① + Detect解码 + NMS + 画框
  ③ 含编码(实战) = ② + 帧编码(JPEG代理;真实 RTSP/H265 见 stream_qgc.py,更重)
脚本对每一档与 30FPS 比较并 PASS/FAIL 告警 —— 报告里要报 ③，否则路演翻车。

⚠️ Orin Nano 这一 SKU 无硬件 NVENC：实战 RTSP 走软编码(x264/x265)吃 CPU，③ 会进一步降，
   报告请用 stream_qgc.py 实测含编码端到端，并备录屏保底轨。

用法(Orin Nano, JetPack 自带 TensorRT；另需 pip install pycuda opencv-python):
  # 最稳:先用 trtexec 建引擎(FP16 或 INT8 混合精度+真实海况帧校准)
  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_int8.engine --int8 --fp16
  python trt_infer_orin.py --engine best_int8.engine --imgsz 640 --source test.mp4 --benchmark
  # 或脚本内从 ONNX 建 FP16 引擎并测速:
  python trt_infer_orin.py --onnx best.onnx --fp16 --imgsz 640 --benchmark
  # 脚本内建 INT8(需校准图目录,取自命题方二开数据的反光/波纹/运动模糊帧):
  python trt_infer_orin.py --onnx best.onnx --int8 --calib-dir ./calib_frames --imgsz 640 --benchmark

兼容 TensorRT 8.x(num_bindings 旧 API) 与 TensorRT 10.x(num_io_tensors 新 API，JetPack 6.x 自带)。
"""
import argparse
import os
import time

FPS_GATE = 30.0  # 赛题硬卡线


# ----------------------------- 引擎构建/加载 -----------------------------
def _make_int8_calibrator(trt, calib_dir, imgsz, cache="int8_calib.cache"):
    """最简 EntropyCalibrator2:从 calib_dir 读图做 letterbox 预处理喂校准。"""
    import numpy as np, cv2, pycuda.driver as cuda

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.files = [os.path.join(calib_dir, f) for f in os.listdir(calib_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if not self.files:
                raise RuntimeError(f"校准目录无图片: {calib_dir}")
            self.idx = 0
            self.dev = cuda.mem_alloc(1 * 3 * imgsz * imgsz * 4)
            self.cache = cache
            print(f"[INT8] 校准集 {len(self.files)} 张 <- {calib_dir}")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            img = cv2.imread(self.files[self.idx]); self.idx += 1
            if img is None:
                return self.get_batch(names)
            blob, _, _ = preprocess(img, imgsz)
            blob = np.ascontiguousarray(blob)
            cuda.memcpy_htod(self.dev, blob)
            return [int(self.dev)]

        def read_calibration_cache(self):
            return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

        def write_calibration_cache(self, c):
            open(self.cache, "wb").write(c)

    return Calib()


def build_or_load(args):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if args.engine:
        with open(args.engine, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())
    # 从 ONNX 构建
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")
    config = builder.create_builder_config()
    # TRT10 用 set_memory_pool_limit;TRT8 同名 API 也在
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if args.fp16 or args.int8:
        config.set_flag(trt.BuilderFlag.FP16)        # INT8 也开 FP16 做混合精度,敏感层回退
    if args.int8:
        if not args.calib_dir:
            raise RuntimeError("--int8 需 --calib-dir 校准图目录(真实海况帧),或改用 trtexec --int8 建好 engine 再 --engine 加载")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_int8_calibrator(trt, args.calib_dir, args.imgsz)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("引擎构建失败")
    return runtime.deserialize_cuda_engine(serialized)


# ----------------------------- TRT8/TRT10 统一执行封装 -----------------------------
class Runner:
    """屏蔽 TensorRT 8.x(binding) 与 10.x(io_tensor) 的 API 差异。"""

    def __init__(self, engine, imgsz):
        import numpy as np, pycuda.driver as cuda
        import tensorrt as trt
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.cuda = cuda
        self.stream = cuda.Stream()
        self.new_api = hasattr(engine, "num_io_tensors")  # TRT10
        self.inputs, self.outputs = [], []
        in_shape = (1, 3, imgsz, imgsz)

        if self.new_api:
            self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            for name in self.names:
                is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_in:
                    self.ctx.set_input_shape(name, in_shape)
            for name in self.names:
                is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self.ctx.get_tensor_shape(name))
                host = np.empty(shape, dtype=np.float32)
                dev = cuda.mem_alloc(host.nbytes)
                self.ctx.set_tensor_address(name, int(dev))
                (self.inputs if is_in else self.outputs).append([name, host, dev, shape])
        else:  # TRT8
            self.bindings = []
            for i in range(engine.num_bindings):
                is_in = engine.binding_is_input(i)
                if is_in and hasattr(self.ctx, "set_binding_shape"):
                    self.ctx.set_binding_shape(i, in_shape)
                shape = in_shape if is_in else tuple(self.ctx.get_binding_shape(i))
                host = np.empty(shape, dtype=np.float32)
                dev = cuda.mem_alloc(host.nbytes)
                self.bindings.append(int(dev))
                (self.inputs if is_in else self.outputs).append([i, host, dev, shape])

    def infer(self, blob):
        """blob: (1,3,H,W) float32 -> 返回 outputs 的 host numpy 列表。含 H2D/exec/D2H/sync。"""
        cuda = self.cuda
        name, host, dev, shape = self.inputs[0]
        host[...] = blob.reshape(shape)
        cuda.memcpy_htod_async(dev, host, self.stream)
        if self.new_api:
            self.ctx.execute_async_v3(self.stream.handle)
        else:
            self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        for _, h, d, _ in self.outputs:
            cuda.memcpy_dtoh_async(h, d, self.stream)
        self.stream.synchronize()
        return [h for _, h, _, _ in self.outputs]


# ----------------------------- 前/后处理 -----------------------------
def preprocess(img, imgsz):
    """letterbox -> RGB CHW [0,1]。返回 blob(1,3,H,W), ratio, (dw,dh)。"""
    import numpy as np, cv2
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype("float32") / 255.0
    return np.ascontiguousarray(blob[None]), r, (dw, dh)


def postprocess(out, conf=0.25, iou=0.5):
    """解码 ultralytics 无NMS导出 (1, 4+nc, N) -> numpy NMS。返回 [x1,y1,x2,y2,score,cls]。

    布局判定:ultralytics 无NMS导出为 (1, 4+nc, N),N(锚点数,通常数千)远大于 4+nc(列)。
    我们用"行<列则转置"把 (4+nc, N) 摆成 (N, 4+nc)。真实 YOLO 输出永远满足 N>>4+nc,
    该启发式稳健;只有人造极小张量(N<=4+nc)才会判反 —— 此时下面的护栏保证不崩(返回空)而非抛异常。
    """
    import numpy as np
    o = out[0]
    a = np.array(o)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        return np.zeros((0, 6), dtype="float32")
    if a.shape[0] < a.shape[1]:   # (4+nc, N) -> (N, 4+nc)
        a = a.T
    if a.shape[1] < 5 or a.shape[0] == 0:  # 列数 < 4(框)+1(至少1类) 或 无锚点 -> 无法解码,返回空
        return np.zeros((0, 6), dtype="float32")
    boxes_xywh, scores_all = a[:, :4], a[:, 4:]
    cls = scores_all.argmax(1)
    score = scores_all.max(1)
    keep = score > conf
    boxes_xywh, score, cls = boxes_xywh[keep], score[keep], cls[keep]
    if len(boxes_xywh) == 0:
        return np.zeros((0, 6), dtype="float32")
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    keep_idx = _nms(xyxy, score, iou)
    return np.concatenate([xyxy[keep_idx], score[keep_idx, None], cls[keep_idx, None]], 1).astype("float32")


def _nms(boxes, scores, iou_thr):
    import numpy as np
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou_thr]
    return np.array(keep, dtype=int)


# ----------------------------- 端到端三档测速 -----------------------------
def benchmark(runner, imgsz, source=None, warmup=20, iters=200):
    import numpy as np, cv2
    cuda = runner.cuda
    # 取一帧真实图(或合成),代表实战分辨率
    frame = None
    if source and os.path.exists(source):
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read(); cap.release()
        if not ok:
            frame = None
    if frame is None:
        frame = (np.random.rand(1080, 1920, 3) * 255).astype("uint8")
        print("[warn] 未提供 --source 真实视频,用合成帧测速(后处理框数偏随机,仅供粗估)")

    blob, _, _ = preprocess(frame, imgsz)
    for _ in range(warmup):
        runner.infer(blob)

    t_inf = t_post = t_enc = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        out = runner.infer(blob)                       # ① 裸推理
        t1 = time.perf_counter()
        dets = postprocess(out)                         # ② + 解码+NMS
        vis = frame.copy()
        for x1, y1, x2, y2, s, c in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        t2 = time.perf_counter()
        cv2.imencode(".jpg", vis)                       # ③ + 编码(JPEG 代理;实战H265更重)
        t3 = time.perf_counter()
        t_inf += t1 - t0; t_post += t2 - t0; t_enc += t3 - t0

    def fps(total):
        ms = total / iters * 1000
        return ms, 1000 / ms

    rows = [("① 裸推理", *fps(t_inf)),
            ("② 含后处理(解码+NMS+画框)", *fps(t_post)),
            ("③ 含编码(实战代理)", *fps(t_enc))]
    free, total = cuda.mem_get_info()
    print(f"\n=== Orin Nano 端到端测速 (imgsz={imgsz}, iters={iters}) ===")
    print(f"{'档位':<28}{'时延ms':>10}{'FPS':>9}{'  vs 30FPS红线'}")
    for name, ms, f in rows:
        verdict = "[PASS]" if f >= FPS_GATE else "[FAIL](出局风险)"
        print(f"{name:<28}{ms:>10.2f}{f:>9.1f}   {verdict}")
    print(f"\n  显存: {(total-free)/1024/1024:.0f} / {total/1024/1024:.0f} MB")
    print("  [WARN] 报告请以『③ 含编码』为准;实战 RTSP/H265 软编码(Orin Nano 无NVENC)比JPEG更重,")
    print("     最终含编码端到端 FPS 请用 stream_qgc.py 实测,并旁路 tegrastats 记功耗。")
    if rows[2][2] < FPS_GATE:
        print(f"\n  [REDLINE] 含编码 {rows[2][2]:.1f}FPS < {FPS_GATE} → 启动降级链:768→640 / 关P2 / 砍P5 / INT8 / Super Mode / 抽帧")


# ----------------------------- 解码逻辑自测(纯 numpy,无需 Orin/cv2/TRT) -----------------------------
def _selftest():
    """焊死 postprocess(布局判定/xywh→xyxy/conf过滤/NMS去重)与 _nms 的纯 numpy 解码逻辑。
    这条解码若现场跑错会让所有框错位,故本地先验证;preprocess 需 cv2,在 Orin 上另行 --benchmark 验。"""
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  OK  " if c else "  XX  ") + m)
        ok = ok and c

    nc, N = 3, 8400
    # 构造一帧真实形状的无NMS输出 (1, 4+nc, N),全零(score 全 0),仅 anchor 0 放一个强框:
    #   中心(100,100) 宽高(20,20),类别 1,score 0.9 → 期望解码出 xyxy=[90,90,110,110], cls=1
    raw = np.zeros((1, 4 + nc, N), dtype="float32")
    raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = 100, 100, 20, 20
    raw[0, 4 + 1, 0] = 0.9

    dets = postprocess([raw])
    check(dets.shape == (1, 6), f"标准布局 (1,{4+nc},{N}) 解出 1 个框({dets.shape})")
    if len(dets):
        x1, y1, x2, y2, s, c = dets[0]
        check(abs(x1-90) < 1e-3 and abs(y1-90) < 1e-3 and abs(x2-110) < 1e-3 and abs(y2-110) < 1e-3,
              f"xywh(100,100,20,20)→xyxy(90,90,110,110) 正确({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        check(abs(s-0.9) < 1e-6 and int(c) == 1, f"score=0.9 cls=1 正确(得 {s:.2f}/{int(c)})")

    # 已转置布局 (1, N, 4+nc) 应解出同一框(转置分支等价)
    dets_t = postprocess([raw[0].T[None]])
    check(dets_t.shape == (1, 6) and abs(dets_t[0][0]-90) < 1e-3,
          "已转置布局 (1,N,4+nc) 解码等价(转置分支正确)")

    # conf 过滤:阈值升到 0.95 → 唯一框(0.9)被滤掉 → 空
    check(postprocess([raw], conf=0.95).shape == (0, 6), "conf=0.95 滤掉 0.9 框 → 空结果")

    # 全低于 conf(全零 score)→ 空,不崩
    check(postprocess([np.zeros((1, 4+nc, N), "float32")]).shape == (0, 6), "全 0 score → 空结果(不崩)")

    # NMS 去重:两个高度重叠框(IoU>thr)只留高分那个
    a = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype="float32")
    sc = np.array([0.9, 0.8], dtype="float32")
    keep = _nms(a, sc, 0.5)
    check(len(keep) == 1 and keep[0] == 0, f"NMS:重叠框留高分(keep={keep.tolist()})")
    # 不重叠两框都保留
    a2 = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype="float32")
    check(len(_nms(a2, np.array([0.9, 0.8], "float32"), 0.5)) == 2, "NMS:不重叠框全保留")
    # 空输入不崩
    check(_nms(np.empty((0, 4)), np.empty((0,)), 0.5).shape == (0,), "NMS:空输入返回空(不崩)")

    # 病态极小张量(列<5,无法解码)→ 返回空而非抛异常(护栏)
    check(postprocess([np.zeros((1, 4+nc, 2), "float32")]).shape == (0, 6),
          "病态极小张量(锚点<列数)→ 返回空(护栏,不抛 argmax 异常)")

    print("\n" + ("OK trt_infer_orin 解码自测通过" if ok else "XX 自测未通过"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="跑解码逻辑自测(纯numpy,无需Orin)")
    ap.add_argument("--onnx")
    ap.add_argument("--engine")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true", help="脚本内建INT8(需 --calib-dir);或用 trtexec 建好 --engine")
    ap.add_argument("--calib-dir", help="INT8 校准图目录(命题方二开数据的反光/波纹/运动模糊帧 300-500张)")
    ap.add_argument("--source", help="测速用真实视频/图片(强烈建议,合成帧后处理不真实)")
    ap.add_argument("--imgsz", type=int, default=640, help="端侧默认640;小目标紧张升768")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)
    if not a.onnx and not a.engine:
        ap.error("需 --onnx 或 --engine 之一")
    try:
        engine = build_or_load(a)
        runner = Runner(engine, a.imgsz)
    except ImportError as e:
        print(f"[需在 Orin Nano 上运行] 缺 tensorrt/pycuda: {e}")
        return
    if a.benchmark:
        benchmark(runner, a.imgsz, a.source)


if __name__ == "__main__":
    main()
