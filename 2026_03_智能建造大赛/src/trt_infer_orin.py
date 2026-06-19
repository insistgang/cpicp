#!/usr/bin/env python3
"""
trt_infer_orin.py · 【在 Jetson Orin Nano 上运行】ONNX→TensorRT FP16 引擎 + FPS/显存测速骨架

用法(在 Orin Nano 上，JetPack 自带 TensorRT):
  # 1) 用 trtexec 构建 FP16 引擎(最稳)：
  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16
  # 2) 测速 + 显存：
  python trt_infer_orin.py --engine best_fp16.engine --imgsz 1024 --benchmark
  # 或直接从 onnx 构建并测速：
  python trt_infer_orin.py --onnx best.onnx --fp16 --imgsz 1024 --benchmark

输出: 平均推理时延(ms)、FPS、峰值显存(MB)。
说明:
  - 这是测速/部署骨架，检测后处理(解码+NMS)留 TODO，按你的 Detect 头输出张量布局补齐。
  - 功耗用 jetson-stats(jtop)/tegrastats 旁路记录(见 README)，本脚本聚焦时延/显存/FPS。
"""
import argparse
import time

EXPLAIN = """
[依赖] 仅 Orin Nano：JetPack 自带 tensorrt；另需 `pip install pycuda`。
[功耗] 另开终端跑: tegrastats   或   sudo jtop   记录 GPU 功率(mW)，与本测速同步取均值。
"""


def build_or_load(args):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    if args.engine:
        with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
            return rt.deserialize_cuda_engine(f.read())
    # 从 ONNX 构建
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)   # 小目标默认 FP16；INT8 需校准集，见 TODO
    engine = builder.build_serialized_network(network, config)
    with trt.Runtime(logger) as rt:
        return rt.deserialize_cuda_engine(engine)


def benchmark(engine, imgsz, warmup=20, iters=200):
    import numpy as np, pycuda.driver as cuda, pycuda.autoinit  # noqa
    import tensorrt as trt
    ctx = engine.create_execution_context()
    # 简化：单输入单/多输出，按引擎绑定分配
    inp_idx = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
    inp_shape = (1, 3, imgsz, imgsz)
    ctx.set_binding_shape(inp_idx, inp_shape) if hasattr(ctx, "set_binding_shape") else None
    bufs, d_bufs = [], []
    for i in range(engine.num_bindings):
        shape = inp_shape if engine.binding_is_input(i) else tuple(ctx.get_binding_shape(i))
        size = int(np.prod(shape))
        host = np.random.rand(size).astype(np.float32)
        dev = cuda.mem_alloc(host.nbytes)
        bufs.append((host, dev, shape))
        d_bufs.append(int(dev))
    h_in, d_in, _ = bufs[inp_idx]
    cuda.memcpy_htod(d_in, h_in)

    for _ in range(warmup):
        ctx.execute_v2(d_bufs)
    t0 = time.time()
    for _ in range(iters):
        ctx.execute_v2(d_bufs)
    cuda.Context.synchronize()
    dt = (time.time() - t0) / iters * 1000
    free, total = cuda.mem_get_info()
    print(f"\n=== Orin Nano 测速 (imgsz={imgsz}, FP16) ===")
    print(f"  平均时延: {dt:.2f} ms   FPS: {1000/dt:.1f}")
    print(f"  显存占用: {(total-free)/1024/1024:.0f} MB / {total/1024/1024:.0f} MB")
    print("  TODO: 后处理(Detect 解码+NMS)未计入；功耗用 tegrastats 旁路记录")
    # TODO[登录核对]: 若赛题硬卡 FPS/功耗阈值，在此与达标线比较并告警。


def main():
    print(EXPLAIN)
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--engine")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    if not a.onnx and not a.engine:
        ap.error("需 --onnx 或 --engine 之一")
    try:
        eng = build_or_load(a)
    except ImportError as e:
        print(f"[需在 Orin Nano 上运行] 缺 tensorrt/pycuda: {e}")
        return
    if a.benchmark:
        benchmark(eng, a.imgsz)


if __name__ == "__main__":
    main()
