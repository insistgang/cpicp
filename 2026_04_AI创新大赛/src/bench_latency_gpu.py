#!/usr/bin/env python3
"""
bench_latency_gpu.py · 端到端单图延时实测(华为赛题一硬指标:<200ms@2060 / CPU<2s)

与 latency_bench.py 的区别:
  - latency_bench.py 测的是"dummy 特征(网格通道均值)+ 最近邻"的下限流水线,验证计时框架;
  - **本脚本测真实流水线**——用 feature_backend 的真后端(GPU 上 TimmBackend / 本机 ClassicBackend)
    在 **2500×2500 输入** 上做 预处理 → 特征提取 → memory bank 最近邻打分 → 阈值判定,
    分档计时并与红线比较给出 PASS/FAIL。检测时间占竞赛分 30%,此基准直接服务该项。

两档自动切换(get_backend(prefer_real=True)):
  - **GPU 档**:GPU 机装好 torch/timm(见 setup_gpu.sh)→ 自动走 TimmBackend(真特征),
    红线 <200ms@2060。__在 2060 级显卡上跑本脚本即出 GPU 档__。
  - **CPU 档**:本机无 torch → 回退 ClassicBackend,红线 <2s。本机即可实测出 CPU 档。

用法:
  python3 bench_latency_gpu.py                # 默认 2500×2500,自动选档,n=10 次
  python3 bench_latency_gpu.py --size 2500 --runs 20 --bank 200
  python3 bench_latency_gpu.py --selftest     # 小尺寸快速自检(供 run_all_selftests.sh)

GPU 档预期(2060,wide_resnet50_2,grid=8,bank≈200):单图总延时数十~一百多 ms,< 200ms 红线。
CPU 档实测(本机,ClassicBackend,见 README):2500px 单图约几十~几百 ms 量级,< 2s 红线。
"""
import argparse
import sys
import time

import numpy as np

from feature_backend import get_backend
from patchcore_lite import build_memory_bank
from anomaly_score import nn_distance

GPU_BUDGET_MS = 200.0   # 官方:2060 及以下 GPU 单图 < 200ms
CPU_BUDGET_MS = 2000.0  # CPU 可挑战 < 2s


def _make_bank(backend, size, bank_imgs, seed):
    """用若干张"正常件"图建 memory bank(模拟 few-shot 启动后的库)。"""
    rng = np.random.RandomState(seed)
    pool = []
    for _ in range(bank_imgs):
        # 轻纹理正常件(真实数据替换后此处换成读图)
        img = (rng.rand(size, size, 3) * 18 + 150).astype(np.uint8)
        pool.append(backend.image_patches(img))
    feats = np.vstack(pool)
    # coreset 压到小 bank,贴合 <200ms 的最近邻预算
    return build_memory_bank(feats, coreset_ratio=0.3, seed=seed)


def run_once(backend, img, bank, thr=1.0):
    """单张图端到端分档计时(ms):特征提取 + 最近邻打分 + 阈值判定。
    预处理(降采样/归一化)已包含在 backend.image_patches 内部,这里单列特征档。"""
    stages = {}
    t = time.perf_counter()
    patches = backend.image_patches(img)                      # 预处理+特征提取
    stages["feature"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    score = float(nn_distance(patches, bank).max())           # image-level 异常打分
    stages["score"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    _ = score >= thr                                          # 阈值判定
    stages["threshold"] = (time.perf_counter() - t) * 1000

    stages["total"] = sum(stages.values())
    return stages


def bench(size=2500, runs=10, bank_imgs=8, seed=0, warmup=2):
    """返回 (backend, is_real, avg_stages, p95_total)。"""
    backend, is_real = get_backend(prefer_real=True, grid=8)
    bank = _make_bank(backend, size, bank_imgs, seed)
    rng = np.random.RandomState(seed + 1)

    # 预热(GPU 首次推理含 CUDA 上下文/cudnn autotune,不计入)
    for _ in range(max(0, warmup)):
        img = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        run_once(backend, img, bank)

    runs_stages = []
    for _ in range(runs):
        img = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        runs_stages.append(run_once(backend, img, bank))

    keys = runs_stages[0].keys()
    avg = {k: sum(r[k] for r in runs_stages) / runs for k in keys}
    p95_total = float(np.percentile([r["total"] for r in runs_stages], 95))
    return backend, is_real, avg, p95_total


def _report(backend, is_real, avg, p95_total, size):
    budget = GPU_BUDGET_MS if is_real else CPU_BUDGET_MS
    arch = "GPU 真特征(TimmBackend)" if is_real else "CPU 经典特征(ClassicBackend, baseline)"
    print(f"  后端           : {backend.name}  [{arch}]")
    print(f"  输入           : {size}×{size}")
    print("  分档平均耗时(ms):" + ", ".join(f"{k}={avg[k]:.2f}" for k in avg))
    print(f"  单图总延时     : mean={avg['total']:.1f}ms  p95={p95_total:.1f}ms")
    passed = avg["total"] < budget
    line = "PASS" if passed else "FAIL"
    tag = f"<{budget:.0f}ms" + ("@2060 GPU" if is_real else " CPU")
    print(f"  红线 {tag:14s}: {'✅ ' + line if passed else '❌ ' + line} (mean={avg['total']:.1f}ms vs 预算 {budget:.0f}ms)")
    if not is_real:
        print(f"  注:本机无 NVIDIA GPU,走 CPU 档(<2s)。GPU 档 <200ms@2060 需在 2060 级显卡上")
        print(f"     装好 torch/timm(bash setup_gpu.sh)后**重跑本脚本**即自动出 GPU 档真特征延时。")
    return passed


def _selftest():
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    # 小尺寸快测(本机走 ClassicBackend),只校验流水线计时自洽与红线判定逻辑。
    backend, is_real, avg, p95 = bench(size=256, runs=5, bank_imgs=4, seed=0, warmup=1)
    print("  分档平均耗时(ms):" + ", ".join(f"{k}={v:.2f}" for k, v in avg.items()))
    stage_sum = avg["feature"] + avg["score"] + avg["threshold"]
    check(avg["total"] > 0 and all(v >= 0 for v in avg.values())
          and abs(avg["total"] - stage_sum) < 1e-6 * max(1.0, avg["total"]),
          f"各阶段计时正常且 total=分阶段之和(total={avg['total']:.3f} vs Σ={stage_sum:.3f})")
    check(not is_real, "本机走 CPU 经典后端(torch 不可用)")
    # 256px 小图在本机必然远小于任一红线
    check(avg["total"] < CPU_BUDGET_MS, f"小图 CPU 档 < 2s(={avg['total']:.2f}ms)")
    check(p95 >= avg["total"] * 0.5, f"p95({p95:.2f}) 与 mean({avg['total']:.2f}) 同量级")
    print("\n" + ("✅ bench_latency_gpu 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2500, help="方形输入边长(官方 2500)")
    ap.add_argument("--runs", type=int, default=10, help="计时重复次数")
    ap.add_argument("--bank", type=int, default=8, help="建库用的正常件张数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    print(f"=== bench_latency_gpu (2500×2500 真实流水线端到端延时) ===")
    backend, is_real, avg, p95 = bench(size=a.size, runs=a.runs, bank_imgs=a.bank, seed=a.seed)
    passed = _report(backend, is_real, avg, p95, a.size)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
