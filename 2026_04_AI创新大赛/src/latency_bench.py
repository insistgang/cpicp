#!/usr/bin/env python3
"""
latency_bench.py · 端到端延时基准(华为赛题一硬指标:单图<200ms@2060 / CPU<2s)

分档计时:预处理 + 特征提取 + 异常打分 + 阈值判定,对 2500×2500 输入。
本机无 NVIDIA GPU,只能测 CPU(目标<2s)并对 GPU<200ms 红线给出提示;
真模型(timm 特征)到位后替换 dummy 特征即可。检测时间占竞赛分30%,此基准直接服务该项。
纯 numpy,`python latency_bench.py` 自测。
"""
import time

GPU_BUDGET_MS = 200.0
CPU_BUDGET_MS = 2000.0


def run_once(img, bank, patch=50):
    """返回各阶段耗时(ms)。dummy 特征=网格 patch 的通道均值。"""
    import numpy as np
    stages = {}
    t = time.perf_counter()
    small = img[::8, ::8].astype(np.float32) / 255.0          # 预处理:降采样
    stages["preprocess"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

    H, W = small.shape[:2]
    ph = max(1, H // patch); feats = []
    for i in range(0, H - ph, ph):
        for j in range(0, W - ph, ph):
            feats.append(small[i:i+ph, j:j+ph].reshape(-1, small.shape[2]).mean(0))
    feats = np.array(feats) if feats else np.zeros((1, small.shape[2]))
    stages["feature"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

    f2 = np.clip((feats*feats).sum(1), -1e6, 1e6)
    b2 = np.clip((bank*bank).sum(1), -1e6, 1e6)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        d2 = f2[:, None] + b2[None, :] - 2*feats@bank.T
    score = float(np.sqrt(np.clip(d2.min(1), 0, None)).max())  # 异常打分(最近邻)
    stages["score"] = (time.perf_counter() - t) * 1000; t = time.perf_counter()

    _ = score >= 1.0                                           # 阈值判定
    stages["threshold"] = (time.perf_counter() - t) * 1000
    stages["total"] = sum(stages.values())
    return stages


def bench(n=20, size=2500, bank_size=200, D=3, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size, D) * 255).astype(np.uint8)
    bank = rng.randn(bank_size, D)
    runs = [run_once(img, bank) for _ in range(n)]
    avg = {k: sum(r[k] for r in runs) / n for k in runs[0]}
    return avg


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    avg = bench(n=10)
    print("  分档平均耗时(ms):" + ", ".join(f"{k}={v:.2f}" for k, v in avg.items()))
    check(avg["total"] > 0 and all(avg[k] >= 0 for k in avg), "各阶段计时正常")
    cpu_pass = avg["total"] < CPU_BUDGET_MS
    print(f"  CPU<{CPU_BUDGET_MS:.0f}ms: {'✅ PASS' if cpu_pass else '❌ FAIL'} (total={avg['total']:.1f}ms)")
    print(f"  GPU<{GPU_BUDGET_MS:.0f}ms 红线:本机无NVIDIA GPU,需在2060级显卡上用真模型复测")
    check(cpu_pass, f"dummy 流水线 CPU 端 <2s(={avg['total']:.1f}ms)")
    print("\n" + ("✅ latency_bench 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
