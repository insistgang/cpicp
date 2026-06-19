#!/usr/bin/env python3
"""
patchcore_lite.py · PatchCore 风格 memory bank + greedy coreset 子采样(华为赛题一)

100 张正常图的 patch 特征量很大,用 greedy coreset(最远点采样)压成小 bank,
既降内存/加速最近邻(利于 <200ms),又基本保留覆盖度。纯 numpy。
真特征接 timm ResNet/WideResNet 的中层 feature(需 torch);本机用随机特征跑通+自测。
`python patchcore_lite.py` 自测。
"""


def greedy_coreset(feats, ratio=0.25, seed=0):
    """最远点采样选 ~ratio 比例的代表性子集。返回选中的行索引。"""
    import numpy as np
    f = np.asarray(feats, float)
    n = len(f)
    k = max(1, int(n * ratio))
    rng = np.random.RandomState(seed)
    start = rng.randint(n)
    selected = [start]
    min_d = ((f - f[start]) ** 2).sum(1)        # 到已选集合的最近距离
    for _ in range(k - 1):
        nxt = int(min_d.argmax())               # 选离已选集合最远的点
        selected.append(nxt)
        d = ((f - f[nxt]) ** 2).sum(1)
        min_d = np.minimum(min_d, d)
    return sorted(set(selected))


def build_memory_bank(normal_feats, coreset_ratio=0.25, seed=0):
    """normal_feats: (M,D) 所有正常 patch 特征 → coreset 子采样后的 memory bank。"""
    import numpy as np
    f = np.asarray(normal_feats, float)
    idx = greedy_coreset(f, coreset_ratio, seed)
    return f[idx]


def _selftest():
    import sys, numpy as np
    from anomaly_score import nn_distance
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(1)
    D = 32
    # 3 个簇的正常特征
    feats = np.vstack([rng.randn(300, D) * 0.2 + c for c in (rng.randn(3, D) * 3)])
    bank = build_memory_bank(feats, coreset_ratio=0.2)

    check(0 < len(bank) <= int(len(feats) * 0.2) + 1, f"coreset 压缩到 ~20%({len(bank)}/{len(feats)})")

    # coreset 应覆盖各簇:正常点到 bank 距离仍小,异常点距离大
    normal_pt = feats[:50]
    anomaly_pt = rng.randn(50, D) * 0.2 + rng.randn(D) * 10
    nd = nn_distance(normal_pt, bank).mean()
    ad = nn_distance(anomaly_pt, bank).mean()
    check(ad > nd * 3, f"coreset 后异常仍可分(异常{ad:.2f} >> 正常{nd:.2f})")

    # 覆盖度:全量 bank vs coreset,对正常点的最近邻距离不应暴涨
    full_nd = nn_distance(normal_pt, feats).mean()
    check(nd < full_nd * 3 + 1.0, f"coreset 覆盖度可接受(coreset{nd:.2f} vs full{full_nd:.2f})")

    print("\n" + ("✅ patchcore_lite 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
