#!/usr/bin/env python3
"""
anomaly_score.py · 工业异常检测打分核(华为赛题一,few-shot AOI)

PatchCore/PaDiM 范式:用 100 张正常样本特征建 memory bank,测试图特征对 bank 的
最近邻距离即异常分(image-level=patch 距离取 max)。无需训练、天然适配少样本。
打分核改自 02/similarity.py 的相似度思想(余弦→最近邻距离)。纯 numpy,`python anomaly_score.py` 自测。
"""


def nn_distance(test_feats, bank):
    """test_feats (N,D), bank (M,D) → (N,) 每个测试特征到 bank 的最近邻 L2 距离。"""
    import numpy as np
    t = np.asarray(test_feats, float); b = np.asarray(bank, float)
    # (N,M) 距离平方 = |t|^2 + |b|^2 - 2 t·b
    d2 = (t * t).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * t @ b.T
    return np.sqrt(np.clip(d2.min(1), 0, None))


def image_anomaly_score(patch_feats, bank):
    """一张图的 patch 特征 (P,D) → image-level 异常分 = 各 patch 最近邻距离的 max(PatchCore 口径)。"""
    import numpy as np
    return float(nn_distance(patch_feats, bank).max())


def batch_image_scores(images_patches, bank):
    """images_patches: list of (P,D);返回每张图 image-level 异常分。"""
    return [image_anomaly_score(p, bank) for p in images_patches]


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    D = 64
    center = rng.randn(D)
    normal_bank = center + rng.randn(200, D) * 0.1            # 正常特征聚簇 → memory bank
    normal_test = center + rng.randn(20, D) * 0.1
    anomaly_test = center + rng.randn(20, D) * 0.1 + rng.randn(D) * 2.0   # 偏离簇

    ns = nn_distance(normal_test, normal_bank)
    as_ = nn_distance(anomaly_test, normal_bank)
    check(as_.mean() > ns.mean() * 3, f"异常分远高于正常({as_.mean():.2f} vs {ns.mean():.2f})")

    # image-level:含1个异常 patch 的图,分应高
    normal_img = center + rng.randn(30, D) * 0.1
    defect_img = normal_img.copy(); defect_img[0] = center + rng.randn(D) * 3.0
    check(image_anomaly_score(defect_img, normal_bank) > image_anomaly_score(normal_img, normal_bank),
          "含缺陷patch的图 image-level 异常分更高")

    print("\n" + ("✅ anomaly_score 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
