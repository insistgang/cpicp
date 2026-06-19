#!/usr/bin/env python3
"""
features.py · CPU 经典特征提取器(CLIP 不可用时的 baseline 后端)

赛题#23 的官方现有基础是 CLIP,但 CLIP 需要 torch/open_clip + GPU。本模块提供一条
**纯 CPU、仅依赖 PIL+numpy+sklearn** 的经典特征回退路径,使整条流水线
(prepare_data→特征→similarity 去重→metrics)能在真实图像像素上端到端跑通,
而不是跑在随机向量上。

特征构成(对图像内容敏感,足以让"同一张/套用"的票据照片向量靠近、不同模板远离):
  1. 颜色直方图   : RGB 三通道各 nbins 的归一化直方图(整体色调/底色)
  2. 灰度梯度统计 : 分块的梯度幅值/方向能量(HOG 式),刻画版式/文字布局
  3. 分块灰度统计 : 每个 grid 块的均值/标准差(粗略空间结构)
  4. 标准化 + PCA : sklearn StandardScaler + PCA 降维去相关(可选,默认开)

⚠️ 这是 **baseline 特征后端**,不是 CLIP。真特征接口(classify.py/embed.py 的
extract_embeddings)在 torch 可用时仍走 CLIP;import 失败时回退到本模块。
"""
import argparse
import json
import os


# ----------------------------- 单图特征 -----------------------------

def _to_arrays(path_or_img, size=(128, 128)):
    """加载并标准化图像,返回 (rgb[H,W,3] float in [0,1], gray[H,W] float)。"""
    import numpy as np
    from PIL import Image
    if isinstance(path_or_img, Image.Image):
        img = path_or_img
    else:
        img = Image.open(path_or_img)
    img = img.convert("RGB").resize(size, Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114])  # 亮度
    return rgb, gray


def color_histogram(rgb, nbins=16):
    """RGB 三通道归一化直方图,拼成 3*nbins 维。"""
    import numpy as np
    feats = []
    edges = np.linspace(0.0, 1.0, nbins + 1)
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c].ravel(), bins=edges)
        h = h.astype(np.float64)
        s = h.sum()
        feats.append(h / s if s > 0 else h)
    return np.concatenate(feats)


def gradient_block_stats(gray, grid=4, n_orient=6):
    """HOG 式:把灰度图分成 grid×grid 块,每块统计 n_orient 个方向的梯度能量。
    返回 grid*grid*n_orient 维。刻画版式/文字笔画的方向分布。"""
    import numpy as np
    # Sobel 梯度
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)  # [-pi, pi]
    # 折叠到 [0, pi)(无符号方向,典型 HOG 做法)
    ang = np.mod(ang, np.pi)
    bin_idx = np.minimum((ang / np.pi * n_orient).astype(int), n_orient - 1)

    H, W = gray.shape
    feats = []
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    for by in range(grid):
        for bx in range(grid):
            y0, y1 = ys[by], ys[by + 1]
            x0, x1 = xs[bx], xs[bx + 1]
            m = mag[y0:y1, x0:x1].ravel()
            b = bin_idx[y0:y1, x0:x1].ravel()
            hist = np.zeros(n_orient)
            if m.size:
                np.add.at(hist, b, m)
            # 块内 L2 归一化(对比度归一)
            norm = np.linalg.norm(hist)
            feats.append(hist / norm if norm > 1e-8 else hist)
    return np.concatenate(feats)


def block_intensity_stats(gray, grid=4):
    """每个 grid 块的灰度均值+标准差,刻画粗略空间结构。返回 2*grid*grid 维。"""
    import numpy as np
    H, W = gray.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    means, stds = [], []
    for by in range(grid):
        for bx in range(grid):
            blk = gray[ys[by]:ys[by + 1], xs[bx]:xs[bx + 1]]
            means.append(float(blk.mean()) if blk.size else 0.0)
            stds.append(float(blk.std()) if blk.size else 0.0)
    return np.concatenate([np.array(means), np.array(stds)])


def extract_single(path_or_img, size=(128, 128), nbins=16, grid=4, n_orient=6):
    """单图经典特征向量(未标准化/未降维的原始拼接特征)。"""
    import numpy as np
    rgb, gray = _to_arrays(path_or_img, size=size)
    ch = color_histogram(rgb, nbins=nbins)
    gh = gradient_block_stats(gray, grid=grid, n_orient=n_orient)
    bi = block_intensity_stats(gray, grid=grid)
    return np.concatenate([ch, gh, bi]).astype(np.float64)


# --------------------------- 批量 + 标准化/PCA ---------------------------

class ClassicFeatureExtractor:
    """经典特征后端:批量提特征 → StandardScaler → (可选)PCA。

    与 CLIP 接口对齐:fit_transform / transform 返回 [N, D] 嵌入,
    交给 similarity.py 做 L2 归一化 + 余弦相似度。

    用法:
        fe = ClassicFeatureExtractor(pca_dim=64)
        embs = fe.fit_transform(image_paths)          # 训练/全量提取
        # 增量: e = fe.transform(new_paths)
    """

    def __init__(self, size=(128, 128), nbins=16, grid=4, n_orient=6,
                 pca_dim=32, standardize=False):
        # 注:standardize 默认 False。这些是结构化非负特征(颜色/梯度直方图),
        # StandardScaler 会把低方差噪声维放大到与判别维同权,反而拉低去重 AUC(已实测);
        # PCA(不标准化)做去相关/降噪即可。CLIP 类高维稠密特征才需要标准化。
        self.size = size
        self.nbins = nbins
        self.grid = grid
        self.n_orient = n_orient
        self.pca_dim = pca_dim
        self.standardize = standardize
        self.scaler = None
        self.pca = None
        self._fitted = False

    def raw_features(self, image_paths):
        import numpy as np
        feats = [extract_single(p, self.size, self.nbins, self.grid, self.n_orient)
                 for p in image_paths]
        return np.vstack(feats) if feats else np.zeros((0, 1))

    def fit_transform(self, image_paths):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X = self.raw_features(image_paths)
        # NumPy 2.0.2 在本机 BLAS 上对**有限输入**的 matmul 也会抛 spurious 浮点警告
        # (与 similarity.py 注释同因),结果有限正确;此处 errstate 局部抑制,不影响数值。
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.standardize:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            # PCA 维度不能超过 min(n_samples, n_features)
            if self.pca_dim and X.shape[0] >= 2 and X.shape[1] >= 2:
                k = min(self.pca_dim, X.shape[0], X.shape[1])
                self.pca = PCA(n_components=k, random_state=0)
                X = self.pca.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, image_paths):
        import numpy as np
        if not self._fitted:
            raise RuntimeError("先调用 fit_transform 再 transform")
        X = self.raw_features(image_paths)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.scaler is not None:
                X = self.scaler.transform(X)
            if self.pca is not None:
                X = self.pca.transform(X)
        return X


def extract_embeddings_classic(image_paths, pca_dim=32, **kw):
    """便捷函数:返回经典特征嵌入 [N, D]。CLIP 不可用时的 extract_embeddings 等价物。
    默认不标准化(见 ClassicFeatureExtractor 注释),PCA 降维去噪。"""
    fe = ClassicFeatureExtractor(pca_dim=pca_dim, **kw)
    return fe.fit_transform(image_paths)


# ------------------------------- 自测 -------------------------------

def _make_test_images(tmpdir):
    """生成 3 张测试图:base、base 的轻微变体(套用)、一张完全不同的图。"""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    # base: 左上深色块 + 文字状条纹
    base = np.full((160, 160, 3), 230, np.uint8)
    base[10:60, 10:120] = [40, 60, 120]
    for y in range(80, 150, 8):
        base[y:y + 3, 20:140] = [30, 30, 30]
    Image.fromarray(base).save(os.path.join(tmpdir, "base.png"))
    # variant: base 加轻微噪声 + JPEG 般压缩感(模拟"套用"后的同一张)
    var = base.astype(np.int16) + rng.randint(-6, 7, base.shape)
    var = np.clip(var, 0, 255).astype(np.uint8)
    Image.fromarray(var).save(os.path.join(tmpdir, "variant.png"))
    # different: 完全不同的版式/底色
    diff = np.full((160, 160, 3), 250, np.uint8)
    diff[100:150, 30:150] = [200, 40, 40]
    for x in range(20, 140, 10):
        diff[20:90, x:x + 2] = [10, 80, 10]
    Image.fromarray(diff).save(os.path.join(tmpdir, "different.png"))
    return [os.path.join(tmpdir, f) for f in ("base.png", "variant.png", "different.png")]


def _selftest():
    import sys, tempfile
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    with tempfile.TemporaryDirectory() as td:
        paths = _make_test_images(td)

        # 单图特征:维度合理、有限
        f0 = extract_single(paths[0])
        check(f0.ndim == 1 and f0.size > 0, f"单图特征维度 {f0.size}")
        check(np.all(np.isfinite(f0)), "单图特征全有限(无 NaN/Inf)")

        # 批量 + PCA
        fe = ClassicFeatureExtractor(pca_dim=2)  # 3 样本 → PCA 最多 2 维
        embs = fe.fit_transform(paths)
        check(embs.shape[0] == 3, f"批量嵌入 N=3(shape={embs.shape})")
        check(np.all(np.isfinite(embs)), "嵌入全有限")

        # 核心:base 与 variant(套用)余弦相似 > base 与 different。
        # 注:StandardScaler 在仅 3 个样本上会过度白化、扭曲余弦,故用**原始拼接特征**
        # (色直方图+梯度+块统计,皆非负/同量纲)直接验证特征本身能区分内容相似性。
        # 真实流水线样本量大,标准化才有意义;此处验证的是提取器对像素内容的敏感性。
        from similarity import cosine_sim_matrix
        raw = fe.raw_features(paths)
        S = cosine_sim_matrix(raw)
        s_var = S[0, 1]   # base ↔ variant(应高)
        s_diff = S[0, 2]  # base ↔ different(应低)
        margin = s_var - s_diff
        check(s_var > s_diff,
              f"套用对相似 > 不同对(base↔variant={s_var:.3f} > base↔different={s_diff:.3f})")
        check(margin > 0.1,
              f"套用 vs 不同 有明显区分裕度(Δ={margin:.3f}>0.1,可设阈值分开)")

        # transform 一致性:同一图重复提取应得几乎相同特征
        e1 = fe.transform([paths[0]])
        e2 = fe.transform([paths[0]])
        check(np.allclose(e1, e2), "同图重复 transform 结果一致(确定性)")

        # 不同尺寸输入也能处理
        from PIL import Image
        big = Image.new("RGB", (300, 200), (123, 200, 50))
        fb = extract_single(big)
        check(np.all(np.isfinite(fb)), "非方形/任意尺寸图可处理")

    print("\n" + ("✅ features 自测通过" if ok else "❌ features 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="经典 CPU 特征提取(CLIP 回退后端)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--images", nargs="+", help="对这些图像提取经典特征")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--out", help="保存嵌入到 .npy")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.images:
        ap.error("需 --images 或 --selftest")
    import numpy as np
    embs = extract_embeddings_classic(a.images, pca_dim=a.pca_dim)
    print(f"✓ 经典特征嵌入 {embs.shape}(后端=PIL+numpy+sklearn,非 CLIP)")
    if a.out:
        np.save(a.out, embs)
        print(f"  已保存 → {a.out}")


if __name__ == "__main__":
    main()
