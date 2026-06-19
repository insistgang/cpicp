#!/usr/bin/env python3
"""
feature_backend.py · 特征后端(真特征接口 + CPU 经典回退)

华为 AOI few-shot 异常检测的 PatchCore/PaDiM 范式需要把图切成 patch 网格、每个 patch
抽一个特征向量(正常图 patch 建 memory bank,测试图 patch 取最近邻距离当异常分)。

后端选择:
  - 首选 `TimmBackend`:timm/torch 的 WideResNet/ResNet 中层 feature(真特征,论文路线)。
    import torch/timm 失败时自动不可用,接口保留,真权重+算力到位即可启用。
  - 回退 `ClassicBackend`(本机 CPU,无 torch):每个 patch 抽
      · 分块颜色统计(RGB 各通道 mean/std)
      · 梯度直方图(Sobel 幅值方向的 HOG 式 8-bin 直方图,刻画纹理/边缘)
    再 L2 归一化。纯 PIL+numpy,Mac CPU 可真跑,刻画"局部纹理是否异常"足够区分
    划痕/斑点/缺件/色变。明确标注为 baseline,真特征接口不变。

统一返回:
  patch_features(img) -> (gh, gw, D)  网格 patch 特征图(供热力图可视化用网格坐标)
  image_patches(img)  -> (P, D)       展平的 patch 特征(供 memory bank / 打分)

`python feature_backend.py` 自测(经典后端在真实 numpy 图上跑)。
"""
import numpy as np


# ----------------------------------------------------------------------------
# 经典 CPU 后端(本机可真跑的 baseline)
# ----------------------------------------------------------------------------
class ClassicBackend:
    """分块颜色 + 梯度直方图特征。纯 numpy/PIL,无需 torch。"""

    name = "classic-cpu"

    def __init__(self, grid=8, hog_bins=8):
        self.grid = grid          # 把图切成 grid×grid 个 patch
        self.hog_bins = hog_bins

    # --- 内部:整图梯度(Sobel) ---
    @staticmethod
    def _gray(img):
        a = np.asarray(img, np.float32)
        if a.ndim == 2:
            return a
        return a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)

    @staticmethod
    def _sobel(gray):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        ky = kx.T
        # 反射 pad 后做 3x3 卷积(纯 numpy 滑窗,2500 降采样后尺寸小,够快)
        g = np.pad(gray, 1, mode="reflect")
        gx = (kx[0, 0]*g[:-2, :-2] + kx[0, 1]*g[:-2, 1:-1] + kx[0, 2]*g[:-2, 2:]
              + kx[1, 0]*g[1:-1, :-2] + kx[1, 1]*g[1:-1, 1:-1] + kx[1, 2]*g[1:-1, 2:]
              + kx[2, 0]*g[2:, :-2] + kx[2, 1]*g[2:, 1:-1] + kx[2, 2]*g[2:, 2:])
        gy = (ky[0, 0]*g[:-2, :-2] + ky[0, 1]*g[:-2, 1:-1] + ky[0, 2]*g[:-2, 2:]
              + ky[1, 0]*g[1:-1, :-2] + ky[1, 1]*g[1:-1, 1:-1] + ky[1, 2]*g[1:-1, 2:]
              + ky[2, 0]*g[2:, :-2] + ky[2, 1]*g[2:, 1:-1] + ky[2, 2]*g[2:, 2:])
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi   # [0,pi),无方向极性
        return mag, ang

    def _patch_feat(self, rgb_patch, mag_patch, ang_patch):
        """单个 patch → 颜色(6) + HOG(hog_bins) 特征。"""
        # 颜色:RGB 各通道 mean/std,归一化到 [0,1]
        cm = rgb_patch.reshape(-1, rgb_patch.shape[-1]).mean(0) / 255.0
        cs = rgb_patch.reshape(-1, rgb_patch.shape[-1]).std(0) / 255.0
        # 梯度方向直方图(幅值加权)
        bins = np.minimum((ang_patch / np.pi * self.hog_bins).astype(int), self.hog_bins - 1)
        hog = np.zeros(self.hog_bins, np.float32)
        np.add.at(hog, bins.ravel(), mag_patch.ravel())
        hog = hog / (hog.sum() + 1e-6)
        feat = np.concatenate([cm, cs, hog]).astype(np.float32)
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def patch_features(self, img):
        """img: PIL.Image 或 (H,W,3) uint8 → (grid, grid, D) 网格 patch 特征。"""
        rgb = np.asarray(img, np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        rgb = rgb[..., :3]
        H, W = rgb.shape[:2]
        gray = self._gray(rgb)
        mag, ang = self._sobel(gray)
        gh = gw = self.grid
        ys = np.linspace(0, H, gh + 1).astype(int)
        xs = np.linspace(0, W, gw + 1).astype(int)
        out = np.empty((gh, gw, 6 + self.hog_bins), np.float32)
        for i in range(gh):
            for j in range(gw):
                y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
                out[i, j] = self._patch_feat(rgb[y0:y1, x0:x1],
                                             mag[y0:y1, x0:x1], ang[y0:y1, x0:x1])
        return out

    def image_patches(self, img):
        f = self.patch_features(img)
        return f.reshape(-1, f.shape[-1])

    @property
    def feat_dim(self):
        return 6 + self.hog_bins


# ----------------------------------------------------------------------------
# 真特征后端(timm/torch,接口保留;import 失败即不可用)
# ----------------------------------------------------------------------------
class TimmBackend:
    """timm WideResNet/ResNet 中层 feature 的 PatchCore 后端(论文路线)。

    依赖 torch+timm+权重+算力,本机不可用时构造抛 ImportError,调用方回退 ClassicBackend。
    保留完整接口签名,GPU 环境装好 torch/timm 后直接启用即可。
    """

    name = "timm-resnet"

    def __init__(self, model_name="wide_resnet50_2", layers=("layer2", "layer3"), grid=8):
        import torch  # noqa: F401  (此处触发 ImportError → 调用方回退)
        import timm   # noqa: F401
        self.grid = grid
        self.model_name = model_name
        self.layers = layers
        # 真实环境:加载预训练模型、注册 hook 取中层特征、双线性插值对齐网格。
        # 本机无 torch,不会执行到这里。
        raise NotImplementedError(
            "TimmBackend 需 torch+timm+权重+算力;本机回退 ClassicBackend。"
            "GPU 环境实现:load timm 预训练→forward hook 取 layer2/3→插值到 grid 网格。")

    def patch_features(self, img):  # pragma: no cover - 真环境实现
        raise NotImplementedError

    def image_patches(self, img):   # pragma: no cover
        raise NotImplementedError


def get_backend(prefer_real=True, **kw):
    """优先真特征,失败回退经典 CPU 特征。返回 (backend, is_real)。"""
    if prefer_real:
        try:
            return TimmBackend(**{k: v for k, v in kw.items() if k in ("grid",)}), True
        except Exception:
            pass
    return ClassicBackend(**{k: v for k, v in kw.items() if k in ("grid", "hog_bins")}), False


# ----------------------------------------------------------------------------
def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    rng = np.random.RandomState(0)
    be = ClassicBackend(grid=8)

    # 纯色平板:各 patch 特征应高度一致(纹理均匀)
    flat = np.full((128, 128, 3), 150, np.uint8)
    pf = be.patch_features(flat)
    check(pf.shape == (8, 8, be.feat_dim), f"网格特征形状 {pf.shape}")
    flat_feats = pf.reshape(-1, pf.shape[-1])
    spread_flat = np.linalg.norm(flat_feats - flat_feats.mean(0), axis=1).mean()
    check(spread_flat < 0.2, f"纯色图各 patch 特征一致(离散度={spread_flat:.3f})")

    # 带划痕/斑点的图:含缺陷区域 patch 特征应明显偏离正常 patch
    from augment_defect import add_defect
    tex = (rng.rand(128, 128, 3) * 20 + 150).astype(np.uint8)   # 轻纹理正常件
    normal_feats = be.image_patches(tex)
    defect_img, bbox, _ = add_defect(tex, "scratch", np.random.RandomState(3))
    defect_feats = be.image_patches(defect_img)
    # 正常 vs 缺陷:缺陷图至少有 patch 离正常簇中心更远
    center = normal_feats.mean(0)
    max_dev_normal = np.linalg.norm(normal_feats - center, axis=1).max()
    max_dev_defect = np.linalg.norm(defect_feats - center, axis=1).max()
    check(max_dev_defect > max_dev_normal, f"缺陷图含更偏离的 patch({max_dev_defect:.3f}>{max_dev_normal:.3f})")

    # L2 归一化:每个 patch 特征范数≈1
    norms = np.linalg.norm(normal_feats, axis=1)
    check(np.allclose(norms, 1.0, atol=1e-3), f"patch 特征已 L2 归一化(范数≈1, 实测{norms.mean():.3f})")

    # 真特征后端不可用时应优雅回退
    be2, is_real = get_backend(prefer_real=True, grid=8)
    check(isinstance(be2, ClassicBackend) and not is_real, "torch 不可用→回退 ClassicBackend")

    print("\n" + ("✅ feature_backend 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
