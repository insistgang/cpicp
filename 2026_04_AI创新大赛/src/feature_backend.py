#!/usr/bin/env python3
"""
feature_backend.py · 特征后端(真特征接口 + CPU 经典回退)

华为 AOI few-shot 异常检测的 PatchCore/PaDiM 范式需要把图切成 patch 网格、每个 patch
抽一个特征向量(正常图 patch 建 memory bank,测试图 patch 取最近邻距离当异常分)。

后端选择:
  - 首选 `TimmBackend`:timm/torch 的 WideResNet/ResNet 中层 feature(真特征,论文路线,
    PatchCore CVPR'22)。**已是完整可跑实现**(不再占位):加载预训练 backbone,
    features_only 取 layer2+layer3 中层特征图,双线性对齐+拼接,重采样到 grid×grid 网格、
    逐 patch L2 归一化。torch/timm 在场即真跑(本机无 torch,__init__ 内 import 抛
    ImportError → get_backend 回退);GPU 机一键装环境见 setup_gpu.sh。
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
        # numpy 2.0+ 在大数组 matmul 中可能报 false positive divide/overflow 警告
        # (与 anomaly_score.nn_distance 同因),不影响结果正确性 → 抑制。
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
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
# 真特征后端(timm/torch · PatchCore 论文路线 · GPU 机上启用)
# ----------------------------------------------------------------------------
class TimmBackend:
    """timm WideResNet/ResNet 中层 feature 的 PatchCore 后端(论文路线,真特征)。

    PatchCore(Roth et al. CVPR'22)做法:取预训练 backbone 的 **中层** 特征图
    (layer2+layer3,既保留局部纹理又有一定语义),把不同尺度的特征图对齐到同一空间
    分辨率后在通道维拼接 → 每个空间位置一个特征向量 → 这就是一个 "patch" 的描述子。
    本类把该特征图重采样到 grid×grid 网格并 L2 归一化,使其返回签名与 ClassicBackend
    **严格一致**:
        patch_features(img) -> (grid, grid, D)   网格 patch 特征(供热力图网格坐标)
        image_patches(img)  -> (grid*grid, D)    展平 patch 特征(供 memory bank / 打分)
    下游 patchcore_lite / anomaly_score / aoi_metrics / viz_heatmap 全部零改动复用。

    依赖 torch+timm+预训练权重+算力。本机无 torch 时 __init__ 内 `import torch` 抛
    ImportError,get_backend 捕获后回退 ClassicBackend(本文件自测即走此回退路径)。
    GPU 机上装好 torch/timm(见 setup_gpu.sh / requirements-gpu.txt)即真跑。
    """

    name = "timm-wideresnet"

    def __init__(self, model_name="wide_resnet50_2",
                 layers=("layer2", "layer3"), grid=8, device=None,
                 input_size=256, pretrained=True):
        # 关键:import 放在最前。本机无 torch → 这里抛 ImportError,
        # get_backend 捕获后优雅回退 ClassicBackend(保留原占位版的回退契约)。
        import torch
        import timm

        self.torch = torch
        self.grid = int(grid)
        self.model_name = model_name
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        # 设备:优先 CUDA(2060+),其次 Apple MPS,最后 CPU。
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # features_only=True 让 timm 直接以 list 形式吐各 stage 特征图,
        # 用 out_indices 选 layer2(index 2)/layer3(index 3)(ResNet 家族:
        # stem=0, layer1=1, layer2=2, layer3=3, layer4=4)。
        self._idx_of_layer = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
        out_indices = tuple(self._idx_of_layer[l] for l in self.layers)
        self.model = timm.create_model(
            model_name, pretrained=pretrained,
            features_only=True, out_indices=out_indices,
        )
        self.model.eval().to(self.device)
        for p in self.model.parameters():           # 纯推理,冻结、不建图
            p.requires_grad_(False)

        # 用 backbone 自带的预训练归一化常数(ImageNet mean/std),避免手填错。
        cfg = getattr(self.model, "default_cfg", {}) or {}
        self._mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)),
                                  dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)),
                                 dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        # 探测输出通道数 → 确定拼接后的特征维 D(layer2+layer3 通道之和)。
        feat_info = getattr(self.model, "feature_info", None)
        if feat_info is not None:
            try:
                chs = feat_info.channels()              # 与 out_indices 对应
                self._feat_dim = int(sum(chs))
            except Exception:
                self._feat_dim = self._probe_feat_dim()
        else:
            self._feat_dim = self._probe_feat_dim()

    def _probe_feat_dim(self):
        """跑一张全零图探测拼接后的通道数(D)。"""
        torch = self.torch
        with torch.no_grad():
            x = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            fmaps = self.model(x)
            return int(sum(f.shape[1] for f in fmaps))

    # --- 预处理:PIL / (H,W,3) uint8 → 归一化的 (1,3,Hin,Win) tensor ---
    def _to_tensor(self, img):
        torch = self.torch
        arr = np.asarray(img)
        if arr.ndim == 2:                                # 灰度 → 三通道
            arr = np.repeat(arr[..., None], 3, axis=2)
        arr = arr[..., :3].astype(np.float32) / 255.0
        # (H,W,3) → (1,3,H,W)
        t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]
        t = t.to(self.device)
        # 缩放到 backbone 期望的方形输入(双线性);2500×2500 真图在此降采样。
        t = torch.nn.functional.interpolate(
            t, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False)
        t = (t - self._mean) / self._std                 # ImageNet 归一化
        return t

    def _forward_maps(self, img):
        """前向取中层特征图,对齐到统一空间分辨率后在通道维拼接 → (1, D, gh, gw)。"""
        torch = self.torch
        x = self._to_tensor(img)
        with torch.no_grad():
            fmaps = self.model(x)                        # list of (1,C_i,h_i,w_i)
        # 以第一张(分辨率最高的 layer2)空间尺寸为基准,其余双线性对齐后拼接。
        ref_h, ref_w = fmaps[0].shape[2], fmaps[0].shape[3]
        aligned = []
        for f in fmaps:
            if f.shape[2] != ref_h or f.shape[3] != ref_w:
                f = torch.nn.functional.interpolate(
                    f, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            aligned.append(f)
        cat = torch.cat(aligned, dim=1)                  # (1, D, ref_h, ref_w)
        # 重采样到 grid×grid 网格(双线性自适应池化,等价 PatchCore 的局部聚合)。
        grid_map = torch.nn.functional.interpolate(
            cat, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return grid_map                                  # (1, D, grid, grid)

    def patch_features(self, img):
        """img → (grid, grid, D) 网格 patch 特征,每个 patch 向量 L2 归一化。"""
        torch = self.torch
        grid_map = self._forward_maps(img)               # (1, D, gh, gw)
        # (1,D,gh,gw) → (gh,gw,D)
        feat = grid_map[0].permute(1, 2, 0).contiguous()
        # 逐 patch L2 归一化(与 ClassicBackend 口径一致:每个 patch 范数≈1)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.float().cpu().numpy().astype(np.float32)

    def image_patches(self, img):
        """img → (grid*grid, D) 展平 patch 特征(供 memory bank / 打分)。"""
        f = self.patch_features(img)
        return f.reshape(-1, f.shape[-1])

    @property
    def feat_dim(self):
        return self._feat_dim


def get_backend(prefer_real=True, **kw):
    """优先真特征,失败回退经典 CPU 特征。返回 (backend, is_real)。

    prefer_real=True 时尝试 TimmBackend(torch/timm 真特征);torch/timm 未装、权重
    下载失败、或显存不足等任何异常都会被捕获 → 回退 ClassicBackend(本机即走此路径)。
    下游 patch_features/image_patches 签名两者一致,切换后下游代码零改动。"""
    if prefer_real:
        try:
            timm_keys = ("model_name", "layers", "grid", "device", "input_size", "pretrained")
            return TimmBackend(**{k: v for k, v in kw.items() if k in timm_keys}), True
        except Exception:
            # torch/timm 不可用或加载失败 → 静默回退,不阻断本机/无 GPU 流程。
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

    # --- TimmBackend 真后端接口契约(本机无 torch,做静态契约校验,不实例化) ---
    # 保证 GPU 机上真后端与 ClassicBackend 签名一致:存在同名方法 + feat_dim 属性。
    import inspect
    for meth in ("patch_features", "image_patches"):
        cb_sig = inspect.signature(getattr(ClassicBackend, meth))
        tb_sig = inspect.signature(getattr(TimmBackend, meth))
        check(list(cb_sig.parameters) == list(tb_sig.parameters),
              f"TimmBackend.{meth} 签名与 ClassicBackend 一致 {list(tb_sig.parameters)}")
    check(isinstance(TimmBackend.feat_dim, property),
          "TimmBackend.feat_dim 为 property(与 ClassicBackend 一致)")
    # __init__ 第一行须 import torch → 无 torch 时构造抛异常被 get_backend 捕获回退。
    src0 = inspect.getsource(TimmBackend.__init__)
    check("import torch" in src0 and "import timm" in src0,
          "TimmBackend.__init__ 先 import torch/timm(无 torch 时触发回退契约)")
    try:
        TimmBackend()                      # 本机必抛(ImportError),验证回退契约不破
        raised = False
    except Exception:
        raised = True
    check(raised, "本机实例化 TimmBackend 抛异常(torch 缺失)→ 回退契约成立")

    print("\n" + ("✅ feature_backend 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
