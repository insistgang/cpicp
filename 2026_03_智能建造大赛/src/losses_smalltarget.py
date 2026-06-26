#!/usr/bin/env python3
"""
losses_smalltarget.py · 小目标友好的度量/损失:NWD + Wise-IoU + Inner-IoU

为什么(直击"像素占比极低的小目标"):
  IoU 对微小框的位置偏移极其敏感(几像素偏移 IoU 就暴跌),导致小目标标签分配/回归不稳。
  - NWD(Normalized Wasserstein Distance):把框建模为高斯,用 Wasserstein 距离衡量相似度,
    对小目标平滑、对位置偏移不敏感 → 更适合做标签分配与辅助损失(arXiv:2110.13389 思路)。
  - Wise-IoU v1:带距离聚焦的 IoU 损失,缓解低质量样本主导(arXiv:2301.10051 思路)。
  - Inner-IoU:用缩放辅助框算 IoU,加速收敛(ratio=1 时退化为标准 IoU)。

本文件 numpy 核心可直接 `python losses_smalltarget.py` 跑自测;另附 torch 版与 ultralytics 接入说明。
收益务必以自己在 4070S 上的 A/B 消融为准,论文只作"参考思路"。
"""
import math


# ----------------------------- numpy 核心(可测) -----------------------------
def _xywh2xyxy(b):
    import numpy as np
    b = np.asarray(b, dtype=float)
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], -1)


def bbox_iou(a, b):
    """标准 IoU,输入 xywh(...,4)。"""
    import numpy as np
    A, B = _xywh2xyxy(a), _xywh2xyxy(b)
    ix1 = np.maximum(A[..., 0], B[..., 0]); iy1 = np.maximum(A[..., 1], B[..., 1])
    ix2 = np.minimum(A[..., 2], B[..., 2]); iy2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (A[..., 2]-A[..., 0]) * (A[..., 3]-A[..., 1])
    ab = (B[..., 2]-B[..., 0]) * (B[..., 3]-B[..., 1])
    return inter / (aa + ab - inter + 1e-9)


def inner_iou(a, b, ratio=0.8):
    """Inner-IoU:对 w,h 按 ratio 缩放出辅助框再算 IoU。ratio=1 等价标准 IoU。"""
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    sa = np.stack([a[..., 0], a[..., 1], a[..., 2]*ratio, a[..., 3]*ratio], -1)
    sb = np.stack([b[..., 0], b[..., 1], b[..., 2]*ratio, b[..., 3]*ratio], -1)
    return bbox_iou(sa, sb)


def nwd(a, b, C=12.8):
    """Normalized Wasserstein Distance,输入 xywh。返回相似度 (0,1],越大越像。
    W2² = (cx_a-cx_b)² + (cy_a-cy_b)² + ((w_a-w_b)/2)² + ((h_a-h_b)/2)²。C 为尺度常数(需按数据集调)。"""
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    w2 = ((a[..., 0]-b[..., 0])**2 + (a[..., 1]-b[..., 1])**2
          + ((a[..., 2]-b[..., 2])/2)**2 + ((a[..., 3]-b[..., 3])/2)**2)
    return np.exp(-np.sqrt(np.maximum(w2, 0.0)) / C)   # 相同框 w2=0 → NWD=1(无 eps 偏差)


def wiou_v1_loss(a, b):
    """Wise-IoU v1 损失:R_WIoU * (1-IoU)。R_WIoU 用中心距/最小包围盒对角线(聚焦)。"""
    import numpy as np
    A, B = _xywh2xyxy(a), _xywh2xyxy(b)
    iou = bbox_iou(a, b)
    cw = np.maximum(A[..., 2], B[..., 2]) - np.minimum(A[..., 0], B[..., 0])
    ch = np.maximum(A[..., 3], B[..., 3]) - np.minimum(A[..., 1], B[..., 1])
    a = np.asarray(a, float); b = np.asarray(b, float)
    cdist = (a[..., 0]-b[..., 0])**2 + (a[..., 1]-b[..., 1])**2
    r_wiou = np.exp(cdist / (cw**2 + ch**2 + 1e-9))     # 距离越大惩罚越大
    return r_wiou * (1.0 - iou)


# ----------------------------- ultralytics 接入说明 -----------------------------
INTEGRATION_NOTE = r"""
[如何接入 ultralytics(在 4070S 上做)]
方案A·标签分配加 NWD(改 TaskAlignedAssigner 的对齐度量,小目标更稳):
  在 ultralytics/utils/tal.py 的 get_box_metrics 里,把 align_metric 的 IoU 项
  替换/混合为 NWD:  metric = score^alpha * (nwd_ratio*nwd + (1-nwd_ratio)*ciou)^beta
方案B·回归损失加 NWD 辅助项(改 ultralytics/utils/loss.py BboxLoss.forward):
  loss_iou = (1 - iou) * weight                # 原有
  loss_nwd = (1 - nwd(pred_xywh, tgt_xywh, C)) * weight
  loss = (1 - r) * loss_iou.sum()/n + r * loss_nwd.sum()/n   # r≈0.5,先 A/B 验证
方案C(最省事)·monkeypatch:训练脚本顶部调用 patch_ultralytics_nwd(r=0.5) 注入(见下)。
务必:每个开关独立 A/B,看 small 桶召回与 mAP,收益不显著就回退。
"""


def patch_ultralytics_nwd(nwd_ratio=0.5, C=12.8):  # pragma: no cover (需 torch+ultralytics)
    """运行时把 NWD 混入 ultralytics 的 bbox_iou(粗粒度注入,便于快速 A/B)。需 torch。"""
    import torch
    from ultralytics.utils import metrics as M
    orig = M.bbox_iou

    def patched(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        iou = orig(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
        b1 = box1 if xywh else _xywh_from_xyxy_torch(box1)
        b2 = box2 if xywh else _xywh_from_xyxy_torch(box2)
        w2 = ((b1[..., 0]-b2[..., 0])**2 + (b1[..., 1]-b2[..., 1])**2
              + ((b1[..., 2]-b2[..., 2])/2)**2 + ((b1[..., 3]-b2[..., 3])/2)**2)
        nwd_t = torch.exp(-torch.sqrt(w2 + eps) / C)
        return (1 - nwd_ratio) * iou + nwd_ratio * nwd_t.unsqueeze(-1) if iou.ndim > nwd_t.ndim else \
               (1 - nwd_ratio) * iou + nwd_ratio * nwd_t
    M.bbox_iou = patched
    print(f"[patch] ultralytics bbox_iou 已混入 NWD(ratio={nwd_ratio}, C={C})")


def _xywh_from_xyxy_torch(b):  # pragma: no cover
    import torch
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], -1)


# ----------------------------- 自测(numpy) -----------------------------
def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  [OK] " if c else "  [FAIL] ") + m); ok = ok and c

    same = [50, 50, 20, 20]
    check(abs(bbox_iou(same, same) - 1.0) < 1e-6, "相同框 IoU=1")
    check(abs(nwd(same, same) - 1.0) < 1e-6, "相同框 NWD=1")
    check(abs(float(inner_iou(same, same, ratio=1.0)) - 1.0) < 1e-6, "Inner-IoU(ratio=1)=标准IoU=1")
    check(float(wiou_v1_loss(same, same)) < 1e-6, "相同框 WIoU 损失≈0")

    far = [500, 500, 20, 20]
    check(float(bbox_iou(same, far)) < 1e-6, "远离不相交框 IoU≈0")
    check(float(nwd(same, far)) < 0.01, "远离框 NWD≈0")

    # NWD 随偏移单调下降,且比 IoU 平滑(微小框小偏移 IoU 掉到0但 NWD 仍有区分度)
    tiny = [50, 50, 4, 4]
    shift = [56, 50, 4, 4]   # 偏移6px,微小框已不相交
    check(float(bbox_iou(tiny, shift)) == 0.0, "微小框偏移6px IoU=0(IoU对小目标过敏)")
    check(0.0 < float(nwd(tiny, shift)) < 1.0, f"同偏移 NWD 仍可区分(NWD={float(nwd(tiny, shift)):.3f}>0)")

    n_near = float(nwd(tiny, [52, 50, 4, 4])); n_far = float(nwd(tiny, [60, 50, 4, 4]))
    check(n_near > n_far, f"NWD 随偏移单调下降({n_near:.3f}>{n_far:.3f})")

    print("\n" + ("[OK] losses_smalltarget 自测通过" if ok else "[FAIL] 自测未通过"))
    print(INTEGRATION_NOTE)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
