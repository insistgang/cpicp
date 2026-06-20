#!/usr/bin/env python3
"""
online_learning.py · 用户反馈驱动的在线/主动学习闭环(华为赛题一明列三大问题之一)

官方要求:"用户反馈驱动优化"——产线操作员对误检/漏检给出反馈,系统据此动态调参
(主动学习),无需离线重训。本模块在 PatchCore memory-bank 范式上实现纯 numpy 闭环:

  · 漏检(False Negative,真缺陷被判正常)→ 这是 bank "覆盖过广"的体现,操作员标注后
    上调判定阈值(让边界更严),使同类漏检在后续被检出。
  · 误检(False Positive,正常件被判缺陷)→ 操作员把该正常 patch 增量并入 memory bank,
    bank 覆盖更全,该正常模式的最近邻距离下降到阈值以下,后续不再误报;阈值同步可松。

阈值更新用对反馈样本的 best-F1 重标定(在"历史校准集 + 新反馈"上),既利用反馈又不被
单个样本带偏。记录反馈前/后在固定验证集上的 F1,产出"反馈→F1 提升"曲线作为专家评分证据。

纯 numpy,不依赖 torch / 真数据。`python online_learning.py` 自测(造可复现的
"先漏检/误检 → 喂反馈 → 指标提升"序列)。
"""
import numpy as np

from anomaly_score import nn_distance
from aoi_metrics import best_threshold, pr_at_threshold


class OnlineDetector:
    """带反馈闭环的 image-level 异常检测器。

    image-level 分 = patch 特征到 memory bank 的最近邻距离的 max(PatchCore 口径)。
    feedback_* 方法接收操作员对单张图的纠正,更新 bank / 校准集 / 阈值。
    """

    def __init__(self, bank, threshold, cal_scores=None, cal_labels=None,
                 max_bank=20000):
        self.bank = np.asarray(bank, float)
        self.threshold = float(threshold)
        # 历史校准集(分数级):用于反馈后在"旧校准+新反馈"上重标定阈值,避免被单点带偏。
        self.cal_scores = (np.asarray(cal_scores, float) if cal_scores is not None
                           else np.empty(0))
        self.cal_labels = (np.asarray(cal_labels, int) if cal_labels is not None
                           else np.empty(0, int))
        self.max_bank = int(max_bank)          # bank 上限(防无界增长,端侧内存约束)
        self.n_feedback = 0

    # ---- 推理 ----
    def image_score(self, patch_feats):
        """patch_feats (P,D) → image-level 异常分。空 patch 视为 0 分(无证据)。"""
        p = np.asarray(patch_feats, float)
        if p.size == 0 or self.bank.size == 0:
            return 0.0
        return float(nn_distance(p, self.bank).max())

    def predict(self, patch_feats):
        """1=判为缺陷, 0=判为正常。"""
        return int(self.image_score(patch_feats) >= self.threshold)

    # ---- 阈值重标定(在历史校准集 + 反馈样本上 best-F1) ----
    def _recalibrate(self, fb_score, fb_label):
        self.cal_scores = np.append(self.cal_scores, float(fb_score))
        self.cal_labels = np.append(self.cal_labels, int(fb_label))
        if self.cal_scores.size >= 2 and len(set(self.cal_labels.tolist())) == 2:
            self.threshold = best_threshold(self.cal_scores, self.cal_labels,
                                            objective="f1")["thr"]

    # ---- 操作员反馈 ----
    def feedback_false_negative(self, patch_feats):
        """漏检:真缺陷被判正常。记录为缺陷样本并重标定阈值(边界变严)。"""
        self._recalibrate(self.image_score(patch_feats), 1)
        self.n_feedback += 1
        return self.threshold

    def feedback_false_positive(self, patch_feats):
        """误检:正常件被判缺陷。把该正常 patch 并入 memory bank(覆盖更全),
        并记录为正常样本重标定阈值。"""
        p = np.asarray(patch_feats, float)
        s_before = self.image_score(p)
        if p.ndim == 2 and p.size and self.bank.size:
            self.bank = np.vstack([self.bank, p])
            if len(self.bank) > self.max_bank:                 # 超限:保留最近的
                self.bank = self.bank[-self.max_bank:]
        # 并库后该正常图的分数会下降;用并库后的分数做校准更贴近上线后行为
        self._recalibrate(self.image_score(p) if self.bank.size else s_before, 0)
        self.n_feedback += 1
        return self.threshold

    # ---- 在固定验证集上评估(供画"反馈→指标"曲线) ----
    def evaluate(self, val_patches, val_labels):
        scores = np.array([self.image_score(p) for p in val_patches])
        m = pr_at_threshold(scores, val_labels, self.threshold)
        return {"f1": m["f1"], "recall": m["recall"],
                "precision": m["precision"], "accuracy": m["accuracy"]}


def feedback_session(detector, feedback_items, val_patches, val_labels):
    """按操作员反馈序列逐条更新,记录每步在固定验证集上的指标 → "反馈→F1"曲线。

    feedback_items: list[(patch_feats, kind)],kind ∈ {"fn","fp"}。
    返回 list[dict],含 step / n_feedback / 指标(step=0 为反馈前基线)。
    """
    history = [{"step": 0, "n_feedback": 0, "threshold": detector.threshold,
                **detector.evaluate(val_patches, val_labels)}]
    for i, (feats, kind) in enumerate(feedback_items, start=1):
        if kind == "fn":
            detector.feedback_false_negative(feats)
        elif kind == "fp":
            detector.feedback_false_positive(feats)
        else:
            raise ValueError(f"未知反馈类型 {kind!r}(应为 'fn' 或 'fp')")
        history.append({"step": i, "n_feedback": detector.n_feedback,
                        "threshold": detector.threshold,
                        **detector.evaluate(val_patches, val_labels)})
    return history


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(0)
    D = 32
    center = rng.randn(D)

    def normal_patches(n, seed):
        r = np.random.RandomState(seed)
        return center + r.randn(n, D) * 0.12          # 正常 patch 簇

    def defect_patches(n, seed, sep=1.4):
        r = np.random.RandomState(seed)
        p = center + r.randn(n, D) * 0.12
        p[0] = center + r.randn(D) * sep              # 注入 1 个异常 patch
        return p

    # ---- 1. 初始 bank 只覆盖"主正常模式",故意漏掉一类"罕见但合法"的正常模式 ----
    # 这模拟真实产线:初始少样本没见过某种合法工件 → 上线后对它误检(FP)。
    main_normal = np.vstack([normal_patches(30, s) for s in range(8)])
    bank = main_normal.copy()

    # 一个初始 bank 没覆盖的"罕见正常模式"(偏移),会被误判为缺陷
    rare_offset = rng.randn(D) * 0.9
    def rare_normal_img(seed):
        r = np.random.RandomState(seed)
        return center + rare_offset + r.randn(20, D) * 0.12

    # 初始阈值:用主正常 + 一批缺陷校准
    init_cal_normal = np.array([nn_distance(normal_patches(20, 100 + s), bank).max()
                                for s in range(20)])
    init_cal_defect = np.array([nn_distance(defect_patches(20, 200 + s), bank).max()
                                for s in range(20)])
    cal_scores = np.concatenate([init_cal_normal, init_cal_defect])
    cal_labels = np.concatenate([np.zeros(20), np.ones(20)])
    thr0 = best_threshold(cal_scores, cal_labels, objective="f1")["thr"]

    det = OnlineDetector(bank, thr0, cal_scores, cal_labels)

    # ---- 2. 固定验证集:正常(含罕见正常模式)+ 缺陷 ----
    val_patches, val_labels = [], []
    for s in range(40):
        val_patches.append(normal_patches(20, 300 + s)); val_labels.append(0)
    for s in range(15):                                  # 罕见正常模式(初始会被误检)
        val_patches.append(rare_normal_img(400 + s)); val_labels.append(0)
    for s in range(40):
        val_patches.append(defect_patches(20, 500 + s)); val_labels.append(1)
    val_labels = np.array(val_labels)

    base = det.evaluate(val_patches, val_labels)
    # 初始罕见正常模式应被误检 → precision 受损
    rare_scores = np.array([det.image_score(rare_normal_img(400 + s)) for s in range(15)])
    n_fp_before = int((rare_scores >= det.threshold).sum())
    check(n_fp_before > 0, f"上线初期对未见过的合法模式有误检(FP={n_fp_before}/15) → 有改进空间")

    # ---- 3. 操作员对 5 个误检的罕见正常件给 FP 反馈 ----
    fp_feedback = [(rare_normal_img(400 + s), "fp") for s in range(5)]
    hist = feedback_session(det, fp_feedback, val_patches, val_labels)

    after = hist[-1]
    check(after["precision"] >= base["precision"],
          f"FP 反馈后 precision 不降反升({base['precision']:.3f}→{after['precision']:.3f})")
    # 关键:反馈过的那批罕见正常件,误检数应下降
    rare_scores2 = np.array([det.image_score(rare_normal_img(400 + s)) for s in range(15)])
    n_fp_after = int((rare_scores2 >= det.threshold).sum())
    check(n_fp_after < n_fp_before, f"FP 反馈后同类误检减少({n_fp_before}→{n_fp_after})")
    check(len(det.bank) > len(bank), f"误检正常 patch 已并入 memory bank({len(bank)}→{len(det.bank)})")

    # ---- 4. F1 整体不退化(闭环是净增益) ----
    check(after["f1"] >= base["f1"] - 1e-9, f"反馈闭环后 F1 不退化({base['f1']:.3f}→{after['f1']:.3f})")

    # ---- 5. 反馈历史曲线结构正确(可画"反馈次数→F1") ----
    check(len(hist) == len(fp_feedback) + 1 and hist[0]["step"] == 0,
          f"反馈历史含基线+每步({len(hist)}点,可绘曲线)")
    check(all("f1" in h and "n_feedback" in h for h in hist), "每步记录 F1/反馈数(曲线就绪)")

    # ---- 6. 边界:空 patch / 空 bank 不崩 ----
    empty_det = OnlineDetector(np.empty((0, D)), 1.0)
    check(empty_det.image_score(np.empty((0, D))) == 0.0, "空 bank/空 patch → 分数 0 不崩")
    check(det.predict(np.empty((0, D))) == 0, "空 patch → 判正常不崩")

    print(f"\n  闭环曲线(step→F1):" + " ".join(f"{h['step']}:{h['f1']:.3f}" for h in hist))
    print("\n" + ("✅ online_learning 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
