#!/usr/bin/env python3
"""
run_real_pipeline.py · 真实合成图像上的 AOI few-shot 异常检测端到端流水线

把之前在随机 numpy 特征上跑的逻辑(feature_backend / patchcore_lite /
fewshot_protocol / anomaly_score / aoi_metrics)接到 synth_aoi 程序化生成的
真实工件图像上,端到端跑通并产出 **真实 AUC / F1 / few-shot 结果 / per-class 检出**
与 per-image 延时、官方竞赛分。特征用经典 CPU 后端(torch 不可用时的 baseline),
真特征接口(TimmBackend)保留。

流程严格按华为官方少样本协议:
  少样本启动 = N 张正常 + M 张缺陷做"迁移"(建库 + 校准阈值) → 测试集 1000+ 张评测。
本机为速度默认 N=100 正常、M=30 缺陷、测试 600+,可 --full 放大到 1000+。

输出:
  - 控制台:backend / AUC / F1 / recall / per-class 检出率 / 延时 / 竞赛分
  - output/pipeline_report.json:全部指标(供文档/PPT 引用)
  - output/pipeline_scores.npz:scores+labels+kinds(供 viz_heatmap / P-R 曲线复用)

`python run_real_pipeline.py`            标准规模(经典 CPU baseline,离线可跑)
`python run_real_pipeline.py --real`     标准规模(timm 真特征,需 torch/timm + 已缓存权重)
`python run_real_pipeline.py --full`     测试集放大到 1000+(贴合官方口径,较慢)
`python run_real_pipeline.py --selftest` 自测(小规模,校验真实可分性 + 产物落盘)
"""
import argparse
import json
import os
import time

import numpy as np

from synth_aoi import gen_dataset, DEFECT_KINDS
from feature_backend import get_backend
from patchcore_lite import build_memory_bank
from anomaly_score import nn_distance
from aoi_metrics import roc_auc, best_threshold, pr_at_threshold, compute_competition_score

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def build_features(images, backend):
    """对每张图抽 patch 特征。返回 list[(P,D)] 与 image-level 网格形状。"""
    return [backend.image_patches(im) for im in images]


def run(n_normal=100, n_defect=30, n_test_normal=400, n_test_defect=200,
        size=160, grid=8, coreset_ratio=0.3, seed=0, save=True, verbose=True,
        report_name="pipeline_report.json", scores_name="pipeline_scores.npz",
        prefer_real=False):
    """端到端跑 few-shot AOI 异常检测,返回指标 dict。

    report_name/scores_name 可改落盘文件名;自测用单独文件名,**不覆盖**标准规模的
    交付物 pipeline_report.json(否则一键自测后磁盘上的 report 会被小规模数据替换,
    与 README 标注的标准规模指标不符)。"""
    backend, is_real = get_backend(prefer_real=prefer_real, grid=grid)

    # ---- 1. 生成 few-shot 训练件 + 独立测试件(不重叠的 seed 段) ----
    train_imgs, train_labs, train_metas = gen_dataset(n_normal, n_defect, size, seed=seed)
    # 测试集另起 seed,与训练件不重叠
    test_imgs, test_labs, test_metas = _gen_test(n_test_normal, n_test_defect, size, seed=seed + 777)

    # ---- 2. 抽真实特征(经典 CPU 后端) ----
    t_feat = time.perf_counter()
    train_patch = build_features(train_imgs, backend)
    test_patch = build_features(test_imgs, backend)
    feat_ms = (time.perf_counter() - t_feat) * 1000

    train_labs = np.asarray(train_labs)
    test_labs = np.asarray(test_labs)

    # ---- 3. PatchCore memory bank:仅用正常件 patch,greedy coreset 压缩 ----
    normal_idx = np.where(train_labs == 0)[0]
    # 留 20% 正常件做阈值校准(不进库)
    n_cal = max(1, int(len(normal_idx) * 0.2))
    bank_normal_idx = normal_idx[n_cal:]
    cal_normal_idx = normal_idx[:n_cal]
    defect_idx = np.where(train_labs == 1)[0]

    normal_patch_pool = np.vstack([train_patch[i] for i in bank_normal_idx])
    bank = build_memory_bank(normal_patch_pool, coreset_ratio=coreset_ratio, seed=seed)

    # ---- 4. 校准阈值:用留出正常 + 30 缺陷件的 image-level 分,best-F1 ----
    cal_imgs_idx = list(cal_normal_idx) + list(defect_idx)
    cal_scores = np.array([nn_distance(train_patch[i], bank).max() for i in cal_imgs_idx])
    cal_labels = np.array([train_labs[i] for i in cal_imgs_idx])
    thr = best_threshold(cal_scores, cal_labels, objective="f1")["thr"]

    # ---- 5. 测试集评测 + per-image 延时(打分阶段) ----
    test_scores = np.empty(len(test_imgs))
    lat = []
    for i, p in enumerate(test_patch):
        t0 = time.perf_counter()
        test_scores[i] = nn_distance(p, bank).max()
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)

    auc = roc_auc(test_scores, test_labs)
    m = pr_at_threshold(test_scores, test_labs, thr)

    # ---- 6. per-class 检出率(缺陷件按真缺陷类型) ----
    test_kinds = [mt["kind"] for mt in test_metas]
    per_class = {}
    for kind in DEFECT_KINDS:
        idx = [i for i, k in enumerate(test_kinds) if k == kind]
        if idx:
            detected = int((test_scores[idx] >= thr).sum())
            per_class[kind] = {"n": len(idx), "detected": detected,
                               "recall": round(detected / len(idx), 4)}

    # ---- 7. 官方竞赛分(准确率20% + 检测时间30% + 方案完整度50%) ----
    # per-image 延时换算到 2500×2500 真分辨率的近似:本机 size 小,按面积线性外推作参考
    scale = (2500.0 / size) ** 2
    est_full_ms = float(lat.mean() * scale + feat_ms / len(test_imgs) * scale)
    comp = compute_competition_score(accuracy=m["accuracy"], latency_ms=est_full_ms,
                                     plan_completeness=0.9, budget_ms=2000.0)  # CPU 预算 2s

    result = {
        "backend": backend.name, "is_real_feature": is_real,
        "feat_dim": backend.feat_dim,
        "fewshot": {"n_normal_train": int((train_labs == 0).sum()),
                    "n_defect_train": int((train_labs == 1).sum()),
                    "n_test": len(test_imgs),
                    "n_test_defect": int((test_labs == 1).sum())},
        "bank_size": int(len(bank)),
        "threshold": round(float(thr), 6),
        "auc": round(float(auc), 4),
        "f1": round(float(m["f1"]), 4),
        "recall": round(float(m["recall"]), 4),
        "precision": round(float(m["precision"]), 4),
        "accuracy": round(float(m["accuracy"]), 4),
        "per_class_recall": per_class,
        "latency_ms_per_image_score": {"mean": round(float(lat.mean()), 4),
                                       "p95": round(float(np.percentile(lat, 95)), 4)},
        "est_latency_2500px_cpu_ms": round(est_full_ms, 1),
        "competition_score": {k: round(v, 4) for k, v in comp.items()},
    }

    if verbose:
        _print_report(result)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, report_name), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        np.savez(os.path.join(OUT_DIR, scores_name),
                 scores=test_scores, labels=test_labs,
                 kinds=np.array(test_kinds), threshold=thr)
        if verbose:
            print(f"\n✓ 产物:{os.path.abspath(os.path.join(OUT_DIR, report_name))}")
            print(f"✓ 产物:{os.path.abspath(os.path.join(OUT_DIR, scores_name))}")

    # 把中间量也带回(viz 复用,不落盘也能拿)
    result["_arrays"] = {"test_scores": test_scores, "test_labels": test_labs,
                         "test_kinds": test_kinds, "bank": bank, "backend": backend,
                         "test_patch": test_patch, "test_imgs": test_imgs,
                         "test_metas": test_metas}
    return result


def _gen_test(n_normal, n_defect, size, seed):
    """测试集:正常 + 4 类均匀缺陷,seed 段与训练不重叠。"""
    return gen_dataset(n_normal, n_defect, size, seed=seed)


def _print_report(r):
    print(f"  特征后端       : {r['backend']} (真特征={r['is_real_feature']}, D={r['feat_dim']})")
    fs = r["fewshot"]
    print(f"  少样本协议     : {fs['n_normal_train']}正+{fs['n_defect_train']}缺 建库/校准 → 测 {fs['n_test']} 张(缺陷 {fs['n_test_defect']})")
    print(f"  memory bank    : {r['bank_size']} 个 patch(coreset 压缩后)")
    print(f"  ── 真实指标 ──")
    print(f"  AUC            : {r['auc']}")
    print(f"  F1 / Recall    : {r['f1']} / {r['recall']}  (Precision {r['precision']}, Acc {r['accuracy']})")
    print(f"  阈值           : {r['threshold']}")
    print(f"  per-class 检出 :")
    for k, v in r["per_class_recall"].items():
        print(f"      {k:9s}: {v['detected']}/{v['n']}  recall={v['recall']}")
    lat = r["latency_ms_per_image_score"]
    print(f"  打分延时/图    : mean={lat['mean']}ms  p95={lat['p95']}ms")
    print(f"  2500px CPU 估算: {r['est_latency_2500px_cpu_ms']}ms")
    cs = r["competition_score"]
    print(f"  竞赛分(参考)  : {cs['competition_score']}  (方案{cs['plan_completeness']}*0.5 + 准确率{cs['accuracy']}*0.2 + 时延{cs['latency_score']}*0.3)")


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    # 小规模真跑(真实合成图 + 真实经典特征)。
    # 落盘到独立的 .selftest.* 文件,**不覆盖**标准规模交付物 pipeline_report.json
    # (后者由 `python run_real_pipeline.py` 生成,README 标注的指标即来自它)。
    rep = os.path.join(OUT_DIR, "pipeline_report.selftest.json")
    npz = os.path.join(OUT_DIR, "pipeline_scores.selftest.npz")
    r = run(n_normal=40, n_defect=20, n_test_normal=80, n_test_defect=60,
            size=128, grid=8, save=True, verbose=False,
            report_name="pipeline_report.selftest.json",
            scores_name="pipeline_scores.selftest.npz",
            prefer_real=False)
    check(not r["is_real_feature"], "自测强制走经典 CPU 特征(离线可复现)")
    check(r["auc"] > 0.85, f"真实合成图上 AUC>0.85(={r['auc']})")
    check(r["recall"] > 0.7, f"缺陷检出率>0.7(={r['recall']})")
    check(r["bank_size"] > 0, f"memory bank 非空(={r['bank_size']})")
    check(len(r["per_class_recall"]) == 4, f"4 类缺陷都有 per-class 检出({list(r['per_class_recall'])})")
    # 产物落盘且非空(自测专用文件,不动标准规模交付物)
    check(os.path.getsize(rep) > 0, "pipeline_report.selftest.json 已落盘且非空")
    check(os.path.getsize(npz) > 0, "pipeline_scores.selftest.npz 已落盘且非空")

    print(f"\n  小规模真实指标:AUC={r['auc']} F1={r['f1']} recall={r['recall']}")
    print("\n" + ("✅ run_real_pipeline 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--full", action="store_true", help="测试集放大到 1000+(贴合官方口径)")
    ap.add_argument("--real", action="store_true", help="启用 timm 真特征(需 torch/timm + 已缓存权重)")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.full:
        print("=== run_real_pipeline (FULL, 测试集 1000+) ===")
        run(n_normal=100, n_defect=30, n_test_normal=700, n_test_defect=320,
            size=160, grid=8, prefer_real=a.real)
    else:
        print("=== run_real_pipeline (标准规模) ===")
        run(n_normal=100, n_defect=30, n_test_normal=400, n_test_defect=200,
            size=160, grid=8, prefer_real=a.real)


if __name__ == "__main__":
    main()
