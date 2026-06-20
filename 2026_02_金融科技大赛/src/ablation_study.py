#!/usr/bin/env python3
"""
ablation_study.py · 赛题#23 官方明确要求的「消融实验」自动化框架

官方技术文档四要素之一是**消融实验**。本模块在真实合成像素(synth_images 产出)上,
对经典 CPU 特征 baseline 做系统消融,产出可直接进技术报告的对照表(CSV):

  ① 特征组成消融  : 只颜色直方图 / 只梯度(HOG式) / 只块统计 / 全拼接
                    → 验证各成分对去重 AUC 的贡献(回答"哪类特征最判别")
  ② PCA 维度消融  : pca_dim ∈ {0(不降维),16,32,64,128} → AUC/最优F1
                    → 验证 PCA 去相关降噪的收益与维度权衡
  ③ 标准化消融    : StandardScaler on/off
                    → 量化 features.py 默认 standardize=False 的影响(本合成集上落在噪声裕度内、
                      方向随 seed 翻转,故不宣称"必降 AUC";真实数据/CLIP 上应重跑定夺)

所有实验复用 features.py / similarity.py / metrics.py 的已测函数,纯 CPU、无 torch/数据依赖。
切到 CLIP 真特征后(算力机),把 backbone 换成 CLIP 嵌入再跑同一套框架即可对照(见 B 类上机包)。

用法:
  python ablation_study.py --selftest                      # 小规模自测(零回归校验)
  python ablation_study.py --out output/synth --csv ablation.csv   # 对已有合成影像跑全量消融
  python ablation_study.py --gen --n-groups 30             # 一键生成合成影像 + 跑消融
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _raw_component_features(paths, component):
    """对一批图像提取**单一成分**的原始特征(未标准化/未降维),用于特征组成消融。
    component ∈ {'color','grad','block','all'}。"""
    import numpy as np
    from features import (color_histogram, gradient_block_stats,
                          block_intensity_stats, _to_arrays, extract_single)
    rows = []
    for p in paths:
        if component == "all":
            rows.append(extract_single(p))
            continue
        rgb, gray = _to_arrays(p)
        if component == "color":
            rows.append(color_histogram(rgb))
        elif component == "grad":
            rows.append(gradient_block_stats(gray))
        elif component == "block":
            rows.append(block_intensity_stats(gray))
        else:
            raise ValueError(f"未知特征成分: {component}")
    return np.vstack(rows) if rows else np.zeros((0, 1))


def _auc_for_embeddings(embs, face_records, pairs):
    """给定嵌入 [N,D] 与相似度对真值,算去重 AUC + 最优 F1 阈值。"""
    from similarity import cosine_sim_matrix
    from metrics import roc_auc, best_threshold
    S = cosine_sim_matrix(embs)
    id2idx = {r["image_path"]: i for i, r in enumerate(face_records)}
    sc, lb = [], []
    for p in pairs:
        a = id2idx.get(p["img_a"]); b = id2idx.get(p["img_b"])
        if a is not None and b is not None:
            sc.append(float(S[a, b])); lb.append(p["label"])
    if not sc:
        return float("nan"), None
    return roc_auc(sc, lb), best_threshold(sc, lb, objective="f1")


def _prep(image_dir, seed=0):
    """加载面签照片 + 相似度对真值。返回 (face_records, face_paths, pairs)。"""
    from prepare_data import load_manifest, build_similarity_pairs
    manifest = os.path.join(image_dir, "manifest.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"未找到 {manifest};先用 synth_images.py 生成合成影像")
    records = load_manifest(manifest)
    face = [r for r in records if r["type_label"] == "面签照片"]
    paths = [os.path.join(image_dir, r["image_path"]) for r in face]
    pairs = build_similarity_pairs(face, neg_per_pos=2, only_face=False, seed=seed)
    return face, paths, pairs


def ablate_feature_composition(image_dir, pca_dim=32, seed=0):
    """消融①:逐成分 vs 全拼接。返回 [{'component','dim','auc','best_f1','best_thr'}, ...]。"""
    from features import ClassicFeatureExtractor
    import numpy as np
    face, paths, pairs = _prep(image_dir, seed)
    rows = []
    for comp in ("color", "grad", "block", "all"):
        raw = _raw_component_features(paths, comp)
        # 统一走 PCA 降维(与主流水线一致),保证维度可比
        fe = ClassicFeatureExtractor(pca_dim=pca_dim, standardize=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if pca_dim and raw.shape[0] >= 2 and raw.shape[1] >= 2:
                from sklearn.decomposition import PCA
                k = min(pca_dim, raw.shape[0], raw.shape[1])
                X = PCA(n_components=k, random_state=0).fit_transform(raw)
            else:
                X = raw
        auc, best = _auc_for_embeddings(X, face, pairs)
        rows.append({
            "experiment": "feature_composition", "component": comp,
            "dim": int(X.shape[1]), "auc": round(float(auc), 4),
            "best_f1": round(best["f1"], 4) if best else None,
            "best_thr": round(best["thr"], 4) if best else None,
        })
    return rows


def ablate_pca_dim(image_dir, dims=(0, 16, 32, 64, 128), seed=0):
    """消融②:PCA 维度扫描(0=不降维)。返回逐维度 AUC/最优F1。"""
    from features import extract_embeddings_classic
    import numpy as np
    face, paths, pairs = _prep(image_dir, seed)
    rows = []
    for d in dims:
        # pca_dim=0 → ClassicFeatureExtractor 跳过 PCA(原始拼接特征)
        embs = extract_embeddings_classic(paths, pca_dim=(d or 0))
        auc, best = _auc_for_embeddings(embs, face, pairs)
        rows.append({
            "experiment": "pca_dim", "pca_dim": d, "dim": int(embs.shape[1]),
            "auc": round(float(auc), 4),
            "best_f1": round(best["f1"], 4) if best else None,
            "best_thr": round(best["thr"], 4) if best else None,
        })
    return rows


def ablate_standardize(image_dir, pca_dim=32, seed=0):
    """消融③:StandardScaler on/off(量化标准化对去重 AUC 的影响;本合成集上差异在噪声裕度内、随 seed 翻转)。"""
    from features import ClassicFeatureExtractor
    face, paths, pairs = _prep(image_dir, seed)
    rows = []
    for std in (False, True):
        fe = ClassicFeatureExtractor(pca_dim=pca_dim, standardize=std)
        embs = fe.fit_transform(paths)
        auc, best = _auc_for_embeddings(embs, face, pairs)
        rows.append({
            "experiment": "standardize", "standardize": std,
            "dim": int(embs.shape[1]), "auc": round(float(auc), 4),
            "best_f1": round(best["f1"], 4) if best else None,
            "best_thr": round(best["thr"], 4) if best else None,
        })
    return rows


def run_all(image_dir, seed=0, verbose=True):
    """跑全部三组消融,返回扁平 rows 列表(可直接写 CSV / 进报告)。"""
    rows = []
    rows += ablate_feature_composition(image_dir, seed=seed)
    rows += ablate_pca_dim(image_dir, seed=seed)
    rows += ablate_standardize(image_dir, seed=seed)
    if verbose:
        _print_table(rows)
    return rows


def _print_table(rows):
    print("\n=== 消融实验结果(经典 CPU 特征 baseline,真实合成像素)===")
    cur = None
    for r in rows:
        if r["experiment"] != cur:
            cur = r["experiment"]
            print(f"\n[{cur}]")
        # 每组消融的可变维度:特征组成→component,维度扫描→pca_dim,标准化→standardize
        label = r.get("component", r.get("pca_dim", r.get("standardize")))
        print(f"  {str(label):>8}  dim={r['dim']:>4}  AUC={r['auc']:.4f}  "
              f"F1={r['best_f1']}  thr={r['best_thr']}")


def write_csv(rows, path):
    cols = ["experiment", "component", "pca_dim", "standardize", "dim",
            "auc", "best_f1", "best_thr"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return os.path.abspath(path)


def _selftest():
    import tempfile
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    with tempfile.TemporaryDirectory() as td:
        from synth_images import generate
        out = os.path.join(td, "synth")
        # 适度规模:足够让消融差异稳定显现,又能秒级跑完
        generate(out, n_groups=22, reuse_frac=0.5, seed=4, size=(128, 128))

        # ① 特征组成:每个成分单独都应可分(AUC>0.7),全拼接不差于最弱单成分
        comp = ablate_feature_composition(out, pca_dim=32, seed=0)
        by = {r["component"]: r["auc"] for r in comp}
        check(set(by) == {"color", "grad", "block", "all"}, "特征组成消融出 4 行(color/grad/block/all)")
        check(all(v > 0.7 for v in by.values()),
              f"各特征成分单独都可分 AUC>0.7({by})")
        check(by["all"] >= min(by["color"], by["grad"], by["block"]) - 1e-6,
              f"全拼接 AUC≥最弱单成分(all={by['all']}, min_single={min(by['color'],by['grad'],by['block'])})")

        # ② PCA 维度:全部应跑通且 AUC 有限;高维(64/128)不应明显劣于低维
        pca = ablate_pca_dim(out, dims=(0, 16, 32, 64), seed=0)
        check(len(pca) == 4 and all(np.isfinite(r["auc"]) for r in pca),
              f"PCA 维度消融 4 个维度全跑通且 AUC 有限")
        aucs = {r["pca_dim"]: r["auc"] for r in pca}
        check(all(v > 0.8 for v in aucs.values()), f"各 PCA 维度 AUC>0.8({aucs})")

        # ③ 标准化:on/off 两行都出,且 AUC 有限(佐证论断的数据基础)
        std = ablate_standardize(out, pca_dim=32, seed=0)
        check(len(std) == 2 and all(np.isfinite(r["auc"]) for r in std),
              f"标准化消融 on/off 两行,AUC 有限({[(r['standardize'], r['auc']) for r in std]})")

        # CSV 落盘非空且可回读
        rows = comp + pca + std
        csv_path = os.path.join(td, "ablation.csv")
        write_csv(rows, csv_path)
        check(os.path.getsize(csv_path) > 0, "消融结果 CSV 落盘非空")
        with open(csv_path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        check(n_lines == len(rows) + 1, f"CSV 行数=实验数+表头({n_lines}={len(rows)}+1)")

    print("\n" + ("✅ ablation_study 自测通过" if ok else "❌ ablation_study 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="赛题#23 消融实验自动化(经典特征 baseline)")
    ap.add_argument("--selftest", action="store_true", help="小规模自测(无需数据)")
    ap.add_argument("--out", help="已有合成影像目录(含 manifest.csv)")
    ap.add_argument("--gen", action="store_true", help="先生成合成影像再跑消融")
    ap.add_argument("--n-groups", type=int, default=30)
    ap.add_argument("--reuse-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="ablation.csv", help="结果 CSV 输出路径")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    image_dir = a.out
    if a.gen:
        from synth_images import generate
        image_dir = a.out or "output/synth_ablation"
        print(f"=== 生成合成影像 → {image_dir} ===")
        generate(image_dir, n_groups=a.n_groups, reuse_frac=a.reuse_frac, seed=a.seed)
    if not image_dir:
        ap.error("需 --out 指定合成影像目录,或 --gen 一键生成,或 --selftest")
    rows = run_all(image_dir, seed=a.seed, verbose=True)
    path = write_csv(rows, a.csv)
    print(f"\n📄 消融结果已写入: {path}")


if __name__ == "__main__":
    main()
