#!/usr/bin/env python3
"""
aoi_prepare.py · AOI 数据准备(华为赛题一少样本切分)

读 manifest(image_path,label,defect_type),按官方协议切出
  few-shot 训练集 = 100 张正常 + 30 张缺陷;测试集 = 其余(目标 1000+)。
含合成清单生成器(无真实数据可联调)。纯标准库,`python aoi_prepare.py --selftest`。
"""
import argparse
import csv
import random
from collections import Counter

DEFECT_TYPES = ("尺寸偏差", "缺件少件", "逻辑顺序错误", "色变", "外观缺陷")


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def split_fewshot(records, n_normal=100, n_defect=30, seed=0):
    """正常 label=0,缺陷 label=1。返回 (train, test)。train=100正+30缺(官方口径)。"""
    rng = random.Random(seed)
    normal = [r for r in records if str(r["label"]) == "0"]
    defect = [r for r in records if str(r["label"]) == "1"]
    rng.shuffle(normal); rng.shuffle(defect)
    train = normal[:n_normal] + defect[:n_defect]
    test = normal[n_normal:] + defect[n_defect:]
    rng.shuffle(test)
    return train, test


def defect_distribution(records):
    return dict(Counter(r.get("defect_type", "") for r in records if str(r["label"]) == "1"))


def synth_manifest(n_normal=900, n_defect=330, seed=0):
    rng = random.Random(seed)
    recs = [{"image_path": f"img_{i:05d}.png", "label": "0", "defect_type": ""} for i in range(n_normal)]
    for j in range(n_defect):
        recs.append({"image_path": f"def_{j:05d}.png", "label": "1",
                     "defect_type": rng.choice(DEFECT_TYPES)})
    rng.shuffle(recs)
    return recs


def write_manifest(records, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "defect_type"])
        w.writeheader(); w.writerows(records)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    recs = synth_manifest(n_normal=900, n_defect=330)
    train, test = split_fewshot(recs)
    n_tr_normal = sum(1 for r in train if r["label"] == "0")
    n_tr_defect = sum(1 for r in train if r["label"] == "1")
    check(n_tr_normal == 100 and n_tr_defect == 30, f"训练集=100正+30缺(={n_tr_normal}+{n_tr_defect})")
    check(len(test) >= 1000, f"测试集≥1000(={len(test)})")
    check(len(train) + len(test) == len(recs), "切分无重叠无遗漏")
    dist = defect_distribution(recs)
    check(set(dist) <= set(DEFECT_TYPES) and sum(dist.values()) == 330, f"缺陷类型分布:{dist}")

    print("\n" + ("✅ aoi_prepare 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--gen-synth")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.gen_synth:
        write_manifest(synth_manifest(), a.gen_synth); print(f"✓ 合成清单→{a.gen_synth}"); return
    if not a.manifest:
        ap.error("需 --manifest 或 --selftest 或 --gen-synth")
    recs = load_manifest(a.manifest)
    train, test = split_fewshot(recs)
    print(f"✓ few-shot 训练 {len(train)}(100正+30缺)/ 测试 {len(test)};缺陷分布 {defect_distribution(recs)}")


if __name__ == "__main__":
    main()
