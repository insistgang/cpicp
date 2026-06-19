#!/usr/bin/env python3
"""
prepare_data.py · 赛题#23 数据准备:清单加载 + 分类集/相似度对构建 + 合成数据生成器

数据清单(manifest)约定(对齐官方"每张影像标注 类型标签+相似度标注+业务标签"):
  CSV 列: image_path, type_label, group_id, customer_id, business_label
    - type_label   : 影像类型(面签合影/证件/权属证明/合同文档…);"面签照片"是分类要筛的正类
    - group_id     : 相同/重复影像归同一 group(相似度任务的正样本来源;同 group = 应判为高相似)
    - customer_id  : 所属客户(用于区分"同客户重复" vs "跨客户套用")
    - business_label: 信贷产品线/业务类型(跨场景复用要求)

用法:
  python prepare_data.py --selftest                      # 合成数据跑通流水线(无需真实数据)
  python prepare_data.py --manifest data/manifest.csv --out data/prepared
"""
import argparse
import csv
import itertools
import random

FACE_SIGN = "面签照片"   # 分类正类


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def build_classification(records):
    """二分类:面签照片(1) vs 其他(0)。返回 [(image_path, label), ...]。"""
    return [(r["image_path"], 1 if r["type_label"] == FACE_SIGN else 0) for r in records]


def build_similarity_pairs(records, neg_per_pos=2, only_face=True, seed=0):
    """相似度对:同 group_id 为正样本(label1),跨 group 为负样本(label0)。
    only_face=True 时仅用面签照片(任务第二步只对筛出的面签照做相似度)。
    每个正样本配 neg_per_pos 个负样本。正样本附 cross_customer 标记(跨客户套用)。"""
    rng = random.Random(seed)
    recs = [r for r in records if (not only_face or r["type_label"] == FACE_SIGN)]
    groups = {}
    for i, r in enumerate(recs):
        groups.setdefault(r["group_id"], []).append(i)

    pos = []
    for gid, idxs in groups.items():
        for a, b in itertools.combinations(idxs, 2):
            pos.append((a, b))
    neg = []
    n_neg_target = len(pos) * neg_per_pos
    gids = list(groups.keys())
    tries = 0
    while len(neg) < n_neg_target and tries < n_neg_target * 20 and len(gids) > 1:
        tries += 1
        g1, g2 = rng.sample(gids, 2)
        a = rng.choice(groups[g1]); b = rng.choice(groups[g2])
        neg.append((a, b))

    def mk(a, b, label):
        return {
            "img_a": recs[a]["image_path"], "img_b": recs[b]["image_path"],
            "cust_a": recs[a]["customer_id"], "cust_b": recs[b]["customer_id"],
            "label": label,
            "cross_customer": label == 1 and recs[a]["customer_id"] != recs[b]["customer_id"],
        }

    pairs = [mk(a, b, 1) for a, b in pos] + [mk(a, b, 0) for a, b in neg]
    return pairs


def synth_manifest(n_groups=20, types=("面签照片", "证件", "权属证明", "合同文档"),
                   reuse_frac=0.3, seed=0):
    """生成合成清单:模拟多类型影像,部分 group 被跨客户套用(reuse_frac)。返回 records 列表。"""
    rng = random.Random(seed)
    records = []
    img = 0
    for g in range(n_groups):
        gid = f"G{g}"
        t = rng.choice(types)
        base_cust = f"C{rng.randint(0, 50)}"
        k = rng.randint(2, 4)            # 同组若干张(重复/相似)
        cross = rng.random() < reuse_frac
        for m in range(k):
            # 被套用的组:部分成员换成不同客户(跨客户套用)
            cust = f"C{rng.randint(51, 99)}" if (cross and m > 0) else base_cust
            records.append({
                "image_path": f"img_{img:05d}.jpg", "type_label": t,
                "group_id": gid, "customer_id": cust,
                "business_label": rng.choice(["微贷", "经营贷", "按揭", "消费贷"]),
            })
            img += 1
    # 加一批各自独立(无重复)的影像,贴近真实分布
    for s in range(n_groups):
        records.append({
            "image_path": f"img_{img:05d}.jpg", "type_label": rng.choice(types),
            "group_id": f"S{s}", "customer_id": f"C{rng.randint(0, 99)}",
            "business_label": rng.choice(["微贷", "经营贷", "按揭", "消费贷"]),
        })
        img += 1
    rng.shuffle(records)
    return records


def write_manifest(records, path):
    cols = ["image_path", "type_label", "group_id", "customer_id", "business_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(records)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    recs = synth_manifest(n_groups=25, reuse_frac=0.4, seed=1)
    check(len(recs) > 0, f"合成清单 {len(recs)} 条影像")

    # 分类集
    clf = build_classification(recs)
    n_face = sum(lbl for _, lbl in clf)
    check(n_face > 0 and n_face < len(clf), f"分类集:面签 {n_face}/{len(clf)}(正负都有)")

    # 相似度对
    pairs = build_similarity_pairs(recs, neg_per_pos=2, only_face=False, seed=1)
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    check(len(pos) > 0 and len(neg) > 0, f"相似度对:正 {len(pos)} / 负 {len(neg)}")

    # 正样本必同 group(用 image→group 反查验证)
    img2g = {r["image_path"]: r["group_id"] for r in recs}
    check(all(img2g[p["img_a"]] == img2g[p["img_b"]] for p in pos), "所有正样本对同 group_id")
    check(all(img2g[p["img_a"]] != img2g[p["img_b"]] for p in neg), "所有负样本对跨 group_id")

    # 应至少构造出"跨客户套用"的正样本(命题方核心诉求)
    cross = [p for p in pos if p["cross_customer"]]
    check(len(cross) > 0, f"含跨客户套用正样本 {len(cross)} 对(可训/可评)")

    # 只用面签:数量应≤全量
    face_pairs = build_similarity_pairs(recs, only_face=True, seed=1)
    check(len(face_pairs) <= len(pairs), f"only_face 子集 {len(face_pairs)}≤全量 {len(pairs)}")

    print("\n" + ("✅ prepare_data 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--gen-synth", help="生成合成清单到指定 csv 路径(供无数据时联调)")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.gen_synth:
        write_manifest(synth_manifest(), a.gen_synth)
        print(f"✓ 合成清单已写入 {a.gen_synth}")
        return
    if not a.manifest:
        ap.error("需 --manifest 或 --selftest 或 --gen-synth")
    recs = load_manifest(a.manifest)
    clf = build_classification(recs)
    pairs = build_similarity_pairs(recs)
    print(f"✓ 分类集 {len(clf)} 条(面签 {sum(l for _,l in clf)});相似度对 {len(pairs)} "
          f"(正 {sum(p['label'] for p in pairs)})")


if __name__ == "__main__":
    main()
