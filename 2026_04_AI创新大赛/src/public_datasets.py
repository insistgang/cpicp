#!/usr/bin/env python3
"""
public_datasets.py · DAGM/MVTec AD 公共数据集 manifest 适配器

把公开工业异常检测数据集统一转换成本项目使用的 manifest:
  image_path,label,defect_type,dataset,category,split

MVTec AD 常见结构:
  <root>/<category>/train/good/*.png
  <root>/<category>/test/good/*.png
  <root>/<category>/test/<defect_type>/*.png

DAGM 2007 常见结构:
  <root>/Class1/Train/*.PNG
  <root>/Class1/Train/Label/*.PNG  # mask/label 存在则判为缺陷

脚本不下载数据,只做本地目录扫描;数据集协议/登录下载由队长手动完成。
"""
import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_DIR_NAMES = {"ground_truth", "gt", "mask", "masks", "label", "labels", "annotations"}
GOOD_TOKENS = {"good", "normal", "ok", "negative", "nondefective", "non-defective"}
BAD_TOKENS = {"defect", "defective", "bad", "anomaly", "positive", "ng", "fault"}


def is_image(path):
    return Path(path).suffix.lower() in IMG_EXTS


def is_mask_dir(path):
    parts = {p.lower() for p in Path(path).parts}
    return bool(parts & MASK_DIR_NAMES)


def write_manifest(records, path):
    fields = ["image_path", "label", "defect_type", "dataset", "category", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def build_mvtec_manifest(root):
    """扫描 MVTec AD 根目录或单个类别目录,返回 manifest records。"""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    category_dirs = [root] if (root / "train").exists() or (root / "test").exists() else [
        p for p in root.iterdir() if p.is_dir() and ((p / "train").exists() or (p / "test").exists())
    ]

    records = []
    for cat_dir in sorted(category_dirs):
        category = cat_dir.name
        for split in ("train", "test"):
            split_dir = cat_dir / split
            if not split_dir.exists():
                continue
            for img in sorted(split_dir.rglob("*")):
                if not img.is_file() or not is_image(img):
                    continue
                rel_parts = img.relative_to(split_dir).parts
                subtype = rel_parts[0] if len(rel_parts) > 1 else img.parent.name
                label = "0" if subtype.lower() in GOOD_TOKENS else "1"
                records.append({
                    "image_path": str(img),
                    "label": label,
                    "defect_type": "" if label == "0" else f"{category}:{subtype}",
                    "dataset": "mvtec_ad",
                    "category": category,
                    "split": split,
                })
    return records


def _norm_stem(stem):
    s = stem.lower()
    s = re.sub(r"(image|img|label|mask|gt|defect|defective)", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _collect_mask_keys(root):
    keys = set()
    for p in Path(root).rglob("*"):
        if p.is_file() and is_image(p) and is_mask_dir(p.parent):
            keys.add(_norm_stem(p.stem))
    return keys


def _infer_dagm_category(path, root):
    rel = Path(path).relative_to(root)
    for part in rel.parts:
        if part.lower().startswith("class"):
            return part
    return rel.parts[0] if len(rel.parts) > 1 else Path(root).name


def build_dagm_manifest(root):
    """扫描 DAGM 2007 根目录,用 mask/label 或路径 token 推断缺陷标签。"""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    mask_keys = _collect_mask_keys(root)
    records = []
    for img in sorted(root.rglob("*")):
        if not img.is_file() or not is_image(img) or is_mask_dir(img.parent):
            continue
        rel_parts = [p.lower() for p in img.relative_to(root).parts]
        rel_tokens = set(rel_parts)
        for part in rel_parts:
            rel_tokens.update(t for t in re.split(r"[^a-z0-9]+", part) if t)
        has_mask = _norm_stem(img.stem) in mask_keys
        token_defect = bool(rel_tokens & BAD_TOKENS)
        token_good = bool(rel_tokens & GOOD_TOKENS)
        label = "1" if has_mask or (token_defect and not token_good) else "0"
        category = _infer_dagm_category(img, root)
        split = "test" if "test" in rel_tokens else "train" if "train" in rel_tokens else ""
        records.append({
            "image_path": str(img),
            "label": label,
            "defect_type": "" if label == "0" else f"{category}:defect",
            "dataset": "dagm2007",
            "category": category,
            "split": split,
        })
    return records


def summarize(records):
    total = len(records)
    defects = sum(1 for r in records if str(r["label"]) == "1")
    normals = total - defects
    cats = sorted({r.get("category", "") for r in records if r.get("category", "")})
    return {"total": total, "normal": normals, "defect": defects, "categories": cats}


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _selftest():
    import sys
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✅ " if cond else "  ❌ ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        mv = base / "mvtec"
        _touch(mv / "bottle" / "train" / "good" / "a.png")
        _touch(mv / "bottle" / "test" / "good" / "b.png")
        _touch(mv / "bottle" / "test" / "scratch" / "c.png")
        _touch(mv / "bottle" / "ground_truth" / "scratch" / "c_mask.png")
        mv_recs = build_mvtec_manifest(mv)
        check(len(mv_recs) == 3, "MVTec:只收 train/test 图片,不把 ground_truth mask 当样本")
        check(sum(r["label"] == "0" for r in mv_recs) == 2, "MVTec:good 判正常")
        check(sum(r["label"] == "1" for r in mv_recs) == 1, "MVTec:test 缺陷子类判异常")
        check(any(r["defect_type"] == "bottle:scratch" for r in mv_recs), "MVTec:缺陷类型含 category/subtype")

        dg = base / "dagm"
        _touch(dg / "Class1" / "Train" / "Image_001.PNG")
        _touch(dg / "Class1" / "Train" / "Image_002.PNG")
        _touch(dg / "Class1" / "Train" / "Label" / "Label_002.PNG")
        dg_recs = build_dagm_manifest(dg)
        check(len(dg_recs) == 2, "DAGM:跳过 Label/mask 目录")
        check(sum(r["label"] == "0" for r in dg_recs) == 1, "DAGM:无 mask 图片判正常")
        check(sum(r["label"] == "1" for r in dg_recs) == 1, "DAGM:有同 stem mask 图片判缺陷")

        out = base / "manifest.csv"
        write_manifest(mv_recs + dg_recs, out)
        check(out.exists() and out.stat().st_size > 0, "manifest 可写出")

    print("\n" + ("✅ public_datasets 自测通过" if ok else "❌ public_datasets 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mvtec", help="MVTec AD 根目录或单个类别目录")
    ap.add_argument("--dagm", help="DAGM 2007 根目录")
    ap.add_argument("--out", default="../output/public_manifest.csv")
    a = ap.parse_args()

    if a.selftest:
        _selftest()

    records = []
    if a.mvtec:
        records.extend(build_mvtec_manifest(a.mvtec))
    if a.dagm:
        records.extend(build_dagm_manifest(a.dagm))
    if not records:
        ap.error("需提供 --mvtec 或 --dagm,或运行 --selftest")

    write_manifest(records, a.out)
    s = summarize(records)
    print(f"✓ manifest -> {os.path.abspath(a.out)}")
    print(f"  total={s['total']} normal={s['normal']} defect={s['defect']} categories={len(s['categories'])}")
    if s["categories"]:
        print("  categories=" + ", ".join(s["categories"][:20]) + (" ..." if len(s["categories"]) > 20 else ""))


if __name__ == "__main__":
    main()
