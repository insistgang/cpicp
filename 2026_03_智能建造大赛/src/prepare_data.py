#!/usr/bin/env python3
"""
prepare_data.py · SeaDronesSee + AFO 下载指引 + 转 YOLO 格式 + 陆/海域划分

用法:
  python prepare_data.py --root ./datasets --step guide      # 只打印下载指引
  python prepare_data.py --root ./datasets --step convert    # 转 YOLO 格式 + 合并 + 划分

设计说明:
  - 大数据集需手动按指引下载(版权/登录限制)，本脚本负责"转换+划分+目录组装"。
  - AFO 推荐直接从 Roboflow 导出 YOLOv8 格式(已是 images/labels 结构)，省去转换。
  - SeaDronesSee 标注为 COCO json，提供 coco→yolo 转换函数。
  - 跨域迁移评价(crossdomain_eval.py)需要"陆域(source)"与"海域(target)"两个域，
    本脚本把 AFO/SeaDronesSee 标为 target(海域)；source(陆域，如违建/VisDrone) 由你另置。

官方核实：赛题7 识别目标为「落水人员 / 船只 / 浮标」三类（《参赛指南》P80）。
CLASS_MAP 已据此定稿，与 configs/searescue.yaml 的 names 一致。
"""
import argparse
import json
import shutil
from pathlib import Path

# 公开数据集类别名 → 官方三类 id（落水人员=0 / 船只=1 / 浮标=2）
# SeaDronesSee 的 swimmer/floater 统一并入「落水人员」；life_jacket 不在官方目标内（丢弃）。
CLASS_MAP = {
    "swimmer": 0, "person": 0, "human": 0, "floater": 0,
    "person in water": 0, "drowning": 0,
    "boat": 1, "sailboat": 1, "kayak": 1, "ship": 1, "vessel": 1,
    "buoy": 2,
    # life_jacket / life jacket：官方三类外，不映射
}


def load_dataset_names(yaml_path: Path):
    """读取某数据集的 data.yaml 的 names（list 或 dict）→ {id: name_lower}。无则 None。"""
    if not yaml_path.exists():
        return None
    import yaml
    d = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v).lower() for i, v in enumerate(names)}
    return None


def remap_label_text(text: str, src_names: dict) -> str:
    """把 YOLO label 每行首列(源类别id) 按 src_names→CLASS_MAP 重映射；官方三类外的行丢弃。"""
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            sid = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        name = src_names.get(sid, "")
        if name not in CLASS_MAP:
            continue
        parts[0] = str(CLASS_MAP[name])
        out.append(" ".join(parts))
    return "\n".join(out)

GUIDE = r"""
================ 数据下载指引（手动） ================
[1] SeaDronesSee（海上搜救基准，COCO 标注）
    主页:  https://seadronessee.cs.uni-tuebingen.de/
    代码:  https://github.com/Ben93kie/SeaDronesSee
    下载 Object Detection v2 的 images + annotations(COCO json)，放到:
      {root}/raw/seadronessee/images, annotations

[2] AFO（航拍漂浮目标，推荐 Roboflow 导出 YOLOv8 格式，免转换）
    Roboflow: https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object
    选择 "YOLOv8" 格式导出，解压到:
      {root}/raw/afo/train,valid,test/images,labels

[3] 陆域 source（用于跨域迁移评价，可选其一）
    - 你的无人机违建数据集（最贴合你的论文），或
    - VisDrone (https://github.com/VisDrone/VisDrone-Dataset) 取 person/people 类
    放到: {root}/raw/source_land/images,labels（YOLO 格式）
=====================================================
"""


def coco_to_yolo(coco_json: Path, img_dir: Path, out_img: Path, out_lbl: Path):
    """SeaDronesSee COCO json → YOLO txt。仅映射 CLASS_MAP 中的类别。"""
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
    imgs = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    n, skipped = 0, 0
    for img_id, im in imgs.items():
        src = img_dir / im["file_name"]
        if not src.exists():
            skipped += 1   # 找不到图就不写 label，避免"有标签无图"污染数据集
            continue
        w, h = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(img_id, []):
            name = cats.get(a["category_id"], "")
            if name not in CLASS_MAP:
                continue  # 跳过官方三类外的类
            cls = CLASS_MAP[name]
            x, y, bw, bh = a["bbox"]  # COCO: x,y,w,h (左上)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        # 图存在才写：空 txt 即负样本（对小目标训练有用）
        (out_lbl / (Path(im["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines), encoding="utf-8")
        shutil.copy(src, out_img / Path(im["file_name"]).name)
        n += 1
    msg = f"  COCO→YOLO: {coco_json.name} 处理 {n} 张"
    if skipped:
        msg += f"（跳过 {skipped} 张：在 {img_dir} 找不到对应图，请核对 images 目录约定）"
    print(msg)


def convert(root: Path):
    """组装成 datasets/searescue/{images,labels}/{train,val,test}（海域 target）。"""
    out = root / "searescue"
    # --- AFO: Roboflow 已是 YOLO，直接合并 ---
    afo = root / "raw" / "afo"
    afo_names = load_dataset_names(afo / "data.yaml")  # Roboflow 导出含 data.yaml
    if (afo / "train" / "images").exists() and afo_names is None:
        print("  [AFO][WARN] 未找到 raw/afo/data.yaml，无法核对类别索引 → 标签将原样拷贝，"
              "类别可能与官方三类错位！请提供 data.yaml 后重跑。")
    split_map = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in split_map.items():
        si = afo / src_split / "images"
        sl = afo / src_split / "labels"
        if si.exists():
            di = out / "images" / dst_split
            dl = out / "labels" / dst_split
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            for f in si.glob("*"):
                shutil.copy(f, di / f.name)
            for f in sl.glob("*.txt"):
                if afo_names:   # 按 AFO data.yaml → CLASS_MAP 重映射首列，丢弃官方三类外目标
                    (dl / f.name).write_text(
                        remap_label_text(f.read_text(encoding="utf-8"), afo_names),
                        encoding="utf-8")
                else:
                    shutil.copy(f, dl / f.name)
            tag = "已按 data.yaml 重映射到官方三类" if afo_names else "原样拷贝(未重映射)"
            print(f"  AFO {src_split}→{dst_split} 合并完成（{tag}）")
    # --- SeaDronesSee: COCO → YOLO ---
    sds = root / "raw" / "seadronessee"
    ann = sds / "annotations"
    if ann.exists():
        for js, split in [("instances_train.json", "train"), ("instances_val.json", "val")]:
            jp = ann / js
            if jp.exists():
                img_dir = sds / "images" / split        # 优先 images/{split}
                if not img_dir.exists():
                    img_dir = sds / "images"            # 兼容扁平 images/ 目录
                coco_to_yolo(jp, img_dir,
                             out / "images" / split, out / "labels" / split)
    print(f"\n[OK] 海域(target)数据集就绪: {out}")
    print("  下一步: python train.py --data configs/searescue.yaml --p2")
    print("  陆域(source)数据置于 raw/source_land/，供 crossdomain_eval.py 使用")


def add_negatives(root: Path, neg_dir: Path, ratio=0.15, split="train"):
    """把纯水面/反光/碎浪/泡沫等**无目标帧**拷入 images/{split} 并写空 txt(负样本)。
    救援场景 recall 优先,但负样本太多会让模型变保守漏检真人 → 数量上限 = ratio×当前正样本数(默认15%)。"""
    out = root / "searescue"
    di, dl = out / "images" / split, out / "labels" / split
    if not di.exists():
        print(f"  [add_negatives] 跳过:{di} 不存在,请先 --step convert")
        return 0
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    pos = len([f for f in di.glob("*") if f.suffix.lower() in exts])
    cap = max(1, int(pos * ratio))
    negs = [f for f in Path(neg_dir).glob("*") if f.suffix.lower() in exts]
    dl.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in negs[:cap]:
        shutil.copy(f, di / f.name)
        (dl / (f.stem + ".txt")).write_text("", encoding="utf-8")  # 空 txt = 负样本
        n += 1
    print(f"  add_negatives: 加入 {n} 张负样本(上限 {cap}={ratio:.0%}×{pos} 正样本) → {split}")
    if len(negs) > cap:
        print(f"    (源目录有 {len(negs)} 张,按比例只取 {cap} 张;调 --neg-ratio 可放宽)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets", type=Path)
    ap.add_argument("--step", choices=["guide", "convert"], default="guide")
    ap.add_argument("--neg-dir", type=Path, help="负样本(纯水面/反光/碎浪)目录,convert 后按比例加入")
    ap.add_argument("--neg-ratio", type=float, default=0.15, help="负样本占正样本的比例上限")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.step == "guide":
        print(GUIDE.format(root=a.root))
    else:
        print(GUIDE.format(root=a.root))
        convert(a.root)
        if a.neg_dir:
            add_negatives(a.root, a.neg_dir, a.neg_ratio)


if __name__ == "__main__":
    main()
