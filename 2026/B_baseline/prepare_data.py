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

TODO[登录核对]: 赛题真实类别确认后，更新 CLASS_MAP 与 configs/searescue.yaml 的 names。
"""
import argparse
import json
import shutil
from pathlib import Path

# SeaDronesSee/AFO 公开类别 → 统一类别 id（占位，见 TODO）
CLASS_MAP = {
    "swimmer": 0, "person": 0, "human": 0,
    "floater": 1,
    "boat": 2, "sailboat": 2, "kayak": 2,
    "buoy": 3,
    "life_jacket": 4, "life jacket": 4,
}

GUIDE = """
================ 数据下载指引（手动） ================
[1] SeaDronesSee（海上搜救基准，COCO 标注）
    主页:  https://seadronessee.cs.uni-tuebingen.de/
    代码:  https://github.com/Ben93kie/SeaDronesSee
    下载 Object Detection v2 的 images + annotations(COCO json)，放到:
      {root}/raw/seadronessee/{images, annotations}

[2] AFO（航拍漂浮目标，推荐 Roboflow 导出 YOLOv8 格式，免转换）
    Roboflow: https://universe.roboflow.com/large-benchmark-datasets/afo-aerial-dataset-of-floating-object
    选择 "YOLOv8" 格式导出，解压到:
      {root}/raw/afo/{train,valid,test}/{images,labels}

[3] 陆域 source（用于跨域迁移评价，可选其一）
    - 你的无人机违建数据集（最贴合你的论文），或
    - VisDrone (https://github.com/VisDrone/VisDrone-Dataset) 取 person/people 类
    放到: {root}/raw/source_land/{images,labels}（YOLO 格式）
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
    n = 0
    for img_id, im in imgs.items():
        w, h = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(img_id, []):
            name = cats.get(a["category_id"], "")
            if name not in CLASS_MAP:
                continue  # 跳过 CLASS_MAP 外的类
            cls = CLASS_MAP[name]
            x, y, bw, bh = a["bbox"]  # COCO: x,y,w,h (左上)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        # 即使无目标也写空 txt（负样本对小目标训练有用）
        (out_lbl / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        src = img_dir / im["file_name"]
        if src.exists():
            shutil.copy(src, out_img / Path(im["file_name"]).name)
            n += 1
    print(f"  COCO→YOLO: {coco_json.name} 处理 {n} 张")


def convert(root: Path):
    """组装成 datasets/searescue/{images,labels}/{train,val,test}（海域 target）。"""
    out = root / "searescue"
    # --- AFO: Roboflow 已是 YOLO，直接合并 ---
    afo = root / "raw" / "afo"
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
                shutil.copy(f, dl / f.name)
            print(f"  AFO {src_split}→{dst_split} 合并完成")
            # TODO: AFO 的类别索引与 CLASS_MAP 不一定一致 → 需按 AFO data.yaml 重映射 labels
            print("  TODO: 核对 AFO 类别索引并按 CLASS_MAP 重映射 labels（见脚本注释）")
    # --- SeaDronesSee: COCO → YOLO ---
    sds = root / "raw" / "seadronessee"
    ann = sds / "annotations"
    if ann.exists():
        for js, split in [("instances_train.json", "train"), ("instances_val.json", "val")]:
            jp = ann / js
            if jp.exists():
                coco_to_yolo(jp, sds / "images" / split,
                             out / "images" / split, out / "labels" / split)
    print(f"\n✓ 海域(target)数据集就绪: {out}")
    print("  下一步: python train.py --data configs/searescue.yaml --p2")
    print("  陆域(source)数据置于 raw/source_land/，供 crossdomain_eval.py 使用")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets", type=Path)
    ap.add_argument("--step", choices=["guide", "convert"], default="guide")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.step == "guide":
        print(GUIDE.format(root=a.root))
    else:
        print(GUIDE.format(root=a.root))
        convert(a.root)


if __name__ == "__main__":
    main()
