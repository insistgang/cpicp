#!/usr/bin/env python3
"""
embed.py · 第二步:面签照片向量化(度量学习)(赛题#23 步骤2)

把筛出的面签照片编码成**判别性向量**,使"同一张/套用"的照片向量距离近、不同的远。
两档:
  ① 直接用 CLIP 图像特征(零训练基线,官方现有基础就是这个)。
  ② 度量学习微调:在 CLIP/DINOv2 特征上加投影头,用相似度对(prepare_data.build_similarity_pairs)
     做对比学习(InfoNCE)/三元组,提升"细微差异套用"的判别力。
输出 embeddings.npy + ids.json,直接喂 similarity.py 做去重检测。

⚠️ 需 torch(+open_clip);py_compile 可过(重依赖在函数内导入)。无数据时可用合成嵌入联调 similarity。
"""
import argparse
import json


def extract_embeddings(image_paths, backbone="clip", model_pack=None, batch=32):
    """返回 np.ndarray [N, D](L2 未归一化,交给 similarity 归一)。"""
    import torch, numpy as np
    from PIL import Image
    if backbone == "clip":
        from classify import load_clip
        model, preprocess, _, device = model_pack or load_clip()
        encode = lambda x: model.encode_image(x)
    else:  # timm DINOv2/ViT
        import timm
        from timm.data import resolve_data_config, create_transform
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(backbone, pretrained=True, num_classes=0).to(device).eval()
        preprocess = create_transform(**resolve_data_config({}, model=model))
        encode = lambda x: model(x)
    embs = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i:i + batch]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            f = encode(imgs)
            embs.append(f.cpu().numpy())
    return np.concatenate(embs, 0)


class ProjectionHead:
    """度量学习投影头训练(InfoNCE 对比学习)。scaffold:接真实数据后在 4090 上跑。"""

    def train(self, base_embs, pair_index, labels, dim=128, epochs=20, lr=1e-3):  # pragma: no cover
        import torch, torch.nn as nn, numpy as np
        X = torch.tensor(base_embs, dtype=torch.float32)
        head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Linear(512, dim))
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        ia = torch.tensor([p[0] for p in pair_index]); ib = torch.tensor([p[1] for p in pair_index])
        y = torch.tensor(labels, dtype=torch.float32)
        for ep in range(epochs):
            z = nn.functional.normalize(head(X), dim=1)
            sim = (z[ia] * z[ib]).sum(1)                 # 余弦
            loss = nn.functional.binary_cross_entropy_with_logits(sim * 10, y)  # 正样本拉近/负样本推远
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                print(f"  ep{ep} loss={loss.item():.4f}")
        self.head = head
        return self

    def transform(self, base_embs):  # pragma: no cover
        import torch, torch.nn as nn
        with torch.no_grad():
            z = self.head(torch.tensor(base_embs, dtype=torch.float32))
            return nn.functional.normalize(z, dim=1).numpy()


def save(embs, image_ids, customer_ids, out_prefix="embeddings"):
    import numpy as np
    np.save(out_prefix + ".npy", embs)
    json.dump({"image_ids": list(image_ids), "customer_ids": list(customer_ids)},
              open(out_prefix + ".json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✓ 保存 {embs.shape} → {out_prefix}.npy / .json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--out", default="embeddings")
    a = ap.parse_args()
    if not a.images:
        ap.error("需 --images(算力机上运行)")
    embs = extract_embeddings(a.images, a.backbone)
    save(embs, a.images, ["?"] * len(a.images), a.out)
    print("  下一步: python similarity.py 用这些嵌入做去重检测")


if __name__ == "__main__":
    main()
