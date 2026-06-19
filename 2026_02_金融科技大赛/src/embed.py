#!/usr/bin/env python3
"""
embed.py · 第二步:面签照片向量化(度量学习)(赛题#23 步骤2)

把筛出的面签照片编码成**判别性向量**,使"同一张/套用"的照片向量距离近、不同的远。
两档:
  ① 直接用 CLIP 图像特征(零训练基线,官方现有基础就是这个)。
  ② 度量学习微调:在 CLIP/DINOv2 特征上加投影头,用相似度对(prepare_data.build_similarity_pairs)
     做对比学习(InfoNCE)/三元组,提升"细微差异套用"的判别力。
输出 embeddings.npy + ids.json,直接喂 similarity.py 做去重检测。

回退策略:`extract_embeddings` 自动检测 torch。torch 可用→走 CLIP/timm 真特征;
torch 不可用(Mac 无 GPU/离线)→**优雅回退到 features.py 的经典 CPU 特征**
(PIL 颜色直方图 + 梯度/分块统计 + sklearn 标准化/PCA),让整条流水线在**真实图像像素**
上端到端跑通(baseline),而非随机向量。真特征接口(extract_embeddings_clip)保留。
"""
import argparse
import json


def torch_available():
    """检测 torch(+open_clip)是否可用。Mac 无 GPU/离线环境通常 False → 走经典特征回退。"""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def extract_embeddings_clip(image_paths, backbone="clip", model_pack=None, batch=32):
    """真特征路径:CLIP / timm 主干提取(需 torch)。返回 [N, D](未归一化)。"""
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


def extract_embeddings(image_paths, backbone="clip", model_pack=None, batch=32,
                       force_backend=None, pca_dim=64):
    """统一入口:torch 可用→CLIP 真特征;否则**优雅回退到经典 CPU 特征**(features.py)。

    force_backend: None=自动;"clip"=强制 CLIP(无 torch 会抛错);"classic"=强制经典特征。
    返回 np.ndarray [N, D](交给 similarity 做 L2 归一化)。
    """
    backend = force_backend
    if backend is None:
        backend = "clip" if torch_available() else "classic"
    if backend == "clip":
        return extract_embeddings_clip(image_paths, backbone=backbone,
                                       model_pack=model_pack, batch=batch)
    # ---- 经典特征回退(baseline):PIL 颜色直方图 + 梯度/分块统计 + 标准化/PCA ----
    from features import extract_embeddings_classic
    print("  [embed] torch/CLIP 不可用 → 回退到经典 CPU 特征(baseline,非 CLIP)")
    return extract_embeddings_classic(image_paths, pca_dim=pca_dim)


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


def _selftest():
    """验证 extract_embeddings 的经典回退路径在真实图像上可跑(无 torch 也能验证)。"""
    import sys, tempfile, os
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    print(f"  torch 可用: {torch_available()}(False 时走经典特征回退)")
    with tempfile.TemporaryDirectory() as td:
        from synth_images import generate
        records, _ = generate(td, n_groups=8, reuse_frac=0.5, seed=1, size=(96, 96))
        paths = [os.path.join(td, r["image_path"]) for r in records]

        # 自动后端(Mac 上应回退经典特征)
        embs = extract_embeddings(paths, force_backend="classic", pca_dim=32)
        check(embs.shape[0] == len(paths), f"经典回退提取 {embs.shape} 行数=图数")
        check(np.all(np.isfinite(embs)), "嵌入全有限")

        # 同 group 副本相似 > 跨 group(真特征语义)
        from similarity import cosine_sim_matrix
        from collections import defaultdict
        g2idx = defaultdict(list)
        for i, r in enumerate(records):
            g2idx[r["group_id"]].append(i)
        multi = [g for g, v in g2idx.items() if g.startswith("G") and len(v) >= 2][0]
        i, j = g2idx[multi][:2]
        other = [k for k, r in enumerate(records) if r["group_id"].startswith("S")][0]
        S = cosine_sim_matrix(embs)
        check(S[i, j] > S[i, other],
              f"经典嵌入:同组 {S[i,j]:.3f} > 跨组 {S[i,other]:.3f}")

        # save 产出文件
        outp = os.path.join(td, "emb")
        save(embs, [r["image_path"] for r in records],
             [r["customer_id"] for r in records], outp)
        check(os.path.getsize(outp + ".npy") > 0 and os.path.getsize(outp + ".json") > 0,
              "嵌入 .npy/.json 落盘非空")

    print("\n" + ("✅ embed 自测通过" if ok else "❌ embed 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="经典回退路径自测(无 torch 可跑)")
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--backend", choices=["auto", "clip", "classic"], default="auto",
                    help="auto=torch 有则 CLIP 否则经典;classic=强制经典 CPU 特征")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--out", default="embeddings")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.images:
        ap.error("需 --images 或 --selftest")
    fb = None if a.backend == "auto" else a.backend
    embs = extract_embeddings(a.images, a.backbone, force_backend=fb, pca_dim=a.pca_dim)
    save(embs, a.images, ["?"] * len(a.images), a.out)
    print("  下一步: python similarity.py 用这些嵌入做去重检测")


if __name__ == "__main__":
    main()
