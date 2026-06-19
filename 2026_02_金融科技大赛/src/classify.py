#!/usr/bin/env python3
"""
classify.py · 第一步:从多类型影像中筛出"面签照片"(赛题#23 步骤1)

官方现有基础用 CLIP 做分类过滤。本模块提供两条路:
  ① 零样本(zero-shot):用 CLIP 图文相似度 + 中文/英文类别提示词,无需训练即可分类(冷启动/无标注时)。
  ② 线性探针(linear-probe):用少量标注训一个 logistic 回归头(精度更高,推荐拿到合成数据后用)。
输出:每张影像 是否面签照片(+各类概率),供 similarity 步骤只对面签照片做去重。

回退策略:torch+open_clip 可用→CLIP 零样本/线性探针(真特征);不可用(Mac 无 GPU/离线)
→**经典特征线性探针**(features.py 的 PIL+sklearn 特征 + LogisticRegression),
在真实合成影像上有监督地筛"面签照片",让分类步骤也能在本地端到端跑通(baseline)。
"""
import argparse

# 类别提示词(零样本):面签照片为正类,其余为干扰类
PROMPTS = {
    "面签照片": ["一张银行信贷面签合影照片", "客户与客户经理面对面签约的合影"],
    "证件": ["一张身份证或证件影像", "an ID card document photo"],
    "权属证明": ["一张房产或抵押物权属证明影像"],
    "合同文档": ["一张合同文本扫描件", "a scanned contract document"],
}
FACE_CLASS = "面签照片"


def load_clip(model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", device=None):
    import torch, open_clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.to(device).eval(), preprocess, tokenizer, device


def zero_shot_classify(image_paths, model_pack=None, batch=32):
    """零样本:返回 [(path, pred_class, p_face), ...]。p_face=判为面签照片的概率。"""
    import torch
    from PIL import Image
    model, preprocess, tokenizer, device = model_pack or load_clip()
    classes = list(PROMPTS.keys())
    with torch.no_grad():
        # 每类提示词取均值作类别文本特征
        text_feats = []
        for c in classes:
            t = tokenizer(PROMPTS[c]).to(device)
            tf = model.encode_text(t)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats.append(tf.mean(0))
        text_feats = torch.stack(text_feats)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        out = []
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i:i + batch]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (100.0 * f @ text_feats.T).softmax(-1)
            for p, row in zip(chunk, logits):
                idx = int(row.argmax())
                out.append((p, classes[idx], float(row[classes.index(FACE_CLASS)])))
    return out


def train_linear_probe(train_paths, train_labels, model_pack=None, C=0.5):
    """线性探针:CLIP 图像特征 + LogisticRegression。labels: 1=面签 0=其他。返回 (clf, model_pack)。"""
    import torch, numpy as np
    from sklearn.linear_model import LogisticRegression
    from PIL import Image
    model, preprocess, tokenizer, device = model_pack or load_clip()
    feats = []
    with torch.no_grad():
        for p in train_paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img); f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy()[0])
    clf = LogisticRegression(C=C, max_iter=1000).fit(np.array(feats), train_labels)
    return clf, (model, preprocess, tokenizer, device)


def filter_face_signing(results, p_thresh=0.5):
    """从分类结果里取出面签照片的路径(供 similarity 步骤)。"""
    return [p for (p, cls, p_face) in results if cls == FACE_CLASS or p_face >= p_thresh]


# ---------------- 经典特征回退:无 torch 时的面签筛选器 ----------------

def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class ClassicFaceClassifier:
    """经典特征 + LogisticRegression 二分类(面签照片 vs 其它)。
    CLIP 不可用时的线性探针回退:有监督(合成数据有 type_label),在真实像素上筛面签。
    """

    def __init__(self, pca_dim=48):
        from features import ClassicFeatureExtractor
        self.fe = ClassicFeatureExtractor(pca_dim=pca_dim)
        self.clf = None

    def fit(self, image_paths, labels):
        """labels: 1=面签照片 0=其它。"""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X = self.fe.fit_transform(image_paths)
        # 抑制 NumPy 2.0.2 本机 BLAS 对有限输入 matmul 的 spurious 浮点警告(见 features.py 注释)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.clf = LogisticRegression(C=1.0, max_iter=2000,
                                          class_weight="balanced").fit(X, labels)
        return self

    def predict(self, image_paths):
        """返回 [(path, pred_class, p_face), ...],与 zero_shot_classify 同形。"""
        import numpy as np
        X = self.fe.transform(image_paths)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            proba = self.clf.predict_proba(X)[:, list(self.clf.classes_).index(1)]
        out = []
        for p, pf in zip(image_paths, proba):
            out.append((p, FACE_CLASS if pf >= 0.5 else "其它", float(pf)))
        return out


def classify_face_signing(image_paths, train_paths=None, train_labels=None, force_backend=None):
    """统一入口:torch→CLIP 零样本;否则→经典特征线性探针(需 train_paths/labels 训练)。
    返回 [(path, pred_class, p_face), ...]。"""
    backend = force_backend or ("clip" if torch_available() else "classic")
    if backend == "clip":
        return zero_shot_classify(image_paths)
    if train_paths is None or train_labels is None:
        raise ValueError("经典回退需 train_paths/train_labels 训练线性探针(合成数据有 type_label)")
    print("  [classify] torch/CLIP 不可用 → 经典特征线性探针回退(baseline)")
    model = ClassicFaceClassifier().fit(train_paths, train_labels)
    return model.predict(image_paths)


def _selftest():
    """在真实合成影像上验证经典回退分类器能筛出面签照片。"""
    import sys, tempfile, os
    import numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    print(f"  torch 可用: {torch_available()}(False 时走经典线性探针回退)")
    with tempfile.TemporaryDirectory() as td:
        from synth_images import generate
        records, _ = generate(td, n_groups=30, reuse_frac=0.3, seed=5, size=(96, 96))
        paths = [os.path.join(td, r["image_path"]) for r in records]
        labels = [1 if r["type_label"] == FACE_CLASS else 0 for r in records]
        n_face = sum(labels)
        check(0 < n_face < len(labels), f"数据:面签 {n_face}/{len(labels)}(正负都有)")

        # 训练/测试切分(简单留出)
        n = len(paths)
        idx = list(range(n))
        import random as _r
        _r.Random(0).shuffle(idx)
        cut = int(n * 0.7)
        tr, te = idx[:cut], idx[cut:]
        tr_p = [paths[i] for i in tr]; tr_y = [labels[i] for i in tr]
        te_p = [paths[i] for i in te]; te_y = [labels[i] for i in te]

        res = classify_face_signing(te_p, train_paths=tr_p, train_labels=tr_y,
                                    force_backend="classic")
        check(len(res) == len(te_p), "对测试集全部出预测")

        pred = [1 if c == FACE_CLASS else 0 for (_, c, _) in res]
        acc = np.mean([int(a == b) for a, b in zip(pred, te_y)])
        # 面签模板视觉与其它差异大,经典特征应明显优于随机
        check(acc > 0.75, f"测试集分类准确率 {acc:.3f} > 0.75(经典特征筛面签有效)")

        # 召回:面签照片大多被筛出
        face_idx = [i for i, y in enumerate(te_y) if y == 1]
        if face_idx:
            recall = np.mean([pred[i] for i in face_idx])
            check(recall > 0.6, f"面签召回 {recall:.3f} > 0.6(不大量漏筛)")

    print("\n" + ("✅ classify 自测通过" if ok else "❌ classify 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="经典回退分类器自测(无 torch 可跑)")
    ap.add_argument("--images", nargs="+", help="待分类影像路径")
    ap.add_argument("--mode", choices=["zeroshot"], default="zeroshot")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.images:
        ap.error("需 --images(CLIP 零样本需算力机装 torch+open_clip)或 --selftest")
    res = zero_shot_classify(a.images)
    faces = filter_face_signing(res)
    print(f"✓ 分类完成 {len(res)} 张,筛出面签照片 {len(faces)} 张")
    for p, c, pf in res:
        print(f"  {c:8s} p_face={pf:.3f}  {p}")


if __name__ == "__main__":
    main()
