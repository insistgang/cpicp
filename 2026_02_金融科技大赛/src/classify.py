#!/usr/bin/env python3
"""
classify.py · 第一步:从多类型影像中筛出"面签照片"(赛题#23 步骤1)

官方现有基础用 CLIP 做分类过滤。本模块提供两条路:
  ① 零样本(zero-shot):用 CLIP 图文相似度 + 中文/英文类别提示词,无需训练即可分类(冷启动/无标注时)。
  ② 线性探针(linear-probe):用少量标注训一个 logistic 回归头(精度更高,推荐拿到合成数据后用)。
输出:每张影像 是否面签照片(+各类概率),供 similarity 步骤只对面签照片做去重。

⚠️ 需 torch + open_clip(算力机上跑);py_compile 可过(重依赖在函数内导入)。
合成数据联调:先用 prepare_data.synth_manifest 的类型标签验证流程逻辑。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", help="待分类影像路径")
    ap.add_argument("--mode", choices=["zeroshot"], default="zeroshot")
    a = ap.parse_args()
    if not a.images:
        ap.error("需 --images(算力机上,装 torch+open_clip 后运行)")
    res = zero_shot_classify(a.images)
    faces = filter_face_signing(res)
    print(f"✓ 分类完成 {len(res)} 张,筛出面签照片 {len(faces)} 张")
    for p, c, pf in res:
        print(f"  {c:8s} p_face={pf:.3f}  {p}")


if __name__ == "__main__":
    main()
