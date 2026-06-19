#!/usr/bin/env python3
"""
synth_images.py · 程序化生成合成金融票据/证照影像(PIL),让流水线跑在真实像素上

赛题#23 的真实合成数据需签 NDA 才能领取。在拿到之前,本模块用 PIL 程序化生成一批
**结构化的金融影像**(不同模板:面签合影/证件/权属证明/合同),并刻意注入两类去重场景:
  - 同客户重复提交 : 同一张影像被同一客户多次提交(轻微 JPEG/裁剪/亮度扰动)
  - ⚠️ 跨客户套用   : 同一张面签照被套用到**不同客户**(命题方最关心的违规)

产出:
  - PNG 影像文件(真实像素,可被 features.py / classify.py 提取)
  - manifest.csv(列对齐 prepare_data.py:image_path,type_label,group_id,customer_id,business_label)

这样 prepare_data → 经典特征 → similarity 去重 → metrics 全程跑在真实图像上,
输出**真实** AUC / Top-k / 去重检出,而非随机向量。

用法:
  python synth_images.py --selftest
  python synth_images.py --out output/synth --n-groups 30 --reuse-frac 0.35
"""
import argparse
import csv
import os
import random

TYPES = ["面签照片", "证件", "权属证明", "合同文档"]
BIZ = ["微贷", "经营贷", "按揭", "消费贷"]
IMG_SIZE = (256, 256)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def _load_font(size):
    from PIL import ImageFont
    for c in _FONT_CANDIDATES:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _palette(rng):
    """每个 group 一套确定性配色(模板风格 + 客户标识),使同 group 影像视觉相近。"""
    base_hue = rng.randint(0, 5)  # random.randint 两端闭区间 → 6 套配色 [0,5]
    bgs = [(245, 245, 250), (250, 246, 240), (242, 248, 244),
           (248, 244, 248), (244, 246, 250), (250, 250, 244)]
    accents = [(40, 70, 140), (150, 60, 40), (40, 110, 70),
               (110, 60, 130), (60, 90, 150), (160, 130, 40)]
    return bgs[base_hue], accents[base_hue]


def _render_template(t_type, group_seed, customer_id, biz, size=IMG_SIZE):
    """渲染一张某类型的金融影像。group_seed 决定版式/配色 → 同 group 看起来像同一张。"""
    import numpy as np
    from PIL import Image, ImageDraw
    rng = random.Random(group_seed)
    bg, accent = _palette(rng)
    W, H = size
    # 模板内含面签合影所需的固定版式偏移(标题条/背景墙/人物/桌子/标牌),针对默认 256px 设计。
    # 画布过小会让这些硬编码偏移越界(PIL rectangle y1<y0 或 randint 空区间),给出清晰报错而非
    # 深层 PIL/randrange 崩溃。96px 即满足全部偏移;所有调用方与默认值均 ≥96。
    if W < 96 or H < 96:
        raise ValueError(f"size 至少 96x96(模板版式偏移需要);收到 {size}。"
                         f"更小尺寸请在生成后用 PIL.resize 缩放,而非直接渲染。")
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    f_title = _load_font(22)
    f_body = _load_font(15)
    f_small = _load_font(12)

    if t_type == "面签照片":
        # 面签合影:两个人形剪影 + 桌子 + 标题条。为让**不同 group 视觉可分**
        # (经典特征才能区分实例),按 group 随机化:人物位置/肤色/衣色/背景墙色。
        wall = (200 + rng.randint(-30, 30), 200 + rng.randint(-30, 30),
                205 + rng.randint(-30, 30))
        d.rectangle([0, 42, W, H - 44], fill=wall)  # 背景墙(每组不同色)
        d.rectangle([0, 0, W, 42], fill=accent)
        d.text((12, 9), "信贷面签合影", font=f_title, fill=(255, 255, 255))
        skin = (180 + rng.randint(-20, 40), 150 + rng.randint(-20, 30), 120 + rng.randint(-20, 30))
        cloth = [(rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200)) for _ in range(2)]
        cxs = (W // 2 + rng.randint(-25, -10), W // 2 + rng.randint(10, 25))
        head_r = rng.randint(18, 26)
        for k, cx in enumerate(cxs):
            hy = rng.randint(78, 100)
            d.ellipse([cx - head_r, hy - head_r, cx + head_r, hy + head_r], fill=skin)  # 头
            d.polygon([(cx - 40, H - 40), (cx + 40, H - 40),
                       (cx + 28, hy + 16), (cx - 28, hy + 16)], fill=cloth[k])         # 身/衣
        # 桌子(每组木色不同)
        wood = (140 + rng.randint(-30, 40), 100 + rng.randint(-20, 30), 60 + rng.randint(-10, 30))
        d.rectangle([20, H - 40, W - 20, H - 18], fill=wood)
        # 每组独有的小标牌(位置/色),进一步增强 group 间可分性
        bx, by = rng.randint(20, W - 60), rng.randint(50, 70)
        d.rectangle([bx, by, bx + 36, by + 16], fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        d.text((14, H - 16), f"客户:{customer_id}  {biz}", font=f_small, fill=(60, 60, 60))

    elif t_type == "证件":
        d.rectangle([24, 30, W - 24, H - 30], outline=accent, width=3, fill=(255, 255, 255))
        d.text((40, 44), "居民身份证", font=f_title, fill=accent)
        # 照片框
        d.rectangle([W - 92, 80, W - 40, 150], fill=(200, 205, 215))
        for i, lab in enumerate(["姓名", "性别", "民族", "住址"]):
            d.text((40, 86 + i * 26), f"{lab}  ******", font=f_body, fill=(40, 40, 40))
        d.text((40, H - 60), f"公民身份号码 {3000 + group_seed % 9999}", font=f_small, fill=(30, 30, 30))

    elif t_type == "权属证明":
        d.rectangle([0, 0, W, 50], fill=accent)
        d.text((12, 14), "不动产权属证明", font=f_title, fill=(255, 240, 200))
        # 红章圆环
        cx, cy = W - 70, 110
        d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], outline=(190, 40, 40), width=3)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(190, 40, 40))
        for i, lab in enumerate(["权利人", "坐落", "面积", "用途", "权利类型"]):
            d.text((24, 74 + i * 28), f"{lab}:________", font=f_body, fill=(35, 35, 35))

    else:  # 合同文档
        d.text((W // 2 - 50, 18), "借款合同", font=f_title, fill=(20, 20, 20))
        d.line([24, 50, W - 24, 50], fill=accent, width=2)
        # 模拟文本行
        rng2 = random.Random(group_seed + 1)
        y = 64
        for _ in range(11):
            x1 = 28
            x2 = W - 28 - rng2.randint(0, 80)
            d.line([x1, y, x2, y], fill=(120, 120, 120), width=2)
            y += 16
        d.text((28, H - 34), f"甲方:{customer_id}", font=f_small, fill=(40, 40, 40))

    # 模板纹理(每 group 固定的轻微噪点底纹,增强 group 内一致性 / group 间差异)
    tex = np.array(img, dtype=np.int16)
    nrng = np.random.RandomState(group_seed)
    tex += nrng.randint(-3, 4, tex.shape)
    img = Image.fromarray(np.clip(tex, 0, 255).astype("uint8"))
    return img


def _perturb(img, seed):
    """对一张影像做**轻微**扰动,模拟"套用/重复提交"时的二次采集(同一张照片被重新
    截图/压缩/微调亮度)。真实套用场景下副本与原图高度相似,故扰动幅度刻意保持小,
    使同组余弦显著高于跨组(否则去重无从谈起)。"""
    import numpy as np
    from PIL import Image, ImageEnhance
    rng = np.random.RandomState(seed)
    # 亮度:±3%
    img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-0.03, 0.03))
    # 对比度:±3%
    img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-0.03, 0.03))
    # 极微裁剪(0~2px)后缩放回原尺寸(模拟截图/重压缩,不破坏版式)
    W, H = img.size
    dx, dy = rng.randint(0, 3), rng.randint(0, 3)
    if dx or dy:
        img = img.crop((dx, dy, W - dx, H - dy)).resize((W, H), Image.BILINEAR)
    # 轻像素噪声 ±2
    a = np.array(img, dtype=np.int16) + rng.randint(-2, 3, (H, W, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def generate(out_dir, n_groups=30, reuse_frac=0.35, seed=0, size=IMG_SIZE):
    """生成合成影像 + manifest。返回 (records, manifest_path)。

    每个 group = 一张"原始影像",派生 k 张(同一张的轻微扰动副本)。
    - cross(按 reuse_frac 比例): 派生副本分配给**不同客户** → 跨客户套用(同 group, 异客户)
    - 否则: 派生副本属同一客户 → 同客户重复
    另加一批各自独立(单张, 无重复)的影像,贴近真实分布(大量正常影像 + 少量违规)。
    """
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    records = []
    img_idx = 0

    for g in range(n_groups):
        gid = f"G{g:03d}"
        t = rng.choice(TYPES)
        base_cust = f"C{rng.randint(0, 40):03d}"
        biz = rng.choice(BIZ)
        k = rng.randint(2, 4)  # 同组副本数(含原图)
        cross = rng.random() < reuse_frac
        group_seed = 10_000 + g  # 决定版式,使同 group 影像相像
        base_img = _render_template(t, group_seed, base_cust, biz, size=size)

        for m in range(k):
            # 第 0 张为原图;其余为扰动副本(模拟二次提交/套用)
            cur = base_img if m == 0 else _perturb(base_img, seed=group_seed * 100 + m)
            # 跨客户套用:副本(m>0)换成不同客户
            cust = (f"C{rng.randint(60, 99):03d}" if (cross and m > 0) else base_cust)
            fname = f"img_{img_idx:05d}.png"
            cur.save(os.path.join(out_dir, fname))
            records.append({
                "image_path": fname,
                "type_label": t,
                "group_id": gid,
                "customer_id": cust,
                "business_label": biz,
            })
            img_idx += 1

    # 独立单张影像(无重复),数量约等于 n_groups,贴近真实分布
    for s in range(n_groups):
        t = rng.choice(TYPES)
        cust = f"C{rng.randint(0, 99):03d}"
        biz = rng.choice(BIZ)
        gseed = 50_000 + s
        img = _render_template(t, gseed, cust, biz, size=size)
        fname = f"img_{img_idx:05d}.png"
        img.save(os.path.join(out_dir, fname))
        records.append({
            "image_path": fname, "type_label": t, "group_id": f"S{s:03d}",
            "customer_id": cust, "business_label": biz,
        })
        img_idx += 1

    rng.shuffle(records)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    cols = ["image_path", "type_label", "group_id", "customer_id", "business_label"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    return records, manifest_path


def _selftest():
    import sys, tempfile
    import numpy as np
    from PIL import Image
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "synth")
        records, mpath = generate(out, n_groups=12, reuse_frac=0.6, seed=3, size=(128, 128))

        check(len(records) > 0, f"生成 {len(records)} 条影像记录")
        check(os.path.exists(mpath), "manifest.csv 已写出")

        # 所有声明的图像文件确实存在且非空
        all_exist = all(os.path.getsize(os.path.join(out, r["image_path"])) > 0 for r in records)
        check(all_exist, "所有 PNG 文件存在且非空")

        # 是真实可读图像(非占位)
        sample = records[0]["image_path"]
        im = Image.open(os.path.join(out, sample))
        check(im.size == (128, 128) and im.mode in ("RGB", "RGBA"),
              f"PNG 可被 PIL 读取(size={im.size}, mode={im.mode})")

        # 必须构造出跨客户套用(同 group 不同 customer)
        from collections import defaultdict
        g2cust = defaultdict(set)
        for r in records:
            if r["group_id"].startswith("G"):
                g2cust[r["group_id"]].add(r["customer_id"])
        cross_groups = [g for g, cs in g2cust.items() if len(cs) > 1]
        check(len(cross_groups) > 0, f"含跨客户套用 group {len(cross_groups)} 个(同组异客户)")

        # 类型分布:面签 + 其它都有(分类任务正负样本)
        types = {r["type_label"] for r in records}
        check("面签照片" in types and len(types) > 1, f"类型多样({sorted(types)})")

        # 关键:同 group 影像在像素上确实相似(经典特征验证生成器有效)
        from features import ClassicFeatureExtractor
        from similarity import cosine_sim_matrix
        # 找一个有 >=2 张的 G 组
        from collections import defaultdict as dd
        g2imgs = dd(list)
        for r in records:
            g2imgs[r["group_id"]].append(r["image_path"])
        multi = [g for g, v in g2imgs.items() if g.startswith("G") and len(v) >= 2][0]
        same_imgs = g2imgs[multi][:2]
        # 一张来自不同 group 的图
        other_img = [r["image_path"] for r in records
                     if r["group_id"] != multi and r["group_id"].startswith("S")][0]
        paths = [os.path.join(out, p) for p in (same_imgs + [other_img])]
        fe = ClassicFeatureExtractor(pca_dim=0)
        raw = fe.raw_features(paths)
        S = cosine_sim_matrix(raw)
        s_same = S[0, 1]   # 同 group 两张(同一张的扰动副本)
        s_other = S[0, 2]  # 跨 group
        check(s_same > s_other,
              f"同组副本像素相似 > 跨组({s_same:.3f} > {s_other:.3f})—生成器有效")

    print("\n" + ("✅ synth_images 自测通过" if ok else "❌ synth_images 自测未通过"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="程序化生成合成金融票据/证照影像")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="output/synth", help="输出目录")
    ap.add_argument("--n-groups", type=int, default=30)
    ap.add_argument("--reuse-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    records, mpath = generate(a.out, n_groups=a.n_groups, reuse_frac=a.reuse_frac, seed=a.seed)
    n_face = sum(1 for r in records if r["type_label"] == "面签照片")
    print(f"✓ 生成 {len(records)} 张合成影像 → {os.path.abspath(a.out)}")
    print(f"  面签照片 {n_face} 张 | manifest: {mpath}")
    print(f"  下一步: python pipeline.py --real-images {a.out}")


if __name__ == "__main__":
    main()
