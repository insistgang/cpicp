#!/usr/bin/env python3
"""
similarity.py · 面签照片向量相似度去重检测(赛题#23 第二步:查"重复使用/跨客户套用")

输入:面签照片的嵌入向量 + 影像id + 客户id(+可选业务id)。
输出:高相似可疑对,并区分两类违规:
  - same_customer_repeat   同一客户重复提交历史面签照(合规上仍需关注)
  - cross_customer_misuse  ⚠️ 跨客户套用同一张面签照(最严重的违规,命题方核心诉求)
纯 numpy(大规模生产用 FAISS,见 build_index 注释)。`python similarity.py` 跑自测。
"""


def l2_normalize(embs):
    import numpy as np
    e = np.asarray(embs, float)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(n, 1e-12, None)


def cosine_sim_matrix(embs):
    import numpy as np
    e = l2_normalize(embs)
    return e @ e.T


def find_suspicious_pairs(embs, image_ids, customer_ids, threshold=0.9, business_ids=None):
    """返回相似度≥threshold 的影像对列表,按相似度降序。每项标注违规类型。"""
    import numpy as np
    S = cosine_sim_matrix(embs)
    n = len(image_ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(S[i, j])
            if s < threshold:
                continue
            same_cust = customer_ids[i] == customer_ids[j]
            pairs.append({
                "img_a": image_ids[i], "img_b": image_ids[j],
                "cust_a": customer_ids[i], "cust_b": customer_ids[j],
                "score": s,
                "type": "same_customer_repeat" if same_cust else "cross_customer_misuse",
                "biz_a": None if business_ids is None else business_ids[i],
                "biz_b": None if business_ids is None else business_ids[j],
            })
    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs


def summarize(pairs):
    """检测汇总视图的数据(官方要求"检测汇总视图,展示检测结果统计")。"""
    cross = [p for p in pairs if p["type"] == "cross_customer_misuse"]
    same = [p for p in pairs if p["type"] == "same_customer_repeat"]
    return {
        "total_suspicious_pairs": len(pairs),
        "cross_customer_misuse": len(cross),       # 最严重
        "same_customer_repeat": len(same),
        "max_score": max((p["score"] for p in pairs), default=0.0),
        "top_cross_customer": cross[:10],          # 供 demo 并排查看
    }


def build_index(embs):
    """大规模检索建议:embs 量大时用 FAISS(IndexFlatIP + L2归一化≈余弦)。
    import faiss; idx = faiss.IndexFlatIP(d); idx.add(l2_normalize(embs)); D,I = idx.search(q, k)
    此处给 numpy 版回退(小规模/离线评测足够)。"""
    return l2_normalize(embs)


def _selftest():
    import sys, numpy as np
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    rng = np.random.RandomState(7)
    D = 32
    base = rng.randn(D)                                   # 一张原始面签照
    reused = base + rng.randn(D) * 0.01                   # 被套用的近乎同一张照(高相似)
    same_cust_dup = base + rng.randn(D) * 0.01            # 同客户重复提交的同一张照
    others = rng.randn(6, D)                              # 6 张不相干影像
    embs = np.vstack([base, reused, same_cust_dup, others])
    image_ids = ["IMG_base", "IMG_reused", "IMG_dup", *[f"IMG_o{i}" for i in range(6)]]
    # base=客户A;reused=客户B(跨客户套用!);dup=客户A(同客户重复);others 各异
    custs = ["A", "B", "A", "C", "D", "E", "F", "G", "H"]

    pairs = find_suspicious_pairs(embs, image_ids, custs, threshold=0.9)
    types = {(p["img_a"], p["img_b"]): p["type"] for p in pairs}

    def has(a, b, t):
        return types.get((a, b)) == t or types.get((b, a)) == t

    check(has("IMG_base", "IMG_reused", "cross_customer_misuse"),
          "跨客户套用(base↔reused,客户A↔B)被标为 cross_customer_misuse")
    check(has("IMG_base", "IMG_dup", "same_customer_repeat"),
          "同客户重复(base↔dup,都客户A)被标为 same_customer_repeat")
    # 不相干影像不应进可疑对
    involved = {p["img_a"] for p in pairs} | {p["img_b"] for p in pairs}
    check(not any(f"IMG_o{i}" in involved for i in range(6)), "不相干影像未被误报")

    summ = summarize(pairs)
    check(summ["cross_customer_misuse"] >= 1 and summ["max_score"] > 0.99,
          f"汇总:跨客户套用 {summ['cross_customer_misuse']} 对,最高相似 {summ['max_score']:.3f}")

    # 阈值升高应减少/不增可疑对(sanity)
    fewer = find_suspicious_pairs(embs, image_ids, custs, threshold=0.99)
    check(len(fewer) <= len(pairs), f"阈值↑可疑对不增({len(pairs)}→{len(fewer)})")

    print("\n" + ("✅ similarity 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
