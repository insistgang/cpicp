#!/usr/bin/env python3
"""
similarity_search.py · 典籍片段语义/字面相似检索(数字人文展示亮点)

复用 02 金融 similarity 的余弦相似思想,改成"典籍片段互文/相似检索":
默认离线 TF-IDF 向量(无需 GPU/模型),保证本机可跑;留 CLIP/句向量接口(真要可换)。
用途:发现一部书内/跨版本的互文片段、主题聚类,作"数字人文作品"的可视化素材。
`python similarity_search.py` 自测。
"""
import math
from collections import Counter

from text_tools import _char_bigrams


def tfidf_matrix(passages):
    """返回 (vectors: list[dict token->weight], vocab_df)。稀疏字典向量,避免 numpy 依赖。"""
    docs = [_char_bigrams(p) for p in passages]
    df = Counter()
    for toks in docs:
        for t in set(toks):
            df[t] += 1
    N = max(1, len(passages))
    vecs = []
    for toks in docs:
        tf = Counter(toks); total = sum(tf.values()) or 1
        v = {t: (c / total) * (math.log((N + 1) / (df[t] + 1)) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
        vecs.append({t: w / norm for t, w in v.items()})
    return vecs


def cosine(a, b):
    """两个稀疏字典向量(已归一化)的余弦 = 点积。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def find_similar(passages, query_idx, topk=3, threshold=0.0):
    vecs = tfidf_matrix(passages)
    q = vecs[query_idx]
    sims = [(j, cosine(q, vecs[j])) for j in range(len(passages)) if j != query_idx]
    sims = [(j, s) for j, s in sims if s >= threshold]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topk]


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    passages = [
        "积财千万，不如薄伎在身。",          # 0
        "积财万贯，不如一技傍身，伎艺可贵。",   # 1 与0高度互文
        "教妇初来，教儿婴孩。",              # 2 教化主题
        "父母威严而有慈，则子女畏慎而生孝。",   # 3 与2同主题(教化)
        "春江潮水连海平，海上明月共潮生。",     # 4 完全无关(诗)
    ]
    top = find_similar(passages, 0, topk=4)
    check(top[0][0] == 1, f"片段0最相似的是互文片段1(得{top[0][0]})")
    check(top[0][1] > 0.0, f"互文相似度>0(={top[0][1]:.3f})")
    # 真正的语义判别: 互文片段1 的相似度必须 *严格高于* 无关诗句4,
    # 不能只靠"诗句恰好在末位"——那只是排序的索引顺序巧合(诗句与教化句同为0时,
    # 末位由原索引决定,不证明算法能区分)。这里直接比两者数值。
    scores = dict(top)
    s_inter = scores.get(1, 0.0)   # 互文片段
    s_poem = scores.get(4, 0.0)    # 无关诗句
    check(s_inter > s_poem,
          f"互文相似度({s_inter:.3f}) 严格 > 无关诗句({s_poem:.3f})")
    # 无关诗句应是最低分(并列最低也接受)
    check(s_poem == min(s for _, s in top),
          f"无关诗句相似度处于最低档(={s_poem:.3f})")

    print("\n" + ("✅ similarity_search 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
