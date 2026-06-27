#!/usr/bin/env python3
"""
text_tools.py · 古文文本处理(数字人文作品的辅助工具,文科赛技术非核心)

赛题2 是人文研究/数字人文,技术只作展示辅助。本模块纯标准库(无 jieba/numpy 依赖,古文无现成分词):
  - 古文分句(按 。！？；等)
  - 字-bigram TF-IDF 关键词(无分词器时对文言文更稳)
  - 导出词云/关键词 JSON(供静态可视化网页读取)
`python text_tools.py` 用内置《颜氏家训》示例段自测。
"""
import json
import math
import re
from collections import Counter

SENT_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")
STOP = set("之乎者也而其以于与為为则乃且夫矣焉哉耳兮的了是在和有不人我你他"
           "所可如自此故亦皆若使为又即既"  # 补:常见文言虚词,避免在词云里盖过教化主题词
           )


def split_sentences(text):
    return [s.strip() for s in SENT_RE.findall(text) if s.strip()]


def _char_bigrams(s):
    chars = [c for c in s if '一' <= c <= '鿿']      # 仅汉字
    toks = [c for c in chars if c not in STOP]
    toks += [chars[i] + chars[i+1] for i in range(len(chars)-1)
             if chars[i] not in STOP and chars[i+1] not in STOP]
    return toks


def tfidf_keywords(passages, topk=20):
    """passages: list[str]。返回 [(token, weight)],按 TF-IDF 汇总降序。"""
    docs_tokens = [_char_bigrams(p) for p in passages]
    df = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    N = max(1, len(passages))
    agg = Counter()
    for toks in docs_tokens:
        tf = Counter(toks); total = sum(tf.values()) or 1
        for t, c in tf.items():
            idf = math.log((N + 1) / (df[t] + 1)) + 1
            agg[t] += (c / total) * idf
    return agg.most_common(topk)


def export_wordcloud_json(passages, path=None, topk=50):
    data = [{"text": t, "weight": round(w, 4)} for t, w in tfidf_keywords(passages, topk)]
    if path:
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return data


SAMPLE = (
    "夫圣贤之书，教人诚孝，慎言检迹，立身扬名，亦已备矣。"
    "积财千万，不如薄伎在身。伎之易习而可贵者，无过读书。"
    "父母威严而有慈，则子女畏慎而生孝矣。"
    "教妇初来，教儿婴孩。"
)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    sents = split_sentences(SAMPLE)
    check(len(sents) >= 4, f"古文分句({len(sents)}句)")

    kws = tfidf_keywords(sents, topk=15)
    check(len(kws) > 0 and all(isinstance(w, float) for _, w in kws), f"TF-IDF 关键词非空({len(kws)})")
    toks = {t for t, _ in kws}
    check(any("孝" in t or "教" in t or "书" in t for t in toks), f"抽到教化主题词(含 孝/教/书):{list(toks)[:6]}")

    data = export_wordcloud_json(sents)
    check(json.dumps(data, ensure_ascii=False) and data[0]["weight"] >= data[-1]["weight"], "词云JSON可序列化且按权重降序")

    print("\n" + ("✅ text_tools 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
