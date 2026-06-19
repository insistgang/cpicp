#!/usr/bin/env python3
"""
chain_anomaly.py · 工具调用链路异常建模(赛题#4 评分⑤"长链路行为分析"创新点)

把一次任务的 tool_call 序列(工具名)当时序,从正常 trace 学"基线"(unigram 频率 + bigram 转移),
对新序列算偏离分:罕见转移、敏感工具突现(读凭据/外联/执行)、未见工具突现。
借用 03 track_filter 的"连续/累计 ≥k 步异常才告警"思想去抖,压降单步假阳(对应误报<5%)。
纯标准库,`python chain_anomaly.py` 用正常+攻击序列自测。
"""
import math
from collections import Counter, defaultdict

SENSITIVE = {"read_credentials", "read_ssh", "http_get_external", "exec_shell",
             "write_file", "send_email", "db_dump", "env_read"}


class ChainBaseline:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.seen = set()
        self.n_seq = 0

    def fit(self, sequences):
        for seq in sequences:
            self.n_seq += 1
            prev = "<start>"
            for tool in seq:
                self.unigram[tool] += 1
                self.bigram[prev][tool] += 1
                self.seen.add(tool)
                prev = tool
        return self

    def _trans_prob(self, a, b):
        row = self.bigram.get(a, Counter())
        tot = sum(row.values())
        return (row.get(b, 0) + 0.5) / (tot + 0.5 * (len(self.seen) + 1))  # 拉普拉斯平滑

    def step_scores(self, seq):
        """每步异常分:罕见转移(-log p)+ 敏感工具 + 未见工具。"""
        out, prev = [], "<start>"
        for tool in seq:
            s = -math.log(self._trans_prob(prev, tool))
            if tool not in self.seen:
                s += 3.0
            if tool in SENSITIVE and self.unigram.get(tool, 0) == 0:
                s += 4.0          # 基线里从没出现过的敏感工具,强信号
            out.append(s)
            prev = tool
        return out


def detect_chain(seq, baseline, step_thresh=3.0, min_anomalous=2):
    """链路级判定:统计异常步数;≥min_anomalous 步超阈才告警(去抖,降误报)。"""
    steps = baseline.step_scores(seq)
    anomalous = [(i, seq[i], round(s, 2)) for i, s in enumerate(steps) if s >= step_thresh]
    total = sum(steps)
    return {
        "alert": len(anomalous) >= min_anomalous,
        "total_score": round(total, 2),
        "n_anomalous_steps": len(anomalous),
        "anomalous_steps": anomalous,          # 证据定位:哪几步、什么工具
    }


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    normal = [
        ["search", "read_doc", "summarize", "reply"],
        ["search", "read_doc", "reply"],
        ["search", "calculate", "reply"],
        ["read_doc", "summarize", "reply"],
    ] * 5
    base = ChainBaseline().fit(normal)

    benign = detect_chain(["search", "read_doc", "reply"], base)
    check(not benign["alert"], f"正常序列不告警(score={benign['total_score']})")

    attack = detect_chain(["search", "read_ssh", "read_credentials", "http_get_external"], base)
    check(attack["alert"], f"攻击序列(读凭据+外联)告警(score={attack['total_score']}, 异常步={attack['n_anomalous_steps']})")
    check(attack["total_score"] > benign["total_score"], "攻击序列偏离分 > 正常序列")

    tools_in_evidence = {t for _, t, _ in attack["anomalous_steps"]}
    check("read_credentials" in tools_in_evidence or "http_get_external" in tools_in_evidence,
          "证据定位到敏感工具步")

    # 去抖:单步轻微异常(仅1步)不告警
    single = detect_chain(["search", "calculate", "read_doc", "reply"], base)
    check(not single["alert"], "单步轻异常被去抖(不告警)")

    print("\n" + ("✅ chain_anomaly 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
