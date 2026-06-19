#!/usr/bin/env python3
"""
build_demo_html.py · 生成单文件攻防演示页(赛题#4 方向C 答辩杀手锏)

读取 ../output/attack_demo.json(闭环记录)+ ../output/roc_pr.png(可选,base64 内嵌),
产出 ../output/demo.html —— 单文件、内嵌全部数据与图、浏览器双击即开,可现场演示/截图。

页面展示:
  顶部总览仪表盘(场景数 / 闭环成功 / 攻击者成功步数 加规则前→后 / 良性零误报 / 决策延时)
  每条攻击一张卡片:战术 → 检测命中(检测器/分数/命中规则/证据 span) →
                     自动生成的防御规则 → 加规则前 vs 加规则后逐步决策对照(allow/alert/block)
  良性对照(零误报)+ ROC/PR 量化基准图。

`python build_demo_html.py` 自测(校验生成的 HTML 非空、含数据锚点、可解析)。
"""
import base64
import html
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
JSON_PATH = os.path.join(OUT_DIR, "attack_demo.json")
PNG_PATH = os.path.join(OUT_DIR, "roc_pr.png")
HTML_PATH = os.path.join(OUT_DIR, "demo.html")


def _ensure_inputs():
    """若闭环 JSON / 基准图缺失,就地生成(让本脚本可独立运行)。"""
    if not os.path.exists(JSON_PATH):
        import attack_demo
        attack_demo.main()
    if not os.path.exists(PNG_PATH):
        try:
            import benchmark_plot
            benchmark_plot.make_plot()
        except Exception as e:  # 基准图非必须,失败也能出页
            print(f"[warn] 基准图生成跳过: {e}")


DECISION_CSS = {"allow": "ok", "alert": "warn", "block": "bad"}
DECISION_TXT = {"allow": "放行 ALLOW", "alert": "告警 ALERT", "block": "阻断 BLOCK"}


def _esc(x):
    return html.escape(str(x))


def _render_evidence(ev):
    if "span" in ev:
        return (f'<span class="rule">{_esc(ev["rule"])}</span> '
                f'<code>span={ev["span"]}</code> '
                f'<span class="snip">“{_esc(ev["snippet"])}”</span>')
    if "step" in ev:
        return (f'<span class="rule">step#{ev["step"]} {_esc(ev["tool"])}</span> '
                f'<code>异常分={ev["step_score"]}</code>')
    return _esc(json.dumps(ev, ensure_ascii=False))


def _render_steps(steps):
    rows = []
    for s in steps:
        d = s["decision"]
        rows.append(
            f'<tr class="{DECISION_CSS[d]}">'
            f'<td><code>{_esc(s["tool"])}</code></td>'
            f'<td><span class="pill {DECISION_CSS[d]}">{DECISION_TXT[d]}</span></td>'
            f'<td>{_esc(", ".join(s.get("tags", [])) or "-")}</td>'
            f'<td>{_esc(s.get("evidence") or "-")}</td>'
            f'<td>{s.get("latency_ms", "-")} ms</td>'
            f'</tr>')
    return "".join(rows)


def _render_scenario(r):
    dets = []
    for d in r["detections"]:
        evs = "".join(f"<li>{_render_evidence(e)}</li>" for e in d["evidence"])
        dets.append(
            f'<div class="det">'
            f'<div class="det-head"><b>{_esc(d["detector"])}</b> '
            f'<span class="verdict">{_esc(d["verdict"])}</span> '
            f'<span class="score">分数 {d["score"]}</span></div>'
            f'<ul class="evlist">{evs}</ul>'
            f'</div>')
    rules = "".join(f'<li><code>{_esc(x)}</code></li>' for x in r["generated_rules"])
    b, a = r["before_defense"], r["after_defense"]
    skill_block = (f'<div class="skill"><b>关联 Skill/MCP 描述:</b> '
                   f'<code>{_esc(r["skill_text"])}</code></div>') if r["skill_text"] else ""
    loop = ('<span class="pill bad">闭环已闭合</span>' if r["loop_closed"]
            else '<span class="pill warn">未闭合</span>')
    return f"""
    <section class="scenario">
      <h3>[{_esc(r['id'])}] {_esc(r['name'])} {loop}</h3>
      <div class="tactic">战术: {_esc(r['tactic'])}　|　最高检测分: <b>{r['max_detection_score']}</b></div>
      <p class="narr">{_esc(r['narrative'])}</p>
      {skill_block}
      <div class="phase"><h4>① 检测命中</h4>{''.join(dets)}</div>
      <div class="phase"><h4>② 自动生成防御策略 ({len(r['generated_rules'])} 条)</h4><ul class="rules">{rules}</ul></div>
      <div class="phase compare">
        <div class="col">
          <h4>③a 加规则前(出厂默认)</h4>
          <table><tr><th>工具</th><th>决策</th><th>标签</th><th>证据</th><th>延时</th></tr>{_render_steps(b['steps'])}</table>
          <div class="kpi">阻断 {b['n_blocked']} / 告警 {b['n_alert']} 步　攻击者得手: <b>{'是' if b['attacker_won'] else '否'}</b></div>
        </div>
        <div class="arrow">→</div>
        <div class="col">
          <h4>③b 加规则后(自适应防御)</h4>
          <table><tr><th>工具</th><th>决策</th><th>标签</th><th>证据</th><th>延时</th></tr>{_render_steps(a['steps'])}</table>
          <div class="kpi">阻断 {a['n_blocked']} / 告警 {a['n_alert']} 步　攻击者得手: <b>{'是' if a['attacker_won'] else '否'}</b></div>
        </div>
      </div>
    </section>"""


def build():
    _ensure_inputs()
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]

    png_tag = ""
    if os.path.exists(PNG_PATH):
        b64 = base64.b64encode(open(PNG_PATH, "rb").read()).decode()
        png_tag = (f'<img class="roc" alt="ROC/PR 基准图" '
                   f'src="data:image/png;base64,{b64}">')

    benign = data["benign"]
    benign_rows = "".join(
        f'<tr class="ok"><td><code>{_esc(x["tool"])}</code></td>'
        f'<td><span class="pill ok">{DECISION_TXT[x["decision"]]}</span></td>'
        f'<td>{x["latency_ms"]} ms</td></tr>' for x in benign["steps"])

    scenarios_html = "".join(_render_scenario(r) for r in data["scenarios"])

    def yn(b):
        return '<span class="pill ok">是</span>' if b else '<span class="pill bad">否</span>'

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Agent 攻防闭环演示 · 网安赛#4</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
         margin:0; background:#0f1419; color:#d8dee4; line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 22px 60px; }}
  h1 {{ text-align:center; color:#fff; margin:0 0 4px; font-size:1.7em; }}
  .sub {{ text-align:center; color:#8b98a5; margin:0 0 24px; font-size:.95em; }}
  .dash {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
          gap:12px; margin:0 0 28px; }}
  .kpi-card {{ background:#1a2230; border:1px solid #2a3550; border-radius:10px;
             padding:14px; text-align:center; }}
  .kpi-card .big {{ font-size:1.7em; font-weight:700; color:#fff; }}
  .kpi-card .lbl {{ font-size:.82em; color:#8b98a5; margin-top:4px; }}
  .delta {{ color:#3ddc84; }}
  h2 {{ color:#fff; border-left:4px solid #4a90d9; padding-left:12px; margin-top:40px; }}
  section.scenario {{ background:#161d29; border:1px solid #283449; border-radius:12px;
                     padding:18px 20px; margin:18px 0; }}
  section.scenario h3 {{ color:#fff; margin:0 0 6px; font-size:1.15em; }}
  .tactic {{ color:#8b98a5; font-size:.9em; margin-bottom:8px; }}
  .narr {{ color:#b8c2cf; margin:6px 0 12px; }}
  .skill {{ background:#241a1a; border:1px solid #5a2d2d; border-radius:8px;
           padding:8px 12px; margin-bottom:12px; font-size:.88em; word-break:break-all; }}
  .phase {{ margin:14px 0; }}
  .phase h4 {{ color:#7fb3ec; margin:0 0 8px; font-size:.98em; }}
  .det {{ background:#10161f; border-radius:8px; padding:10px 12px; margin:6px 0; }}
  .det-head .verdict {{ background:#5a2d2d; color:#ff9b9b; padding:1px 8px;
                       border-radius:4px; font-size:.8em; margin-left:6px; }}
  .det-head .score {{ color:#ffd479; margin-left:8px; font-size:.85em; }}
  ul.evlist {{ margin:6px 0 0; padding-left:18px; font-size:.86em; color:#9fb0c0; }}
  ul.evlist .rule {{ color:#7fb3ec; }}
  ul.evlist .snip {{ color:#e0a060; }}
  ul.rules {{ margin:4px 0; padding-left:18px; }}
  ul.rules code {{ color:#3ddc84; }}
  .compare {{ display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; }}
  .compare .col {{ flex:1 1 320px; }}
  .compare .arrow {{ align-self:center; font-size:2em; color:#4a90d9; }}
  table {{ width:100%; border-collapse:collapse; font-size:.84em; margin:4px 0; }}
  th,td {{ border:1px solid #283449; padding:5px 8px; text-align:left; }}
  th {{ background:#1c2738; color:#a8b6c5; }}
  code {{ background:#0c1118; padding:1px 5px; border-radius:3px; color:#c8d4e0; }}
  .pill {{ padding:2px 9px; border-radius:10px; font-size:.82em; font-weight:600; white-space:nowrap; }}
  .pill.ok {{ background:#13351f; color:#3ddc84; }}
  .pill.warn {{ background:#3a2f10; color:#ffd479; }}
  .pill.bad {{ background:#3a1414; color:#ff7b7b; }}
  tr.bad td {{ background:#1c1010; }}
  tr.warn td {{ background:#1c1808; }}
  .kpi {{ font-size:.85em; color:#9fb0c0; margin-top:6px; }}
  .roc {{ width:100%; border-radius:10px; border:1px solid #283449; background:#fff; margin-top:8px; }}
  .foot {{ text-align:center; color:#5a6675; margin-top:40px; font-size:.82em; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>通用 AI Agent 攻防闭环演示</h1>
  <p class="sub">"华为杯"第五届中国研究生网络安全创新大赛 · 揭榜题#4 · 红队攻击 → 检测命中 → 自动防御 → 攻击失败</p>

  <div class="dash">
    <div class="kpi-card"><div class="big">{s['n_scenarios']}</div><div class="lbl">合成攻击场景</div></div>
    <div class="kpi-card"><div class="big">{s['n_loops_closed']}/{s['n_scenarios']}</div><div class="lbl">闭环成功</div></div>
    <div class="kpi-card"><div class="big delta">{s['attacker_success_before']}→{s['attacker_success_after']}</div><div class="lbl">攻击者得手步数(加规则前→后)</div></div>
    <div class="kpi-card"><div class="big delta">{s['blocked_steps_before']}→{s['blocked_steps_after']}</div><div class="lbl">拦截步数(加规则前→后)</div></div>
    <div class="kpi-card"><div class="big">{yn(s['benign_zero_false_positive'])}</div><div class="lbl">良性零误报</div></div>
    <div class="kpi-card"><div class="big">{s['max_decision_latency_ms']}ms</div><div class="lbl">最大决策延时(<1s {yn(s['latency_under_1s'])})</div></div>
  </div>

  <h2>一、红队攻击 → 检测 → 自动防御 → 复测闭环</h2>
  {scenarios_html}

  <h2>二、良性任务对照(零误报)</h2>
  <section class="scenario">
    <p class="narr">同一拦截器跑正常业务任务,应全程放行且链路不告警,证明防御不伤可用性。</p>
    <table><tr><th>工具</th><th>决策</th><th>延时</th></tr>{benign_rows}</table>
    <div class="kpi">全部放行: {'是' if benign['all_allowed'] else '否'}　链路异常告警: {'是' if benign['chain_alert'] else '否'}　最大延时 {benign['max_latency_ms']} ms</div>
  </section>

  <h2>三、检测内核量化基准(ROC / PR)</h2>
  {png_tag or '<p class="narr">(基准图未生成,运行 benchmark_plot.py 后重建)</p>'}

  <p class="foot">本页为离线攻防演示,数据/图均由 src 内已自测检测内核在合成轨迹上真实跑出并内嵌,
  浏览器双击即开。官方靶场到手后接真 Agent 即可复测。</p>
</div>
</body>
</html>"""

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.abspath(HTML_PATH), len(page)


def _selftest():
    import sys
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m); ok = ok and c

    path, n = build()
    check(os.path.exists(path) and os.path.getsize(path) > 8000,
          f"demo.html 已生成且非空({os.path.getsize(path)} bytes)")
    txt = open(path, encoding="utf-8").read()
    check("<!DOCTYPE html>" in txt and "</html>" in txt, "HTML 结构完整")
    # 关键锚点:每个场景 id 都出现
    data = json.load(open(JSON_PATH, encoding="utf-8"))
    for r in data["scenarios"]:
        check(f"[{r['id']}]" in txt, f"页面含场景 {r['id']}")
    check("闭环已闭合" in txt, "页面体现闭环结论")
    check("加规则前" in txt and "加规则后" in txt, "页面含加规则前/后对照")
    check("data:image/png;base64," in txt or "未生成" in txt, "基准图已内嵌或占位")
    # 无未转义的脏数据导致标签破裂(粗检:script 注入风险)
    check("<script>" not in txt, "无内联脚本(纯静态可截图)")

    print(f"\nHTML 路径: {path}  ({n} chars)")
    print("\n" + ("✅ build_demo_html 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        path, n = build()
        print(f"demo.html 已生成: {path}  ({n} chars)")
    else:
        _selftest()
