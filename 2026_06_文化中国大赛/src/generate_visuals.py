#!/usr/bin/env python3
"""
generate_visuals.py · 数字人文可视化素材生成器

为《颜氏家训》书籍生命史项目生成静态可视化数据(JSON/HTML):
  - 版本流变时间轴数据 (timeline.json)
  - 传播下沉路径数据 (diffusion.json)
  - 教化主题词云数据 (wordcloud.json)
  - 互文网络数据 (network.json)
  - 静态可视化网页 (visual.html) — 供嵌入PPT或独立展示

纯标准库,无外部依赖。`python generate_visuals.py` 自测。
"""
import json
import os
import sys
from text_tools import export_wordcloud_json
from similarity_search import find_similar

# 输出目录用 __file__ 相对(项目根/output), 与 render_figures/build_pptx 一致,
# 避免从 src/ 内直接运行时在 src/output/ 误建文件。
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUTDIR = os.path.join(PROJ, "output")


# ============ 核心数据:书籍生命史 ============

VERSION_TIMELINE = [
    {"era": "南北朝·梁陈",   "year": "约589",    "form": "手写抄本",       "desc": "颜之推于乱世成书,以抄本在士族间流传",       "note": "成书"},
    {"era": "唐",          "year": "7-9世纪",  "form": "宫廷/寺院抄本",   "desc": "《颜氏家训》入《隋书·经籍志》,为士族家学教材", "note": "官方著录"},
    {"era": "宋",          "year": "约960-1279", "form": "雕版刻本",       "desc": "首次雕版印刷,成本降低,开始走出士族",         "note": "刻本起点"},
    {"era": "明",          "year": "1368-1644", "form": "坊刻普及本",     "desc": "民间书坊大量翻刻,价格亲民,进入蒙学/家塾",     "note": "下沉关键"},
    {"era": "清",          "year": "1644-1912", "form": "家刻/族谱引用",  "desc": "家训条文被族谱、家规直接引用,融入日常家礼",   "note": "民间日用"},
    {"era": "民国",        "year": "1912-1949", "form": "标点排印本",     "desc": "现代标点排印,白话注释,面向大众读者",         "note": "现代转型"},
    {"era": "当代",        "year": "1950-至今",  "form": "数字化/数据库",  "desc": "中华经典古籍库、爱如生等数字化,全球可检索",   "note": "数字活化"},
]

DIFFUSION_PATH = [
    {"level": "士族家学",    "audience": "门阀士族",    "channel": "手抄/家传",      "example": "颜氏家族内部传承,作为教子教材"},
    {"level": "蒙学读物",    "audience": "私塾学童",    "channel": "坊刻/书坊",      "example": "与《三字经》《弟子规》并列家塾教材"},
    {"level": "族谱家规",    "audience": "宗族成员",    "channel": "族谱刻印/口头",   "example": "家训条文被摘录进各姓族谱、家规"},
    {"level": "民间日用",    "audience": "普通百姓",    "channel": "乡规民约/节庆",   "example": "治家、教子理念融入婚丧嫁娶、年节礼仪"},
]

# 互文网络:核心节点+边
NETWORK_NODES = [
    {"id": "颜氏家训",    "group": "核心",    "era": "南北朝"},
    {"id": "朱子家训",    "group": "家训",    "era": "清"},
    {"id": "弟子规",     "group": "蒙书",    "era": "清"},
    {"id": "三字经",     "group": "蒙书",    "era": "宋"},
    {"id": "温公家范",   "group": "家训",    "era": "宋"},
    {"id": "袁氏世范",   "group": "家训",    "era": "宋"},
]
NETWORK_EDGES = [
    {"source": "颜氏家训", "target": "朱子家训", "relation": "主题继承",     "strength": 0.8},
    {"source": "颜氏家训", "target": "弟子规",  "relation": "教化理念影响", "strength": 0.6},
    {"source": "颜氏家训", "target": "三字经",  "relation": "蒙学共现",     "strength": 0.5},
    {"source": "颜氏家训", "target": "温公家范", "relation": "体例借鉴",    "strength": 0.7},
    {"source": "颜氏家训", "target": "袁氏世范", "relation": "主题继承",    "strength": 0.75},
    {"source": "朱子家训", "target": "弟子规",  "relation": "蒙学共现",    "strength": 0.55},
]

# 经典片段(用于 TF-IDF/互文演示)。
# ─ 代码内置的回退样本; 若 src/corpus/yanshi_selected.txt 存在则优先用该文件,
#   便于队员把校勘好的全本直接放进去而无需改代码(见 load_corpus)。
_FALLBACK_PASSAGES = [
    "夫圣贤之书，教人诚孝，慎言检迹，立身扬名，亦已备矣。",
    "积财千万，不如薄伎在身。伎之易习而可贵者，无过读书。",
    "父母威严而有慈，则子女畏慎而生孝矣。",
    "教妇初来，教儿婴孩。",
    "夫风化者，自上而行于下者也，自先而施于后者也。",
    "是以与善人居，如入芝兰之室，久而自芳也。",
    "人生小幼，精神专利，长成已后，思虑散逸，固须早教，勿失机也。",
    "然则可学而不可学者，其身体；可忧而不可忧者，其年寿。",
]

CORPUS_PATH = os.path.join(HERE, "corpus", "yanshi_selected.txt")


def load_corpus(path=CORPUS_PATH):
    """优先从语料文件读取片段(每行一段, # 注释/空行忽略); 文件缺失或为空则回退内置样本。
    返回 (passages, source_label)。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        passages = [ln for ln in lines if ln and not ln.startswith("#")]
        if passages:
            return passages, os.path.relpath(path, PROJ)
    except (OSError, UnicodeDecodeError):
        pass
    return list(_FALLBACK_PASSAGES), "内置回退样本"


PASSAGES, _CORPUS_SRC = load_corpus()


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_json(data, path):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {path}")


def generate_all_jsons(outdir=OUTDIR):
    """生成全部 JSON 数据文件。"""
    print("=== 生成可视化数据文件 ===")
    print(f"  语料来源: {_CORPUS_SRC} ({len(PASSAGES)} 段)")
    write_json(VERSION_TIMELINE, os.path.join(outdir, "timeline.json"))
    write_json(DIFFUSION_PATH,   os.path.join(outdir, "diffusion.json"))
    write_json(export_wordcloud_json(PASSAGES, topk=50),
               os.path.join(outdir, "wordcloud.json"))
    write_json({"nodes": NETWORK_NODES, "edges": NETWORK_EDGES},
               os.path.join(outdir, "network.json"))

    # 互文检索结果(每个片段各自的最相似片段)
    sims = []
    for i in range(len(PASSAGES)):
        top = find_similar(PASSAGES, i, topk=1, threshold=0.0)
        if top:
            sims.append({"query": PASSAGES[i], "match": PASSAGES[top[0][0]], "score": round(top[0][1], 4)})
    write_json(sims, os.path.join(outdir, "similarity_demo.json"))
    print("=== 全部 JSON 生成完毕 ===\n")


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>《颜氏家训》书籍生命史 · 数字人文可视化</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: "Noto Serif SC", "SimSun", serif; margin: 0; padding: 0; background: #f7f3eb; color: #333; }
  .container { max-width: 960px; margin: 0 auto; padding: 24px; }
  h1 { text-align: center; border-bottom: 2px solid #8b4513; padding-bottom: 12px; color: #5a2d0c; }
  h2 { color: #5a2d0c; margin-top: 32px; border-left: 4px solid #c19a6b; padding-left: 12px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
  .timeline { position: relative; padding-left: 24px; border-left: 3px solid #c19a6b; margin: 16px 0; }
  .timeline-item { margin: 16px 0; position: relative; }
  .timeline-item::before { content: ""; position: absolute; left: -31px; top: 4px; width: 12px; height: 12px; border-radius: 50%; background: #8b4513; }
  .era { font-weight: bold; color: #5a2d0c; }
  .year { color: #888; font-size: 0.9em; margin-left: 8px; }
  .tag { display: inline-block; background: #c19a6b; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 8px; }
  .wordcloud { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 16px 0; }
  .word { display: inline-block; padding: 4px 10px; border-radius: 4px; background: #e8dcc8; color: #5a2d0c; transition: transform 0.2s; }
  .word:hover { transform: scale(1.1); background: #d4c4a8; }
  .network { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 16px 0; }
  .node { padding: 8px 16px; border-radius: 20px; border: 2px solid #8b4513; background: #fff; color: #5a2d0c; font-weight: bold; }
  .node.core { background: #8b4513; color: #fff; }
  .edge-list { font-size: 0.9em; color: #666; }
  .diffusion { display: flex; flex-wrap: wrap; gap: 12px; }
  .diff-step { flex: 1 1 200px; background: #faf8f3; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }
  .diff-step h4 { margin: 0 0 8px; color: #8b4513; }
  .footer { text-align: center; margin-top: 40px; color: #999; font-size: 0.85em; }
</style>
</head>
<body>
<div class="container">
  <h1>《颜氏家训》书籍生命史</h1>
  <p style="text-align:center;color:#666;">数字人文可视化 · 赛题2「典籍与日常」</p>

  <h2>一、版本流变时间轴</h2>
  <div class="timeline" id="timeline"></div>

  <h2>二、传播下沉路径</h2>
  <div class="diffusion" id="diffusion"></div>

  <h2>三、教化主题词云</h2>
  <div class="wordcloud" id="wordcloud"></div>

  <h2>四、互文知识网络</h2>
  <div class="network" id="network"></div>
  <div class="edge-list" id="edge-list"></div>

  <div class="footer">本作品为数字人文展示,技术仅作辅助,文化内涵为核心。</div>
</div>
<script>
// ============ 数据(内嵌,无需外部JSON) ============
const TIMELINE = {TIMELINE_DATA};
const DIFFUSION = {DIFFUSION_DATA};
const WORDCLOUD = {WORDCLOUD_DATA};
const NETWORK = {NETWORK_DATA};

// ============ 渲染 ============
function renderTimeline() {
  const el = document.getElementById('timeline');
  el.innerHTML = TIMELINE.map(item => `
    <div class="timeline-item">
      <span class="era">${item.era}</span>
      <span class="year">${item.year}</span>
      <span class="tag">${item.form}</span>
      <div>${item.desc}</div>
    </div>
  `).join('');
}

function renderDiffusion() {
  const el = document.getElementById('diffusion');
  el.innerHTML = DIFFUSION.map(d => `
    <div class="diff-step">
      <h4>${d.level}</h4>
      <div><b>受众:</b> ${d.audience}</div>
      <div><b>渠道:</b> ${d.channel}</div>
      <div><b>例:</b> ${d.example}</div>
    </div>
  `).join('');
}

function renderWordcloud() {
  const el = document.getElementById('wordcloud');
  const maxW = Math.max(...WORDCLOUD.map(w => w.weight));
  const minW = Math.min(...WORDCLOUD.map(w => w.weight));
  el.innerHTML = WORDCLOUD.map(w => {
    const size = 0.8 + 1.4 * (w.weight - minW) / (maxW - minW || 1);
    return `<span class="word" style="font-size:${size.toFixed(2)}em">${w.text}</span>`;
  }).join('');
}

function renderNetwork() {
  const el = document.getElementById('network');
  const edgesEl = document.getElementById('edge-list');
  el.innerHTML = NETWORK.nodes.map(n =>
    `<span class="node ${n.group === '核心' ? 'core' : ''}">${n.id}</span>`
  ).join('');
  edgesEl.innerHTML = '<b>互文关系:</b> ' + NETWORK.edges.map(e =>
    `${e.source}→${e.target} (${e.relation})`
  ).join(' · ');
}

renderTimeline();
renderDiffusion();
renderWordcloud();
renderNetwork();
</script>
</body>
</html>
'''


def generate_html(outdir=OUTDIR):
    """生成单个静态 HTML 文件(内嵌全部数据,无需外部JSON)。"""
    ensure_dir(os.path.join(outdir, "visual.html"))
    wc = export_wordcloud_json(PASSAGES, topk=50)
    html = HTML_TEMPLATE
    html = html.replace("{TIMELINE_DATA}", json.dumps(VERSION_TIMELINE, ensure_ascii=False))
    html = html.replace("{DIFFUSION_DATA}", json.dumps(DIFFUSION_PATH, ensure_ascii=False))
    html = html.replace("{WORDCLOUD_DATA}", json.dumps(wc, ensure_ascii=False))
    html = html.replace("{NETWORK_DATA}", json.dumps({"nodes": NETWORK_NODES, "edges": NETWORK_EDGES}, ensure_ascii=False))

    path = os.path.join(outdir, "visual.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] {path} (单文件,内嵌数据,可直接浏览器打开)")


def _selftest():
    ok = True

    def check(c, m):
        nonlocal ok
        print(("  ✅ " if c else "  ❌ ") + m)
        ok = ok and c

    # 1. 数据完整性
    check(len(VERSION_TIMELINE) == 7, f"时间轴7个节点(={len(VERSION_TIMELINE)})")
    check(len(DIFFUSION_PATH) == 4, f"传播路径4层(={len(DIFFUSION_PATH)})")
    check(len(NETWORK_NODES) == 6, f"网络节点6个(={len(NETWORK_NODES)})")
    check(len(NETWORK_EDGES) == 6, f"网络边6条(={len(NETWORK_EDGES)})")

    # 1b. 语料加载: 优先文件、缺失回退, 两条路径都验证一遍
    file_psg, file_src = load_corpus()
    check(len(file_psg) >= 8, f"语料加载非空({len(file_psg)} 段, 来源: {file_src})")
    missing_psg, missing_src = load_corpus(os.path.join(HERE, "corpus", "_nonexistent_.txt"))
    check(missing_psg == _FALLBACK_PASSAGES and missing_src == "内置回退样本",
          "语料文件缺失时正确回退到内置样本")

    # 2. 生成文件(用 __file__ 相对的 OUTDIR, 不受运行目录影响)
    outdir = OUTDIR
    generate_all_jsons(outdir)
    generate_html(outdir)

    for fname in ["timeline.json", "diffusion.json", "wordcloud.json", "network.json", "similarity_demo.json", "visual.html"]:
        fpath = os.path.join(outdir, fname)
        check(os.path.exists(fpath), f"文件存在: {fname}")

    # 3. visual.html 内容检查
    with open(os.path.join(outdir, "visual.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    check("颜氏家训" in html_content, "HTML包含作品名")
    check("<script>" in html_content, "HTML包含内嵌JS")
    check("TIMELINE" in html_content, "HTML内嵌时间轴数据")

    # 4. JSON 数据有效性
    with open(os.path.join(outdir, "wordcloud.json"), "r", encoding="utf-8") as f:
        wc = json.load(f)
    check(len(wc) > 0 and "text" in wc[0] and "weight" in wc[0], "词云JSON结构正确")

    with open(os.path.join(outdir, "network.json"), "r", encoding="utf-8") as f:
        net = json.load(f)
    check("nodes" in net and "edges" in net, "网络JSON结构正确")

    print("\n" + ("✅ generate_visuals 自测通过" if ok else "❌ 自测未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
