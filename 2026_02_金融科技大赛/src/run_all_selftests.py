#!/usr/bin/env python3
"""
run_all_selftests.py · 赛题#23 全模块自测脚本

在无 GPU/无真实数据环境下跑通所有纯逻辑自测,验证(11 模块):
  - prepare_data: 合成数据 + 分类集/相似度对构建
  - metrics: 阈值扫描 P/R/F1 + AUC + Top-k 检索
  - similarity: 余弦相似度去重 + 跨客户套用检测
  - app_demo: 检测汇总视图数据层逻辑
  - features: 经典 CPU 特征提取器(CLIP 回退后端)
  - synth_images: 程序化合成金融影像生成器
  - classify: 分类筛面签(经典特征线性探针回退)
  - embed: 嵌入提取(经典特征回退)
  - pipeline: 端到端流水线(模拟向量 + 真实像素两条路径)
  - viz_dedup: 去重结果可视化(相似度热图 + 可疑对拼图)
  - ablation: 消融实验框架(特征组成 / PCA 维度 / 标准化 on-off)

用法:
  python run_all_selftests.py              # 运行全部自测
  python run_all_selftests.py --verbose    # 逐模块详细输出
  python run_all_selftests.py --html       # 生成 HTML 报告
"""
import argparse
import subprocess
import sys
import os
import time
import json

# 用 sys.executable 而非字面量 "python3":保证子进程与本运行器同一解释器,
# 不受 PATH 中 python3 指向不同(如 Homebrew 3.14 无 numpy vs 系统 3.9 有 numpy)影响。
PY = sys.executable
MODULES = [
    ("prepare_data", [PY, "prepare_data.py", "--selftest"]),
    ("metrics",      [PY, "metrics.py"]),
    ("similarity",   [PY, "similarity.py"]),
    ("app_demo",     [PY, "app_demo.py", "--demo"]),
    ("features",     [PY, "features.py", "--selftest"]),       # 经典 CPU 特征(CLIP 回退后端)
    ("synth_images", [PY, "synth_images.py", "--selftest"]),   # 合成金融影像生成器
    ("classify",     [PY, "classify.py", "--selftest"]),       # 分类(经典线性探针回退)
    ("embed",        [PY, "embed.py", "--selftest"]),          # 嵌入(经典特征回退)
    ("pipeline",     [PY, "pipeline.py", "--selftest"]),       # 端到端(模拟向量+真实像素)
    ("viz_dedup",    [PY, "viz_dedup.py", "--selftest"]),      # 去重可视化(热图+拼图)
    ("ablation",     [PY, "ablation_study.py", "--selftest"]), # 消融实验框架(特征组成/PCA维度/标准化)
]


def safe_console(text):
    """Return text printable on the current console without UnicodeEncodeError."""
    enc = sys.stdout.encoding or "utf-8"
    return str(text).encode(enc, errors="replace").decode(enc, errors="replace")


def run_module(name, cmd, verbose=False):
    """运行单个模块,返回 (通过?, stdout, stderr, 耗时秒)。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"[RUN] {name}")
        print(f"  命令: {' '.join(cmd)}")
    t0 = time.time()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=child_env,
    )
    elapsed = time.time() - t0
    passed = proc.returncode == 0
    if verbose:
        print(safe_console(proc.stdout))
        if proc.stderr:
            print("[stderr]", safe_console(proc.stderr))
        print(f"  结果: {'[OK]' if passed else '[FAIL]'} ({elapsed:.2f}s)")
    return passed, proc.stdout, proc.stderr, elapsed


def generate_html_report(results, out_path="selftest_report.html"):
    """生成 HTML 自测报告。"""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rows = ""
    for r in results:
        status = "[OK] 通过" if r["passed"] else "[FAIL] 失败"
        color = "#2ecc71" if r["passed"] else "#e74c3c"
        stderr = r["stderr"] or "(无)"
        rows += f"""
        <tr>
          <td><b>{r['name']}</b></td>
          <td style="color:{color}">{status}</td>
          <td>{r['elapsed']:.2f}s</td>
          <td><pre>{r['stdout'][-500:]}</pre></td>
          <td><pre>{stderr[:500]}</pre></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>赛题#23 自测报告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #f5f5f7; }}
  .card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 1200px; margin: 0 auto; }}
  h1 {{ margin-top: 0; }}
  .summary {{ font-size: 1.2em; margin: 16px 0; padding: 12px 16px; border-radius: 8px; background: {'#d4edda' if passed == total else '#f8d7da'}; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
  th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #e0e0e0; font-size: 0.9em; }}
  th {{ background: #f0f0f5; }}
  pre {{ background: #f8f8fa; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; margin: 0; }}
  .timestamp {{ color: #888; font-size: 0.85em; margin-top: 8px; }}
</style>
</head>
<body>
<div class="card">
  <h1>赛题#23 金融影像智能相似度检测 · 全模块自测报告</h1>
  <div class="summary">
    <b>{passed}/{total}</b> 模块通过 · 状态: {'[OK] 全部通过' if passed == total else '[FAIL] 存在失败'}
  </div>
  <table>
    <tr><th>模块</th><th>状态</th><th>耗时</th><th>输出(末尾500字)</th><th>错误</th></tr>
    {rows}
  </table>
  <div class="timestamp">生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] HTML 报告已生成: {os.path.abspath(out_path)}")


def main():
    ap = argparse.ArgumentParser(description="赛题#23 全模块自测")
    ap.add_argument("--verbose", "-v", action="store_true", help="逐模块详细输出")
    ap.add_argument("--html", action="store_true", help="生成 HTML 报告")
    ap.add_argument("--json", action="store_true", help="输出 JSON 结果")
    a = ap.parse_args()

    print("=" * 60)
    print("赛题#23 金融影像智能相似度检测 · 全模块自测")
    print("=" * 60)

    results = []
    all_passed = True
    for name, cmd in MODULES:
        passed, stdout, stderr, elapsed = run_module(name, cmd, verbose=a.verbose)
        results.append({
            "name": name,
            "passed": passed,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed": elapsed,
        })
        all_passed = all_passed and passed
        if not a.verbose:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {name:12s} ({elapsed:.2f}s)")

    print("\n" + "=" * 60)
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    if all_passed:
        print(f"[OK] 全部 {total} 个模块自测通过")
    else:
        print(f"[FAIL] {passed_count}/{total} 通过,存在失败模块")
        for r in results:
            if not r["passed"]:
                print(f"   -> {r['name']}: 返回码非0")
                if r["stderr"]:
                    print(f"     stderr: {r['stderr'][:200]}")
    print("=" * 60)

    if a.html:
        generate_html_report(results)

    if a.json:
        print(json.dumps([{k: v for k, v in r.items() if k in ("name", "passed", "elapsed")}
                          for r in results], ensure_ascii=False, indent=2))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
