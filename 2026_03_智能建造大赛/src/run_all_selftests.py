#!/usr/bin/env python3
"""
run_all_selftests.py · 一键运行所有本地可执行的自测（Mac 无 GPU 环境）

用法:
  python3 run_all_selftests.py

返回码: 0 = 全部通过, 1 = 有失败项

覆盖模块:
  - geolocate.py      (8项几何自测)
  - track_filter.py   (5项时序滤波自测)
  - augment_water.py  (5项增广自测)
  - losses_smalltarget.py (9项损失自测)
  - stream_qgc.py     (模块接线 + GStreamer 检查)
  - crossdomain_eval.py (随机特征域差流程)
  - prepare_data.py   (guide 输出格式检查)
  - tools/gen_water_scene.py    (合成海面+GT-Glint 增广配图自测)
  - tools/gen_report_figs.py    (分桶召回/PR/FPS-精度 图表自测)
  - tools/gen_tech_plan_docx.py (技术方案 docx 结构自测)
  - tools/gen_perf_report_docx.py (性能报告 docx 结构/OOXML/内嵌图自测)
  - configs/*.yaml    (YAML 语法合法性)

注意: train.py / eval.py / export_onnx.py / trt_infer_orin.py 需真实数据/GPU/Orin,
      不在本地自测范围内(由 Makefile 在 Docker 中跑)。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

TESTS = [
    ("geolocate",        [sys.executable, "geolocate.py"]),
    ("track_filter",     [sys.executable, "track_filter.py"]),
    ("augment_water",    [sys.executable, "augment_water.py", "--selftest"]),
    ("losses_smalltarget", [sys.executable, "losses_smalltarget.py"]),
    ("stream_qgc",       [sys.executable, "stream_qgc.py", "--selftest"]),
    ("crossdomain_eval", [sys.executable, "crossdomain_eval.py", "--demo"]),
    ("prepare_data_guide", [sys.executable, "prepare_data.py", "--root", "./datasets", "--step", "guide"]),
    ("gen_water_scene",  [sys.executable, "tools/gen_water_scene.py", "--selftest"]),
    ("gen_report_figs",  [sys.executable, "tools/gen_report_figs.py", "--selftest"]),
    ("gen_tech_plan_docx", [sys.executable, "tools/gen_tech_plan_docx.py", "--selftest"]),
    ("gen_perf_report_docx", [sys.executable, "tools/gen_perf_report_docx.py", "--selftest"]),
]


def check_yaml(path: Path):
    """验证 YAML 语法可解析。"""
    try:
        import yaml
        yaml.safe_load(path.read_text(encoding="utf-8"))
        return True, f"{path.name} YAML 语法 OK"
    except Exception as e:
        return False, f"{path.name} YAML 解析失败: {e}"


def main():
    results = []
    fails = 0

    print("=" * 60)
    print("03 智能建造大赛 · 本地自测总控")
    print("=" * 60)

    for name, cmd in TESTS:
        print(f"\n>>> [{name}] {' '.join(cmd)}")
        try:
            r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0
            if not ok:
                fails += 1
            # 打印 stdout 最后几行(结果摘要)
            lines = r.stdout.strip().splitlines()
            for ln in lines[-8:]:
                print("  " + ln)
            if r.stderr.strip():
                # 过滤掉常见警告
                err = r.stderr.strip()
                if "RuntimeWarning" not in err:
                    print(f"  [stderr] {err[:200]}")
            results.append((name, ok))
        except subprocess.TimeoutExpired:
            print(f"  ❌ {name} 超时")
            fails += 1
            results.append((name, False))
        except Exception as e:
            print(f"  ❌ {name} 异常: {e}")
            fails += 1
            results.append((name, False))

    # YAML 检查
    print("\n>>> [configs YAML 语法检查]")
    for yf in sorted((HERE / "configs").glob("*.yaml")):
        ok, msg = check_yaml(yf)
        if not ok:
            fails += 1
        print(f"  {'✅' if ok else '❌'} {msg}")
        results.append((yf.name, ok))

    # 汇总
    print("\n" + "=" * 60)
    print("汇总:")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  总计: {len(results)} 项, 通过 {len(results)-fails} 项, 失败 {fails} 项")
    print("=" * 60)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
