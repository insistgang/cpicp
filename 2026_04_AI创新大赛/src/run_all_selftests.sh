#!/usr/bin/env bash
# 一键回归:跑 04 全部模块自测(纯 numpy/PIL/sklearn/matplotlib,本机可跑)
set -e
cd "$(dirname "$0")"

# 选一个真带 numpy 的解释器:PATH 上的 python3 可能是无 numpy 的 Homebrew 版,
# 自动回退到带科学栈的(本机为系统 /usr/bin/python3 3.9.6)。可用 PY 环境变量覆盖。
pick_py() {
  for c in "$PY" python3 /usr/bin/python3 python; do
    [ -n "$c" ] || continue
    if "$c" -c 'import numpy' >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  echo "❌ 找不到带 numpy 的 python3(请装 numpy 或用 PY=<解释器> 指定)" >&2; return 1
}
PYBIN="$(pick_py)"
echo "解释器: $PYBIN ($("$PYBIN" --version 2>&1))"

# 无参自测(直接 python f.py)
for f in anomaly_score patchcore_lite aoi_metrics fewshot_protocol latency_bench illegal_build_pipeline feature_backend online_learning; do
  echo "=== $f ==="; "$PYBIN" "$f.py"
done
# --selftest 形式
for f in aoi_prepare augment_defect synth_aoi run_real_pipeline viz_heatmap; do
  echo "=== $f --selftest ==="; "$PYBIN" "$f.py" --selftest
done
echo "✅ 04 全部自测通过"
