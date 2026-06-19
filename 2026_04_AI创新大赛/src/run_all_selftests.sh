#!/usr/bin/env bash
# 一键回归:跑 04 全部模块自测(纯 numpy/PIL/sklearn/matplotlib,本机可跑)
set -e
cd "$(dirname "$0")"
# 无参自测(直接 python f.py)
for f in anomaly_score patchcore_lite aoi_metrics fewshot_protocol latency_bench illegal_build_pipeline feature_backend; do
  echo "=== $f ==="; python3 "$f.py"
done
# --selftest 形式
for f in aoi_prepare augment_defect synth_aoi run_real_pipeline viz_heatmap; do
  echo "=== $f --selftest ==="; python3 "$f.py" --selftest
done
echo "✅ 04 全部自测通过"
