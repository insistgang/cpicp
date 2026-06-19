#!/usr/bin/env bash
# 一键回归:跑 04 全部模块自测(纯 numpy/标准库,本机可跑)
set -e
cd "$(dirname "$0")"
for f in anomaly_score patchcore_lite aoi_metrics fewshot_protocol latency_bench illegal_build_pipeline; do
  echo "=== $f ==="; python3 "$f.py"
done
for f in aoi_prepare augment_defect; do
  echo "=== $f --selftest ==="; python3 "$f.py" --selftest
done
echo "✅ 04 全部自测通过"
