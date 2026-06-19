#!/bin/bash
# run_all_selftests.sh · 06 文化中国大赛 全模块自测脚本
# 依次运行 src/ 下所有带 _selftest 的 Python 模块

set -euo pipefail

cd "$(dirname "$0")/.."
SRC_DIR="src"

PASS=0
FAIL=0

run_test() {
    local mod="$1"
    echo "========================================"
    echo "Testing: $mod"
    echo "========================================"
    if python3 "$SRC_DIR/$mod"; then
        echo ""
        echo "  ✅ $mod 通过"
        ((PASS++))
    else
        echo ""
        echo "  ❌ $mod 失败"
        ((FAIL++))
    fi
    echo ""
}

run_test "text_tools.py"
run_test "similarity_search.py"
run_test "generate_visuals.py"
run_test "render_figures.py"
run_test "build_pptx.py"

echo "========================================"
echo "自测汇总: 通过=$PASS  失败=$FAIL"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
