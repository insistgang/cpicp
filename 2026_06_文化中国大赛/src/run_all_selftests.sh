#!/bin/bash
# run_all_selftests.sh · 06 文化中国大赛 全模块自测脚本
# 依次运行 src/ 下所有带 _selftest 的 Python 模块

set -euo pipefail

cd "$(dirname "$0")/.."
SRC_DIR="src"

# 选解释器: render_figures/build_pptx 依赖 matplotlib+pptx(本机仅装在某个 python3 上)。
# 优先用调用者显式指定的 $PYTHON(若它能 import matplotlib 与 pptx); 否则探测候选,
# 选第一个能 import 两者的; 都找不到则回退 python3。
pick_python() {
    local probe='import matplotlib, pptx'
    for cand in "${PYTHON:-}" python3 /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
        [ -n "$cand" ] || continue
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "$probe" >/dev/null 2>&1; then
            echo "$cand"; return 0
        fi
    done
    echo python3
}
PY="$(pick_python)"
echo "使用解释器: $PY ($("$PY" --version 2>&1))"
echo ""

PASS=0
FAIL=0

run_test() {
    local mod="$1"
    echo "========================================"
    echo "Testing: $mod"
    echo "========================================"
    if "$PY" "$SRC_DIR/$mod"; then
        echo ""
        echo "  ✅ $mod 通过"
        ((++PASS))
    else
        echo ""
        echo "  ❌ $mod 失败"
        ((++FAIL))
    fi
    echo ""
}

run_test "text_tools.py"
run_test "similarity_search.py"
run_test "originality_check.py"
run_test "generate_visuals.py"
run_test "render_figures.py"
run_test "build_pptx.py"

echo "========================================"
echo "自测汇总: 通过=$PASS  失败=$FAIL"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
