#!/usr/bin/env python3
"""
run_all_selftests.py · 01 网安赛 10 模块一键自测

运行 src/ 下所有模块的 selftest,汇总结果到终端。
用法: python3 run_all_selftests.py
"""
import subprocess
import sys

MODULES = [
    "audit_schema",
    "asset_scanner",
    "config_risk_scanner",
    "malicious_skill_detector",
    "prompt_injection_detector",
    "chain_anomaly",
    "metrics_eval",
    "interceptor",
    "defense_policy",
    "fake_agent",
]


def main():
    passed = 0
    failed = 0
    results = []

    for name in MODULES:
        print(f"=== {name} ===")
        try:
            rc = subprocess.run(
                [sys.executable, f"{name}.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            ok = rc.returncode == 0
            if ok:
                passed += 1
                print(f"  ✅ {name} 自测通过")
            else:
                failed += 1
                print(f"  ❌ {name} 自测失败 (exit={rc.returncode})")
                if rc.stderr:
                    print(f"     stderr: {rc.stderr[:200]}")
            results.append((name, ok, rc.stdout, rc.stderr))
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  ❌ {name} 超时")
            results.append((name, False, "", "timeout"))
        except Exception as e:
            failed += 1
            print(f"  ❌ {name} 异常: {e}")
            results.append((name, False, "", str(e)))
        print("")

    print("=" * 50)
    print(f"汇总: {passed}/{len(MODULES)} 通过, {failed}/{len(MODULES)} 失败")
    if failed == 0:
        print("🎉 全部模块自测通过!")
    else:
        print("⚠️  以下模块需修复:")
        for name, ok, _, _ in results:
            if not ok:
                print(f"   - {name}")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
