"""
Week 08 示例代码测试

测试所有示例代码是否能正常运行。
"""
import subprocess
import sys
from pathlib import Path


def run_example(script_name: str) -> bool:
    """运行示例脚本并检查是否成功"""
    script_path = Path(__file__).parent.parent / 'examples' / script_name
    print(f"Testing {script_name}...", end=" ")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_path.parent)
        )

        if result.returncode == 0:
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL")
            print(f"  Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_solution() -> bool:
    """测试 solution.py"""
    solution_path = Path(__file__).parent.parent / 'starter_code' / 'solution.py'
    print(f"Testing solution.py...", end=" ")

    try:
        result = subprocess.run(
            [sys.executable, str(solution_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL")
            print(f"  Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Week 08 示例代码测试")
    print("=" * 60)

    examples = [
        "01_confidence_interval_basics.py",
        "02_ci_interpretation.py",
        "03_bootstrap_method.py",
        "04_bootstrap_ci_methods.py",
        "05_permutation_test.py",
        "08_statlab_ci.py",
    ]

    results = []

    for example in examples:
        results.append(run_example(example))

    results.append(test_solution())

    print("\n" + "=" * 60)
    print(f"测试结果: {sum(results)}/{len(results)} 通过")
    print("=" * 60)

    if all(results):
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败，请检查")


if __name__ == "__main__":
    main()
