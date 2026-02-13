"""
Week 16 示例代码测试套件

本测试文件验证 examples/ 目录下的示例代码是否可运行。
"""

import subprocess
import sys
from pathlib import Path

import pytest

# 获取 examples 目录路径
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestReproducibleTemplate:
    """测试可复现分析模板"""

    def test_template_runs_without_error(self):
        """测试模板脚本可正常运行"""
        script_path = EXAMPLES_DIR / "01_reproducible_template.py"
        assert script_path.exists(), f"脚本不存在: {script_path}"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 检查是否成功运行
        assert result.returncode == 0, f"脚本运行失败: {result.stderr}"
        assert "可复现分析模板执行完成" in result.stdout

    def test_template_outputs_version_info(self):
        """测试模板输出版本信息"""
        script_path = EXAMPLES_DIR / "01_reproducible_template.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert "依赖版本" in result.stdout
        assert "numpy" in result.stdout.lower()


class TestPresentationTemplate:
    """测试展示模板"""

    def test_template_file_exists(self):
        """测试模板文件存在"""
        script_path = EXAMPLES_DIR / "02_presentation_template.py"
        assert script_path.exists(), f"脚本不存在: {script_path}"


class TestAddStoryToReport:
    """测试添加数据故事到报告"""

    def test_template_file_exists(self):
        """测试模板文件存在"""
        script_path = EXAMPLES_DIR / "03_add_story_to_report.py"
        assert script_path.exists(), f"脚本不存在: {script_path}"


class TestAddReproducibility:
    """测试添加可复现章节"""

    def test_template_file_exists(self):
        """测试模板文件存在"""
        script_path = EXAMPLES_DIR / "04_add_reproducibility_to_report.py"
        assert script_path.exists(), f"脚本不存在: {script_path}"


class TestExportHtml:
    """测试 HTML 导出"""

    def test_template_file_exists(self):
        """测试模板文件存在"""
        script_path = EXAMPLES_DIR / "05_export_html.py"
        assert script_path.exists(), f"脚本不存在: {script_path}"


class TestSolutionCode:
    """测试作业参考实现"""

    def test_solution_file_exists(self):
        """测试参考实现文件存在"""
        solution_path = EXAMPLES_DIR.parent / "starter_code" / "solution.py"
        assert solution_path.exists(), f"参考实现不存在: {solution_path}"

    def test_solution_has_main_function(self):
        """测试参考实现有主函数"""
        solution_path = EXAMPLES_DIR.parent / "starter_code" / "solution.py"
        content = solution_path.read_text(encoding="utf-8")
        assert "def main()" in content, "参考实现缺少 main() 函数"
        assert 'if __name__ == "__main__"' in content, "参考实现缺少入口点"


class TestFileStructure:
    """测试文件结构完整性"""

    def test_all_example_files_exist(self):
        """测试所有示例文件存在"""
        expected_files = [
            "01_reproducible_template.py",
            "02_presentation_template.py",
            "03_add_story_to_report.py",
            "04_add_reproducibility_to_report.py",
            "05_export_html.py",
        ]

        for filename in expected_files:
            filepath = EXAMPLES_DIR / filename
            assert filepath.exists(), f"示例文件不存在: {filepath}"

    def test_all_example_files_are_python(self):
        """测试所有示例文件都是有效的 Python 文件"""
        for filepath in EXAMPLES_DIR.glob("*.py"):
            content = filepath.read_text(encoding="utf-8")
            # 简单检查是否包含 Python 代码特征
            assert "def " in content or "import " in content, f"文件不是有效的 Python 文件: {filepath}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
