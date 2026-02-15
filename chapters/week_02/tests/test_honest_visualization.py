"""
Week 02 测试：诚实可视化（Honest Visualization）

测试覆盖：
1. exercise_5_honest_visualization - 诚实可视化示例

由于 solution.py 主要是参考答案实现，这里的测试主要是烟雾测试，
确保示例代码能运行并生成正确的图表。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")


# =============================================================================
# Test: 诚实可视化示例
# =============================================================================

class TestHonestVisualizationExample:
    """测试诚实可视化示例函数"""

    def test_function_exists(self):
        """检查函数存在"""
        assert hasattr(solution, 'exercise_5_honest_visualization')

    def test_run_example_creates_files(self, tmp_path: Path):
        """
        运行示例并检查生成的文件

        期望：应生成诚实可视化对比图
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        solution.exercise_5_honest_visualization(output_dir)

        # 检查文件是否生成
        expected_file = output_dir / "honest_viz_solution.png"
        assert expected_file.exists(), f"应生成 {expected_file}"

        # 检查文件大小（应该非空）
        assert expected_file.stat().st_size > 0


# =============================================================================
# Test: 诚实可视化原则验证
# =============================================================================

class TestHonestVisualizationPrinciples:
    """测试诚实可视化的原则"""

    def test_compare_truncated_vs_honest(self):
        """
        对比截断 Y 轴与诚实 Y 轴的差异

        期望：应能说明视觉差异
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("seaborn 或 matplotlib 不可用")

        penguins = sns.load_dataset("penguins")
        mean_mass = penguins.groupby("species")["body_mass_g"].mean()

        # 创建两张对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 截断 Y 轴
        axes[0].bar(mean_mass.index, mean_mass.values)
        axes[0].set_ylim(3000, 5500)
        truncated_ylim = axes[0].get_ylim()

        # 诚实 Y 轴
        axes[1].bar(mean_mass.index, mean_mass.values)
        axes[1].set_ylim(0, 6000)
        honest_ylim = axes[1].get_ylim()

        plt.close()

        # 验证 Y 轴范围确实不同
        assert truncated_ylim[0] > 0, "截断图 Y 轴应从非零值开始"
        assert honest_ylim[0] == 0, "诚实图 Y 轴应从 0 开始"

    def test_penguins_species_comparison(self):
        """
        测试 Penguins 物种比较

        期望：Gentoo 应该是最重的
        """
        try:
            import seaborn as sns
        except ImportError:
            pytest.skip("seaborn 不可用")

        penguins = sns.load_dataset("penguins")
        mean_mass = penguins.groupby("species")["body_mass_g"].mean()

        # 验证 Gentoo 最重
        assert mean_mass['Gentoo'] > mean_mass['Adelie']
        assert mean_mass['Gentoo'] > mean_mass['Chinstrap']

        # 验证实际数值
        assert 5000 < mean_mass['Gentoo'] < 5200  # 约 5076g
        assert 3600 < mean_mass['Adelie'] < 3800    # 约 3700g
        assert 3700 < mean_mass['Chinstrap'] < 3800 # 约 3733g
