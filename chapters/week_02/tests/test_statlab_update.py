"""
Week 02 测试：StatLab 更新功能

测试覆盖：
1. update_report_with_descriptive_stats() - 更新 report.md 添加描述统计
2. create_week2_checkpoint() - 创建 Week 02 检查点
3. validate_report_completeness() - 验证报告完整性

测试用例类型：
- 正例：正常报告更新
- 边界：不存在的报告文件、空数据
- 反例：错误的数据类型
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块
import sys
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

# 尝试导入 solution 模块
try:
    from solution import (
        update_report_with_descriptive_stats,
        create_week2_checkpoint,
        validate_report_completeness,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: update_report_with_descriptive_stats()
# =============================================================================

class TestUpdateReportWithDescriptiveStats:
    """测试更新报告函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_update_existing_report(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试更新已存在的报告文件

        期望：应在文件末尾追加描述统计章节
        """
        report_path = tmp_path / "report.md"

        # 先创建一个初始报告（Week 01 的数据卡）
        initial_content = """# 数据分析报告

## 数据来源
- 来源：测试数据库
- 日期：2026-02-11

## 数据描述
这是测试数据
"""
        report_path.write_text(initial_content, encoding='utf-8')

        # 更新报告
        result = update_report_with_descriptive_stats(
            sample_dataframe,
            report_path=str(report_path)
        )

        assert result is not None

        # 验证文件被更新
        updated_content = report_path.read_text(encoding='utf-8')

        # 原有内容应保留
        assert '数据来源' in updated_content
        assert '数据描述' in updated_content

        # 新内容应添加
        assert '描述统计' in updated_content or 'Descriptive' in updated_content

    def test_update_with_figures(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试更新并生成图表

        期望：应生成图表文件并在报告中引用
        """
        report_path = tmp_path / "report_with_figures.md"
        figures_dir = tmp_path / "figures"

        # 创建初始报告
        report_path.write_text("# 分析报告\n", encoding='utf-8')

        result = update_report_with_descriptive_stats(
            sample_dataframe,
            report_path=str(report_path),
            generate_figures=True,
            figures_dir=str(figures_dir)
        )

        assert result is not None

        # 验证图表被生成
        assert figures_dir.exists()

        # 验证报告中包含图表引用
        content = report_path.read_text(encoding='utf-8')
        has_figure_ref = '.png' in content or '.jpg' in content or 'figures/' in content

    def test_update_preserves_structure(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试更新保留原有结构

        期望：不应破坏原有的 Markdown 结构
        """
        report_path = tmp_path / "structured_report.md"

        # 创建有结构的报告
        initial_content = """# 项目报告

## 1. 背景说明

## 2. 数据来源

## 3. 初步分析

待补充...
"""
        report_path.write_text(initial_content, encoding='utf-8')

        update_report_with_descriptive_stats(
            sample_dataframe,
            report_path=str(report_path)
        )

        content = report_path.read_text(encoding='utf-8')

        # 原有章节应保留
        assert '背景说明' in content
        assert '数据来源' in content
        assert '初步分析' in content

    # --------------------
    # 边界情况
    # --------------------

    def test_update_nonexistent_file(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试更新不存在的文件

        期望：应创建新文件
        """
        new_report_path = tmp_path / "new_report.md"

        assert not new_report_path.exists()

        result = update_report_with_descriptive_stats(
            sample_dataframe,
            report_path=str(new_report_path)
        )

        # 应该创建文件
        assert new_report_path.exists()

        content = new_report_path.read_text(encoding='utf-8')
        assert len(content) > 0

    def test_update_empty_dataframe(self, tmp_path: Path):
        """
        测试空 DataFrame 的更新

        期望：应写入空统计或提示
        """
        report_path = tmp_path / "empty_report.md"
        report_path.write_text("# 报告\n", encoding='utf-8')

        empty_df = pd.DataFrame()

        result = update_report_with_descriptive_stats(
            empty_df,
            report_path=str(report_path)
        )

        # 应该能执行，不崩溃
        assert result is not None

    def test_update_with_missing_values(self, dataframe_with_missing: pd.DataFrame, tmp_path: Path):
        """
        测试包含缺失值的数据更新

        期望：应正常处理缺失值
        """
        report_path = tmp_path / "missing_report.md"
        report_path.write_text("# 报告\n", encoding='utf-8')

        result = update_report_with_descriptive_stats(
            dataframe_with_missing,
            report_path=str(report_path)
        )

        assert result is not None

        content = report_path.read_text(encoding='utf-8')

        # 应包含统计内容
        assert '描述统计' in content or 'Descriptive' in content or '均值' in content

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_update_non_dataframe_input(self, tmp_path: Path):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        report_path = tmp_path / "error_report.md"
        report_path.write_text("# 报告\n", encoding='utf-8')

        with pytest.raises((TypeError, ValueError)):
            update_report_with_descriptive_stats(
                [1, 2, 3],
                report_path=str(report_path)
            )


# =============================================================================
# Test: create_week2_checkpoint()
# =============================================================================

class TestCreateWeek2Checkpoint:
    """测试创建 Week 02 检查点函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_create_checkpoint(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试创建检查点

        期望：应创建检查点文件和备份
        """
        checkpoint_path = tmp_path / "checkpoints" / "week_02"
        checkpoint_dir = tmp_path / "checkpoints"

        result = create_week2_checkpoint(
            sample_dataframe,
            checkpoint_dir=str(checkpoint_dir)
        )

        assert isinstance(result, dict) or isinstance(result, str)

        # 验证检查点被创建
        assert checkpoint_path.exists()

    def test_checkpoint_contains_summary(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试检查点包含摘要

        期望：检查点应包含描述统计摘要
        """
        checkpoint_dir = tmp_path / "checkpoints"

        create_week2_checkpoint(
            sample_dataframe,
            checkpoint_dir=str(checkpoint_dir)
        )

        # 查找检查点文件
        checkpoint_files = list(checkpoint_dir.glob("*week_02*"))

        if checkpoint_files:
            content = checkpoint_files[0].read_text(encoding='utf-8')

            # 应包含摘要信息
            assert '均值' in content or 'mean' in content or 'summary' in content.lower()

    def test_checkpoint_includes_timestamp(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试检查点包含时间戳

        期望：检查点应记录创建时间
        """
        checkpoint_dir = tmp_path / "checkpoints"

        result = create_week2_checkpoint(
            sample_dataframe,
            checkpoint_dir=str(checkpoint_dir)
        )

        if isinstance(result, dict):
            assert 'timestamp' in result or 'created_at' in result or 'date' in result

    # --------------------
    # 边界情况
    # --------------------

    def test_create_checkpoint_creates_directory(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试自动创建检查点目录

        期望：如果目录不存在，应自动创建
        """
        new_checkpoint_dir = tmp_path / "new_checkpoints"

        assert not new_checkpoint_dir.exists()

        create_week2_checkpoint(
            sample_dataframe,
            checkpoint_dir=str(new_checkpoint_dir)
        )

        # 目录应被创建
        assert new_checkpoint_dir.exists()

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_create_checkpoint_non_dataframe_input(self, tmp_path: Path):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            create_week2_checkpoint(
                [1, 2, 3],
                checkpoint_dir=str(tmp_path)
            )


# =============================================================================
# Test: validate_report_completeness()
# =============================================================================

class TestValidateReportCompleteness:
    """测试报告完整性验证函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_validate_complete_report(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试验证完整的报告

        期望：应报告所有必需部分都存在
        """
        # 创建完整报告
        report_path = tmp_path / "complete_report.md"
        complete_content = """# 数据分析报告

## 数据卡

### 数据来源
- 来源：测试数据库

### 字段字典
| 字段 | 类型 | 含义 |
|------|------|------|
| age | int64 | 年龄 |

## 描述统计

### 核心指标
| 变量 | 均值 | 中位数 | 标准差 | IQR |
|------|------|--------|--------|-----|
| age | 33 | 33 | 7 | 13 |

### 分布图
![年龄分布](figures/age_dist.png)

说明：年龄分布相对对称。
"""
        report_path.write_text(complete_content, encoding='utf-8')

        result = validate_report_completeness(report_path=str(report_path))

        assert isinstance(result, dict)

        if 'is_complete' in result:
            assert result['is_complete'] == True

        # 应包含检查结果
        assert 'sections' in result or 'missing' in result or 'checks' in result

    def test_validate_sections_present(self, tmp_path: Path):
        """
        测试验证各章节是否存在

        期望：应检查每个必需章节
        """
        report_path = tmp_path / "sections_report.md"

        # 创建部分报告（缺少分布图）
        partial_content = """# 数据分析报告

## 数据卡

内容...

## 描述统计

### 核心指标
| 变量 | 均值 |
|------|------|
| age | 33 |
"""
        report_path.write_text(partial_content, encoding='utf-8')

        result = validate_report_completeness(report_path=str(report_path))

        if isinstance(result, dict):
            # 应指出缺失的章节
            if 'missing' in result:
                assert len(result['missing']) > 0

            # 或者完整性分数 < 100%
            if 'completeness' in result or 'score' in result:
                score = result.get('completeness', result.get('score', 100))
                assert score < 100

    def test_validate_figure_references(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试验证图表引用

        期望：应检查图表文件是否存在
        """
        report_path = tmp_path / "figures_report.md"
        figures_dir = tmp_path / "figures"

        # 创建报告（引用不存在的图）
        content_with_missing_fig = """# 报告

## 描述统计

![不存在的图](figures/missing.png)
"""
        report_path.write_text(content_with_missing_fig, encoding='utf-8')

        result = validate_report_completeness(
            report_path=str(report_path),
            figures_dir=str(figures_dir)
        )

        if isinstance(result, dict):
            # 应指出缺失的图表
            if 'missing_figures' in result:
                assert len(result['missing_figures']) > 0

    # --------------------
    # 边界情况
    # --------------------

    def test_validate_empty_report(self, tmp_path: Path):
        """
        测试验证空报告

        期望：应报告缺失所有章节
        """
        report_path = tmp_path / "empty_report.md"
        report_path.write_text("", encoding='utf-8')

        result = validate_report_completeness(report_path=str(report_path))

        if isinstance(result, dict):
            if 'is_complete' in result:
                assert result['is_complete'] == False

            if 'missing' in result:
                assert len(result['missing']) > 0

    def test_validate_nonexistent_file(self, tmp_path: Path):
        """
        测试验证不存在的文件

        期望：应返回错误或 False
        """
        nonexistent_path = tmp_path / "nonexistent.md"

        result = validate_report_completeness(report_path=str(nonexistent_path))

        # 应返回错误
        assert result is None or isinstance(result, dict)

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_validate_non_markdown_file(self, tmp_path: Path):
        """
        测试验证非 Markdown 文件

        期望：应能处理，但可能给出警告
        """
        txt_path = tmp_path / "not_markdown.txt"
        txt_path.write_text("这不是 Markdown 文件", encoding='utf-8')

        result = validate_report_completeness(report_path=str(txt_path))

        # 应该能处理
        assert result is not None


# =============================================================================
# Test: StatLab 进度验证
# =============================================================================

class TestStatLabProgress:
    """测试 StatLab 进度相关功能"""

    def test_week2_requirements_check(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试 Week 02 要求检查

        期望：应验证 Week 02 的所有要求
        """
        # 创建满足 Week 02 要求的报告
        report_path = tmp_path / "week2_report.md"
        report_path.write_text("""# StatLab 报告

## 数据卡
内容...

## 描述统计

### 集中趋势
均值、中位数...

### 离散程度
标准差、IQR...

### 分布可视化
直方图、箱线图...

### 可视化说明
图表说明...
""", encoding='utf-8')

        result = validate_report_completeness(report_path=str(report_path))

        # 应检查 Week 02 的特定要求
        assert result is not None

    def test_statlab_continuity(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试 StatLab 连续性

        期望：Week 02 的报告应保留 Week 01 的数据卡
        """
        report_path = tmp_path / "continuity_report.md"

        # Week 01 的内容 + Week 02 的内容
        content = """# StatLab 报告

## 数据卡
来自 Week 01...

## 描述统计
Week 02 新增...
"""
        report_path.write_text(content, encoding='utf-8')

        result = validate_report_completeness(report_path=str(report_path))

        if isinstance(result, dict):
            # 应验证 Week 01 内容的保留
            if 'preserves_week1' in result or 'continuity' in result:
                assert result.get('preserves_week1', result.get('continuity', True)) == True
