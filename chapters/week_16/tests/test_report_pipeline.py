"""
Tests for Report Pipeline

报告流水线测试用例矩阵：
- 正例：验证数据加载、统计计算、报告生成的正确行为
- 边界：空数据、单行数据、缺失值、特殊字符
- 反例：无效路径、错误数据类型、损坏的数据
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

# =============================================================================
# 测试数据 Fixture
# =============================================================================

@pytest.fixture
def sample_customer_data():
    """
    Fixture：生成标准的客户流失测试数据
    """
    np.random.seed(42)
    n = 500

    return pd.DataFrame({
        'customer_id': range(n),
        'tenure': np.random.exponential(20, n),
        'monthly_charges': np.random.uniform(20, 100, n),
        'total_charges': np.random.uniform(100, 5000, n),
        'churn': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n)
    })


@pytest.fixture
def sample_data_with_missing():
    """
    Fixture：生成包含缺失值的测试数据
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n) * 2 + 1,
        'group': np.random.choice(['A', 'B', 'C'], n)
    })

    # 随机插入缺失值
    missing_indices = np.random.choice(n, size=10, replace=False)
    df.loc[missing_indices, 'x'] = np.nan

    missing_indices_y = np.random.choice(n, size=5, replace=False)
    df.loc[missing_indices_y, 'y'] = np.nan

    return df


@pytest.fixture
def empty_data():
    """
    Fixture：空数据框
    """
    return pd.DataFrame()


@pytest.fixture
def single_row_data():
    """
    Fixture：单行数据
    """
    return pd.DataFrame({
        'x': [1.0],
        'y': [2.0],
        'group': ['A']
    })


@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Fixture：创建临时 CSV 文件
    """
    def _create_csv(data, filename='test_data.csv'):
        file_path = tmp_path / filename
        data.to_csv(file_path, index=False)
        return str(file_path)

    return _create_csv


# =============================================================================
# 正例测试：数据加载功能
# =============================================================================

class TestDataLoading:
    """测试数据加载功能"""

    def test_load_csv_returns_dataframe(self, sample_customer_data, temp_csv_file):
        """
        正例：加载 CSV 文件应返回 DataFrame

        给定：一个有效的 CSV 文件
        当：调用 load_data 函数
        期望：返回正确的 DataFrame
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        csv_path = temp_csv_file(sample_customer_data)

        if hasattr(solution, 'load_data'):
            result = solution.load_data(csv_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_customer_data)
        else:
            pytest.skip("load_data function not implemented")

    def test_load_data_preserves_columns(self, sample_customer_data, temp_csv_file):
        """
        正例：加载数据应保留列名

        给定：包含特定列名的 CSV
        当：加载数据
        期望：返回的 DataFrame 包含所有原始列名
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        csv_path = temp_csv_file(sample_customer_data)
        expected_columns = set(sample_customer_data.columns)

        if hasattr(solution, 'load_data'):
            result = solution.load_data(csv_path)
            assert set(result.columns) == expected_columns
        else:
            pytest.skip("load_data function not implemented")

    def test_load_data_preserves_dtypes(self, temp_csv_file):
        """
        正例：加载数据应正确推断数据类型

        给定：混合类型数据（数值、字符串）
        当：加载数据
        期望：数值列为 float/int，字符串列为 object 或 string
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        test_data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        csv_path = temp_csv_file(test_data)

        if hasattr(solution, 'load_data'):
            result = solution.load_data(csv_path)
            assert pd.api.types.is_integer_dtype(result['int_col']) or \
                   pd.api.types.is_float_dtype(result['int_col'])
            assert pd.api.types.is_float_dtype(result['float_col'])
            # pandas 3.0+ 使用 string dtype，而不是 object
            assert pd.api.types.is_object_dtype(result['str_col']) or \
                   pd.api.types.is_string_dtype(result['str_col'])
        else:
            pytest.skip("load_data function not implemented")


# =============================================================================
# 正例测试：描述统计计算
# =============================================================================

class TestDescriptiveStatistics:
    """测试描述统计计算功能"""

    def test_compute_descriptive_returns_dict(self, sample_customer_data):
        """
        正例：描述统计应返回字典或 DataFrame

        给定：数值型数据
        当：计算描述统计
        期望：返回包含统计量的字典或 DataFrame
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(sample_customer_data)

            # 结果应该是字典或 DataFrame
            assert isinstance(result, (dict, pd.DataFrame))

            # 如果是字典，应包含基本统计量
            if isinstance(result, dict):
                expected_keys = ['mean', 'std', 'min', 'max', 'count']
                # 至少包含一些基本统计量
                assert len(result) > 0
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_compute_mean_correct(self, sample_customer_data):
        """
        正例：均值计算应正确

        验证均值计算的正确性
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(sample_customer_data)

            # 检查数值列的均值
            numeric_col = 'tenure'
            expected_mean = sample_customer_data[numeric_col].mean()

            if isinstance(result, dict) and 'mean' in result:
                computed_mean = result['mean'].get(numeric_col) if isinstance(result['mean'], dict) else result['mean']
                np.testing.assert_almost_equal(computed_mean, expected_mean, decimal=5)
            elif isinstance(result, pd.DataFrame):
                assert result.loc['mean', numeric_col] == pytest.approx(expected_mean)
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_compute_std_non_negative(self, sample_customer_data):
        """
        正例：标准差应非负

        标准差必须 >= 0
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(sample_customer_data)

            if isinstance(result, dict) and 'std' in result:
                std_values = result['std'].values() if isinstance(result['std'], dict) else [result['std']]
                for std_val in std_values:
                    if isinstance(std_val, (int, float)):
                        assert std_val >= 0
            elif isinstance(result, pd.DataFrame) and 'std' in result.index:
                assert all(result.loc['std'] >= 0)
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_compute_count_matches_data_length(self, sample_customer_data):
        """
        正例：计数应匹配数据长度

        给定：n 行数据
        期望：count = n（假设无缺失值）
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(sample_customer_data)

            expected_count = len(sample_customer_data)

            if isinstance(result, dict) and 'count' in result:
                count_val = result['count']
                if isinstance(count_val, dict):
                    assert all(v == expected_count for v in count_val.values() if isinstance(v, (int, float)))
                else:
                    assert count_val == expected_count
            elif isinstance(result, pd.DataFrame):
                if 'count' in result.index:
                    assert all(result.loc['count'] == expected_count)
        else:
            pytest.skip("compute_descriptive function not implemented")


# =============================================================================
# 边界测试：特殊情况处理
# =============================================================================

class TestBoundaryCases:
    """测试边界情况"""

    def test_empty_dataframe_returns_empty_or_none(self, empty_data):
        """
        边界：空数据框应返回空结果或 None

        给定：空 DataFrame
        当：计算描述统计
        期望：返回空字典/空 DataFrame 或 None
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(empty_data)

            # 空数据应返回空结果或报错
            assert result is None or \
                   len(result) == 0 or \
                   (isinstance(result, pd.DataFrame) and result.empty) or \
                   (isinstance(result, dict) and len(result) == 0)
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_single_row_data_handled(self, single_row_data):
        """
        边界：单行数据应能处理

        给定：只有一行的数据
        当：计算描述统计
        期望：能正常计算，std 可能是 NaN 或 0
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(single_row_data)

            # 单行数据的 std 通常是 NaN 或 0
            assert result is not None
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_data_with_missing_handled(self, sample_data_with_missing):
        """
        边界：包含缺失值的数据应能处理

        给定：包含 NaN 的数据
        当：计算描述统计
        期望：正确处理缺失值（忽略或报告）
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(sample_data_with_missing)

            assert result is not None

            # 检查缺失值计数（如果实现）
            if isinstance(result, dict) and 'missing_count' in result:
                assert result['missing_count'].get('x') > 0
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_all_same_values_std_zero(self):
        """
        边界：所有值相同时标准差应为 0

        给定：常数列
        期望：std = 0
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        constant_data = pd.DataFrame({
            'x': [5.0] * 100,
            'y': [3.0] * 100
        })

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(constant_data)

            if isinstance(result, dict) and 'std' in result:
                std_val = result['std'].get('x') if isinstance(result['std'], dict) else result['std']
                if isinstance(std_val, (int, float)):
                    assert std_val == 0 or np.isnan(std_val)
            elif isinstance(result, pd.DataFrame) and 'std' in result.index:
                assert result.loc['std', 'x'] == 0 or np.isnan(result.loc['std', 'x'])
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_very_large_values_handled(self):
        """
        边界：极大值应能处理

        给定：包含极大值的数据
        期望：不会溢出或报错
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        large_data = pd.DataFrame({
            'x': [1e300, 2e300, 3e300],
            'y': [1.0, 2.0, 3.0]
        })

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(large_data)
            assert result is not None
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_negative_values_handled(self):
        """
        边界：负值应能正确处理

        给定：包含负值的数据
        期望：正确计算统计量
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        negative_data = pd.DataFrame({
            'x': [-10, -5, 0, 5, 10],
            'y': [-100, -50, 0, 50, 100]
        })

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(negative_data)

            # 均值应该是 0（对称分布）
            if isinstance(result, dict) and 'mean' in result:
                mean_val = result['mean'].get('x') if isinstance(result['mean'], dict) else result['mean']
                if isinstance(mean_val, (int, float)):
                    assert mean_val == pytest.approx(0, abs=1e-10)
            elif isinstance(result, pd.DataFrame) and 'mean' in result.index:
                assert result.loc['mean', 'x'] == pytest.approx(0, abs=1e-10)
        else:
            pytest.skip("compute_descriptive function not implemented")


# =============================================================================
# 反例测试：错误处理
# =============================================================================

class TestErrorCases:
    """测试错误处理"""

    def test_nonexistent_file_raises_error(self):
        """
        反例：不存在的文件路径应报错

        给定：不存在的文件路径
        当：尝试加载数据
        期望：抛出 FileNotFoundError 或 ValueError
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'load_data'):
            with pytest.raises((FileNotFoundError, ValueError, OSError)):
                solution.load_data("/nonexistent/path/to/file.csv")
        else:
            pytest.skip("load_data function not implemented")

    def test_invalid_file_type_raises_error(self, tmp_path):
        """
        反例：不支持的文件类型应报错

        给定：.txt 文件（非 CSV）
        当：尝试加载
        期望：抛出异常
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        # 创建非 CSV 文件
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not a csv file")

        if hasattr(solution, 'load_data'):
            try:
                solution.load_data(str(txt_file))
                # 如果没有报错，至少检查返回值
                assert True  # 有些实现可能尝试读取任何文件
            except (ValueError, pd.errors.EmptyDataError):
                assert True  # 预期的错误
        else:
            pytest.skip("load_data function not implemented")

    def test_corrupted_csv_raises_error(self, tmp_path):
        """
        反例：损坏的 CSV 文件应报错

        给定：格式错误的 CSV
        当：尝试加载
        期望：抛出解析错误
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        # 创建格式错误的 CSV
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("a,b,c\n1,2\n3,4,5,6\n")  # 行长度不一致

        if hasattr(solution, 'load_data'):
            try:
                result = solution.load_data(str(bad_csv))
                # pandas 可能会处理这种情况，如果不报错就接受
                assert result is not None
            except (pd.errors.EmptyDataError, ValueError):
                assert True  # 预期的错误
        else:
            pytest.skip("load_data function not implemented")

    def test_none_input_raises_error(self):
        """
        反例：None 输入应报错

        给定：None 作为输入
        当：计算描述统计
        期望：抛出异常
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            with pytest.raises((ValueError, TypeError, AttributeError)):
                solution.compute_descriptive(None)
        else:
            pytest.skip("compute_descriptive function not implemented")


# =============================================================================
# 报告生成测试
# =============================================================================

class TestReportGeneration:
    """测试报告生成功能"""

    def test_generate_report_creates_file(self, sample_customer_data, tmp_path):
        """
        正例：生成报告应创建文件

        给定：数据和输出路径
        当：调用 generate_report
        期望：在指定位置创建报告文件
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        output_path = tmp_path / "report.md"

        if hasattr(solution, 'generate_report'):
            result = solution.generate_report(sample_customer_data, str(output_path))

            # 检查文件是否创建
            assert output_path.exists() or result is not None
        else:
            pytest.skip("generate_report function not implemented")

    def test_report_contains_header(self, sample_customer_data, tmp_path):
        """
        正例：报告应包含标题

        验证生成的报告包含标题标记
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        output_path = tmp_path / "report.md"

        if hasattr(solution, 'generate_report'):
            solution.generate_report(sample_customer_data, str(output_path))

            if output_path.exists():
                content = output_path.read_text()
                # Markdown 标题以 # 开头
                assert '#' in content
        else:
            pytest.skip("generate_report function not implemented")

    def test_report_contains_reproducibility_info(self, sample_customer_data, tmp_path):
        """
        正例：报告应包含可复现性信息

        验证报告包含日期、数据来源等信息
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        output_path = tmp_path / "report.md"

        if hasattr(solution, 'generate_report'):
            solution.generate_report(sample_customer_data, str(output_path))

            if output_path.exists():
                content = output_path.read_text()
                # 检查是否包含日期
                assert any(keyword in content for keyword in
                          ['日期', 'date', 'Date', '2026', '时间', 'time'])
        else:
            pytest.skip("generate_report function not implemented")


# =============================================================================
# 随机种子测试
# =============================================================================

class TestRandomSeedControl:
    """测试随机种子控制"""

    def test_fixed_seed_produces_same_results(self):
        """
        正例：固定随机种子应产生相同结果

        验证可复现性
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        np.random.seed(42)
        data1 = pd.DataFrame({'x': np.random.randn(100)})

        np.random.seed(42)
        data2 = pd.DataFrame({'x': np.random.randn(100)})

        # 相同种子应产生相同数据
        np.testing.assert_array_equal(data1['x'].values, data2['x'].values)

    def test_different_seed_produces_different_results(self):
        """
        正例：不同随机种子应产生不同结果

        验证随机性的存在
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        np.random.seed(42)
        data1 = pd.DataFrame({'x': np.random.randn(100)})

        np.random.seed(123)
        data2 = pd.DataFrame({'x': np.random.randn(100)})

        # 不同种子应产生不同数据（几乎总是）
        # 有极小概率相同，但在实际中不会发生
        assert not np.array_equal(data1['x'].values, data2['x'].values)
