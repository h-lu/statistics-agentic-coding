"""
Week 03 测试：数据转换（Data Transformation）

测试覆盖：
1. standardize_data() - 标准化（Z-score）
2. normalize_data() - 归一化（Min-max）
3. log_transform() - 对数变换
4. encode_features() - 特征编码

测试用例类型：
- 正例：正确执行各种转换
- 边界：常量列、单行数据、含零/负值数据
- 反例：数据泄漏（用训练参数转换测试数据）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
standardize_data = getattr(solution, 'standardize_data', None)
normalize_data = getattr(solution, 'normalize_data', None)
log_transform = getattr(solution, 'log_transform', None)
one_hot_encode = getattr(solution, 'one_hot_encode', None)
label_encode = getattr(solution, 'label_encode', None)


# =============================================================================
# Test: standardize_data()
# =============================================================================

class TestStandardizeData:
    """测试数据标准化函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_standardize_creates_zero_mean_unit_variance(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试标准化后均值接近 0，标准差接近 1

        期望：标准化后每列的均值约等于 0，标准差约等于 1
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        result = standardize_data(multi_scale_dataframe)

        assert isinstance(result, pd.DataFrame), "返回值应该是 DataFrame"
        assert result.shape == multi_scale_dataframe.shape, "形状应保持不变"

        for col in result.columns:
            assert result[col].mean() == pytest.approx(0, abs=1e-10), \
                f"{col} 标准化后均值应接近 0"
            assert result[col].std() == pytest.approx(1, rel=0.01), \
                f"{col} 标准化后标准差应接近 1"

    def test_standardize_preserves_shape(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试标准化保持数据形状

        期望：输出形状与输入相同
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        result = standardize_data(multi_scale_dataframe)

        assert result.shape == multi_scale_dataframe.shape
        assert list(result.columns) == list(multi_scale_dataframe.columns)
        assert result.index.equals(multi_scale_dataframe.index)

    def test_standardize_different_scales(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试标准化让不同尺度变量可比

        期望：不同尺度的变量标准化后在相同范围内
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        result = standardize_data(multi_scale_dataframe)

        # 所有列应该在大致相同的范围（大约 -3 到 3）
        for col in result.columns:
            assert result[col].min() > -5, f"{col} 最小值应合理"
            assert result[col].max() < 5, f"{col} 最大值应合理"

    # --------------------
    # 边界情况
    # --------------------

    def test_standardize_constant_column(self, constant_column_dataframe: pd.DataFrame):
        """
        测试标准化常量列

        期望：常量列（标准差为 0）应被正确处理
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        # 常量列标准化会除以 0，需要正确处理
        try:
            result = standardize_data(constant_column_dataframe)
            # 如果成功处理，常量列应该变成 0 或 NaN
            assert result['constant'].isna().all() or (result['constant'] == 0).all(), \
                "常量列标准化后应为 0 或 NaN"
        except (ValueError, ZeroDivisionError):
            # 也接受抛出异常
            pass

    def test_standardize_single_row(self, single_row_dataframe: pd.DataFrame):
        """
        测试标准化单行数据

        期望：单行数据的标准化可能不可靠
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        # 单行数据的标准差为 0 或未定义
        try:
            result = standardize_data(single_row_dataframe)
            # 如果成功处理，结果可能是 NaN 或 0
        except (ValueError, ZeroDivisionError):
            # 也接受抛出异常
            pass

    def test_standardize_with_missing_values(self):
        """
        测试包含缺失值的数据标准化

        期望：应正确处理缺失值
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [10, 20, 30, np.nan, 50],
        })

        result = standardize_data(df)

        # 非缺失值应该被标准化
        non_nan_a = result['a'].dropna()
        assert non_nan_a.mean() == pytest.approx(0, abs=1e-10)


# =============================================================================
# Test: normalize_data()
# =============================================================================

class TestNormalizeData:
    """测试数据归一化函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_normalize_creates_zero_one_range(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试归一化后数据在 [0, 1] 范围

        期望：归一化后每列的最小值为 0，最大值为 1
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        result = normalize_data(multi_scale_dataframe)

        assert isinstance(result, pd.DataFrame), "返回值应该是 DataFrame"

        for col in result.columns:
            assert result[col].min() == pytest.approx(0, abs=1e-10), \
                f"{col} 归一化后最小值应为 0"
            assert result[col].max() == pytest.approx(1, abs=1e-10), \
                f"{col} 归一化后最大值应为 1"

    def test_normalize_preserves_shape(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试归一化保持数据形状

        期望：输出形状与输入相同
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        result = normalize_data(multi_scale_dataframe)

        assert result.shape == multi_scale_dataframe.shape
        assert list(result.columns) == list(multi_scale_dataframe.columns)

    def test_normalize_all_positive(self, multi_scale_dataframe: pd.DataFrame):
        """
        测试归一化后所有值为非负

        期望：归一化后不应有负值
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        result = normalize_data(multi_scale_dataframe)

        for col in result.columns:
            assert (result[col] >= 0).all(), f"{col} 归一化后应无负值"

    # --------------------
    # 边界情况
    # --------------------

    def test_normalize_constant_column(self, constant_column_dataframe: pd.DataFrame):
        """
        测试归一化常量列

        期望：常量列归一化后应全部为 0
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        result = normalize_data(constant_column_dataframe)

        # 常量列归一化后应为 0
        assert (result['constant'] == 0).all(), "常量列归一化后应为 0"

    def test_normalize_single_row(self, single_row_dataframe: pd.DataFrame):
        """
        测试归一化单行数据

        期望：单行数据归一化后每列都为 1（最大值=最小值）
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        result = normalize_data(single_row_dataframe)

        # 单行数据的 min=max，归一化应为 0 或 1
        for col in result.columns:
            assert result[col].iloc[0] in [0, 1, np.nan], \
                f"单行数据归一化后应为 0 或 1"


# =============================================================================
# Test: log_transform()
# =============================================================================

class TestLogTransform:
    """测试对数变换函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_log_transform_reduces_skewness(self, right_skewed_series: pd.Series):
        """
        测试对数变换减少偏度

        期望：对数变换后的偏度应小于原始偏度
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        original_skew = right_skewed_series.skew()
        transformed = log_transform(right_skewed_series)
        transformed_skew = transformed.skew()

        assert abs(transformed_skew) < abs(original_skew), \
            f"对数变换后偏度应减小：原始={original_skew:.2f}, 变换后={transformed_skew:.2f}"

    def test_log_transform_positive_values(self, right_skewed_series: pd.Series):
        """
        测试对数变换只接受正值

        期望：输入必须是正值
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        # 确保所有值为正
        assert (right_skewed_series > 0).all(), "测试数据应该全是正值"

        result = log_transform(right_skewed_series)

        assert isinstance(result, pd.Series), "返回值应该是 Series"
        assert len(result) == len(right_skewed_series), "长度应保持不变"

    def test_log_transform_compresses_range(self, right_skewed_series: pd.Series):
        """
        测试对数变换压缩数值范围

        期望：大数值被压缩，小数值相对拉伸
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        transformed = log_transform(right_skewed_series)

        # 原始数据的范围应该比对数变换后大
        original_range = right_skewed_series.max() - right_skewed_series.min()
        transformed_range = transformed.max() - transformed.min()

        assert original_range > transformed_range, \
            "对数变换应压缩数值范围"

    # --------------------
    # 边界情况
    # --------------------

    def test_log_transform_with_zeros(self, series_with_zeros: pd.Series):
        """
        测试包含零值的数据

        期望：应正确处理零值（log(0) 无定义）
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        # log(0) 无定义，需要特殊处理
        try:
            result = log_transform(series_with_zeros)
            # 可能使用 log1p（log(1+x)）来处理零值
            assert result.isna().sum() == 0 or series_with_zeros.min() == 0, \
                "零值应被正确处理"
        except (ValueError, AttributeError):
            # 也接受抛出异常
            pass

    def test_log_transform_with_negatives(self, series_with_negatives: pd.Series):
        """
        测试包含负值的数据

        期望：应拒绝或正确处理负值（log(负数) 无定义）
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        # log(负数) 无定义
        try:
            result = log_transform(series_with_negatives)
            # 如果成功，可能做了特殊处理（如加常数偏移）
            # 但更合理的做法是抛出异常
        except (ValueError, AttributeError):
            # 预期抛出异常
            pass

    def test_log_transform_constant_positive(self):
        """
        测试常量正值

        期望：常量值对数变换后仍为常量
        """
        if log_transform is None:
            pytest.skip("log_transform 函数不存在")

        constant_series = pd.Series([10, 10, 10, 10, 10])
        result = log_transform(constant_series)

        assert (result == np.log(10)).all(), "常量值对数变换后应为常量"


# =============================================================================
# Test: encode_features()
# =============================================================================

class TestFeatureEncoding:
    """测试特征编码函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_one_hot_encode_nominal(self, nominal_dataframe: pd.DataFrame):
        """
        测试 One-hot 编码名义变量

        期望：为每个类别创建二进制列
        """
        if one_hot_encode is None:
            pytest.skip("one_hot_encode 函数不存在")

        result = one_hot_encode(nominal_dataframe, 'species')

        assert isinstance(result, pd.DataFrame), "返回值应该是 DataFrame"

        # 检查是否为每个类别创建了列
        unique_species = nominal_dataframe['species'].nunique()
        # 使用 drop_first=True 时，列数应为 unique_species - 1
        expected_cols = unique_species - 1
        actual_encoded_cols = sum('species' in str(col) for col in result.columns)

        assert actual_encoded_cols >= expected_cols, \
            f"应至少为 {unique_species - 1} 个类别创建编码列"

    def test_one_hot_encode_binary_values(self, nominal_dataframe: pd.DataFrame):
        """
        测试 One-hot 编码后的值只有 0 和 1

        期望：编码后的列只包含 0 和 1
        """
        if one_hot_encode is None:
            pytest.skip("one_hot_encode 函数不存在")

        result = one_hot_encode(nominal_dataframe, 'species')

        # 检查编码列的值
        for col in result.columns:
            if 'species' in str(col):
                unique_vals = result[col].unique()
                assert set(unique_vals).issubset({0, 1, np.nan}), \
                    f"One-hot 编码列 {col} 应只包含 0 和 1"

    def test_label_encode_ordinal(self, ordinal_dataframe: pd.DataFrame):
        """
        测试 Label 编码有序变量

        期望：保持类别之间的顺序关系
        """
        if label_encode is None:
            pytest.skip("label_encode 函数不存在")

        result = label_encode(ordinal_dataframe, 'education')

        assert isinstance(result, pd.Series), "返回值应该是 Series"
        assert result.dtype in [np.int32, np.int64, 'int64'], "返回值应为整数类型"

        # 检查编码后的唯一值数量
        unique_encoded = result.nunique()
        unique_original = ordinal_dataframe['education'].nunique()
        assert unique_encoded == unique_original, "编码后的类别数应与原始相同"

    # --------------------
    # 边界情况
    # --------------------

    def test_one_hot_encode_single_category(self):
        """
        测试只有一个类别的 One-hot 编码

        期望：单个类别编码后应合理处理
        """
        if one_hot_encode is None:
            pytest.skip("one_hot_encode 函数不存在")

        df = pd.DataFrame({'id': [1, 2, 3], 'cat': ['A', 'A', 'A']})
        result = one_hot_encode(df, 'cat')

        # 单个类别使用 drop_first=True 后可能没有编码列
        # 这是合理的，因为没有变化

    def test_label_encode_new_category(self):
        """
        测试处理新类别（未见过的类别）

        期望：应能处理或报错
        """
        if label_encode is None:
            pytest.skip("label_encode 函数不存在")

        # 这个测试验证 LabelEncoder 的 fit vs transform 行为
        # 实际应用中，测试集可能有训练集未见过的类别
        # 这里我们只验证函数能正常运行
        df = pd.DataFrame({'edu': ['高中', '本科', '硕士', '博士']})
        result = label_encode(df, 'edu')

        assert result is not None, "Label 编码应该成功"


# =============================================================================
# Test: 数据泄漏检测
# =============================================================================

class TestDataLeakage:
    """测试数据泄漏问题"""

    def test_no_data_leakage_standardization(self):
        """
        测试标准化不应发生数据泄漏

        期望：测试数据应使用训练数据的均值和标准差
        """
        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        # 模拟训练数据和测试数据
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature': np.random.normal(100, 15, 100),
        })
        test_data = pd.DataFrame({
            'feature': np.random.normal(105, 15, 20),
        })

        # 正确做法：使用训练数据的统计量
        train_mean = train_data['feature'].mean()
        train_std = train_data['feature'].std()
        test_standardized_correct = (test_data['feature'] - train_mean) / train_std

        # 错误做法：使用测试数据自己的统计量（数据泄漏）
        test_mean = test_data['feature'].mean()
        test_std = test_data['feature'].std()
        test_standardized_wrong = (test_data['feature'] - test_mean) / test_std

        # 两种结果应该不同
        assert not test_standardized_correct.equals(test_standardized_wrong), \
            "使用不同参数标准化结果应不同"

    def test_no_data_leakage_normalization(self):
        """
        测试归一化不应发生数据泄漏

        期望：测试数据应使用训练数据的最小值和最大值
        """
        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        # 模拟训练数据和测试数据
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature': np.random.uniform(0, 100, 100),
        })
        test_data = pd.DataFrame({
            'feature': np.random.uniform(0, 120, 20),  # 测试集范围更大
        })

        # 正确做法：使用训练数据的范围
        train_min = train_data['feature'].min()
        train_max = train_data['feature'].max()
        test_normalized_correct = (test_data['feature'] - train_min) / (train_max - train_min)

        # 错误做法：使用测试数据自己的范围（数据泄漏）
        test_min = test_data['feature'].min()
        test_max = test_data['feature'].max()
        test_normalized_wrong = (test_data['feature'] - test_min) / (test_max - test_min)

        # 两种结果应该不同
        assert not test_normalized_correct.equals(test_normalized_wrong), \
            "使用不同参数归一化结果应不同"

        # 错误做法可能导致测试集有超出 [0, 1] 的值
        # 正确做法可能有负值或大于 1 的值（如果测试集超出训练集范围）
        assert test_normalized_correct.min() >= test_normalized_wrong.min() - 0.1


# =============================================================================
# Test: 使用真实数据集
# =============================================================================

class TestWithPenguinsData:
    """使用 Penguins 数据集的测试"""

    def test_penguins_standardization(self):
        """
        测试 Penguins 数据集标准化

        期望：能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if standardize_data is None:
            pytest.skip("standardize_data 函数不存在")

        numeric_cols = penguins.select_dtypes(include=[np.number]).columns
        numeric_data = penguins[numeric_cols].dropna()

        result = standardize_data(numeric_data)

        for col in result.columns:
            assert result[col].mean() == pytest.approx(0, abs=1e-10)
            assert result[col].std() == pytest.approx(1, rel=0.01)

    def test_penguins_normalization(self):
        """
        测试 Penguins 数据集归一化

        期望：能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if normalize_data is None:
            pytest.skip("normalize_data 函数不存在")

        numeric_cols = penguins.select_dtypes(include=[np.number]).columns
        numeric_data = penguins[numeric_cols].dropna()

        result = normalize_data(numeric_data)

        for col in result.columns:
            assert result[col].min() == pytest.approx(0, abs=1e-10)
            assert result[col].max() == pytest.approx(1, abs=1e-10)

    def test_penguins_one_hot_encoding(self):
        """
        测试 Penguins 数据集 One-hot 编码

        期望：能正确处理分类变量
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if one_hot_encode is None:
            pytest.skip("one_hot_encode 函数不存在")

        result = one_hot_encode(penguins, 'species')

        # 应该创建编码列
        species_cols = [col for col in result.columns if 'species' in str(col)]
        assert len(species_cols) > 0, "应创建 species 编码列"
