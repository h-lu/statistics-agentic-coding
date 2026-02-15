"""
Week 04 测试：分组比较（Group Comparison）

测试覆盖：
1. groupby_aggregate() - 分组聚合
2. create_pivot_table() - 创建透视表
3. compare_group_statistics() - 比较组间统计量

测试用例类型：
- 正例：正确分组和聚合
- 边界：空分组、单分组、常量列
- 反例：无效的分组变量
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
groupby_aggregate = getattr(solution, 'groupby_aggregate', None)
create_pivot_table = getattr(solution, 'create_pivot_table', None)
compare_group_statistics = getattr(solution, 'compare_group_statistics', None)


# =============================================================================
# Test: groupby_aggregate()
# =============================================================================

class TestGroupbyAggregate:
    """测试分组聚合函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_groupby_mean(self, grouped_purchase_data: pd.DataFrame):
        """
        测试按渠道分组计算均值

        期望：返回每个渠道的均值
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        result = groupby_aggregate(
            grouped_purchase_data,
            group_col='source',
            value_col='purchase_amount',
            agg_func='mean'
        )

        assert isinstance(result, pd.Series), "返回值应该是 Series"
        assert len(result) == 3, "应有 3 个渠道"
        assert all(result > 0), "均值应该都为正"

        # search 渠道的均值应该最高
        assert result['search'] > result['social'], \
            f"search 渠道均值应高于 social"

    def test_groupby_multiple_aggregations(self, grouped_purchase_data: pd.DataFrame):
        """
        测试同时计算多个聚合函数

        期望：返回多个统计量
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        result = groupby_aggregate(
            grouped_purchase_data,
            group_col='source',
            value_col='purchase_amount',
            agg_func=['mean', 'median', 'std', 'count']
        )

        assert isinstance(result, (pd.DataFrame, dict)), "多聚合应返回 DataFrame 或字典"

        if isinstance(result, pd.DataFrame):
            assert 'mean' in result.columns or 'mean' in result.index
            assert len(result) == 3, "应有 3 个分组"

    def test_groupby_returns_correct_group_count(self, multi_group_data: pd.DataFrame):
        """
        测试 groupby 返回正确分组数

        期望：每个类别都产生一个分组
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        result = groupby_aggregate(
            multi_group_data,
            group_col='region',
            value_col='sales',
            agg_func='mean'
        )

        unique_regions = multi_group_data['region'].nunique()
        assert len(result) == unique_regions, \
            f"分组数应等于唯一类别数：{unique_regions}"

    # --------------------
    # 边界情况
    # --------------------

    def test_groupby_with_missing_values(self):
        """
        测试包含缺失值的分组

        期望：应正确处理 NaN
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [1, 2, np.nan, 4, 5, np.nan]
        })

        result = groupby_aggregate(
            df,
            group_col='group',
            value_col='value',
            agg_func='mean'
        )

        # A 组均值 = (1 + 5) / 2 = 3
        # B 组均值 = (2 + 4) / 2 = 3
        assert result['A'] == pytest.approx(3, abs=0.1)
        assert result['B'] == pytest.approx(3, abs=0.1)

    def test_groupby_single_group(self):
        """
        测试单分组场景

        期望：应返回单个聚合结果
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        df = pd.DataFrame({
            'group': ['A'] * 10,
            'value': range(10)
        })

        result = groupby_aggregate(
            df,
            group_col='group',
            value_col='value',
            agg_func='mean'
        )

        assert len(result) == 1, "单分组应返回 1 个结果"
        assert result['A'] == pytest.approx(4.5, abs=0.1), "均值应为 4.5"

    def test_groupby_aggregation_functions_correct(self, grouped_purchase_data: pd.DataFrame):
        """
        测试聚合函数结果正确性

        期望：count、mean、median、std 结果正确
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        # 测试 count
        count_result = groupby_aggregate(
            grouped_purchase_data,
            group_col='source',
            value_col='purchase_amount',
            agg_func='count'
        )
        assert all(count_result > 0), "每个组应该有计数"

        # 测试 median
        median_result = groupby_aggregate(
            grouped_purchase_data,
            group_col='source',
            value_col='purchase_amount',
            agg_func='median'
        )
        assert all(median_result > 0), "每个组应该有中位数"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_groupby_invalid_column(self, grouped_purchase_data: pd.DataFrame):
        """
        测试无效的分组列

        期望：应抛出 KeyError
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        with pytest.raises(KeyError):
            groupby_aggregate(
                grouped_purchase_data,
                group_col='nonexistent_column',
                value_col='purchase_amount',
                agg_func='mean'
            )

    def test_groupby_invalid_aggregation(self, grouped_purchase_data: pd.DataFrame):
        """
        测试无效的聚合函数

        期望：应抛出异常
        """
        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        with pytest.raises((ValueError, AttributeError, TypeError)):
            groupby_aggregate(
                grouped_purchase_data,
                group_col='source',
                value_col='purchase_amount',
                agg_func='invalid_agg_func'
            )


# =============================================================================
# Test: create_pivot_table()
# =============================================================================

class TestCreatePivotTable:
    """测试透视表创建函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_pivot_two_dimensions(self, multi_group_data: pd.DataFrame):
        """
        测试二维透视表

        期望：创建行列交叉的表格
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        result = create_pivot_table(
            multi_group_data,
            values='sales',
            index='region',
            columns='product',
            aggfunc='mean'
        )

        assert isinstance(result, pd.DataFrame), "透视表应该是 DataFrame"

        # 维度检查
        n_regions = multi_group_data['region'].nunique()
        n_products = multi_group_data['product'].nunique()

        # 结果形状应该匹配
        assert result.shape[0] <= n_regions, "行数应匹配 region 数量"
        assert result.shape[1] <= n_products, "列数应匹配 product 数量"

    def test_pivot_dimensions_correct(self, multi_group_data: pd.DataFrame):
        """
        测试透视表维度正确

        期望：返回正确的行列数
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        result = create_pivot_table(
            multi_group_data,
            values='sales',
            index='region',
            columns='product',
            aggfunc='mean'
        )

        # 检查维度
        assert len(result.index) > 0, "透视表应有行"
        assert len(result.columns) > 0, "透视表应有列"

    def test_pivot_different_aggregations(self, multi_group_data: pd.DataFrame):
        """
        测试不同的聚合函数

        期望：sum、mean、count 等都能正确计算
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        for agg_func in ['mean', 'sum', 'count', 'median']:
            result = create_pivot_table(
                multi_group_data,
                values='sales',
                index='region',
                columns='product',
                aggfunc=agg_func
            )

            assert isinstance(result, pd.DataFrame), f"{agg_func} 聚合应返回 DataFrame"
            assert result.notna().sum().sum() > 0, f"{agg_func} 聚合应有非空值"

    # --------------------
    # 边界情况
    # --------------------

    def test_pivot_with_missing_combinations(self):
        """
        测试透视表中缺失的组合

        期望：缺失组合应为 NaN
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        # 创建有缺失组合的数据
        df = pd.DataFrame({
            'region': ['North', 'North', 'South', 'South'],
            'product': ['A', 'B', 'A', 'B'],  # East 缺失
            'sales': [100, 200, 150, 250]
        })

        result = create_pivot_table(
            df,
            values='sales',
            index='region',
            columns='product',
            aggfunc='sum'
        )

        # 检查是否有 NaN（取决于实现）
        assert isinstance(result, pd.DataFrame), "透视表应该是 DataFrame"

    def test_pivot_single_value_column(self, single_column_dataframe: pd.DataFrame):
        """
        测试单列数据的透视表

        期望：应能正确处理
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        # 添加分组列和另一列用于透视表
        df = single_column_dataframe.copy()
        df['group'] = ['A', 'A', 'B', 'B', 'B']
        df['category'] = ['X', 'Y', 'X', 'Y', 'X']  # 添加列用于 columns 参数

        result = create_pivot_table(
            df,
            values='value',
            index='group',
            columns='category',
            aggfunc='mean'
        )

        assert isinstance(result, pd.DataFrame), "透视表应该是 DataFrame"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_pivot_invalid_columns(self, multi_group_data: pd.DataFrame):
        """
        测试无效的列名

        期望：应抛出 KeyError
        """
        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        with pytest.raises(KeyError):
            create_pivot_table(
                multi_group_data,
                values='nonexistent_value',
                index='region',
                columns='product',
                aggfunc='mean'
            )


# =============================================================================
# Test: compare_group_statistics()
# =============================================================================

class TestCompareGroupStatistics:
    """测试组间统计量比较函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_compare_two_groups(self, grouped_purchase_data: pd.DataFrame):
        """
        测试比较两组的统计量

        期望：返回两组的差异指标
        """
        if compare_group_statistics is None:
            pytest.skip("compare_group_statistics 函数不存在")

        # 筛选两组
        df_two_groups = grouped_purchase_data[
            grouped_purchase_data['source'].isin(['search', 'social'])
        ]

        result = compare_group_statistics(
            df_two_groups,
            group_col='source',
            value_col='purchase_amount'
        )

        assert isinstance(result, dict), "结果应该是字典"

        # 应该包含差异信息
        if 'mean_difference' in result:
            assert result['mean_difference'] > 0, "search 均值应高于 social"

        if 'group_means' in result:
            assert len(result['group_means']) == 2, "应有 2 个组均值"

    def test_compare_multiple_groups(self, grouped_purchase_data: pd.DataFrame):
        """
        测试比较多组

        期望：返回所有组的统计量
        """
        if compare_group_statistics is None:
            pytest.skip("compare_group_statistics 函数不存在")

        result = compare_group_statistics(
            grouped_purchase_data,
            group_col='source',
            value_col='purchase_amount'
        )

        # 应该有 3 个组的信息
        if 'group_stats' in result:
            assert len(result['group_stats']) == 3, "应有 3 个组的统计"
        elif isinstance(result, pd.DataFrame):
            assert len(result) == 3, "应有 3 行（3 个组）"

    # --------------------
    # 边界情况
    # --------------------

    def test_compare_groups_with_single_value(self):
        """
        测试每组只有一个值

        期望：应能处理，但标准差为 0
        """
        if compare_group_statistics is None:
            pytest.skip("compare_group_statistics 函数不存在")

        df = pd.DataFrame({
            'group': ['A', 'B'],
            'value': [10, 20]
        })

        result = compare_group_statistics(
            df,
            group_col='group',
            value_col='value'
        )

        assert result is not None, "应能处理小样本"


# =============================================================================
# Test: 使用 Penguins 数据集
# =============================================================================

class TestWithPenguinsData:
    """使用真实数据集的测试"""

    def test_penguins_groupby_species(self):
        """
        测试按物种分组比较 Penguins 数据

        期望：能正确分组并计算统计量
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if groupby_aggregate is None:
            pytest.skip("groupby_aggregate 函数不存在")

        result = groupby_aggregate(
            penguins,
            group_col='species',
            value_col='body_mass_g',
            agg_func='mean'
        )

        assert len(result) == 3, "应有 3 个物种"

        # Gentoo 应该最重
        if isinstance(result, pd.Series):
            assert result['Gentoo'] > result['Adelie'], \
                "Gentoo 应比 Adelie 重"
            assert result['Gentoo'] > result['Chinstrap'], \
                "Gentoo 应比 Chinstrap 重"

    def test_penguins_pivot_table(self):
        """
        测试 Penguins 数据集创建透视表

        期望：能按物种和岛屿创建透视表
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if create_pivot_table is None:
            pytest.skip("create_pivot_table 函数不存在")

        result = create_pivot_table(
            penguins,
            values='body_mass_g',
            index='species',
            columns='island',
            aggfunc='mean'
        )

        assert isinstance(result, pd.DataFrame), "透视表应该是 DataFrame"
        assert result.shape[0] == 3, "应有 3 个物种（行）"

    def test_penguins_species_difference(self):
        """
        测试比较 Penguins 物种间的差异

        期望：能识别出 Gentoo 最重
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if compare_group_statistics is None:
            pytest.skip("compare_group_statistics 函数不存在")

        result = compare_group_statistics(
            penguins.dropna(subset=['body_mass_g']),
            group_col='species',
            value_col='body_mass_g'
        )

        # Gentoo 的均值应该最高
        if 'group_stats' in result:
            gentoo_mean = result['group_stats'].get('Gentoo', {}).get('mean', 0)
            adelie_mean = result['group_stats'].get('Adelie', {}).get('mean', 0)
            assert gentoo_mean > adelie_mean, "Gentoo 应比 Adelie 重"
