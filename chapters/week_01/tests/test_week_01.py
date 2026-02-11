"""
Week 01 测试用例

测试覆盖范围：
1. classify_question() - 统计三问分类器
2. detect_data_type() - 数据类型检测器
3. create_data_card() - 数据卡生成器
4. get_df_info() - pandas 基础信息获取

每个功能包含：
- 正例（happy path）
- 边界情况
- 反例（错误输入或应拒绝的情况）
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块
import sys
from pathlib import Path

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

from solution import (
    QuestionType,
    DataType,
    classify_question,
    detect_data_type,
    create_data_card,
    get_df_info,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_dataframe():
    """创建用于测试的示例 DataFrame"""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 35, 28, 22],
        "city": ["北京", "上海", "深圳", "北京", "上海"],
        "salary": [8000.0, 12000.0, 15000.0, 9000.0, 8500.0],
        "is_vip": [0, 1, 1, 0, 0],
    })


@pytest.fixture
def dataframe_with_missing():
    """创建包含缺失值的 DataFrame"""
    return pd.DataFrame({
        "age": [25, 30, None, 28, 22],
        "city": ["北京", None, "深圳", "北京", "上海"],
        "salary": [8000.0, 12000.0, 15000.0, None, 8500.0],
    })


@pytest.fixture
def empty_dataframe():
    """创建空 DataFrame"""
    return pd.DataFrame()


# ===========================================================================
# 1. classify_question() 测试
# ===========================================================================

class TestClassifyQuestion:
    """测试统计三问分类器"""

    # --------------------
    # 正例（happy path）
    # --------------------

    @pytest.mark.parametrize(
        "question, expected_type",
        [
            # 描述性问题
            ("这批用户的平均消费金额是多少？", QuestionType.DESCRIPTION),
            ("数据的中位数是多少？", QuestionType.DESCRIPTION),
            ("用户的年龄分布是什么样的？", QuestionType.DESCRIPTION),
            ("有多少用户来自北京？", QuestionType.DESCRIPTION),
            ("前10名的用户是谁？", QuestionType.DESCRIPTION),
            ("用户的消费占比如何？", QuestionType.DESCRIPTION),
            # 推断性问题
            ("根据这1000个样本，推断全国用户的平均消费", QuestionType.INFERENCE),
            ("从样本推断总体的范围", QuestionType.INFERENCE),
            ("两个版本的差异是否显著？", QuestionType.INFERENCE),
            ("A/B测试中哪个版本更好？", QuestionType.INFERENCE),
            ("样本和总体的差异有多大？", QuestionType.INFERENCE),
            # 预测性问题
            ("预测这个新用户下周会不会购买", QuestionType.PREDICTION),
            ("下个月的销售额会是多少？", QuestionType.PREDICTION),
            ("未来三个月的用户增长趋势", QuestionType.PREDICTION),
            ("这个新用户会购买吗？", QuestionType.PREDICTION),
            ("将来的股价会怎么走？", QuestionType.PREDICTION),
        ],
    )
    def test_classify_known_questions(self, question, expected_type):
        """测试已知问题的分类"""
        # 如果传入的是字符串元组形式，转换为字符串
        if isinstance(question, tuple):
            question = question[0]
        result = classify_question(question)
        assert result == expected_type, f"问题 '{question}' 分类错误：期望 {expected_type}，得到 {result}"

    # --------------------
    # 边界情况
    # --------------------

    def test_classify_empty_question(self):
        """测试空问题"""
        result = classify_question("")
        assert result is None

    def test_classify_question_with_no_keywords(self):
        """测试不包含关键词的问题"""
        result = classify_question("这是什么意思？")
        assert result is None

    def test_classify_case_insensitive(self):
        """测试大小写不敏感"""
        result1 = classify_question("预测用户行为")
        result2 = classify_question("预测用户行为".upper())
        assert result1 == result2 == QuestionType.PREDICTION

    def test_classify_priority(self):
        """测试关键词优先级：预测 > 推断 > 描述"""
        # 同时包含预测和推断关键词 → 预测优先
        result = classify_question("预测未来样本的推断范围")
        assert result == QuestionType.PREDICTION

        # 同时包含推断和描述关键词 → 推断优先
        result = classify_question("推断总体的平均分布")
        assert result == QuestionType.INFERENCE

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_classify_none_input(self):
        """测试 None 输入"""
        with pytest.raises(AttributeError):
            classify_question(None)

    def test_classify_non_string_input(self):
        """测试非字符串输入"""
        with pytest.raises(AttributeError):
            classify_question(123)


# ===========================================================================
# 2. detect_data_type() 测试
# ===========================================================================

class TestDetectDataType:
    """测试数据类型检测器"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_detect_continuous_numerical(self):
        """测试连续数值型"""
        series = pd.Series([170.5, 165.2, 180.3, 175.8, 168.9, 172.1, 178.5, 169.3,
                           174.8, 176.2, 171.5, 173.9, 177.4, 170.8, 179.1])
        result = detect_data_type(series)
        assert result == DataType.NUMERICAL_CONTINUOUS

    def test_detect_discrete_numerical(self):
        """测试离散数值型"""
        series = pd.Series([18, 25, 30, 42, 55, 28, 33, 45, 22, 37, 29, 48, 31,
                           26, 41, 35, 23, 39, 27, 50])
        result = detect_data_type(series)
        assert result == DataType.NUMERICAL_DISCRETE

    def test_detect_categorical_nominal(self):
        """测试无序类别型"""
        series = pd.Series(["北京", "上海", "深圳", "北京", "上海"])
        result = detect_data_type(series)
        assert result == DataType.CATEGORICAL_NOMINAL

    def test_detect_categorical_with_domain_hint(self):
        """测试带业务提示的类型检测"""
        # 性别编码为 0/1，应该是类别
        series = pd.Series([0, 1, 0, 1, 1])
        result = detect_data_type(series, domain_hint="性别编码")
        assert result == DataType.CATEGORICAL_NOMINAL

    def test_detect_count_data_with_hint(self):
        """测试计数数据（带业务提示）"""
        series = pd.Series([1, 2, 3, 1, 2, 5, 1])
        result = detect_data_type(series, domain_hint="用户购买次数count")
        assert result == DataType.NUMERICAL_DISCRETE

    # --------------------
    # 边界情况
    # --------------------

    def test_detect_empty_series(self):
        """测试空 Series"""
        series = pd.Series([], dtype=float)
        result = detect_data_type(series)
        # 空数值系列，唯一值=0 <= 10，应该是 CATEGORICAL_NOMINAL
        assert result == DataType.CATEGORICAL_NOMINAL

    def test_detect_single_unique_value(self):
        """测试只有一个唯一值"""
        series = pd.Series([5, 5, 5, 5, 5])
        result = detect_data_type(series)
        assert result == DataType.CATEGORICAL_NOMINAL

    def test_detect_with_missing_values(self):
        """测试包含缺失值的 Series"""
        series = pd.Series([18, 25, None, 42, 55, 33, 29, 47, 22, 38, 26, 44,
                           31, 36, 49, 23, 41, 28, 37, 50])
        result = detect_data_type(series)
        assert result == DataType.NUMERICAL_DISCRETE

    def test_detect_many_unique_integers(self):
        """测试唯一值很多的整数（>100）"""
        series = pd.Series(range(150))
        result = detect_data_type(series)
        # 唯一值 > 100，虽然都是整数，但超过阈值
        assert result == DataType.NUMERICAL_CONTINUOUS

    def test_detect_binary_encoded(self):
        """测试二值编码（0/1）"""
        series = pd.Series([0, 1, 0, 1, 1, 0, 1])
        result = detect_data_type(series)
        # 唯一值 <= 10，默认为类别
        assert result == DataType.CATEGORICAL_NOMINAL

    # --------------------
    # 反例（不需要特别拒绝，但需验证行为）
    # --------------------

    def test_detect_mixed_type_series(self):
        """测试混合类型 Series（pandas 会自动转换）"""
        # pandas 会把混合类型转为 object
        series = pd.Series([1, "a", 3, "b"])
        result = detect_data_type(series)
        assert result == DataType.CATEGORICAL_NOMINAL


# ===========================================================================
# 3. create_data_card() 测试
# ===========================================================================

class TestCreateDataCard:
    """测试数据卡生成器"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_create_data_card_basic(self, sample_dataframe):
        """测试基本数据卡生成"""
        card = create_data_card(
            df=sample_dataframe,
            title="测试数据卡",
            data_source="测试来源",
            description="这是测试数据",
        )

        # 验证必需的章节存在
        assert "# 测试数据卡" in card
        assert "## 数据来源" in card
        assert "## 数据描述" in card
        assert "## 统计三问" in card
        assert "## 样本规模" in card
        assert "## 字段字典" in card
        assert "## 缺失概览" in card
        assert "## 使用限制与注意事项" in card

    def test_create_data_card_with_field_meanings(self, sample_dataframe):
        """测试带字段含义的数据卡"""
        field_meanings = {
            "user_id": "用户ID",
            "age": "年龄",
            "city": "城市",
        }
        card = create_data_card(
            df=sample_dataframe,
            title="测试数据卡",
            data_source="测试来源",
            description="测试",
            field_meanings=field_meanings,
        )

        # 验证字段含义被正确插入
        assert "用户ID" in card
        assert "年龄" in card
        assert "城市" in card

    def test_create_data_card_with_time_range(self, sample_dataframe):
        """测试带时间范围的数据卡"""
        card = create_data_card(
            df=sample_dataframe,
            title="测试数据卡",
            data_source="测试来源",
            description="测试",
            time_range="2025-01-01 至 2025-12-31",
        )

        assert "## 时间范围" in card
        assert "2025-01-01 至 2025-12-31" in card

    def test_create_data_card_with_analysis_type(self, sample_dataframe):
        """测试不同分析类型的数据卡"""
        for analysis_type in ["描述（Description）", "推断（Inference）", "预测（Prediction）"]:
            card = create_data_card(
                df=sample_dataframe,
                title="测试数据卡",
                data_source="测试来源",
                description="测试",
                analysis_type=analysis_type,
            )
            assert analysis_type in card

    def test_create_data_card_with_limitations(self, sample_dataframe):
        """测试带使用限制的数据卡"""
        limitations = "本数据仅用于测试，不能用于生产环境。"
        card = create_data_card(
            df=sample_dataframe,
            title="测试数据卡",
            data_source="测试来源",
            description="测试",
            limitations=limitations,
        )

        assert limitations in card

    # --------------------
    # 边界情况
    # --------------------

    def test_create_data_card_empty_dataframe(self):
        """测试空 DataFrame 的数据卡"""
        df = pd.DataFrame()
        card = create_data_card(
            df=df,
            title="空数据卡",
            data_source="测试",
            description="空数据测试",
        )

        assert "# 空数据卡" in card
        assert "**行数**：0" in card

    def test_create_data_card_with_missing_values(self, dataframe_with_missing):
        """测试包含缺失值的数据卡"""
        card = create_data_card(
            df=dataframe_with_missing,
            title="含缺失值数据卡",
            data_source="测试",
            description="测试",
        )

        # 验证缺失概览
        assert "## 缺失概览" in card
        assert "age" in card or "city" in card or "salary" in card

    def test_create_data_card_all_missing_column(self):
        """测试某列全为缺失值的情况"""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [None, None, None],
        })
        card = create_data_card(
            df=df,
            title="测试",
            data_source="测试",
            description="测试",
        )

        # col2 应该出现在缺失概览中
        assert "col2" in card

    def test_create_data_card_no_missing_values(self, sample_dataframe):
        """测试无缺失值的数据卡"""
        card = create_data_card(
            df=sample_dataframe,
            title="测试",
            data_source="测试",
            description="测试",
        )

        assert "✅ 无缺失值" in card

    # --------------------
    # 反例（处理极端情况）
    # --------------------

    def test_create_data_card_empty_strings(self, sample_dataframe):
        """测试空字符串参数"""
        card = create_data_card(
            df=sample_dataframe,
            title="",
            data_source="",
            description="",
        )

        # 即使是空字符串也应该能生成
        assert "# " in card

    def test_create_data_card_none_field_meanings(self, sample_dataframe):
        """测试 field_meanings 为 None"""
        card = create_data_card(
            df=sample_dataframe,
            title="测试",
            data_source="测试",
            description="测试",
            field_meanings=None,
        )

        # 应该显示"待补充"
        assert "待补充" in card


# ===========================================================================
# 4. get_df_info() 测试
# ===========================================================================

class TestGetDfInfo:
    """测试 DataFrame 信息获取函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_get_df_info_basic(self, sample_dataframe):
        """测试基本信息获取"""
        info = get_df_info(sample_dataframe)

        assert info["shape"] == (5, 5)
        assert set(info["columns"]) == {"user_id", "age", "city", "salary", "is_vip"}
        assert "user_id" in info["dtypes"]
        assert "city" in info["dtypes"]
        assert info["null_counts"]["user_id"] == 0
        assert info["memory_usage"] > 0

    def test_get_df_info_with_missing(self, dataframe_with_missing):
        """测试包含缺失值的 DataFrame"""
        info = get_df_info(dataframe_with_missing)

        assert info["null_counts"]["age"] == 1
        assert info["null_counts"]["city"] == 1
        assert info["null_counts"]["salary"] == 1

    # --------------------
    # 边界情况
    # --------------------

    def test_get_df_info_empty_dataframe(self, empty_dataframe):
        """测试空 DataFrame"""
        info = get_df_info(empty_dataframe)

        assert info["shape"] == (0, 0)
        assert info["columns"] == []
        assert info["dtypes"] == {}
        assert info["null_counts"] == {}

    def test_get_df_info_single_column(self):
        """测试单列 DataFrame"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        info = get_df_info(df)

        assert info["shape"] == (3, 1)
        assert info["columns"] == ["a"]

    def test_get_df_info_single_row(self):
        """测试单行 DataFrame"""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        info = get_df_info(df)

        assert info["shape"] == (1, 3)

    def test_get_df_info_large_dataset(self):
        """测试较大数据集"""
        df = pd.DataFrame({
            "col1": range(10000),
            "col2": range(10000),
        })
        info = get_df_info(df)

        assert info["shape"] == (10000, 2)
        assert info["memory_usage"] > 0


# ===========================================================================
# 综合测试
# ===========================================================================

class TestIntegration:
    """集成测试：测试多个函数协作"""

    def test_end_to_end_data_card_workflow(self):
        """端到端测试：从原始数据到数据卡"""
        # 1. 创建模拟数据
        df = pd.DataFrame({
            "user_id": range(1, 101),
            "age": np.random.randint(18, 65, 100),
            "city": np.random.choice(["北京", "上海", "深圳"], 100),
            "purchase_count": np.random.poisson(5, 100),
        })

        # 添加一些缺失值
        df.loc[np.random.choice(100, 10), "age"] = None
        df.loc[np.random.choice(100, 5), "city"] = None

        # 2. 检测每列的数据类型
        types = {}
        for col in df.columns:
            hint = "购买次数count" if col == "purchase_count" else None
            types[col] = detect_data_type(df[col], hint)

        # 3. 生成数据卡
        field_meanings = {
            "user_id": "用户ID",
            "age": "年龄",
            "city": "城市",
            "purchase_count": "购买次数",
        }
        card = create_data_card(
            df=df,
            title="端到端测试数据卡",
            data_source="测试生成",
            description="集成测试数据",
            field_meanings=field_meanings,
        )

        # 4. 验证
        assert "**行数**：100" in card
        assert "age" in card
        assert "purchase_count" in card
        # 缺失值数量格式：**age**：X 个缺失
        assert ("个缺失" in card)  # 确保有缺失值被记录


# ===========================================================================
# 运行 pytest 的命令
# ===========================================================================
# python3 -m pytest chapters/week_01/tests/test_week_01.py -v
# python3 -m pytest chapters/week_01/tests/test_week_01.py -v -k "classify_question"
# python3 -m pytest chapters/week_01/tests/test_week_01.py -v --tb=short
