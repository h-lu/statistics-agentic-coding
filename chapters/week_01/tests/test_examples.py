"""
Week 01 示例代码测试

运行方式：
    pytest chapters/week_01/tests/test_examples.py -v
    或
    python3 -m pytest chapters/week_01/tests/test_examples.py -v
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# 添加 examples 目录到路径
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

# 导入示例模块
import importlib

# 导入 starter_code solution
starter_dir = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_dir))


# ===================================================================
# 测试 01_three_questions.py
# ===================================================================

def test_three_questions_module():
    """测试统计三问分类器模块可以正常导入"""
    import importlib
    three_questions = importlib.import_module("01_three_questions")
    assert hasattr(three_questions, "QuestionType")
    assert hasattr(three_questions, "classify_question")


def test_classify_question():
    """测试问题分类功能"""
    three_questions = importlib.import_module("01_three_questions")

    # 测试描述性问题
    q1 = "这批用户的平均消费是多少？"
    result = three_questions.classify_question(q1)
    assert result == three_questions.QuestionType.DESCRIPTION

    # 测试推断性问题
    q2 = "根据样本，总体用户的平均消费范围是多少？"
    result = three_questions.classify_question(q2)
    assert result in [three_questions.QuestionType.INFERENCE, three_questions.QuestionType.DESCRIPTION]

    # 测试预测性问题
    q3 = "这个新用户下周会购买吗？"
    result = three_questions.classify_question(q3)
    assert result == three_questions.QuestionType.PREDICTION


# ===================================================================
# 测试 02_data_types.py
# ===================================================================

def test_data_types_module():
    """测试数据类型检测器模块可以正常导入"""
    data_types = importlib.import_module("02_data_types")
    assert hasattr(data_types, "DataType")
    assert hasattr(data_types, "detect_data_type")


def test_detect_data_type():
    """测试数据类型检测功能"""
    data_types = importlib.import_module("02_data_types")

    # 测试连续数值
    s1 = pd.Series([1.5, 2.3, 3.7, 4.1])
    result = data_types.detect_data_type(s1)
    assert result in [data_types.DataType.NUMERICAL_CONTINUOUS,
                      data_types.DataType.NUMERICAL_DISCRETE]

    # 测试分类型（字符串）
    s2 = pd.Series(["北京", "上海", "深圳"])
    result = data_types.detect_data_type(s2)
    assert result == data_types.DataType.CATEGORICAL_NOMINAL

    # 测试编码的类别（0/1）
    s3 = pd.Series([0, 1, 0, 1, 1])
    result = data_types.detect_data_type(s3)
    assert result in [data_types.DataType.CATEGORICAL_NOMINAL,
                      data_types.DataType.NUMERICAL_DISCRETE]


# ===================================================================
# 测试 03_pandas_basics.py
# ===================================================================

def test_pandas_basics_module():
    """测试 pandas 基础操作模块可以正常导入"""
    pandas_basics = importlib.import_module("03_pandas_basics")
    assert hasattr(pandas_basics, "create_sample_df")
    assert hasattr(pandas_basics, "demo_select_column")


def test_create_sample_df():
    """测试示例 DataFrame 创建"""
    pandas_basics = importlib.import_module("03_pandas_basics")
    df = pandas_basics.create_sample_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4  # 4 行
    assert df.shape[1] == 4  # 4 列
    assert list(df.columns) == ["name", "age", "city", "salary"]


# ===================================================================
# 测试 04_data_card.py
# ===================================================================

def test_data_card_module():
    """测试数据卡生成器模块可以正常导入"""
    data_card = importlib.import_module("04_data_card")
    assert hasattr(data_card, "create_data_card")


def test_create_data_card():
    """测试数据卡生成功能"""
    data_card = importlib.import_module("04_data_card")

    # 创建测试数据
    df = pd.DataFrame({
        "name": ["张三", "李四"],
        "age": [25, 30],
        "city": ["北京", "上海"],
    })

    # 生成数据卡
    card = data_card.create_data_card(
        df=df,
        title="测试数据卡",
        data_source="测试",
        description="这是一个测试数据集",
    )

    # 验证输出
    assert isinstance(card, str)
    assert "测试数据卡" in card
    assert "2" in card  # 行数
    assert "3" in card  # 列数
    assert "name" in card
    assert "age" in card


# ===================================================================
# 测试 99_statlab.py
# ===================================================================

def test_statlab_module():
    """测试 StatLab 模块可以正常导入"""
    statlab = importlib.import_module("99_statlab")
    assert hasattr(statlab, "generate_statlab_data_card")
    assert hasattr(statlab, "create_sample_data")


def test_create_sample_data(tmp_path):
    """测试 StatLab 示例数据创建"""
    statlab = importlib.import_module("99_statlab")

    # 在临时目录创建测试数据
    test_file = tmp_path / "test_users.csv"
    statlab.create_sample_data(str(test_file))

    # 验证文件存在
    assert test_file.exists()

    # 验证数据可以读取
    df = pd.read_csv(test_file)
    assert df.shape[0] == 1000  # 默认 1000 行
    assert "user_id" in df.columns


def test_generate_statlab_data_card():
    """测试 StatLab 数据卡生成"""
    statlab = importlib.import_module("99_statlab")

    # 创建测试数据
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    })

    metadata = {
        "title": "测试报告",
        "source": "测试来源",
        "collection_date": "2025-01-01",
        "description": "测试描述",
        "analysis_type": "描述（Description）",
        "field_meanings": {
            "id": "ID",
            "value": "值",
        },
    }

    # 生成报告
    report = statlab.generate_statlab_data_card(df, metadata)

    # 验证输出
    assert isinstance(report, str)
    assert "测试报告" in report
    assert "3" in report  # 行数
    assert "id" in report
    assert "value" in report


# ===================================================================
# 测试 starter_code/solution.py
# ===================================================================

def test_solution_module():
    """测试参考实现模块可以正常导入"""
    solution = importlib.import_module("solution")
    assert hasattr(solution, "classify_analysis_goal")
    assert hasattr(solution, "detect_column_type")
    assert hasattr(solution, "explore_dataset")
    assert hasattr(solution, "generate_data_card")


def test_solution_classify_analysis_goal():
    """测试参考实现的问题分类功能"""
    solution = importlib.import_module("solution")

    # 测试描述性问题
    result = solution.classify_analysis_goal("平均消费是多少？")
    assert result == "description"

    # 测试预测性问题
    result = solution.classify_analysis_goal("下周会购买吗？")
    assert result == "prediction"


def test_solution_detect_column_type():
    """测试参考实现的列类型检测功能"""
    solution = importlib.import_module("solution")

    # 测试数值型
    s1 = pd.Series([1, 2, 3, 4, 5])
    result = solution.detect_column_type(s1, "年龄")
    assert result["technical"] == "numerical"
    assert result["can_calculate_mean"] == True

    # 测试分类型（有业务提示）
    s2 = pd.Series([1, 2, 1, 2])
    result = solution.detect_column_type(s2, "性别（1=男，2=女）")
    assert result["statistical"] in ["nominal", "discrete"]


def test_solution_explore_dataset():
    """测试参考实现的数据探索功能"""
    solution = importlib.import_module("solution")

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })

    result = solution.explore_dataset(df)

    assert result["shape"] == (3, 2)
    assert "a" in result["columns"]
    assert "b" in result["columns"]
    assert result["dtypes"]["a"] == "int64"


def test_solution_generate_data_card():
    """测试参考实现的数据卡生成功能"""
    solution = importlib.import_module("solution")

    df = pd.DataFrame({
        "col1": [1, 2, 3],
    })

    card = solution.generate_data_card(
        df=df,
        title="测试",
        data_source="测试来源",
        description="测试描述",
    )

    assert isinstance(card, str)
    assert "测试" in card
    assert "col1" in card


# ===================================================================
# 集成测试：验证输出文件
# ===================================================================

def test_example_outputs():
    """测试示例脚本可以正常输出内容（不报错即可）"""
    # 注意：这里只验证模块可以导入和核心函数可以调用
    # 不真正运行 main() 函数（因为可能有交互输入）

    modules_to_test = [
        "01_three_questions",
        "02_data_types",
        "03_pandas_basics",
        "04_data_card",
        "99_statlab",
    ]

    for module_name in modules_to_test:
        module = importlib.import_module(module_name)
        assert module is not None


# ===================================================================
# 运行所有测试
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
