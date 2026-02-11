"""
Week 01: Solution Template

这是学生需要实现的函数模板。包含以下核心功能：
1. classify_question(): 统计三问分类器
2. detect_data_type(): 数据类型检测器
3. create_data_card(): 数据卡生成器

测试用例会测试这些函数的实现。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 1. 统计三问分类器
# ---------------------------------------------------------------------------

class QuestionType(Enum):
    """统计分析的三类目标"""
    DESCRIPTION = "description"
    INFERENCE = "inference"
    PREDICTION = "prediction"


def classify_question(question: str) -> QuestionType | None:
    """
    根据问题文本判断其属于哪类统计问题（描述/推断/预测）

    Args:
        question: 分析问题文本

    Returns:
        QuestionType 枚举值，如果无法分类则返回 None

    Examples:
        >>> classify_question("这批用户的平均消费金额是多少？")
        QuestionType.DESCRIPTION
        >>> classify_question("根据这1000个样本，全国用户的平均消费金额落在什么范围？")
        QuestionType.INFERENCE
        >>> classify_question("这个新用户下周会不会购买？")
        QuestionType.PREDICTION
    """
    # TODO: 实现分类逻辑
    # 提示：
    # 1. 描述性关键词: "平均", "中位数", "分布", "占比", "有多少", "前几", "排名"
    # 2. 推断性关键词: "推断", "总体", "样本", "范围", "差异是否", "显著", "a/b"
    # 3. 预测性关键词: "预测", "会", "将", "未来", "下周", "下个月", "新用户", "会不会"
    #
    # 注意：预测性关键词优先级最高，推断性其次，描述性最低

    question_lower = question.lower()

    desc_keywords = ["平均", "中位数", "分布", "占比", "有多少", "前几", "前", "排名"]
    infer_keywords = ["推断", "总体", "样本", "范围", "差异是否", "显著", "a/b"]
    pred_keywords = ["预测", "会", "将", "未来", "下周", "下个月", "新用户", "会不会"]

    desc_score = sum(1 for kw in desc_keywords if kw in question_lower)
    infer_score = sum(1 for kw in infer_keywords if kw in question_lower)
    pred_score = sum(1 for kw in pred_keywords if kw in question_lower)

    if pred_score > 0:
        return QuestionType.PREDICTION
    elif infer_score > 0:
        return QuestionType.INFERENCE
    elif desc_score > 0:
        return QuestionType.DESCRIPTION
    return None


# ---------------------------------------------------------------------------
# 2. 数据类型检测器
# ---------------------------------------------------------------------------

class DataType(Enum):
    """数据类型分类"""
    NUMERICAL_CONTINUOUS = "numerical_continuous"  # 连续数值
    NUMERICAL_DISCRETE = "numerical_discrete"      # 离散数值
    CATEGORICAL_NOMINAL = "categorical_nominal"    # 无序类别
    CATEGORICAL_ORDINAL = "categorical_ordinal"    # 有序类别


def detect_data_type(series: pd.Series, domain_hint: str | None = None) -> DataType:
    """
    根据数据特征检测列的类型

    Args:
        series: pandas Series
        domain_hint: 业务语义提示（如"这是性别列"）

    Returns:
        DataType 枚举值

    Examples:
        >>> import pandas as pd
        >>> s1 = pd.Series([18, 25, 30, 42, 55])
        >>> detect_data_type(s1)
        DataType.NUMERICAL_DISCRETE
        >>> s2 = pd.Series([170.5, 165.2, 180.3])
        >>> detect_data_type(s2)
        DataType.NUMERICAL_CONTINUOUS
        >>> s3 = pd.Series(["北京", "上海", "深圳"])
        >>> detect_data_type(s3)
        DataType.CATEGORICAL_NOMINAL
    """
    # TODO: 实现类型检测逻辑
    # 提示：
    # 1. 如果不是数值类型 → CATEGORICAL_NOMINAL
    # 2. 如果是数值类型：
    #    - 检查是否有浮点数（浮点数 → NUMERICAL_CONTINUOUS）
    #    - 唯一值很少（<= 10）且是整数：
    #      - 唯一值 <= 2 → CATEGORICAL_NOMINAL（可能是编码类别）
    #      - domain_hint 包含 "count" → NUMERICAL_DISCRETE
    #      - 其他 → NUMERICAL_DISCRETE（如年龄范围）
    #    - 唯一值很多：
    #      - 整数且唯一值 < 100 → NUMERICAL_DISCRETE
    #      - 其他 → NUMERICAL_CONTINUOUS

    if not pd.api.types.is_numeric_dtype(series):
        return DataType.CATEGORICAL_NOMINAL

    unique_count = series.nunique()
    total_count = len(series)

    # 检查是否有浮点数
    has_floats = not (series.dropna() % 1 == 0).all()

    if unique_count <= 10:
        if has_floats:
            return DataType.NUMERICAL_CONTINUOUS
        elif domain_hint and "count" in domain_hint.lower():
            return DataType.NUMERICAL_DISCRETE
        elif unique_count <= 2:
            return DataType.CATEGORICAL_NOMINAL
        else:
            return DataType.NUMERICAL_DISCRETE
    else:
        all_ints = (series.dropna() % 1 == 0).all()

        if all_ints and unique_count < 100:
            return DataType.NUMERICAL_DISCRETE
        else:
            return DataType.NUMERICAL_CONTINUOUS


# ---------------------------------------------------------------------------
# 3. 数据卡生成器
# ---------------------------------------------------------------------------

def create_data_card(
    df: pd.DataFrame,
    title: str,
    data_source: str,
    description: str,
    field_meanings: dict[str, str] | None = None,
    time_range: str | None = None,
    analysis_type: str = "描述（Description）",
    limitations: str | None = None,
) -> str:
    """
    生成数据卡的 Markdown 文本

    Args:
        df: pandas DataFrame
        title: 数据集标题
        data_source: 数据来源描述
        description: 数据描述
        field_meanings: 字段含义字典 {列名: 业务含义}
        time_range: 时间范围
        analysis_type: 分析类型（描述/推断/预测）
        limitations: 使用限制说明

    Returns:
        Markdown 格式的数据卡文本

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [25, 30, 35], "city": ["北京", "上海", "深圳"]})
        >>> card = create_data_card(df, "测试数据卡", "测试来源", "测试描述")
        >>> "# 测试数据卡" in card
        True
        >>> "## 数据来源" in card
        True
        >>> "age" in card
        True
    """
    from datetime import datetime

    card_lines = [
        f"# {title}",
        "",
        "## 数据来源",
        f"- **来源**：{data_source}",
        f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 数据描述",
        description,
        "",
        "## 统计三问",
        f"本周的分析目标属于：**{analysis_type}**",
        "",
        "**三类目标的区别：**",
        "- **描述（Description）**：说明数据本身的特点，结论只适用于这批数据",
        "- **推断（Inference）**：从样本推断总体，结论带有不确定性",
        "- **预测（Prediction）**：对未来或未见样本做出判断，需要建模",
        "",
        "## 样本规模",
        f"- **行数**：{df.shape[0]:,}",
        f"- **列数**：{df.shape[1]}",
        "",
    ]

    # 时间范围
    if time_range:
        card_lines.extend([
            "## 时间范围",
            time_range,
            "",
        ])

    # 字段字典
    card_lines.extend([
        "## 字段字典",
        "",
        "| 字段名 | 数据类型 | 业务含义 | 缺失率 |",
        "|--------|----------|----------|--------|",
    ])

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = df[col].isna().mean() * 100
        meaning = field_meanings.get(col, "待补充") if field_meanings else "待补充"
        card_lines.append(f"| {col} | {dtype} | {meaning} | {missing_rate:.1f}% |")

    card_lines.append("")

    # 缺失概览
    card_lines.extend([
        "## 缺失概览",
        "",
    ])

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    if len(missing_summary) > 0:
        for col, count in missing_summary.items():
            rate = count / len(df) * 100
            card_lines.append(f"- **{col}**：{count} 个缺失 ({rate:.1f}%)")
    else:
        card_lines.append("- ✅ 无缺失值")

    card_lines.append("")

    # 使用限制
    if limitations:
        card_lines.extend([
            "## 使用限制与注意事项",
            "",
            limitations,
            "",
        ])
    else:
        card_lines.extend([
            "## 使用限制与注意事项",
            "",
            "待补充：本数据集能回答什么问题？不能回答什么问题？",
            "",
        ])

    return "\n".join(card_lines)


# ---------------------------------------------------------------------------
# 4. pandas 基础操作辅助函数
# ---------------------------------------------------------------------------

def get_df_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    获取 DataFrame 的基本信息

    Args:
        df: pandas DataFrame

    Returns:
        包含以下信息的字典：
        - shape: (行数, 列数)
        - columns: 列名列表
        - dtypes: 每列的数据类型
        - null_counts: 每列的缺失值数量
        - memory_usage: 内存使用量（字节）

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> info = get_df_info(df)
        >>> info["shape"]
        (3, 2)
        >>> info["columns"]
        ['a', 'b']
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isna().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }


# ---------------------------------------------------------------------------
# 别名函数（为兼容测试文件期望的函数名）
# ---------------------------------------------------------------------------

def classify_analysis_goal(question: str) -> str:
    """classify_question 的字符串返回版本（兼容测试）"""
    result = classify_question(question)
    if result is None:
        return "description"
    return result.value


def detect_column_type(series: pd.Series, business_meaning: str | None = None) -> dict[str, Any]:
    """detect_data_type 的字典返回版本（兼容测试）"""
    dtype = detect_data_type(series, business_meaning)

    # 确定技术类型和是否可计算均值
    if dtype in [DataType.NUMERICAL_CONTINUOUS, DataType.NUMERICAL_DISCRETE]:
        technical = "numerical"
        can_calc_mean = True
    else:
        technical = "categorical"
        can_calc_mean = False

    # 映射枚举到简化的 statistical 值
    statistical_map = {
        DataType.NUMERICAL_CONTINUOUS: "continuous",
        DataType.NUMERICAL_DISCRETE: "discrete",
        DataType.CATEGORICAL_NOMINAL: "nominal",
        DataType.CATEGORICAL_ORDINAL: "ordinal",
    }
    statistical = statistical_map.get(dtype, "unknown")

    # 特殊情况：编码的类别不应计算均值
    if business_meaning and any(kw in business_meaning for kw in ["性别", "类别", "是否"]):
        can_calc_mean = False

    return {
        "technical": technical,
        "statistical": statistical,
        "can_calculate_mean": can_calc_mean,
    }


def explore_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """get_df_info 的增强版本（兼容测试）"""
    base_info = get_df_info(df)

    # 添加 numeric_summary
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        base_info["numeric_summary"] = df[numeric_cols].describe().to_dict()

    return base_info


def generate_data_card(
    df: pd.DataFrame,
    title: str,
    data_source: str,
    description: str,
    field_meanings: dict[str, str] | None = None,
    time_range: str | None = None,
    analysis_type: str = "描述（Description）",
) -> str:
    """create_data_card 的别名（兼容测试）"""
    return create_data_card(
        df=df,
        title=title,
        data_source=data_source,
        description=description,
        field_meanings=field_meanings,
        time_range=time_range,
        analysis_type=analysis_type,
    )

