"""
示例：数据类型检测器（数值型/分类型，连续/离散）。

运行方式：python3 chapters/week_01/examples/02_data_types.py
预期输出：stdout 输出典型列的数据类型判断和分析建议。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class DataType(Enum):
    """数据类型分类"""
    NUMERICAL_CONTINUOUS = "numerical_continuous"  # 连续数值
    NUMERICAL_DISCRETE = "numerical_discrete"      # 离散数值
    CATEGORICAL_NOMINAL = "categorical_nominal"    # 无序类别
    CATEGORICAL_ORDINAL = "categorical_ordinal"    # 有序类别


@dataclass
class ColumnInfo:
    """列的类型信息"""
    name: str
    dtype: DataType
    reason: str
    methods: list[str]  # 适用的分析方法


def detect_data_type(series: pd.Series, domain_hint: str | None = None) -> DataType:
    """
    根据数据特征检测列的类型（启发式方法）

    注意：这不是完美的，因为 pandas 的 dtype 不反映业务语义。
    例如：性别编码成 0/1 时，dtype 是 int64，但业务上是分类型。

    Args:
        series: pandas Series
        domain_hint: 业务语义提示（如"这是性别列"）

    Returns:
        DataType 枚举值
    """
    # 1. 检查是否是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 非数值型 → 分类型
        # TODO: 可以进一步判断是否有序（如"低中高"）
        return DataType.CATEGORICAL_NOMINAL

    # 2. 数值类型需要进一步判断
    unique_count = series.nunique()
    total_count = len(series)

    # 简单的启发式规则：
    # - 唯一值很少（如 < 10）且是整数 → 可能是编码的类别
    # - 唯一值很多 → 可能是真正的数值

    # 首先检查是否有浮点数（有浮点数通常是连续数值）
    has_floats = not (series.dropna() % 1 == 0).all()

    if unique_count <= 10:
        # 唯一值少
        if has_floats:
            # 有浮点数，即使是少量也是连续数值
            return DataType.NUMERICAL_CONTINUOUS
        elif domain_hint and "count" in domain_hint.lower():
            # 有业务提示明确是计数
            return DataType.NUMERICAL_DISCRETE
        elif unique_count <= 2:
            # 0/1 或 1/2 通常是编码的类别
            return DataType.CATEGORICAL_NOMINAL
        else:
            # 少量整数，可能是离散数值（如年龄范围）或类别
            # 默认视为离散数值
            return DataType.NUMERICAL_DISCRETE
    else:
        # 唯一值多，判断是连续还是离散
        # 如果值都是整数，可能是离散（如年龄、订单数）
        # 如果有浮点数，可能是连续（如身高、体重、金额）

        # 检查是否所有非空值都是整数
        all_ints = (series.dropna() % 1 == 0).all()

        if all_ints and unique_count < 100:
            return DataType.NUMERICAL_DISCRETE
        else:
            return DataType.NUMERICAL_CONTINUOUS


def get_analysis_methods(dtype: DataType) -> list[str]:
    """根据数据类型返回适用的分析方法"""
    methods = {
        DataType.NUMERICAL_CONTINUOUS: [
            "均值、中位数、标准差",
            "直方图、箱线图",
            "相关系数、回归分析",
            "t 检验、ANOVA（需满足正态假设）"
        ],
        DataType.NUMERICAL_DISCRETE: [
            "频数统计、众数",
            "条形图",
            "泊松回归、计数模型",
        ],
        DataType.CATEGORICAL_NOMINAL: [
            "频数、比例",
            "条形图、饼图",
            "卡方检验",
            "不能计算均值！"
        ],
        DataType.CATEGORICAL_ORDINAL: [
            "频数、中位数",
            "条形图",
            "秩相关（Spearman）",
        ]
    }
    return methods.get(dtype, [])


def demo_with_fake_data() -> None:
    """使用假数据演示类型检测"""
    print("=" * 70)
    print("数据类型检测演示（基于假数据）")
    print("=" * 70)

    # 创建假数据
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, 100),           # 年龄：离散数值
        "height": np.random.normal(170, 10, 100),        # 身高：连续数值
        "gender": np.random.choice([0, 1], 100),         # 性别：分类型（编码）
        "city": np.random.choice(["北京", "上海", "深圳"], 100),  # 城市：分类型
        "purchase_count": np.random.poisson(3, 100),     # 购买次数：离散数值（计数）
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], 100),  # 满意度：有序分类
        "is_vip": np.random.choice([0, 1], 100),         # 是否 VIP：分类型
    })

    # 业务提示
    domain_hints = {
        "gender": "性别编码，0=女，1=男",
        "city": "城市名称",
        "purchase_count": "用户年度购买次数",
        "satisfaction": "用户满意度评分 1-5",
        "is_vip": "是否 VIP 会员",
    }

    for col in df.columns:
        hint = domain_hints.get(col)
        dtype = detect_data_type(df[col], hint)
        methods = get_analysis_methods(dtype)

        type_names = {
            DataType.NUMERICAL_CONTINUOUS: "连续数值",
            DataType.NUMERICAL_DISCRETE: "离散数值",
            DataType.CATEGORICAL_NOMINAL: "无序类别",
            DataType.CATEGORICAL_ORDINAL: "有序类别",
        }

        print(f"\n列名：{col}")
        print(f"  pandas dtype: {df[col].dtype}")
        print(f"  业务类型: {type_names[dtype]}")
        print(f"  适用方法: {', '.join(methods[:2])}...")

        # 特殊提示
        if col == "gender":
            print(f"  ⚠️  注意：虽然 dtype 是 int，但业务上是类别，不能计算均值！")
        elif col == "purchase_count":
            print(f"  ℹ️  这是计数数据，常用泊松分布建模")


def bad_example() -> None:
    """坏示例：对类别数据计算均值"""
    print("\n" + "=" * 70)
    print("❌ 坏示例：对类别数据计算均值")
    print("=" * 70)

    df = pd.DataFrame({
        "gender": [0, 0, 1, 1, 1, 0],  # 0=女，1=男
    })

    print(f"\n数据：{df['gender'].tolist()} (0=女, 1=男)")
    print(f"均值：{df['gender'].mean():.2f}")
    print(f"\n这个 0.5 有什么意义？")
    print(f'  - 不能说"平均性别是 0.5"')
    print(f'  - 不能说"班级一半男一半女"（实际比例是 3/6 = 0.5）')
    print(f"  - 正确做法：计算频数或比例")


def good_example() -> None:
    """好示例：正确处理类别数据"""
    print("\n" + "=" * 70)
    print("✅ 好示例：正确处理类别数据")
    print("=" * 70)

    df = pd.DataFrame({
        "gender": ["女", "女", "男", "男", "男", "女"],
    })

    print(f"\n数据：{df['gender'].tolist()}")
    print(f"\n频数统计：")
    print(df['gender'].value_counts())
    print(f"\n比例统计：")
    print((df['gender'].value_counts(normalize=True) * 100).round(1))
    print(f"\n✅ 这才是类别数据的正确分析方式！")


def main() -> None:
    """主函数"""
    demo_with_fake_data()
    bad_example()
    good_example()


if __name__ == "__main__":
    main()
