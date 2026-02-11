"""
示例：pandas 基础操作 —— DataFrame 索引机制详解。

运行方式：python3 chapters/week_01/examples/03_pandas_basics.py
预期输出：stdout 输出 pandas 选择操作的对比和常见错误。
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def create_sample_df() -> pd.DataFrame:
    """创建示例 DataFrame"""
    df = pd.DataFrame({
        "name": ["张三", "李四", "王五", "赵六"],
        "age": [25, 30, 35, 28],
        "city": ["北京", "上海", "深圳", "杭州"],
        "salary": [8000, 12000, 15000, 9000]
    })
    return df


def demo_select_column(df: pd.DataFrame) -> None:
    """演示列选择"""
    print("=" * 70)
    print("1. 列选择（df['列名']）")
    print("=" * 70)

    print("\n选择一列：")
    print(df['name'])
    print(f"  返回类型：{type(df['name'])}")

    print("\n选择多列：")
    print(df[['name', 'age']])


def demo_select_row(df: pd.DataFrame) -> None:
    """演示行选择 —— 这是容易出错的地方"""
    print("\n" + "=" * 70)
    print("2. 行选择（loc vs iloc）")
    print("=" * 70)

    print("\n❌ 坏示例：df[0] 会报错（选择列，不是行）")
    print("  df[0] → KeyError: 0")
    print("  解释：pandas 把 0 当作列名，而不是行位置")

    print("\n✅ 正确做法：用 iloc 按位置选择行")
    print("df.iloc[0] → 选择第 1 行")
    print(df.iloc[0])

    print("\n✅ 正确做法：用 loc 按标签选择行")
    print("df.loc[0] → 选择索引标签为 0 的行")
    print(df.loc[0])

    print("\n✅ 选择多行：")
    print("df.iloc[1:3] → 第 2、3 行")
    print(df.iloc[1:3])


def demo_select_both(df: pd.DataFrame) -> None:
    """演示同时选择行和列"""
    print("\n" + "=" * 70)
    print("3. 同时选择行和列")
    print("=" * 70)

    print("\n选择第 2 行的 name 列：")
    print("df.loc[1, 'name'] =", df.loc[1, 'name'])
    print("df.iloc[1, 0] =", df.iloc[1, 0])

    print("\n选择第 2-3 行的 name 和 age 列：")
    print(df.loc[1:2, ['name', 'age']])


def demo_filter(df: pd.DataFrame) -> None:
    """演示条件筛选"""
    print("\n" + "=" * 70)
    print("4. 条件筛选")
    print("=" * 70)

    print("\n选择 age > 28 的行：")
    mask = df['age'] > 28
    print(f"筛选条件：{mask.tolist()}")
    print(df[mask])

    print("\n链式写法：df[df['age'] > 28]")
    print(df[df['age'] > 28])

    print("\n多条件筛选：年龄>25 且城市是北京/上海")
    print(df[(df['age'] > 25) & (df['city'].isin(['北京', '上海']))])


def demo_common_mistakes(df: pd.DataFrame) -> None:
    """演示常见错误"""
    print("\n" + "=" * 70)
    print("5. 常见错误与陷阱")
    print("=" * 70)

    print("\n❌ 错误 1：用 df[0] 选择第 1 行")
    print("  会抛出 KeyError，因为 pandas 把 0 当作列名")

    print("\n❌ 错误 2：链式赋值（SettingWithCopyWarning）")
    print("  df[df['age'] > 30]['salary'] = 10000  # 警告！")
    print("  正确做法：")
    print("  df.loc[df['age'] > 30, 'salary'] = 10000")

    print("\n⚠️  注意 3：iloc 切片是左闭右开，loc 切片是左闭右闭")
    print("  df.iloc[0:2]  → 第 1、2 行（索引 0, 1）")
    print(df.iloc[0:2])
    print("  df.loc[0:2]   → 索引标签 0、1、2 的行")
    print(df.loc[0:2])


def demo_index_reset(df: pd.DataFrame) -> None:
    """演示索引重置后的影响"""
    print("\n" + "=" * 70)
    print("6. 索引重置后的影响")
    print("=" * 70)

    print("\n原始 DataFrame：")
    print(df)

    print("\n删除第 1 行后：")
    df_dropped = df.drop(0)
    print(df_dropped)
    print("注意：索引标签还是 1, 2, 3，不是 0, 1, 2")

    print("\n此时 df_dropped.iloc[0] vs df_dropped.loc[0]：")
    print(f"  df_dropped.iloc[0] = {df_dropped.iloc[0]['name']}  # 第 1 行")
    try:
        print(f"  df_dropped.loc[0] = KeyError  # 索引标签 0 已被删除！")
    except KeyError:
        print("  df_dropped.loc[0] → KeyError（索引标签 0 不存在）")

    print("\n重置索引后：")
    df_reset = df_dropped.reset_index(drop=True)
    print(df_reset)
    print("现在 iloc[0] 和 loc[0] 都指向同一行了")


def demo_basic_info(df: pd.DataFrame) -> None:
    """演示基本数据查看方法"""
    print("\n" + "=" * 70)
    print("7. 基本数据查看方法")
    print("=" * 70)

    print("\ndf.shape → (行数, 列数)")
    print(f"  {df.shape}")

    print("\ndf.columns → 列名列表")
    print(f"  {df.columns.tolist()}")

    print("\ndf.dtypes → 每列的数据类型")
    print(df.dtypes)

    print("\ndf.head(2) → 前 2 行")
    print(df.head(2))

    print("\ndf.describe() → 数值列的统计摘要")
    print(df.describe())

    print("\ndf.info() → 完整的数据信息")
    df.info()


def main() -> None:
    """主函数"""
    df = create_sample_df()

    print("示例 DataFrame：")
    print(df)
    print()

    demo_select_column(df)
    demo_select_row(df)
    demo_select_both(df)
    demo_filter(df)
    demo_common_mistakes(df)
    demo_index_reset(df)
    demo_basic_info(df)


if __name__ == "__main__":
    main()
