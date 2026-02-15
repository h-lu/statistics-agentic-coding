"""
示例：演示 pandas 基础操作（read_csv、shape、dtypes、head）及常见错误恢复。

运行方式：python3 chapters/week_01/examples/03_pandas_basics.py
预期输出：展示数据加载、基本信息检查、类型转换、常见错误与恢复方式
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
from pathlib import Path


def demo_basic_operations(df: pd.DataFrame) -> None:
    """演示 pandas 基础操作"""
    print("=" * 70)
    print("Pandas 基础操作")
    print("=" * 70)
    print()

    # 1. 查看数据形状
    print("1. 数据规模（shape）：")
    print(f"   行数：{df.shape[0]}，列数：{df.shape[1]}")
    print()

    # 2. 查看前几行
    print("2. 前 3 行数据（head）：")
    print(df.head(3))
    print()

    # 3. 查看数据类型
    print("3. 数据类型（dtypes）：")
    print(df.dtypes)
    print()

    # 4. 缺失值统计
    print("4. 缺失值统计（isna().sum()）：")
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            rate = round(count / len(df) * 100, 1)
            print(f"   - {col}: {count} ({rate}%)")
    else:
        print("   无缺失值")
    print()


def demo_type_conversion(df: pd.DataFrame) -> None:
    """演示类型转换"""
    print("=" * 70)
    print("类型转换：告诉 pandas '这列是什么'")
    print("=" * 70)
    print()

    print("转换前的类型：")
    print(df[["species", "island", "sex"]].dtypes)
    print()

    # 转换分类型数据
    df_converted = df.copy()
    df_converted["species"] = df_converted["species"].astype("category")
    df_converted["island"] = df_converted["island"].astype("category")
    df_converted["sex"] = df_converted["sex"].astype("category")

    print("转换后的类型：")
    print(df_converted[["species", "island", "sex"]].dtypes)
    print()

    # 内存使用对比
    print("内存使用对比：")
    print(f"   转换前: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"   转换后: {df_converted.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print()


def demo_common_errors() -> None:
    """演示常见错误与恢复方式"""
    print("=" * 70)
    print("常见错误与恢复方式")
    print("=" * 70)
    print()

    # 错误 1：路径问题
    print("1. 路径问题（FileNotFoundError）")
    print("-" * 70)
    print("   ❌ 错误：路径写死")
    print('   df = pd.read_csv("/Users/xiaobei/Desktop/data.csv")')
    print()
    print("   ✅ 正确：使用相对路径或 pathlib")
    print('   df = pd.read_csv("data/data.csv")')
    print('   df = pd.read_csv(Path("data") / "data.csv")')
    print()

    # 错误 2：编码问题
    print("2. 编码问题（UnicodeDecodeError）")
    print("-" * 70)
    print("   如果文件包含中文，可能会遇到编码错误：")
    print('   df = pd.read_csv("data.csv", encoding="utf-8")  # 默认')
    print('   df = pd.read_csv("data.csv", encoding="gbk")    # 中文 Windows')
    print('   df = pd.read_csv("data.csv", encoding="gb18030")  # 更广泛的中文支持')
    print()

    # 错误 3：日期没被解析
    print("3. 日期没被解析")
    print("-" * 70)
    print("   pandas 可能把日期列当成字符串：")
    print('   df = pd.read_csv("data.csv", parse_dates=["date_column"])')
    print()


def main() -> None:
    # 方法 1：使用 seaborn 内置数据集（最简单，无路径问题）
    print("方法 1：使用 seaborn 内置数据集")
    print("-" * 70)
    penguins = sns.load_dataset("penguins")
    demo_basic_operations(penguins)
    demo_type_conversion(penguins)
    demo_common_errors()

    # 方法 2：从本地文件读取（示例代码，不实际执行）
    print()
    print("=" * 70)
    print("方法 2：从本地文件读取（示例）")
    print("=" * 70)
    print()
    print("# 假设你的文件在 data/penguins.csv")
    print('df = pd.read_csv("data/penguins.csv")           # 相对路径')
    print('df = pd.read_csv("/absolute/path/to/data.csv")  # 绝对路径')
    print('df = pd.read_csv("../data/penguins.csv")        # 上级目录')
    print()


if __name__ == "__main__":
    main()
