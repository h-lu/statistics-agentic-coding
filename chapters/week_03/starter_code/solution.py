"""
Week 03 作业参考实现。

本文件提供作业的参考答案，学生在遇到困难时可以查看。
建议先自己尝试完成作业，实在想不出来再参考本文件。

作业要求概述：
1. 检测和处理缺失值（MCAR/MAR/MNAR）
2. 使用 IQR 和 Z-score 检测异常值
3. 数据转换（标准化/归一化/对数变换）
4. 特征编码（One-hot/Label encoding）

注意：本实现只包含基础要求，不覆盖进阶/挑战部分。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# 核心函数（测试期望的函数）
# =============================================================================

def detect_missing_pattern(df: pd.DataFrame) -> Dict[str, float]:
    """
    检测缺失值的模式和缺失率

    参数：
        df: pandas DataFrame

    返回：
        包含每列缺失率的字典
    """
    missing_rates = {}
    for col in df.columns:
        missing_rates[col] = df[col].isna().mean()
    return missing_rates


def handle_missing_strategy(
    df: pd.DataFrame,
    column: str,
    strategy: Literal['median', 'mean', 'drop', 'constant', 'ffill'] = 'median',
    fill_value: Optional[Any] = None
) -> pd.Series:
    """
    处理缺失值的不同策略

    参数：
        df: pandas DataFrame
        column: 要处理的列名
        strategy: 处理策略
            - 'median': 中位数填充
            - 'mean': 均值填充
            - 'drop': 删除包含缺失值的行
            - 'constant': 常量填充（需要指定 fill_value）
            - 'ffill': 前向填充
        fill_value: 常量填充时的值

    返回：
        处理后的 Series
    """
    data = df[column].copy()

    if strategy == 'median':
        fill_value = data.median()
        return data.fillna(fill_value)
    elif strategy == 'mean':
        fill_value = data.mean()
        return data.fillna(fill_value)
    elif strategy == 'drop':
        return data.dropna()
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("常量填充需要指定 fill_value")
        return data.fillna(fill_value)
    elif strategy == 'ffill':
        return data.ffill()
    else:
        raise ValueError(f"未知的策略: {strategy}")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成缺失值摘要表

    参数：
        df: pandas DataFrame

    返回：
        包含缺失数量和缺失率的 DataFrame
    """
    missing = df.isna().sum()
    missing_rate = df.isna().mean() * 100

    summary = pd.DataFrame({
        'missing_count': missing,
        'missing_%': missing_rate.round(1)
    })

    return summary[summary['missing_count'] > 0].sort_values('missing_count', ascending=False)


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    使用 IQR 规则检测异常值

    参数：
        series: pandas Series，数值型数据
        multiplier: IQR 乘数，默认 1.5

    返回：
        布尔 Series，True 表示该值为异常值
    """
    series = series.dropna()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25

    if iqr == 0:
        # 常量数据没有异常值
        return pd.Series([False] * len(series), index=series.index)

    lower = q25 - multiplier * iqr
    upper = q75 + multiplier * iqr

    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    使用 Z-score 检测异常值

    参数：
        series: pandas Series，数值型数据
        threshold: Z-score 阈值，默认 3.0

    返回：
        布尔 Series，True 表示该值为异常值
    """
    series = series.dropna()
    mean = series.mean()
    std = series.std()

    if std == 0:
        # 常量数据没有异常值
        return pd.Series([False] * len(series), index=series.index)

    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化数据（Z-score normalization）

    参数：
        df: pandas DataFrame，数值型数据

    返回：
        标准化后的 DataFrame
    """
    result = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.number, 'int64', 'float64']:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                result[col] = (df[col] - mean) / std
            else:
                result[col] = 0
    return result


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    归一化数据（Min-max normalization）

    参数：
        df: pandas DataFrame，数值型数据

    返回：
        归一化后的 DataFrame
    """
    result = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.number, 'int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                result[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                result[col] = 0
    return result


def log_transform(series: pd.Series) -> pd.Series:
    """
    对数变换

    参数：
        series: pandas Series，必须为正值

    返回：
        对数变换后的 Series

    注意：
        如果数据包含零值，使用 log1p（log(1+x)）
        如果数据包含负值，抛出异常
    """
    if (series < 0).any():
        raise ValueError("对数变换要求数据必须为非负值")

    if (series == 0).any():
        # 使用 log1p 处理零值
        return np.log1p(series)
    else:
        return np.log(series)


def one_hot_encode(df: pd.DataFrame, column: str, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot 编码

    参数：
        df: pandas DataFrame
        column: 要编码的列名
        drop_first: 是否删除第一列（避免多重共线性）

    返回：
        编码后的 DataFrame
    """
    encoded = pd.get_dummies(df, columns=[column], prefix=column, drop_first=drop_first)
    return encoded


def label_encode(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Label 编码

    参数：
        df: pandas DataFrame
        column: 要编码的列名

    返回：
        编码后的 Series
    """
    # 使用类别出现顺序进行编码（避免依赖 sklearn）
    unique_categories = df[column].unique()
    category_map = {cat: i for i, cat in enumerate(unique_categories)}
    return pd.Series(df[column].map(category_map), index=df.index, name=f'{column}_encoded')


# =============================================================================
# 示例函数
# =============================================================================

def exercise_1_missing_mechanism() -> None:
    """
    作业题 1：缺失值机制诊断

    要求：
    - 加载数据集
    - 检测缺失值模式
    - 判断缺失机制（MCAR/MAR/MNAR）
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("=== 缺失值机制诊断 ===")

    # 检测缺失模式
    missing = missing_summary(penguins)
    print("\n缺失值摘要：")
    print(missing)

    # 分析缺失模式
    for col in missing.index:
        missing_rate = missing.loc[col, 'missing_%'] / 100
        print(f"\n{col}:")
        print(f"  缺失率: {missing_rate:.1%}")

        if missing_rate < 0.05:
            print("  建议处理: 删除（缺失率较低）")
        elif missing_rate < 0.30:
            print("  建议处理: 填充（中位数/众数）")
        else:
            print("  建议处理: 谨慎！可能需要删除整列或深入调查")


def exercise_2_missing_strategies() -> None:
    """
    作业题 2：缺失值处理策略

    要求：
    - 对比不同填充策略的效果
    - 观察对统计量的影响
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("\n=== 缺失值处理策略 ===")

    # 选择有缺失值的列
    col_with_missing = 'bill_length_mm'
    original = penguins[col_with_missing].dropna()

    print(f"\n原始数据统计 ({col_with_missing}):")
    print(f"  均值: {original.mean():.2f}")
    print(f"  中位数: {original.median():.2f}")
    print(f"  标准差: {original.std():.2f}")

    # 测试不同策略
    strategies = ['median', 'mean', 'ffill']
    for strategy in strategies:
        filled = handle_missing_strategy(penguins, col_with_missing, strategy=strategy)
        print(f"\n{strategy.capitalize()} 填充后:")
        print(f"  均值: {filled.mean():.2f}")
        print(f"  中位数: {filled.median():.2f}")
        print(f"  标准差: {filled.std():.2f}")


def exercise_3_outlier_detection(output_dir: Path) -> None:
    """
    作业题 3：异常值检测

    要求：
    - 使用 IQR 规则检测异常值
    - 使用 Z-score 检测异常值
    - 对比两种方法的结果
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("\n=== 异常值检测 ===")

    # 选择数值列
    col = 'body_mass_g'
    data = penguins[col].dropna()

    # IQR 方法
    outliers_iqr = detect_outliers_iqr(data)
    print(f"\nIQR 规则检测到 {outliers_iqr.sum()} 个异常值")

    # Z-score 方法
    outliers_zscore = detect_outliers_zscore(data)
    print(f"Z-score 规则检测到 {outliers_zscore.sum()} 个异常值")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：箱线图
    axes[0].boxplot(data.dropna(), vert=False)
    axes[0].set_xlabel("Body Mass (g)")
    axes[0].set_title("Boxplot (IQR Rule)")
    axes[0].axvline(data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25)),
                    color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)),
                    color='red', linestyle='--', alpha=0.5)

    # 右图：直方图
    axes[1].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(data.mean() - 3 * data.std(), color='red', linestyle='--', label='-3 SD')
    axes[1].axvline(data.mean() + 3 * data.std(), color='red', linestyle='--', label='+3 SD')
    axes[1].set_xlabel("Body Mass (g)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Histogram (Z-score Rule)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "outlier_detection.png", dpi=100, facecolor='white')
    plt.close()
    print(f"\n图表已保存到 {output_dir / 'outlier_detection.png'}")


def exercise_4_data_transformation(output_dir: Path) -> None:
    """
    作业题 4：数据转换

    要求：
    - 标准化数据
    - 归一化数据
    - 对数变换（如果适用）
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("\n=== 数据转换 ===")

    # 选择数值列
    numeric_cols = ['bill_length_mm', 'body_mass_g']
    data = penguins[numeric_cols].dropna()

    print("\n原始数据统计:")
    print(data.describe().round(2))

    # 标准化
    data_std = standardize_data(data)
    print("\n标准化后统计:")
    print(data_std.describe().round(2))

    # 归一化
    data_norm = normalize_data(data)
    print("\n归一化后统计:")
    print(data_norm.describe().round(2))

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 原始数据
    data['bill_length_mm'].plot(kind='hist', ax=axes[0], alpha=0.5, label='bill_length_mm')
    data['body_mass_g'].plot(kind='hist', ax=axes[0], alpha=0.5, label='body_mass_g')
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Original: Different Scales")
    axes[0].legend()

    # 标准化后
    data_std['bill_length_mm'].plot(kind='hist', ax=axes[1], alpha=0.5, label='bill_length_mm')
    data_std['body_mass_g'].plot(kind='hist', ax=axes[1], alpha=0.5, label='body_mass_g')
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Standardized: Same Scale")
    axes[1].legend()

    # 归一化后
    data_norm['bill_length_mm'].plot(kind='hist', ax=axes[2], alpha=0.5, label='bill_length_mm')
    data_norm['body_mass_g'].plot(kind='hist', ax=axes[2], alpha=0.5, label='body_mass_g')
    axes[2].set_xlabel("Normalized Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Normalized: All in [0,1]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "data_transformation.png", dpi=100, facecolor='white')
    plt.close()
    print(f"\n图表已保存到 {output_dir / 'data_transformation.png'}")


def exercise_5_feature_encoding() -> None:
    """
    作业题 5：特征编码

    要求：
    - One-hot 编码名义变量
    - Label 编码有序变量
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("\n=== 特征编码 ===")

    # One-hot 编码物种
    print("\nOne-hot 编码 (species):")
    encoded = one_hot_encode(penguins, 'species', drop_first=True)
    species_cols = [col for col in encoded.columns if 'species' in str(col)]
    print(f"编码后列: {species_cols}")
    print(f"编码后形状: {encoded[species_cols].shape}")

    # Label 编码岛屿
    print("\nLabel 编码 (island):")
    labeled = label_encode(penguins, 'island')
    print(f"编码映射:")
    for i, island in enumerate(penguins['island'].unique()):
        print(f"  {island} -> {labeled[penguins['island'] == island].iloc[0]}")


def exercise_6_cleaning_log() -> None:
    """
    作业题 6：清洗日志

    要求：
    - 记录每一个清洗决策
    - 生成可审计的清洗日志
    """
    try:
        penguins = sns.load_dataset("penguins")
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print("\n=== 清洗决策日志 ===")

    # 清洗日志
    cleaning_log = []

    # 检测缺失值
    missing = missing_summary(penguins)
    for col in missing.index:
        n_missing = missing.loc[col, 'missing_count']
        rate = missing.loc[col, 'missing_%'] / 100
        cleaning_log.append({
            'variable': col,
            'issue': 'missing_values',
            'action': 'keep_as_is',
            'reason': f'缺失率 {rate:.1%} < 5%，保持缺失值',
            'n_affected': n_missing,
        })

    # 检测异常值
    for col in penguins.select_dtypes(include=[np.number]).columns:
        data = penguins[col].dropna()
        outliers = detect_outliers_iqr(data)
        n_outliers = outliers.sum()
        if n_outliers > 0:
            cleaning_log.append({
                'variable': col,
                'issue': 'outliers',
                'action': 'keep_as_is',
                'reason': f'IQR 规则检测到 {n_outliers} 个候选异常值，在合理范围内',
                'n_affected': n_outliers,
            })

    # 打印日志
    log_df = pd.DataFrame(cleaning_log)
    print("\n清洗日志:")
    print(log_df.to_string())


def main() -> None:
    """运行所有作业题的参考解答"""
    print("=" * 60)
    print("Week 03 作业参考实现")
    print("=" * 60)

    output_dir = setup_output_dir()

    exercise_1_missing_mechanism()
    exercise_2_missing_strategies()
    exercise_3_outlier_detection(output_dir)
    exercise_4_data_transformation(output_dir)
    exercise_5_feature_encoding()
    exercise_6_cleaning_log()

    print("\n" + "=" * 60)
    print("所有作业题完成！")
    print(f"图表已保存到: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
