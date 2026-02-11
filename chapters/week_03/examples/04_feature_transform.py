#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：特征变换与编码

本例演示：
1. StandardScaler 标准化（Z-score 标准化）
2. MinMaxScaler 归一化（缩放到指定范围）
3. OneHotEncoder 独热编码（处理分类变量）
4. 封装成可复用函数的工程实践

运行方式：python3 chapters/week_03/examples/04_feature_transform.py
预期输出：各种变换前后的数据对比，以及使用建议
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成包含数值和分类特征的示例数据

    模拟场景：用户画像数据
    - age: 年龄，范围 18-80
    - income: 收入，范围 3000-50000，右偏分布
    - score: 信用评分，范围 300-850
    - city: 城市，分类变量
    - membership: 会员等级，有序分类
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': rng.integers(18, 80, n),
        'income': np.clip(rng.lognormal(8.5, 0.5, n), 3000, 50000).round(2),
        'score': rng.integers(300, 850, n),
        'city': rng.choice(['北京', '上海', '广州', '深圳', '杭州'], n),
        'membership': rng.choice(['普通', '银卡', '金卡', '白金'], n),
    })

    return df


def apply_standard_scaler(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """
    应用 StandardScaler（Z-score 标准化）

    变换公式：z = (x - mean) / std

    适用场景：
    - 数据近似正态分布
    - 需要消除量纲影响（如聚类、PCA）
    - 算法对异常值不敏感（如线性回归、逻辑回归）

    注意：对异常值敏感，极端值会影响 mean 和 std
    """
    df_result = df.copy()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[columns])

    for i, col in enumerate(columns):
        df_result[f'{col}_standardized'] = scaled_values[:, i]

    return df_result, scaler


def apply_minmax_scaler(df: pd.DataFrame, columns: list[str], feature_range: tuple = (0, 1)) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    应用 MinMaxScaler（归一化到指定范围）

    变换公式：x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range

    适用场景：
    - 需要固定范围（如神经网络输入）
    - 数据分布不均匀，有明显边界
    - 需要保留 0 值的意义

    注意：对异常值非常敏感，极端值会压缩其他数据
    """
    df_result = df.copy()

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_values = scaler.fit_transform(df[columns])

    for i, col in enumerate(columns):
        df_result[f'{col}_minmax'] = scaled_values[:, i]

    return df_result, scaler


def apply_onehot_encoder(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    应用 OneHotEncoder（独热编码）

    将分类变量转换为二进制向量
    例如：['北京', '上海'] -> [[1,0], [0,1]]

    适用场景：
    - 名义分类变量（无顺序关系）
    - 需要输入数值型特征的模型（如线性模型、神经网络）

    注意：
    - 会产生稀疏矩阵，高基数特征需谨慎
    - 使用 drop='first' 可避免多重共线性（回归场景）
    """
    df_result = df.copy()

    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_values = encoder.fit_transform(df[columns])

    # 生成新列名
    feature_names = encoder.get_feature_names_out(columns)

    # 添加到结果 DataFrame
    for i, name in enumerate(feature_names):
        df_result[name] = encoded_values[:, i]

    return df_result, encoder


def compare_transformations(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    对比不同变换方法对数据分布的影响
    """
    original = df[column]

    # StandardScaler
    standard_scaler = StandardScaler()
    standardized = standard_scaler.fit_transform(original.values.reshape(-1, 1)).flatten()

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    minmaxed = minmax_scaler.fit_transform(original.values.reshape(-1, 1)).flatten()

    comparison = pd.DataFrame({
        'original': original,
        'standardized': standardized,
        'minmax_0_1': minmaxed
    })

    # 统计摘要
    stats = pd.DataFrame({
        'original': [original.min(), original.max(), original.mean(), original.std()],
        'standardized': [standardized.min(), standardized.max(), standardized.mean(), standardized.std()],
        'minmax_0_1': [minmaxed.min(), minmaxed.max(), minmaxed.mean(), minmaxed.std()]
    }, index=['min', 'max', 'mean', 'std'])

    return comparison, stats.round(4)


def create_preprocessing_pipeline():
    """
    创建一个可复用的预处理流水线配置

    返回一个字典，定义了各列应该使用的预处理方法
    """
    return {
        'numeric_standard': {
            'columns': ['age', 'score'],
            'transformer': 'StandardScaler',
            'rationale': '近似正态分布，用于聚类或距离计算'
        },
        'numeric_minmax': {
            'columns': ['income'],
            'transformer': 'MinMaxScaler',
            'rationale': '右偏分布，需要缩放到固定范围（如神经网络）'
        },
        'categorical_onehot': {
            'columns': ['city'],
            'transformer': 'OneHotEncoder',
            'rationale': '名义分类变量，无顺序关系'
        },
        'categorical_ordinal': {
            'columns': ['membership'],
            'transformer': 'OrdinalEncoder',
            'rationale': '有序分类变量，可保留顺序信息'
        }
    }


def inverse_transform_example():
    """
    演示逆变换：将标准化后的数据还原

    这在解释模型结果时非常有用
    """
    print("\n【逆变换示例】")

    # 原始数据
    data = np.array([[1000], [2000], [3000], [4000], [5000]])
    print(f"原始数据: {data.flatten()}")

    # 标准化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    print(f"标准化后: {scaled.flatten().round(4)}")

    # 逆变换
    original_back = scaler.inverse_transform(scaled)
    print(f"逆变换后: {original_back.flatten()}")

    print("✓ 逆变换成功还原原始数据")


def main() -> None:
    """主函数"""
    df = generate_sample_data(n=500, seed=42)

    print("=" * 70)
    print("特征变换与编码")
    print("=" * 70)

    print("\n【原始数据样本】")
    print(df.head())

    print("\n【数值特征统计】")
    numeric_cols = ['age', 'income', 'score']
    print(df[numeric_cols].describe().round(2))

    # 对比变换效果
    print("\n" + "=" * 70)
    print("变换效果对比（income 字段）")
    print("=" * 70)

    comparison, stats = compare_transformations(df, 'income')
    print("\n统计摘要：")
    print(stats)

    # 应用所有变换
    print("\n" + "=" * 70)
    print("应用所有变换")
    print("=" * 70)

    # 标准化
    df_standard, scaler_standard = apply_standard_scaler(df, ['age', 'score'])
    print("\n1. StandardScaler 已应用")
    print(f"   age 标准化后均值: {df_standard['age_standardized'].mean():.4f}, 标准差: {df_standard['age_standardized'].std():.4f}")

    # 归一化
    df_minmax, scaler_minmax = apply_minmax_scaler(df_standard, ['income'], feature_range=(0, 1))
    print("\n2. MinMaxScaler 已应用")
    print(f"   income 归一化后范围: [{df_minmax['income_minmax'].min():.4f}, {df_minmax['income_minmax'].max():.4f}]")

    # 独热编码
    df_encoded, encoder = apply_onehot_encoder(df_minmax, ['city'])
    print("\n3. OneHotEncoder 已应用")
    city_cols = [col for col in df_encoded.columns if col.startswith('city_')]
    print(f"   生成 {len(city_cols)} 个新列: {city_cols}")

    # 显示编码后的样本
    print("\n【编码后数据样本（部分列）】")
    display_cols = ['user_id', 'age', 'age_standardized', 'income_minmax'] + city_cols[:3]
    print(df_encoded[display_cols].head())

    # 逆变换演示
    inverse_transform_example()

    # 预处理配置
    print("\n" + "=" * 70)
    print("预处理流水线配置示例")
    print("=" * 70)

    pipeline_config = create_preprocessing_pipeline()
    for key, config in pipeline_config.items():
        print(f"\n【{key}】")
        print(f"  列: {config['columns']}")
        print(f"  变换器: {config['transformer']}")
        print(f"  理由: {config['rationale']}")

    print("\n" + "=" * 70)
    print("小北的困惑：")
    print("=" * 70)
    print("'StandardScaler 和 MinMaxScaler 到底用哪个？'")
    print("'我的数据有异常值，用哪个更好？'")

    print("\n老潘的建议：")
    print("'有异常值优先用 StandardScaler，或者先处理异常值。'")
    print("'MinMaxScaler 对异常值太敏感，一个极端值会把其他数据压得很小。'")
    print("'做聚类、PCA 用 StandardScaler；神经网络输入用 MinMaxScaler。'")

    print("\n阿码的发现：")
    print("'OneHotEncoder 生成了好多新列！'")
    print("'如果 city 有 100 个城市，是不是会生成 100 列？'")
    print("'那高基数分类变量怎么办？'")


if __name__ == "__main__":
    main()
