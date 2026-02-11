#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 03 作业参考解决方案

本文件整合本周所有技术，实现从原始数据到清洗后数据的完整 pipeline：
1. 缺失值分析与机制判断
2. 填充策略选择与实施
3. 异常值检测与分类处理
4. 特征变换（标准化/编码）
5. 清洗日志生成

这是给学生的参考实现，当他们在作业中遇到困难时可以查看。

运行方式：python3 chapters/week_03/starter_code/solution.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from pathlib import Path
from datetime import datetime
from typing import Any


# ==================== 数据生成 ====================

def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成带有各种数据质量问题的示例数据

    包含：
    - MCAR 缺失（age）
    - MAR 缺失（income，与 registration_days 相关）
    - 异常值（负数消费、VIP 高消费）
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': rng.integers(18, 80, n),
        'registration_days': np.clip(rng.exponential(100, n), 1, None).round(0).astype(int),
        'total_spend': rng.lognormal(7, 1, n),
        'city': rng.choice(['北京', '上海', '广州', '深圳'], n),
        'income': np.clip(rng.lognormal(8.5, 0.5, n) * 1000, 3000, None),
    })

    df['total_spend'] = np.clip(df['total_spend'], 0, None).round(2)
    df['income'] = df['income'].round(2)

    # MCAR: 随机缺失 age
    df.loc[rng.choice(df.index, 25, replace=False), 'age'] = np.nan

    # MAR: income 缺失与 registration_days 相关
    mar_prob = 1 / (1 + np.exp(-(df['registration_days'] - 150) / 50))
    df.loc[rng.random(n) < mar_prob * 0.4, 'income'] = np.nan

    # 异常值
    df.loc[rng.choice(df.index, 3, replace=False), 'total_spend'] = -999
    df.loc[rng.choice(df.index, 5, replace=False), 'total_spend'] = df['total_spend'] * 5 + 50000

    return df


# ==================== 缺失值处理 ====================

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    分析缺失值情况

    返回缺失概览表，包含缺失数量和缺失率
    """
    missing_count = df.isna().sum()
    missing_rate = (df.isna().mean() * 100).round(2)

    overview = pd.DataFrame({
        'missing_count': missing_count,
        'missing_rate_%': missing_rate,
        'dtype': df.dtypes
    })

    return overview[overview['missing_count'] > 0].sort_values('missing_rate_%', ascending=False)


def impute_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'median',
    group_by: str | None = None
) -> pd.DataFrame:
    """
    填充缺失值

    参数：
        column: 要填充的列
        strategy: 'mean', 'median', 'mode'
        group_by: 如果提供，按该列分组填充

    返回：
        填充后的 DataFrame

    抛出：
        ValueError: 如果 strategy 不是 'mean', 'median', 'mode' 之一
    """
    valid_strategies = ['mean', 'median', 'mode']
    if strategy not in valid_strategies:
        raise ValueError(f"无效的策略 '{strategy}'。必须是以下之一: {valid_strategies}")

    df = df.copy()

    if group_by:
        # 分组填充
        if strategy == 'mean':
            df[column] = df.groupby(group_by)[column].transform(lambda x: x.fillna(x.mean()))
        elif strategy == 'median':
            df[column] = df.groupby(group_by)[column].transform(lambda x: x.fillna(x.median()))
        else:  # mode
            df[column] = df.groupby(group_by)[column].transform(
                lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else x)
            )
    else:
        # 全局填充
        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
        else:  # mode
            df[column] = df[column].fillna(df[column].mode()[0] if len(df[column].mode()) > 0 else df[column])

    return df


# ==================== 异常值处理 ====================

def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    使用 IQR 方法检测异常值

    参数：
        multiplier: 1.5 为常规异常值，3.0 为极端异常值

    返回：
        异常值掩码（True 表示异常）
    """
    data = df[column].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return (df[column] < lower_bound) | (df[column] > upper_bound)


def classify_outlier(value: float) -> str:
    """
    根据业务规则对异常值进行分类

    分类：
    - suspicious: 可疑值（负数或极高）
    - VIP: 高消费真实用户
    - normal: 正常值
    """
    if value < 0:
        return 'suspicious'
    if value >= 50000:
        return 'VIP'
    return 'normal'


def handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    分类处理异常值

    - suspicious: 删除
    - VIP: 保留并标记
    """
    df = df.copy()
    df['outlier_category'] = df[column].apply(classify_outlier)

    # 删除可疑值
    df = df[df['outlier_category'] != 'suspicious'].copy()

    return df


# ==================== 特征变换 ====================

def standardize_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """
    标准化数值特征（Z-score）

    返回：
        (变换后的 DataFrame, fitted scaler)

    注意：
        如果某列标准差为0（常数列），会抛出 ValueError
    """
    df = df.copy()

    # 检查常数列
    for col in columns:
        if df[col].std() == 0:
            raise ValueError(f"列 '{col}' 是常数列（标准差为0），无法标准化")

    scaler = StandardScaler()
    df[[f'{c}_std' for c in columns]] = scaler.fit_transform(df[columns])
    return df, scaler


def normalize_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    归一化数值特征到 [0, 1]

    返回：
        (变换后的 DataFrame, fitted scaler)
    """
    df = df.copy()
    scaler = MinMaxScaler()
    df[[f'{c}_norm' for c in columns]] = scaler.fit_transform(df[columns])
    return df, scaler


def encode_categorical(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    独热编码分类特征

    返回：
        (变换后的 DataFrame, fitted encoder)
    """
    df = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded = encoder.fit_transform(df[[column]])

    feature_names = [f'{column}_{cat}' for cat in encoder.categories_[0]]
    df[feature_names] = encoded

    return df, encoder


# ==================== 清洗日志 ====================

def create_cleaning_log(
    operations: list[dict],
    initial_shape: tuple | None = None,
    final_shape: tuple | None = None
) -> str:
    """
    生成 Markdown 格式的清洗日志

    参数：
        operations: 操作记录列表，每个记录包含 field/problem/strategy/rationale/alternatives/impact
        initial_shape: 初始数据形状（可选）
        final_shape: 最终数据形状（可选）

    返回：
        Markdown 字符串
    """
    lines = [
        "# 数据清洗决策日志\n",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    ]

    if initial_shape and final_shape:
        lines.append(f"**数据形状**: {initial_shape} → {final_shape}\n")

    for i, op in enumerate(operations, 1):
        lines.append(f"## 决策 {i}：{op.get('field', '未指定')}")
        lines.append(f"- **问题描述**：{op.get('problem', '未描述')}")
        lines.append(f"- **处理策略**：{op.get('strategy', '未指定')}")
        lines.append(f"- **选择理由**：{op.get('rationale', '未说明')}")
        lines.append(f"- **替代方案**：{op.get('alternatives', '未考虑')}")
        lines.append(f"- **影响评估**：{op.get('impact', '未评估')}")
        lines.append("")

    return '\n'.join(lines)


# ==================== 完整 Pipeline ====================

def cleaning_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    完整的数据清洗 Pipeline

    返回：
        (清洗后的 DataFrame, 清洗日志 Markdown)
    """
    initial_shape = df.shape
    operations = []
    df = df.copy()

    # 处理空 DataFrame
    if df.empty:
        return df, create_cleaning_log([], initial_shape, df.shape)

    # 动态检测可用的数值列用于异常值检测
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_col = None
    for col in ['total_spend', 'income', 'score', 'age']:
        if col in numeric_cols:
            outlier_col = col
            break

    # 步骤 1: 处理异常值（如果有合适的列）
    if outlier_col:
        df['outlier_category'] = df[outlier_col].apply(classify_outlier)
        suspicious_count = (df['outlier_category'] == 'suspicious').sum()

        if suspicious_count > 0:
            operations.append({
                'field': outlier_col,
                'problem': f'检测到 {int(suspicious_count)} 个负数或异常值',
                'strategy': '删除包含异常值的行',
                'rationale': '数值不应为负数，判断为数据错误',
                'alternatives': '保留并标记（如果负数有特殊含义）',
                'impact': f'删除了 {int(suspicious_count)} 条可疑记录'
            })
            df = df[df['outlier_category'] != 'suspicious'].copy()

        # 步骤 2: 标记 VIP
        vip_count = (df['outlier_category'] == 'VIP').sum()
        if vip_count > 0:
            operations.append({
                'field': outlier_col,
                'problem': f'检测到 {int(vip_count)} 个 VIP 高值用户',
                'strategy': '保留并标记 VIP 用户',
                'rationale': '高值用户为真实 VIP，保留并标记用于后续分析',
                'alternatives': '删除（会损失高价值客户信息）',
                'impact': f'保留了 {int(vip_count)} 条 VIP 记录'
            })

    # 步骤 3: 填充 age 缺失（如果存在 age 列）
    if 'age' in df.columns:
        age_missing = df['age'].isna().sum()
        if age_missing > 0:
            df = impute_missing(df, 'age', strategy='median')
            operations.append({
                'field': 'age',
                'problem': f'缺失率 {age_missing/len(df)*100:.1f}%，MCAR 机制',
                'strategy': '全局中位数填充',
                'rationale': '缺失率低，MCAR 机制，中位数对极端值稳健',
                'alternatives': '删除（会损失样本）；均值填充（受极端值影响）',
                'impact': f'填充了 {int(age_missing)} 个缺失值'
            })

    # 步骤 4: 分组填充 income（如果存在 income 和 city 列）
    if 'income' in df.columns and 'city' in df.columns:
        income_missing = df['income'].isna().sum()
        if income_missing > 0:
            df = impute_missing(df, 'income', strategy='median', group_by='city')
            operations.append({
                'field': 'income',
                'problem': f'缺失率 {income_missing/len(df)*100:.1f}%，MAR 机制',
                'strategy': '按城市分组中位数填充',
                'rationale': 'MAR 机制，分组填充保留地域收入差异',
                'alternatives': '全局填充（会抹平地域差异）；删除（会损失样本）',
                'impact': f'填充了 {int(income_missing)} 个缺失值'
            })

    # 步骤 5: 标准化（使用实际存在的数值列）
    available_numeric = [c for c in ['age', 'income', 'total_spend', 'score'] if c in df.columns]
    if len(available_numeric) >= 2:  # 至少需要2个列才需要标准化
        df, _ = standardize_features(df, available_numeric)
        operations.append({
            'field': ', '.join(available_numeric),
            'problem': '特征尺度差异大',
            'strategy': 'StandardScaler 标准化',
            'rationale': '消除量纲影响，使不同特征可公平比较',
            'alternatives': 'MinMaxScaler（适合有界数据）；RobustScaler（如果有异常值）',
            'impact': f'{len(available_numeric)} 个特征已标准化，均值为 0，标准差为 1'
        })

    # 步骤 6: 编码（使用实际存在的类别列）
    cat_col = None
    for col in ['city', 'user_level']:
        if col in df.columns:
            cat_col = col
            break
    if cat_col:
        df, _ = encode_categorical(df, cat_col)
        operations.append({
            'field': cat_col,
            'problem': '分类型变量需要转换为数值',
            'strategy': 'OneHotEncoder 独热编码',
            'rationale': f'{cat_col} 是 nominal 类别，无顺序关系，适合 one-hot 编码',
            'alternatives': 'LabelEncoder（会引入虚假顺序关系）',
            'impact': f'{cat_col} 已编码为二元特征'
        })

    # 生成日志
    log = create_cleaning_log(operations, initial_shape, df.shape)

    return df, log


# ==================== 主函数 ====================

def main() -> None:
    """主函数：演示完整流程"""
    print("=" * 70)
    print("Week 03 作业参考解决方案")
    print("=" * 70)

    # 生成数据
    print("\n【1】生成示例数据")
    df = generate_sample_data(n=500, seed=42)
    print(f"数据形状: {df.shape}")
    print(f"列: {list(df.columns)}")

    # 缺失值分析
    print("\n【2】缺失值分析")
    missing = analyze_missing(df)
    print(missing)

    # 执行清洗 Pipeline
    print("\n【3】执行清洗 Pipeline")
    df_cleaned, log = cleaning_pipeline(df)
    print(f"清洗后形状: {df_cleaned.shape}")

    # 显示清洗日志
    print("\n【4】清洗日志")
    print(log)

    # 保存结果
    output_dir = Path('chapters/week_03/starter_code/output')
    output_dir.mkdir(exist_ok=True)

    df_cleaned.to_csv(output_dir / 'cleaned_data.csv', index=False)
    with open(output_dir / 'cleaning_log.md', 'w', encoding='utf-8') as f:
        f.write(log)

    print(f"\n结果已保存到: {output_dir}/")

    print("\n" + "=" * 70)
    print("关键要点回顾：")
    print("=" * 70)
    print("1. 缺失值：先分析机制（MCAR/MAR/MNAR），再选策略")
    print("2. 异常值：分类处理比统一删除更合理")
    print("3. 特征变换：StandardScaler 用于聚类/回归，MinMaxScaler 用于神经网络")
    print("4. 清洗日志：记录每个决策的理由，确保可复现")


if __name__ == "__main__":
    main()
