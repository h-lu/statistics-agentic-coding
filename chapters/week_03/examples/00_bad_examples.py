#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反例：数据清洗中的常见错误

本文件展示 Week 03 概念的错误用法，供读者对比学习。
这些代码可以运行，但会产生误导性结果或隐藏问题。

运行方式：python3 chapters/week_03/examples/00_bad_examples.py
预期输出：错误示范的输出，以及正确做法的对比说明
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_sample_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成示例数据"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'age': np.clip(rng.normal(35, 10, n), 18, None).round(0),
        'income': np.clip(rng.lognormal(8.5, 0.5, n), 3000, None),
        'city': rng.choice(['北京', '上海', '广州'], n),
    })
    # 添加缺失值
    df.loc[rng.choice(df.index, 20, replace=False), 'income'] = np.nan
    return df


# ==================== 错误 1：不分析缺失机制直接删除 ====================

def bad_delete_all_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    ❌ 错误做法：不分析缺失机制，直接删除所有含缺失的行

    问题：
    1. 如果缺失是 MAR 或 MNAR，删除会引入选择偏差
    2. 可能损失大量样本
    3. 没有记录删除理由
    """
    return df.dropna()


def good_analyze_then_handle(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 正确做法：先分析缺失机制，再选择策略

    步骤：
    1. 计算缺失率
    2. 分析缺失与观测变量的关系（判断 MAR）
    3. 根据机制选择填充或删除
    4. 记录决策
    """
    missing_rate = df['income'].isna().mean()

    if missing_rate < 0.05:
        # 缺失率低，可以删除
        return df.dropna(subset=['income'])
    else:
        # 缺失率高，应该填充
        median_income = df['income'].median()
        df = df.copy()
        df['income'] = df['income'].fillna(median_income)
        return df


# ==================== 错误 2：用均值填充所有缺失值 ====================

def bad_mean_imputation_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    ❌ 错误做法：对所有列统一使用均值填充

    问题：
    1. 忽略列之间的差异
    2. 对右偏分布（如收入），均值会被极端值拉高
    3. 对分类变量使用均值无意义
    4. 降低方差，扭曲相关性
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def good_contextual_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 正确做法：根据列的特性选择填充策略

    - 数值且对称：均值或中位数
    - 数值且偏态：中位数
    - 分类变量：众数或单独类别
    - 有分组结构：分组填充
    """
    df = df.copy()

    # income 是右偏的，用中位数
    df['income'] = df['income'].fillna(df['income'].median())

    return df


# ==================== 错误 3：统一删除所有异常值 ====================

def bad_remove_all_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    ❌ 错误做法：用 IQR 检测到的异常值全部删除

    问题：
    1. VIP 高消费用户是真实存在的，删除会扭曲业务理解
    2. 某些异常值可能是数据错误，需要修正而非删除
    3. 没有区分不同类型的异常
    """
    q1, q3 = df['income'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return df[(df['income'] >= lower) & (df['income'] <= upper)]


def good_classify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 正确做法：对异常值分类处理

    - suspicious（可疑）：核实后删除或修正
    - VIP（真实高消费）：保留并标记
    - normal：正常范围
    """
    df = df.copy()

    def classify(value):
        if pd.isna(value):
            return 'missing'
        if value < 0:
            return 'suspicious'
        if value > 50000:
            return 'VIP'
        return 'normal'

    df['outlier_category'] = df['income'].apply(classify)

    # 只删除可疑值，保留 VIP
    return df[df['outlier_category'] != 'suspicious']


# ==================== 错误 4：先标准化再划分训练/测试集 ====================

def bad_scale_before_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    ❌ 错误做法：在划分训练/测试集之前对整个数据集进行标准化

    问题：
    1. 数据泄漏：测试集信息通过均值/标准差泄漏到训练过程
    2. 模型评估结果过于乐观
    3. 生产环境无法复现（新数据没有训练集的均值/标准差）
    """
    scaler = StandardScaler()
    df = df.copy()
    df['income_scaled'] = scaler.fit_transform(df[['income']])
    return df


def good_scale_after_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    ✅ 正确做法：先划分数据集，再用训练集的参数转换测试集

    步骤：
    1. 划分训练/测试集
    2. 在训练集上 fit scaler
    3. 用同一个 scaler transform 训练集和测试集
    """
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # 只在训练集上 fit
    train_scaled['income_scaled'] = scaler.fit_transform(train_df[['income']])

    # 用训练集的参数 transform 测试集
    test_scaled['income_scaled'] = scaler.transform(test_df[['income']])

    return train_scaled, test_scaled


# ==================== 错误 5：不记录清洗过程 ====================

def bad_no_documentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    ❌ 错误做法：清洗过程没有任何记录

    问题：
    1. 无法复现分析
    2. 无法审计决策
    3. 三个月后自己也会忘记做了什么
    4. 团队协作时他人无法理解数据处理方式
    """
    df = df.dropna(subset=['age'])
    df = df[df['income'] > 0]
    df['income'] = df['income'].fillna(df['income'].median())
    return df


def good_with_logging(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    ✅ 正确做法：记录每一步操作

    记录内容：
    - 做了什么操作
    - 为什么这样做
    - 影响了多少行
    - 考虑过什么替代方案
    """
    log = []
    initial_rows = len(df)

    # 删除 age 缺失
    age_missing = df['age'].isna().sum()
    df = df.dropna(subset=['age'])
    log.append({
        'operation': '删除缺失值',
        'column': 'age',
        'reason': 'age 是关键变量，缺失行无法用于分析',
        'rows_affected': age_missing
    })

    # 填充 income
    income_missing = df['income'].isna().sum()
    df = df.copy()
    df['income'] = df['income'].fillna(df['income'].median())
    log.append({
        'operation': '填充缺失值',
        'column': 'income',
        'reason': '缺失率低，中位数稳健',
        'rows_affected': income_missing
    })

    return df, log


# ==================== 主函数：对比展示 ====================

def main() -> None:
    """主函数：展示错误做法与正确做法的对比"""
    print("=" * 70)
    print("数据清洗常见错误（反例）")
    print("=" * 70)

    df = generate_sample_data(n=100, seed=42)
    print(f"\n原始数据: {len(df)} 行")
    print(f"income 缺失: {df['income'].isna().sum()} 个")

    # 错误 1
    print("\n" + "=" * 70)
    print("错误 1：不分析缺失机制直接删除")
    print("=" * 70)
    df_bad = bad_delete_all_missing(df)
    print(f"❌ 直接删除后: {len(df_bad)} 行（损失了 {len(df) - len(df_bad)} 行）")
    print("   问题：没有分析缺失是否是随机的，可能引入偏差")

    df_good = good_analyze_then_handle(df)
    print(f"✅ 分析后处理: {len(df_good)} 行")
    print("   正确：根据缺失率决定删除或填充")

    # 错误 2
    print("\n" + "=" * 70)
    print("错误 2：对所有列统一使用均值填充")
    print("=" * 70)
    print("❌ 统一均值填充:")
    print("   问题：对右偏分布会高估典型值，对分类变量无意义")

    print("\n✅ 根据上下文选择策略:")
    print("   - 右偏数值：用中位数")
    print("   - 分类变量：用众数或单独类别")

    # 错误 3
    print("\n" + "=" * 70)
    print("错误 3：统一删除所有异常值")
    print("=" * 70)
    df_bad = bad_remove_all_outliers(df)
    print(f"❌ 删除所有异常值后: {len(df_bad)} 行")
    print("   问题：VIP 用户是真实存在的，删除会扭曲业务理解")

    df_good = good_classify_outliers(df)
    print(f"✅ 分类处理后: {len(df_good)} 行")
    print("   正确：只删除可疑值，保留 VIP")

    # 错误 4
    print("\n" + "=" * 70)
    print("错误 4：先标准化再划分训练/测试集")
    print("=" * 70)
    print("❌ 先标准化再划分:")
    print("   问题：数据泄漏！测试集信息通过均值/标准差泄漏")

    print("\n✅ 正确做法:")
    print("   1. 先划分训练/测试集")
    print("   2. 在训练集上 fit scaler")
    print("   3. 用同一个 scaler transform 测试集")

    # 错误 5
    print("\n" + "=" * 70)
    print("错误 5：不记录清洗过程")
    print("=" * 70)
    df_bad = bad_no_documentation(df)
    print(f"❌ 无记录清洗后: {len(df_bad)} 行")
    print("   问题：无法复现，无法审计，三个月后自己也忘了")

    df_good, log = good_with_logging(df)
    print(f"✅ 有记录清洗后: {len(df_good)} 行")
    print(f"   操作记录: {len(log)} 条")
    for entry in log:
        print(f"     - {entry['operation']} ({entry['column']}): {entry['reason']}")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("数据清洗不是技术问题，是决策问题。")
    print("每一个操作都要有理由，每一个理由都要被记录。")
    print("这样你的分析才是可复现、可审计、可信任的。")


if __name__ == "__main__":
    main()
