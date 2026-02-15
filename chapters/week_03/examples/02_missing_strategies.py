"""
示例：缺失值处理策略（删除 vs 填充 vs 预测）。

本例演示三种常见的缺失值处理策略：
1. 删除（Listwise deletion / Selective deletion）
2. 填充（均值/中位数/前向/常量）
3. 预测填充（简介，Week 09 详述）

运行方式：python3 chapters/week_03/examples/02_missing_strategies.py
预期输出：
- output/missing_strategies_comparison.png：不同策略的对比图
- 控制台输出：各策略的统计结果和影响

核心知识点：
- 缺失率 < 5%：删除（影响不大）
- 缺失率 5%-30%：填充（中位数更稳健）
- 缺失率 > 30%：小心！可能需要删除整列
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# 创建带缺失值的示例数据
# =============================================================================

def create_sample_data_with_missing() -> pd.DataFrame:
    """创建带缺失值的示例数据（模拟年龄和收入）"""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.randint(20000, 100000, size=100)
    })
    # 随机让15%的age缺失
    df.loc[np.random.choice(df.index, size=15, replace=False), "age"] = np.nan
    # 随机让20%的income缺失
    df.loc[np.random.choice(df.index, size=20, replace=False), "income"] = np.nan
    return df


# =============================================================================
# 策略1：删除（Listwise deletion）
# =============================================================================

def strategy_listwise_deletion(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略1：删除所有含缺失值的行

    适用场景：缺失率较低（<5%），或缺失不是随机的
    优点：简单，不引入虚假数据
    缺点：会损失数据，样本量减少可能导致结论不稳定
    """
    print("\n" + "="*60)
    print("【策略1】Listwise Deletion（删除所有含缺失值的行）")
    print("="*60)

    original_len = len(df)
    df_dropped = df.dropna()
    dropped_count = original_len - len(df_dropped)

    print(f"\n原始数据规模：{original_len} 行")
    print(f"删除后数据规模：{len(df_dropped)} 行")
    print(f"丢弃了 {dropped_count} 行数据 ({dropped_count/original_len*100:.1f}%)")

    info = {
        "name": "Listwise Deletion",
        "rows_after": len(df_dropped),
        "rows_dropped": dropped_count,
        "drop_rate": dropped_count / original_len * 100
    }

    return df_dropped, info


def strategy_selective_deletion(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略1a：选择性删除（只删除在关键变量上缺失的行）

    适用场景：某些变量更重要，必须完整
    优点：保留更多数据
    缺点：可能仍然有缺失值需要处理
    """
    print("\n" + "="*60)
    print("【策略1a】Selective Deletion（只要求关键变量完整）")
    print("="*60)

    # 假设 age 是关键变量，income 可以缺失
    df_dropped = df.dropna(subset=["age"])
    dropped_count = len(df) - len(df_dropped)

    print(f"\n原始数据规模：{len(df)} 行")
    print(f"选择性删除后：{len(df_dropped)} 行")
    print(f"丢弃了 {dropped_count} 行 ({dropped_count/len(df)*100:.1f}%)")
    print("说明：只要求 age 不为空，income 可以缺失")

    info = {
        "name": "Selective Deletion (age only)",
        "rows_after": len(df_dropped),
        "rows_dropped": dropped_count,
        "drop_rate": dropped_count / len(df) * 100
    }

    return df_dropped, info


# =============================================================================
# 策略2：填充（Imputation）
# =============================================================================

def strategy_mean_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略2.1：均值填充

    适用场景：数据近似对称分布，MCAR
    优点：保持均值不变
    缺点：会低估方差，对异常值敏感
    """
    print("\n" + "="*60)
    print("【策略2.1】Mean Imputation（均值填充）")
    print("="*60)

    df_filled = df.copy()

    # 记录原始统计
    original_age_mean = df['age'].mean()
    original_income_mean = df['income'].mean()

    # 均值填充
    df_filled['age'] = df_filled['age'].fillna(df_filled['age'].mean())
    df_filled['income'] = df_filled['income'].fillna(df_filled['income'].mean())

    # 比较填充前后
    new_age_mean = df_filled['age'].mean()
    new_income_mean = df_filled['income'].mean()

    print(f"\nAge - 原始均值: {original_age_mean:.1f}, 填充后均值: {new_age_mean:.1f}")
    print(f"Income - 原始均值: {original_income_mean:.1f}, 填充后均值: {new_income_mean:.1f}")

    info = {
        "name": "Mean Imputation",
        "mean_age": new_age_mean,
        "mean_income": new_income_mean
    }

    return df_filled, info


def strategy_median_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略2.2：中位数填充（推荐）

    适用场景：数据有偏态或有异常值
    优点：对极端值稳健，不会引入假极端值
    缺点：会低估方差
    """
    print("\n" + "="*60)
    print("【策略2.2】Median Imputation（中位数填充）✓ 推荐")
    print("="*60)

    df_filled = df.copy()

    # 记录原始统计
    original_age_median = df['age'].median()
    original_income_median = df['income'].median()

    # 中位数填充
    df_filled['age'] = df_filled['age'].fillna(df_filled['age'].median())
    df_filled['income'] = df_filled['income'].fillna(df_filled['income'].median())

    # 比较填充前后
    new_age_median = df_filled['age'].median()
    new_income_median = df_filled['income'].median()

    print(f"\nAge - 原始中位数: {original_age_median:.1f}, 填充后中位数: {new_age_median:.1f}")
    print(f"Income - 原始中位数: {original_income_median:.1f}, 填充后中位数: {new_income_median:.1f}")

    info = {
        "name": "Median Imputation",
        "median_age": new_age_median,
        "median_income": new_income_median
    }

    return df_filled, info


def strategy_forward_fill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略2.3：前向填充（Forward fill）

    适用场景：时间序列数据
    优点：保留时间依赖性
    缺点：如果缺失很长，会传播错误值
    """
    print("\n" + "="*60)
    print("【策略2.3】Forward Fill（前向填充）")
    print("="*60)

    df_filled = df.copy()

    # 按索引排序后前向填充
    df_sorted = df_filled.sort_index()
    df_filled['age'] = df_sorted['age'].ffill().fillna(df_sorted['age'].median())
    df_filled['income'] = df_sorted['income'].ffill().fillna(df_sorted['income'].median())

    print(f"\n前向填充后，剩余缺失值：")
    print(f"  Age: {df_filled['age'].isna().sum()} 个")
    print(f"  Income: {df_filled['income'].isna().sum()} 个")
    print("说明：前向填充后仍有缺失的用中位数填充")

    info = {
        "name": "Forward Fill",
        "remaining_missing_age": df_filled['age'].isna().sum(),
        "remaining_missing_income": df_filled['income'].isna().sum()
    }

    return df_filled, info


def strategy_constant_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    策略2.4：常量填充（标记为"未知"）

    适用场景：缺失本身有意义，或需要保留"缺失"信息
    优点：显式标记缺失，便于后续分析
    缺点：引入新值，可能影响统计
    """
    print("\n" + "="*60)
    print("【策略2.4】Constant Imputation（常量填充，标记为'未知'）")
    print("="*60)

    df_filled = df.copy()

    # 用 -1 标记缺失（"未知"）
    df_filled['age'] = df_filled['age'].fillna(-1)
    df_filled['income'] = df_filled['income'].fillna(-1)

    print(f"\n用 -1 标记缺失值")
    print(f"Age 中 -1 的数量: {(df_filled['age'] == -1).sum()}")
    print(f"Income 中 -1 的数量: {(df_filled['income'] == -1).sum()}")

    info = {
        "name": "Constant Imputation (-1)",
        "unknown_age": (df_filled['age'] == -1).sum(),
        "unknown_income": (df_filled['income'] == -1).sum()
    }

    return df_filled, info


# =============================================================================
# 策略3：预测填充（简介）
# =============================================================================

def strategy_model_based_imputation_intro() -> None:
    """
    策略3：预测填充（Model-based imputation）简介

    用其他变量预测缺失值。Week 09 会详细讨论。
    常见方法：
    - KNN 填充：用相似样本的值填充
    - 回归填充：建立回归模型预测
    - 多重插补（Multiple Imputation）：生成多个可能的值

    注意：预测填充不是万能的，模型不准会引入更大误差
    """
    print("\n" + "="*60)
    print("【策略3】Model-based Imputation（预测填充）")
    print("="*60)
    print("\n本方法将在 Week 09 详细讨论。")
    print("\n常见方法：")
    print("  1. KNN 填充：用 k 个最相似样本的平均值/中位数填充")
    print("  2. 回归填充：建立回归模型预测缺失值")
    print("  3. 多重插补（Multiple Imputation）：生成多个可能值，综合结果")
    print("\n注意事项：")
    print("  - 需要其他变量与缺失变量相关")
    print("  - 预测模型不准会引入更大误差")
    print("  - 适合 MAR（与观测变量相关）的情况")


# =============================================================================
# 可视化对比
# =============================================================================

def plot_strategy_comparison(df_original: pd.DataFrame,
                             df_median: pd.DataFrame,
                             df_mean: pd.DataFrame,
                             output_dir: Path) -> None:
    """可视化对比不同策略的效果"""
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 0. 原始数据（有缺失）
    df_original['age'].dropna().hist(bins=20, edgecolor='black', alpha=0.7, ax=axes[0, 0])
    axes[0, 0].set_xlabel("年龄 (Age)")
    axes[0, 0].set_ylabel("频数 (Frequency)")
    axes[0, 0].set_title("原始数据（忽略缺失）\n(Original: Missing ignored)")

    # 1. 中位数填充
    df_median['age'].hist(bins=20, edgecolor='black', alpha=0.7, ax=axes[0, 1])
    axes[0, 1].axvline(df_median['age'].median(), color='red', linestyle='--',
                       label=f"Median: {df_median['age'].median():.0f}")
    axes[0, 1].set_xlabel("年龄 (Age)")
    axes[0, 1].set_ylabel("频数 (Frequency)")
    axes[0, 1].set_title("中位数填充\n(Median Imputation)")
    axes[0, 1].legend()

    # 2. 均值填充
    df_mean['age'].hist(bins=20, edgecolor='black', alpha=0.7, ax=axes[1, 0])
    axes[1, 0].axvline(df_mean['age'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_mean['age'].mean():.0f}")
    axes[1, 0].set_xlabel("年龄 (Age)")
    axes[1, 0].set_ylabel("频数 (Frequency)")
    axes[1, 0].set_title("均值填充\n(Mean Imputation)")
    axes[1, 0].legend()

    # 3. 对比图：三个策略的分布
    axes[1, 1].hist(df_original['age'].dropna(), bins=20, alpha=0.3,
                    label='Original (ignore missing)', edgecolor='black')
    axes[1, 1].hist(df_median['age'], bins=20, alpha=0.3,
                    label='Median filled', edgecolor='black')
    axes[1, 1].hist(df_mean['age'], bins=20, alpha=0.3,
                    label='Mean filled', edgecolor='black')
    axes[1, 1].set_xlabel("年龄 (Age)")
    axes[1, 1].set_ylabel("频数 (Frequency)")
    axes[1, 1].set_title("三种策略对比\n(Comparison)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "missing_strategies_comparison.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图表已保存：{output_dir / 'missing_strategies_comparison.png'}")


# =============================================================================
# 老潘的经验法则
# =============================================================================

def print_laopan_rule(df: pd.DataFrame) -> None:
    """输出老潘的经验法则"""
    print("\n" + "="*60)
    print("老潘的经验法则：如何选择缺失值处理策略")
    print("="*60)

    for col in df.columns:
        missing_rate = df[col].isna().mean()
        print(f"\n【{col}】缺失率：{missing_rate*100:.1f}%")

        if missing_rate < 0.05:
            print("  → 建议：直接删除（Listwise deletion）")
            print("  → 理由：缺失率低，删除影响不大")
        elif missing_rate < 0.3:
            print("  → 建议：中位数填充（Median imputation）")
            print("  → 理由：删除会损失数据，中位数填充更稳健")
        else:
            print("  → 警告：缺失率过高（>30%）！")
            print("  → 建议：")
            print("    1. 深入调查缺失原因（可能是 MNAR）")
            print("    2. 考虑删除整列")
            print("    3. 使用高级方法（如模型预测）")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "="*60)
    print("缺失值处理策略示例")
    print("="*60)

    output_dir = setup_output_dir()

    # 创建示例数据
    df = create_sample_data_with_missing()

    print(f"\n示例数据概览：")
    print(df.head(10))
    print(f"\n各列缺失率：")
    print(df.isna().mean().round(2))

    # 应用各种策略
    _, info_delete = strategy_listwise_deletion(df)
    _, info_selective = strategy_selective_deletion(df)
    df_mean, info_mean = strategy_mean_imputation(df.copy())
    df_median, info_median = strategy_median_imputation(df.copy())
    strategy_forward_fill(df.copy())
    strategy_constant_imputation(df.copy())
    strategy_model_based_imputation_intro()

    # 老潘的经验法则
    print_laopan_rule(df)

    # 可视化对比
    plot_strategy_comparison(df, df_median, df_mean, output_dir)

    print("\n" + "="*60)
    print("核心结论")
    print("="*60)
    print("1. 缺失率 < 5%：直接删除（影响不大）")
    print("2. 缺失率 5%-30%：中位数填充（对异常值稳健）")
    print("3. 缺失率 > 30%：小心！需要深入调查")
    print("\n选择策略的关键：先诊断缺失机制（见示例01），再决定如何处理")


if __name__ == "__main__":
    main()
