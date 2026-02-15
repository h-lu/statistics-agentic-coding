"""
示例：缺失值机制诊断（MCAR/MAR/MNAR）。

本例演示如何诊断缺失值的机制：
1. 缺失值对统计量的影响（错误的填充方式）
2. 缺失模式可视化
3. MCAR/MAR/MNAR 的判断方法

运行方式：python3 chapters/week_03/examples/01_missing_mechanism.py
预期输出：
- output/missing_fill_zero.png：对比填充0前后的分布
- 控制台输出：缺失值统计和诊断信息

核心知识点：
- 把缺失值填成0会严重扭曲统计量
- 缺失值本身就是信息，需要诊断而非盲目填充
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# 演示1：把缺失值填成0的影响（小北的错误）
# =============================================================================

def demonstrate_wrong_filling() -> pd.DataFrame:
    """
    演示把缺失值填成0的错误做法。

    小北拿到一份年龄数据，发现有缺失值，立刻填成0。
    结果：均值和中位数都被严重拉低。
    """
    print("="*60)
    print("【演示1】小北的错误：把缺失值填成0")
    print("="*60)

    # 创建模拟年龄数据（使用 float 类型以支持 NaN）
    np.random.seed(42)
    age_data = np.random.randint(18, 70, size=100).astype(float)
    # 随机让15%的值变成缺失
    mask = np.random.random(100) < 0.15
    age_data_missing = age_data.copy()
    age_data_missing[mask] = np.nan

    df = pd.DataFrame({"age": age_data_missing})

    print("\n原始数据的统计（忽略缺失值）：")
    print(f"  均值：{df['age'].mean():.1f} 岁")
    print(f"  中位数：{df['age'].median():.1f} 岁")
    print(f"  缺失率：{df['age'].isna().mean() * 100:.1f}%")

    # 错误做法：直接填0
    df_wrong = df.copy()
    df_wrong['age_filled_wrong'] = df_wrong['age'].fillna(0)

    print("\n错误做法（填充0）的统计：")
    print(f"  均值：{df_wrong['age_filled_wrong'].mean():.1f} 岁")
    print(f"  中位数：{df_wrong['age_filled_wrong'].median():.1f} 岁")

    mean_change = df['age'].mean() - df_wrong['age_filled_wrong'].mean()
    median_change = df['age'].median() - df_wrong['age_filled_wrong'].median()

    print(f"\n影响：均值降低了 {mean_change:.1f} 岁，中位数降低了 {median_change:.1f} 岁")
    print("结论：把缺失值填成0会严重扭曲数据的真实分布！")

    return df


def plot_wrong_filling_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """可视化对比：正确做法 vs 错误做法"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：正确做法（保留缺失或用合理填充）
    df['age'].hist(bins=20, edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].set_xlabel("年龄 (Age)")
    axes[0].set_ylabel("频数 (Frequency)")
    axes[0].set_title("正确做法：忽略缺失值\n(Correct: Missing values ignored)")

    # 右图：错误做法（填0）
    df_wrong = df.copy()
    df_wrong['age_filled_wrong'] = df_wrong['age'].fillna(0)
    df_wrong['age_filled_wrong'].hist(bins=20, edgecolor='black', alpha=0.7, ax=axes[1])
    axes[1].set_xlabel("年龄 (Age)")
    axes[1].set_ylabel("频数 (Frequency)")
    axes[1].set_title("错误做法：填充为0\n(Wrong: Missing values filled with 0)")

    plt.tight_layout()
    plt.savefig(output_dir / "missing_fill_zero.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图表已保存：{output_dir / 'missing_fill_zero.png'}")


# =============================================================================
# 演示2：缺失模式诊断
# =============================================================================

def diagnose_missing_mechanism(df: pd.DataFrame, var: str) -> None:
    """
    诊断缺失值的模式，判断可能是 MCAR/MAR/MNAR。

    MCAR (Missing Completely At Random)：缺失与任何变量无关
    MAR (Missing At Random)：缺失与观测变量有关，但与未观测变量无关
    MNAR (Missing Not At Random)：缺失与未观测变量本身有关
    """
    print(f"\n{'='*60}")
    print(f"【演示2】诊断 '{var}' 的缺失模式")
    print("="*60)

    # 1. 缺失率
    missing_rate = df[var].isna().mean()
    print(f"\n1. 缺失率：{missing_rate * 100:.1f}%")

    # 2. 缺失位置分布
    missing_indices = df[df[var].isna()].index
    print(f"\n2. 缺失位置（前10个）：{missing_indices[:10].tolist()}")

    # 3. MCAR 简单检验：如果缺失是随机的，位置应该均匀分布
    #    检查缺失之间是否有聚集现象
    if len(missing_indices) > 1:
        gaps = np.diff(missing_indices)
        avg_gap = gaps.mean()
        print(f"\n3. 缺失之间的平均间隔：{avg_gap:.1f} 行")
        print("   （如果间隔相对均匀，可能是MCAR；如果有聚集，可能是MAR）")

    # 4. 高缺失率警告（MNAR 指标）
    if missing_rate > 0.3:
        print(f"\n⚠️  警告：缺失率超过30%，可能是MNAR（非随机缺失）")
        print("   建议：深入调查缺失原因，考虑是否需要删除整列或使用高级填充方法")
    elif missing_rate > 0.1:
        print(f"\n⚠️  注意：缺失率在10%-30%之间，可能是MAR（随机缺失）")
        print("   建议：检查缺失是否与其他变量相关")
    else:
        print(f"\n✓ 缺失率较低（<10%），可能是MCAR或轻微MAR")
        print("   建议：可以直接删除或用中位数填充")

    # 5. 与其他变量的关系（MAR 检验）
    print(f"\n4. MAR检验：检查缺失是否与其他变量相关")
    # 检查"有缺失"和"无缺失"的组在其他变量上的差异
    df_with_missing_flag = df.copy()
    df_with_missing_flag[f'{var}_is_missing'] = df[var].isna()

    # 如果有其他数值变量，比较差异
    other_numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c != var]
    if other_numeric_cols:
        print("\n   与其他变量的关系：")
        for col in other_numeric_cols[:3]:  # 只检查前3个
            group_means = df_with_missing_flag.groupby(f'{var}_is_missing')[col].mean()
            if len(group_means) == 2:
                diff = abs(group_means[True] - group_means[False])
                pct_diff = diff / group_means[False] * 100 if group_means[False] != 0 else 0
                print(f"     - {col}: 有缺失组均值 vs 无缺失组均值差异 = {diff:.1f} ({pct_diff:.1f}%)")


# =============================================================================
# 演示3：在真实数据集上应用
# =============================================================================

def demonstrate_on_real_data() -> None:
    """在 Palmer Penguins 数据集上演示缺失值诊断"""
    print(f"\n{'='*60}")
    print("【演示3】真实数据集：Palmer Penguins")
    print("="*60)

    penguins = sns.load_dataset("penguins")

    print("\n整体缺失情况：")
    missing_summary = penguins.isna().sum()
    missing_rate = (penguins.isna().mean() * 100).round(1)
    for col in penguins.columns:
        if missing_summary[col] > 0:
            print(f"  {col}: {missing_summary[col]} 个缺失 ({missing_rate[col]}%)")

    # 诊断 bill_length_mm 的缺失
    if penguins['bill_length_mm'].isna().any():
        diagnose_missing_mechanism(penguins, 'bill_length_mm')

    # 诊断 sex 的缺失（可能是 MAR：与物种或岛屿相关）
    if penguins['sex'].isna().any():
        print(f"\n{'='*60}")
        print("【演示4】检验 MAR 假设：sex 缺失是否与 species 相关？")
        print("="*60)

        # 检查每个物种的 sex 缺失率
        for species in penguins['species'].unique():
            species_data = penguins[penguins['species'] == species]
            missing_rate = species_data['sex'].isna().mean() * 100
            print(f"{species}: sex 缺失率 = {missing_rate:.1f}%")

        # 如果缺失率在不同物种间差异很大，可能是 MAR
        species_missing_rates = penguins.groupby('species')['sex'].apply(lambda x: x.isna().mean() * 100)
        max_diff = species_missing_rates.max() - species_missing_rates.min()
        if max_diff > 5:
            print(f"\n⚠️  不同物种的缺失率差异达 {max_diff:.1f}%，可能是 MAR（与物种相关）")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "="*60)
    print("缺失值机制诊断示例")
    print("="*60)

    output_dir = setup_output_dir()

    # 演示1：错误填充的影响
    df = demonstrate_wrong_filling()
    plot_wrong_filling_comparison(df, output_dir)

    # 演示2-4：真实数据集上的诊断
    demonstrate_on_real_data()

    print("\n" + "="*60)
    print("核心结论")
    print("="*60)
    print("1. 缺失值不是'空白'，而是信息")
    print("2. 不要一上来就填充，先诊断缺失机制")
    print("3. MCAR：可以安全删除或简单填充")
    print("4. MAR：需要考虑与其他变量的关系")
    print("5. MNAR：最危险，需要深入调查")
    print("\n下一步：根据缺失机制选择合适的处理策略（见示例02）")


if __name__ == "__main__":
    main()
