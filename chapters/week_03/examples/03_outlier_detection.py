"""
示例：异常值检测方法（IQR vs Z-score）。

本例演示两种常见的异常值检测方法：
1. IQR 规则（基于箱线图，稳健，不依赖分布假设）
2. Z-score 规则（假设正态分布，敏感但不稳健）

运行方式：python3 chapters/week_03/examples/03_outlier_detection.py
预期输出：
- output/outlier_detection_methods.png：两种方法的对比图
- output/outlier_detection_penguins.png：企鹅数据的异常值
- 控制台输出：检测到的异常值统计

核心知识点：
- IQR 规则稳健，适用于任意分布
- Z-score 假设正态分布，数据偏态时不准确
- 统计规则只给出"候选异常值"，需要结合业务判断
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats


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
# 方法1：IQR 规则
# =============================================================================

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    用 IQR 规则检测异常值

    Args:
        series: 输入数据
        multiplier: IQR 倍数，默认 1.5（较宽松），3.0 用于极端异常值

    Returns:
        Boolean Series，True 表示该值是异常值

    原理：
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - 下界 = Q1 - multiplier * IQR
    - 上界 = Q3 + multiplier * IQR
    """
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower = q25 - multiplier * iqr
    upper = q75 + multiplier * iqr
    return (series < lower) | (series > upper)


def demo_iqr_method(data: pd.Series, var_name: str = "value") -> dict:
    """演示 IQR 方法"""
    print(f"\n{'='*60}")
    print(f"【方法1】IQR 规则检测 '{var_name}' 的异常值")
    print("="*60)

    # 计算统计量
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    print(f"\n统计量：")
    print(f"  Q25 (25th percentile): {q25:.1f}")
    print(f"  Q75 (75th percentile): {q75:.1f}")
    print(f"  IQR (Q75 - Q25): {iqr:.1f}")
    print(f"  下界 (Q25 - 1.5×IQR): {lower:.1f}")
    print(f"  上界 (Q75 + 1.5×IQR): {upper:.1f}")

    # 检测异常值
    outliers = detect_outliers_iqr(data)
    outlier_values = data[outliers]

    print(f"\n检测结果：")
    print(f"  检测到 {outliers.sum()} 个异常值（共 {len(data)} 个观测值）")
    if len(outlier_values) > 0:
        print(f"  异常值：{outlier_values.tolist()[:10]}")
        if len(outlier_values) > 10:
            print(f"    （只显示前10个，共{len(outlier_values)}个）")

    return {
        "method": "IQR (1.5×IQR)",
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "n_outliers": outliers.sum(),
        "outlier_values": outlier_values.tolist() if len(outlier_values) <= 10 else outlier_values.tolist()[:10]
    }


# =============================================================================
# 方法2：Z-score 规则
# =============================================================================

def detect_outliers_zscore(series: pd.Series, threshold: float = 3) -> pd.Series:
    """
    用 Z-score 检测异常值

    Args:
        series: 输入数据
        threshold: Z-score 阈值，默认 3（约 99.7% 数据在 ±3 SD 之间）

    Returns:
        Boolean Series，True 表示该值是异常值

    原理：
    - Z-score = (x - mean) / std
    - |Z-score| > threshold → 异常值
    - 假设数据近似正态分布

    注意：
    - 均值和标准差都会被极端值影响
    - 如果数据有偏态，Z-score 会不准确
    """
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def demo_zscore_method(data: pd.Series, var_name: str = "value") -> dict:
    """演示 Z-score 方法"""
    print(f"\n{'='*60}")
    print(f"【方法2】Z-score 规则检测 '{var_name}' 的异常值")
    print("="*60)

    # 计算统计量
    mean = data.mean()
    std = data.std()
    lower = mean - 3 * std
    upper = mean + 3 * std

    print(f"\n统计量：")
    print(f"  均值 (Mean): {mean:.1f}")
    print(f"  标准差 (Std Dev): {std:.1f}")
    print(f"  下界 (Mean - 3×SD): {lower:.1f}")
    print(f"  上界 (Mean + 3×SD): {upper:.1f}")

    # 检测异常值
    outliers = detect_outliers_zscore(data)
    outlier_values = data[outliers]
    outlier_z_scores = np.abs((data[outliers] - mean) / std)

    print(f"\n检测结果：")
    print(f"  检测到 {outliers.sum()} 个异常值（共 {len(data)} 个观测值）")
    if len(outlier_values) > 0:
        print(f"  异常值：{outlier_values.tolist()[:10]}")
        print(f"  对应的 Z-score：{outlier_z_scores.tolist()[:10]}")

    return {
        "method": "Z-score (±3 SD)",
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "n_outliers": outliers.sum(),
        "outlier_values": outlier_values.tolist() if len(outlier_values) <= 10 else outlier_values.tolist()[:10]
    }


# =============================================================================
# 可视化对比
# =============================================================================

def plot_method_comparison(data: pd.Series, var_name: str,
                           iqr_result: dict, zscore_result: dict,
                           output_dir: Path) -> None:
    """可视化对比两种方法"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：箱线图（IQR 规则）
    axes[0].boxplot(data.dropna(), vert=True)
    axes[0].set_ylabel(f"{var_name}")
    axes[0].set_title("箱线图 (IQR 规则)\nBoxplot (IQR Rule)")
    axes[0].set_xticks([])

    # 标注 IQR 边界
    axes[0].axhline(y=iqr_result['lower'], color='red', linestyle='--',
                    alpha=0.5, label=f"下界: {iqr_result['lower']:.0f}")
    axes[0].axhline(y=iqr_result['upper'], color='red', linestyle='--',
                    alpha=0.5, label=f"上界: {iqr_result['upper']:.0f}")
    axes[0].legend(fontsize=8)

    # 右图：带阈值线的直方图（Z-score 规则）
    axes[1].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(zscore_result['lower'], color='red', linestyle='--',
                    label=f"-3 SD: {zscore_result['lower']:.0f}")
    axes[1].axvline(zscore_result['upper'], color='red', linestyle='--',
                    label=f"+3 SD: {zscore_result['upper']:.0f}")
    axes[1].set_xlabel(f"{var_name}")
    axes[1].set_ylabel("频数 (Frequency)")
    axes[1].set_title("直方图 + Z-score 阈值\nHistogram with Z-score Threshold")
    axes[1].legend()

    plt.tight_layout()
    filename = f"outlier_detection_{var_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图表已保存：{output_dir / filename}")


# =============================================================================
# 在真实数据集上演示
# =============================================================================

def demonstrate_on_penguins() -> None:
    """在 Palmer Penguins 数据集上演示异常值检测"""
    print(f"\n{'='*60}")
    print("【真实数据集演示】Palmer Penguins")
    print("="*60)

    penguins = sns.load_dataset("penguins")

    # 1. 体重 (body_mass_g)
    print(f"\n{'─'*60}")
    print("变量：体重 (body_mass_g)")
    print("─"*60)

    body_mass = penguins["body_mass_g"].dropna()
    iqr_result = demo_iqr_method(body_mass, "body_mass_g")
    zscore_result = demo_zscore_method(body_mass, "body_mass_g")

    output_dir = setup_output_dir()
    plot_method_comparison(body_mass, "Body Mass (g)", iqr_result, zscore_result, output_dir)

    # 2. 嘴峰长度 (bill_length_mm)
    print(f"\n{'─'*60}")
    print("变量：嘴峰长度 (bill_length_mm)")
    print("─"*60)

    bill_length = penguins["bill_length_mm"].dropna()
    iqr_result = demo_iqr_method(bill_length, "bill_length_mm")

    # 3. 按物种分组检测（更合理的做法）
    print(f"\n{'='*60}")
    print("【进阶】按物种分组检测异常值")
    print("="*60)
    print("\n不同物种的体重分布不同，应该分组检测异常值\n")

    for species in penguins["species"].unique():
        species_data = penguins[penguins["species"] == species]["body_mass_g"].dropna()
        print(f"\n{species}:")
        outliers = detect_outliers_iqr(species_data)
        n_outliers = outliers.sum()
        if n_outliers > 0:
            outlier_values = species_data[outliers]
            print(f"  检测到 {n_outliers} 个异常值: {outlier_values.tolist()}")
        else:
            print(f"  无异常值")


# =============================================================================
# 创建带异常值的示例数据
# =============================================================================

def create_data_with_outliers() -> pd.Series:
    """创建带异常值的示例数据"""
    np.random.seed(42)
    # 正态分布数据
    normal_data = np.random.normal(loc=50, scale=10, size=100)
    # 添加一些异常值
    data_with_outliers = np.append(normal_data, [120, 130, 5, -5])
    return pd.Series(data_with_outliers)


# =============================================================================
# 对比两种方法
# =============================================================================

def compare_methods_on_synthetic_data() -> None:
    """在人工生成的数据上对比两种方法"""
    print(f"\n{'='*60}")
    print("【方法对比】人工数据（含明显异常值）")
    print("="*60)

    data = create_data_with_outliers()

    print("\n数据描述：")
    print(f"  样本量：{len(data)}")
    print(f"  均值：{data.mean():.1f}")
    print(f"  标准差：{data.std():.1f}")
    print(f"  偏度：{data.skew():.2f}")

    iqr_result = demo_iqr_method(data, "synthetic_value")
    zscore_result = demo_zscore_method(data, "synthetic_value")

    print(f"\n{'='*60}")
    print("方法对比总结")
    print("="*60)
    print(f"\nIQR 规则：检测到 {iqr_result['n_outliers']} 个异常值")
    print(f"Z-score 规则：检测到 {zscore_result['n_outliers']} 个异常值")

    if iqr_result['n_outliers'] != zscore_result['n_outliers']:
        print("\n⚠️  两种方法检测结果不同！")
        print("  → IQR 规则更稳健，不依赖分布假设")
        print("  → Z-score 假设正态分布，偏态数据时不准确")


# =============================================================================
# 老潘的建议
# =============================================================================

def print_laopan_advice() -> None:
    """输出老潘的建议"""
    print(f"\n{'='*60}")
    print("老潘的建议：IQR vs Z-score，该用哪个？")
    print("="*60)
    print("\n【IQR 规则】✓ 推荐")
    print("  优点：稳健，对分布形态不做假设")
    print("  适用：不知道数据是否正态，或数据有偏态")
    print("  场景：探索性分析（EDA）")

    print("\n【Z-score 规则】")
    print("  优点：精确，有概率解释（99.7% 在 ±3 SD 之间）")
    print("  缺点：不稳健，均值和标准差会被极端值影响")
    print("  适用：数据确实近似正态分布")
    print("  场景：验证性分析（已知分布假设）")

    print("\n【老潘的经验法则】")
    print("  1. 先用箱线图（IQR 规则）看一眼")
    print("  2. 如果有离群点，标记出来")
    print("  3. 如果有理由相信数据应该正态，可以同时用 Z-score 对照")
    print("  4. 更关键的是：结合业务规则判断")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "="*60)
    print("异常值检测方法示例")
    print("="*60)

    # 在人工数据上演示
    compare_methods_on_synthetic_data()

    # 在真实数据集上演示
    demonstrate_on_penguins()

    # 老潘的建议
    print_laopan_advice()

    print("\n" + "="*60)
    print("核心结论")
    print("="*60)
    print("1. IQR 规则稳健，适用于任意分布")
    print("2. Z-score 假设正态分布，偏态数据时不准确")
    print("3. 统计规则只给出'候选异常值'")
    print("4. 真正的关键：结合业务规则判断（见示例04）")


if __name__ == "__main__":
    main()
