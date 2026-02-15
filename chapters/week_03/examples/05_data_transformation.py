"""
示例：数据转换与编码（标准化、归一化、对数变换、One-hot）。

本例演示常见的数据转换方法：
1. 标准化（Z-score）：让不同尺度的变量可比
2. 归一化（Min-max）：缩放到 [0, 1] 区间
3. 对数变换：压缩大数值，改善右偏
4. 特征编码：One-hot vs Label encoding

运行方式：python3 chapters/week_03/examples/05_data_transformation.py
预期输出：
- output/transformation_comparison.png：标准化 vs 归一化对比
- output/log_transform_demo.png：对数变换前后对比
- 控制台输出：转换前后的统计量

核心知识点：
- 标准化适合统计分析（回归、检验）
- 归一化适合机器学习（神经网络）
- 对数变换只适合右偏的正值数据
- 名义变量用 One-hot，有序变量用 Label encoding
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


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
# 1. 标准化 vs 归一化
# =============================================================================

def demonstrate_standardization_vs_normalization() -> None:
    """演示标准化和归一化的区别"""
    print("="*60)
    print("【演示1】标准化 vs 归一化")
    print("="*60)

    penguins = sns.load_dataset("penguins")

    # 选择两个数值型变量（不同尺度）
    numeric_cols = ["bill_length_mm", "body_mass_g"]
    data = penguins[numeric_cols].dropna()

    print("\n原始数据统计：")
    print(data.describe().round(1))

    # 方法1：标准化（Z-score）
    scaler_std = StandardScaler()
    data_standardized = pd.DataFrame(
        scaler_std.fit_transform(data),
        columns=numeric_cols
    )

    print("\n标准化后统计（均值≈0，标准差≈1）：")
    print(data_standardized.describe().round(2))

    # 方法2：归一化（Min-max）
    scaler_norm = MinMaxScaler()
    data_normalized = pd.DataFrame(
        scaler_norm.fit_transform(data),
        columns=numeric_cols
    )

    print("\n归一化后统计（最小值≈0，最大值≈1）：")
    print(data_normalized.describe().round(2))

    # 可视化对比
    plot_transformation_comparison(data, data_standardized, data_normalized,
                                   numeric_cols, setup_output_dir())


def plot_transformation_comparison(data_orig: pd.DataFrame,
                                   data_std: pd.DataFrame,
                                   data_norm: pd.DataFrame,
                                   cols: list,
                                   output_dir: Path) -> None:
    """可视化对比三种转换方法"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左图：原始数据（不同尺度）
    for col in cols:
        data_orig[col].plot(kind="hist", alpha=0.5, label=col, ax=axes[0],
                           edgecolor="black")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("原始数据\n(Different Scales)")
    axes[0].legend()

    # 中图：标准化后（相同尺度）
    for col in cols:
        data_std[col].plot(kind="hist", alpha=0.5, label=col, ax=axes[1],
                          edgecolor="black")
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("标准化后\n(Same Scale, mean=0, std=1)")
    axes[1].legend()

    # 右图：归一化后（都在 [0,1]）
    for col in cols:
        data_norm[col].plot(kind="hist", alpha=0.5, label=col, ax=axes[2],
                           edgecolor="black")
    axes[2].set_xlabel("Normalized Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("归一化后\n(All in [0,1])")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "transformation_comparison.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图表已保存：{output_dir / 'transformation_comparison.png'}")


# =============================================================================
# 2. 对数变换
# =============================================================================

def demonstrate_log_transform() -> None:
    """演示对数变换改善偏态"""
    print("\n" + "="*60)
    print("【演示2】对数变换：让右偏数据变对称")
    print("="*60)

    # 创建右偏分布（模拟收入数据）
    np.random.seed(42)
    income_data = np.random.lognormal(mean=10, sigma=0.5, size=1000)

    print("\n原始数据统计（右偏）：")
    print(f"  均值：{income_data.mean():.0f}")
    print(f"  中位数：{np.median(income_data):.0f}")
    print(f"  标准差：{income_data.std():.0f}")
    print(f"  偏度：{pd.Series(income_data).skew():.2f}")
    print(f"  说明：偏度 > 0 表示右偏（右边的尾巴较长）")

    # 对数变换
    log_income = np.log(income_data)

    print("\n对数变换后统计（更对称）：")
    print(f"  均值：{log_income.mean():.2f}")
    print(f"  中位数：{np.median(log_income):.2f}")
    print(f"  标准差：{log_income.std():.2f}")
    print(f"  偏度：{pd.Series(log_income).skew():.2f}")
    print(f"  说明：偏度接近 0 表示分布对称")

    # 可视化对比
    plot_log_transform(income_data, log_income, setup_output_dir())


def plot_log_transform(data_orig: np.ndarray, data_log: np.ndarray,
                       output_dir: Path) -> None:
    """可视化对数变换的效果"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：原始数据（右偏）
    axes[0].hist(data_orig, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("收入 (Income)")
    axes[0].set_ylabel("频数 (Frequency)")
    axes[0].set_title("原始数据：右偏分布\n(Right-skewed)")

    # 右图：对数变换后（更对称）
    axes[1].hist(data_log, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("log(收入) (Log Income)")
    axes[1].set_ylabel("频数 (Frequency)")
    axes[1].set_title("对数变换后：更对称\n(More Symmetric)")

    plt.tight_layout()
    plt.savefig(output_dir / "log_transform_demo.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图表已保存：{output_dir / 'log_transform_demo.png'}")


# =============================================================================
# 3. 特征编码：One-hot vs Label
# =============================================================================

def demonstrate_feature_encoding() -> None:
    """演示特征编码方法"""
    print("\n" + "="*60)
    print("【演示3】特征编码：One-hot vs Label")
    print("="*60)

    penguins = sns.load_dataset("penguins")

    print("\n原始分类变量：")
    print(penguins[["species", "island", "sex"]].drop_duplicates().head(10))

    # One-hot 编码
    print("\n─"*60)
    print("方法1：One-hot 编码（推荐用于名义变量）")
    print("─"*60)

    species_onehot = pd.get_dummies(penguins["species"], prefix="species",
                                     drop_first=True)
    print("\nOne-hot 编码结果（species）：")
    print(species_onehot.head())
    print("\n说明：")
    print("  - 为每个类别创建一个二进制列（0 或 1）")
    print("  - drop_first=True 避免多重共线性（Week 09 详述）")
    print("  - 适合名义变量（无顺序关系，如 species）")

    # Label 编码
    print("\n─"*60)
    print("方法2：Label 编码（仅用于有序变量）")
    print("─"*60)

    le = LabelEncoder()
    penguins_copy = penguins.copy()
    penguins_copy["species_encoded"] = le.fit_transform(penguins_copy["species"])

    print("\nLabel 编码结果（species）：")
    print(penguins_copy[["species", "species_encoded"]].drop_duplicates()
          .sort_values("species_encoded"))
    print("\n说明：")
    print("  - 把类别映射为整数（0、1、2...）")
    print("  - 引入'顺序关系'（0 < 1 < 2），但 species 是名义变量！")
    print("  - 警告：如果用线性回归，模型会误以为 Gentoo(2) 是 Chinstrap(1) 的'两倍'")
    print("  - 仅适合有序变量（如 education_level: 高中 < 本科 < 硕士 < 博士）")

    # 创建有序变量示例
    print("\n有序变量示例（education_level）：")
    education_data = pd.DataFrame({
        "person_id": [1, 2, 3, 4, 5],
        "education": ["高中", "本科", "博士", "硕士", "本科"]
    })
    print(education_data)

    # 定义顺序
    education_order = {"高中": 0, "本科": 1, "硕士": 2, "博士": 3}
    education_data["education_encoded"] = education_data["education"].map(education_order)

    print("\nLabel 编码结果（education）：")
    print(education_data.sort_values("education_encoded"))
    print("\n说明：education 是有序变量，可以用 Label 编码")


# =============================================================================
# 老潘的经验法则
# =============================================================================

def print_laopan_rules() -> None:
    """输出老潘的经验法则"""
    print("\n" + "="*60)
    print("老潘的经验法则：数据转换与编码")
    print("="*60)

    print("\n【标准化 vs 归一化】")
    print("  标准化（Z-score）：")
    print("    - 对新数据更稳健")
    print("    - 保留'相对位置'信息")
    print("    - 适合统计分析（回归、检验、聚类）")
    print()
    print("  归一化（Min-max）：")
    print("    - 结果直观（都在 [0, 1]）")
    print("    - 对新值敏感（新极端值会重新缩放）")
    print("    - 适合神经网络、深度学习")
    print()
    print("  老潘的建议：")
    print("    → 做统计分析用标准化")
    print("    → 做机器学习用归一化")
    print("    → 不要同时用，会让数据失去意义")

    print("\n【对数变换】")
    print("  适用场景：右偏的正值数据（如收入、价格、计数）")
    print("  不适用场景：")
    print("    - 数据有负值或零（log(0) 无定义）")
    print("    - 数据已经近似对称")
    print("  老潘的建议：")
    print("    → 先看偏度，再决定是否变换")
    print("    → 变换后重新检查分布")

    print("\n【特征编码】")
    print("  One-hot 编码：")
    print("    - 适合名义变量（无顺序，如 species、island）")
    print("    - 优点：不引入虚假的顺序关系")
    print("    - 缺点：类别多时会增加很多列")
    print()
    print("  Label 编码：")
    print("    - 只适合有序变量（有顺序，如 education_level）")
    print("    - 警告：对名义变量会引入虚假顺序！")
    print()
    print("  老潘的建议：")
    print("    → 名义变量用 One-hot")
    print("    → 有序变量用 Label")
    print("    → 不要'为了省列数'用 Label 编码名义变量")


# =============================================================================
# 何时需要数据转换？
# =============================================================================

def when_to_transform() -> None:
    """解释何时需要数据转换"""
    print("\n" + "="*60)
    print("何时需要数据转换？")
    print("="*60)

    print("\n不要'为了用而用'。数据转换的目的是让数据更适合分析。\n")

    print("【需要标准化的场景】")
    print("  ✓ 不同尺度的变量需要比较（如身高cm vs 体重kg）")
    print("  ✓ 基于距离的算法（KNN、K-means、SVM）")
    print("  ✓ 回归分析（系数可比）")

    print("\n【需要对数变换的场景】")
    print("  ✓ 数据有严重偏态（偏度 > 1 或 < -1）")
    print("  ✓ 方差不齐（不同组的方差差异大）")
    print("  ✓ 残差不满足正态假设（回归诊断）")

    print("\n【不需要转换的场景】")
    print("  ✗ 决策树、随机森林（对尺度不敏感）")
    print("  ✗ 数据已经近似正态分布")
    print("  ✗ 转换会让解释变复杂（如'对数收入'不如'收入'直观）")

    print("\n阿码问：'那我用 AI 自动判断是否需要转换？'")
    print("\n技术上可以。AI 可以用统计检验（如 Shapiro-Wilk）告诉你")
    print("这份数据不符合正态分布，然后建议对数变换。")
    print("\n但这里有个问题：正态性不是所有分析的前提。")
    print("很多现代方法（如决策树、随机森林）对分布形态不敏感。")
    print("\n老潘说：'不要为了变换而变换。先问：你的分析目标是什么？'")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "="*60)
    print("数据转换与编码示例")
    print("="*60)

    # 演示1：标准化 vs 归一化
    demonstrate_standardization_vs_normalization()

    # 演示2：对数变换
    demonstrate_log_transform()

    # 演示3：特征编码
    demonstrate_feature_encoding()

    # 老潘的经验法则
    print_laopan_rules()

    # 何时需要转换
    when_to_transform()

    print("\n" + "="*60)
    print("核心结论")
    print("="*60)
    print("1. 标准化让不同尺度的变量可比")
    print("2. 归一化缩放到 [0,1]，适合神经网络")
    print("3. 对数变换改善右偏，只适合正值数据")
    print("4. 名义变量用 One-hot，有序变量用 Label")
    print("5. 不要'为了用而用'，先问分析目标是什么")


if __name__ == "__main__":
    main()
