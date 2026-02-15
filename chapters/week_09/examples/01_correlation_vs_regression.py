"""
示例：相关 vs 回归——从"描述关系"到"量化关系"。

本例演示相关分析（correlation）和回归分析（regression）的区别：
- 相关回答"有多强的关系"（r 在 -1 到 1 之间）
- 回归回答"什么关系"（y = a + bx，量化 x 每增加 1 单位，y 的变化）

运行方式：python3 chapters/week_09/examples/01_correlation_vs_regression.py
预期输出：
  - stdout 输出相关系数和回归系数的对比
  - 展示散点图 + 回归线
  - 保存图表到 images/01_scatter_with_regression.png

核心概念：
  - Pearson 相关系数 r：描述线性关系强度和方向
  - 回归方程 y = a + bx：量化 y 如何随 x 变化
  - OLS（最小二乘法）：让残差平方和最小的拟合方法
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
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


def generate_ad_data(n: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    生成模拟广告投入与销售额数据

    参数:
        n: 样本量
        random_state: 随机种子

    返回:
        (ad_spend, sales): 广告投入和销售额数组
    """
    np.random.seed(random_state)
    ad_spend = np.random.normal(loc=50, scale=15, size=n)
    # sales = 10 + 0.5 * ad_spend + 噪声
    sales = 10 + 0.5 * ad_spend + np.random.normal(loc=0, scale=5, size=n)
    return ad_spend, sales


def correlation_example(x: np.ndarray, y: np.ndarray) -> None:
    """展示相关分析"""
    print("=" * 60)
    print("相关分析（Correlation）")
    print("=" * 60)

    r, p_value = stats.pearsonr(x, y)

    print(f"\nPearson 相关系数 r = {r:.4f}")
    print(f"p 值 = {p_value:.4f}")

    print("\n相关系数告诉你：")
    print("  - r = 0.75：广告投入和销售额有较强的正线性关系")
    print("  - r > 0：投入越多，销售越高")
    print("  - |r| 接近 1：关系很强")

    print("\n但相关系数的局限：")
    print("  - ❌ 不能告诉你'广告投入增加 1 万元，销售会增加多少'")
    print("  - ❌ 不能直接用于预测")
    print("  - ❌ 不能判断因果关系")


def regression_example(x: np.ndarray, y: np.ndarray) -> dict:
    """
    展示回归分析

    返回:
        dict 包含回归结果
    """
    print("\n" + "=" * 60)
    print("回归分析（Regression）")
    print("=" * 60)

    # 手动计算回归系数
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 斜率 b = Cov(x,y) / Var(x)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator

    # 截距 a = y_mean - b * x_mean
    intercept = y_mean - slope * x_mean

    # 计算 R²
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\n回归方程: sales = {intercept:.2f} + {slope:.4f} × ad_spend")
    print(f"  截距 (a): {intercept:.2f}")
    print(f"  斜率 (b): {slope:.4f}")
    print(f"  R²: {r_squared:.4f}")

    print("\n回归系数告诉你：")
    print("  - 斜率 = 0.5：广告投入每增加 1 万元，销售额平均增加 0.5 万元")
    print("  - 截距 = 10：广告投入为 0 时，基础销售额约 10 万元")
    print("  - R² = 0.56：模型解释了销售额 56% 的变异")

    print("\n回归的优势：")
    print("  - ✅ 量化关系：知道'增加多少'")
    print("  - ✅ 可以预测：给定 x，预测 y")
    print("  - ✅ 可以推断：检验系数是否显著")

    return {
        'intercept': intercept,
        'slope': slope,
        'r_squared': r_squared,
        'y_pred': y_pred
    }


def bad_example_correlation_only(x: np.ndarray, y: np.ndarray) -> None:
    """❌ 坏例子：只报告相关系数"""
    print("\n" + "=" * 60)
    print("❌ 坏例子：只报告相关系数")
    print("=" * 60)

    r, _ = stats.pearsonr(x, y)

    print(f"\n报告：广告投入和销售额的相关系数是 r = {r:.2f}")
    print("\n问题：")
    print("  - 决策者问：'我多投 10 万广告，能多卖多少？'")
    print("  - 你回答：'关系很强' → 没有量化影响")
    print("  - 决策者不满意：'我要的是数字，不是形容词'")


def good_example_with_regression(x: np.ndarray, y: np.ndarray, reg_result: dict) -> None:
    """✅ 好例子：相关 + 回归"""
    print("\n" + "=" * 60)
    print("✅ 好例子：相关 + 回归")
    print("=" * 60)

    r, _ = stats.pearsonr(x, y)

    print(f"\n报告：")
    print(f"  1. 广告投入和销售额呈强正相关（r = {r:.2f}）")
    print(f"  2. 回归分析：sales = {reg_result['intercept']:.2f} + {reg_result['slope']:.4f} × ad_spend")
    print(f"  3. 解读：广告投入每增加 1 万元，销售额平均增加 {reg_result['slope']:.2f} 万元")
    print(f"  4. 模型解释力：R² = {reg_result['r_squared']:.2f}")

    print("\n这样报告更好，因为：")
    print("  - 相关告诉决策者'有关系'")
    print("  - 回归告诉决策者'什么关系'")
    print("  - 可以用于预算决策和预测")


def plot_scatter_with_regression(x: np.ndarray, y: np.ndarray,
                                  reg_result: dict) -> None:
    """绘制散点图 + 回归线"""
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：只看相关（散点图）
    ax1.scatter(x, y, alpha=0.6, s=50, color='#2E86AB')
    ax1.set_xlabel('广告投入（万元）', fontsize=12)
    ax1.set_ylabel('销售额（万元）', fontsize=12)
    ax1.set_title('散点图：广告投入 vs 销售额\n'
                 '(相关告诉你"有多强的关系")', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # 标注相关系数
    r, _ = stats.pearsonr(x, y)
    ax1.text(0.05, 0.95, f'r = {r:.3f}',
             transform=ax1.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右图：加上回归线
    ax2.scatter(x, y, alpha=0.6, s=50, color='#2E86AB', label='观测值')
    ax2.plot(x, reg_result['y_pred'], color='#A23B72',
             linewidth=2.5, label=f"回归线\ny = {reg_result['intercept']:.2f} + {reg_result['slope']:.2f}x")
    ax2.set_xlabel('广告投入（万元）', fontsize=12)
    ax2.set_ylabel('销售额（万元）', fontsize=12)
    ax2.set_title('回归分析：加上回归线\n'
                 '(回归告诉你"什么关系")', fontsize=13)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 标注 R²
    ax2.text(0.05, 0.95, f"R² = {reg_result['r_squared']:.3f}",
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_scatter_with_regression.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/01_scatter_with_regression.png")


def compare_correlation_regression() -> None:
    """对比相关和回归的回答问题"""
    print("\n" + "=" * 60)
    print("相关 vs 回归：回答不同问题")
    print("=" * 60)

    print("\n决策者问：'如果广告投入增加 10 万元，销售额会增加多少？'\n")

    print("❌ 用相关回答：")
    print("  '广告投入和销售额的相关系数是 0.75，说明关系很强。'")
    print("  → 决策者：'我要的是数字，不是形容词！'\n")

    print("✅ 用回归回答：")
    print("  '根据回归分析，广告投入每增加 1 万元，")
    print("   销售额平均增加 0.5 万元。")
    print("   因此增加 10 万元广告，销售额预计增加 5 万元。'")
    print("  → 决策者：'这个信息有用！'\n")

    print("结论：")
    print("  - 相关：描述'有多强的关系'")
    print("  - 回归：量化'什么关系' + 预测")


def main() -> None:
    """主函数"""
    print('相关 vs 回归：从"描述关系"到"量化关系"\n')

    # 生成数据
    ad_spend, sales = generate_ad_data(n=100, random_state=42)

    # 相关分析
    correlation_example(ad_spend, sales)

    # 回归分析
    reg_result = regression_example(ad_spend, sales)

    # 坏例子 vs 好例子
    bad_example_correlation_only(ad_spend, sales)
    good_example_with_regression(ad_spend, sales, reg_result)

    # 对比相关和回归
    compare_correlation_regression()

    # 绘图
    plot_scatter_with_regression(ad_spend, sales, reg_result)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("\n关键要点：")
    print("  1. 相关系数 r：描述线性关系强度（-1 到 1）")
    print("  2. 回归方程 y = a + bx：量化 y 如何随 x 变化")
    print("  3. 斜率 b：x 每增加 1 单位，y 的变化量")
    print("  4. R²：模型解释的方差比例（0 到 1）")
    print("  5. 相关 + 回归配合使用才是正道")
    print("\n在报告中：")
    print("  ❌ '相关系数是 0.75'")
    print("  ✅ '广告投入和销售额呈强正相关（r = 0.75）；")
    print("      回归分析显示，广告投入每增加 1 万元，")
    print("      销售额平均增加 0.5 万元（95% CI: [0.4, 0.6]，p < 0.001）'")
    print()


if __name__ == "__main__":
    main()
