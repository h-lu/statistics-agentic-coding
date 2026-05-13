"""
示例：贝叶斯定理演示——条件概率与"逆向概率"计算

本例演示贝叶斯定理的核心概念：P(A|B) = P(B|A) × P(A) / P(B)
用一个医疗检测的经典场景展示"逆向概率"的计算。

运行方式：python3 chapters/week_14/examples/01_bayes_theorem_demo.py

预期输出：
- 打印贝叶斯定理各组成部分的计算结果
- 打印常见误解的对比（正例 vs 反例）
- 生成一张先验/后验对比的条形图到 images/
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===== 图表中文字体配置 =====
def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei', 'Droid Sans Fallback', 'Droid Sans Fallback']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


# ===== 正例：正确的贝叶斯计算 =====
def bayes_theorem_correct(prior: float, sensitivity: float, false_positive: float) -> dict:
    """
    正确应用贝叶斯定理计算 P(患病|检测阳性)

    参数:
        prior: P(患病) - 先验概率（患病率）
        sensitivity: P(检测阳性|患病) - 灵敏度/真阳性率
        false_positive: P(检测阳性|健康) - 假阳性率

    返回:
        包含各概率分量的字典
    """
    # P(检测阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)
    # 这是全概率公式，归一化常数
    p_positive = sensitivity * prior + false_positive * (1 - prior)

    # 贝叶斯定理：P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
    posterior = (sensitivity * prior) / p_positive

    return {
        '先验 P(患病)': prior,
        '似然 P(阳性|患病)': sensitivity,
        '似然 P(阳性|健康)': false_positive,
        '证据 P(阳性)': p_positive,
        '后验 P(患病|阳性)': posterior
    }


# ===== 反例：常见误解 =====
def common_misunderstanding(prior: float, sensitivity: float, false_positive: float) -> dict:
    """
    常见误解：直接把灵敏度当成后验概率

    这是很多人会犯的错误：以为"检测阳性且真患病的概率"=灵敏度
    实际上灵敏度是 P(检测阳性|患病)，不是 P(患病|检测阳性)

    返回:
        错误计算的"假后验"和正确的后验对比
    """
    # 错误：直接用灵敏度作为"阳性时患病的概率"
    wrong_posterior = sensitivity

    # 正确：需要用同一场景的假阳性率计算，不能偷偷更换参数
    p_positive = sensitivity * prior + false_positive * (1 - prior)
    correct_posterior = (sensitivity * prior) / p_positive

    return {
        '错误理解（直接用灵敏度）': wrong_posterior,
        '正确理解（贝叶斯后验）': correct_posterior,
        '差异': wrong_posterior - correct_posterior
    }


# ===== 流失率场景（贯穿案例） =====
def churn_rate_bayes_example():
    """
    贯穿案例：用贝叶斯定理估计流失概率

    场景：公司历史流失率 15%（先验）
          某预测模型说"这个客户会流失"（似然=0.8）
          模型对非流失客户误报率 20%
    """
    prior_churn = 0.15      # P(流失)
    model_sensitivity = 0.8  # P(预测流失|真流失)
    model_false_alarm = 0.2  # P(预测流失|真不流失)

    result = bayes_theorem_correct(prior_churn, model_sensitivity, model_false_alarm)

    print("=" * 50)
    print("流失率场景：模型预测流失，实际流失的概率是多少？")
    print("=" * 50)
    for key, value in result.items():
        if key == '后验 P(患病|阳性)':
            print(f"后验 P(流失|预测流失): {value:.1%}")
        elif key == '先验 P(患病)':
            print(f"先验 P(流失): {value:.1%}")
        elif key == '似然 P(阳性|患病)':
            print(f"模型对流失客户的召回率: {value:.1%}")
        elif key == '似然 P(阳性|健康)':
            print(f"模型对不流失客户的误报率: {value:.1%}")

    print(f"\n结论：即使模型预测流失，实际流失概率只有 {result['后验 P(患病|阳性)']:.1%}")
    print(f"原因：先验流失率较低（{prior_churn:.1%}），大量假阳性稀释了结果")
    print()

    return result


# ===== 可视化：先验 vs 后验 =====
def plot_prior_posterior(prior: float, posterior: float, output_dir: Path) -> None:
    """
    绘制先验和后验的对比条形图

    参数:
        prior: 先验概率
        posterior: 后验概率
        output_dir: 图片输出目录
    """
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['先验\nP(流失)', '后验\nP(流失|预测流失)']
    values = [prior * 100, posterior * 100]
    colors = ['#3498db', '#e74c3c']

    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

    # 在柱子上标注数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('概率 (%)', fontsize=12)
    ax.set_title('贝叶斯更新：先验如何被数据更新', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)

    # 添加说明文字
    info_text = (f"先验：历史流失率 {prior:.1%}\n"
                 f"后验：给定模型预测流失后的\n"
                 f"实际流失概率 {posterior:.1%}")
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_bayes_update.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '01_bayes_update.png'}")


# ===== 主函数 =====
def main() -> None:
    output_dir = Path(__file__).parent.parent / 'images'

    # 示例 1：医疗检测场景（经典）
    print("\n" + "=" * 60)
    print("示例 1：医疗检测——贝叶斯定理的直觉")
    print("=" * 60)

    # 假设：某疾病患病率 1%，检测灵敏度 99%，假阳性率 5%
    prior_disease = 0.01
    sensitivity = 0.99
    false_positive = 0.05

    result = bayes_theorem_correct(prior_disease, sensitivity, false_positive)

    print(f"\n先验 P(患病) = {prior_disease:.1%}")
    print(f"灵敏度 P(检测阳性|患病) = {sensitivity:.1%}")
    print(f"假阳性率 P(检测阳性|健康) = {false_positive:.1%}")
    print(f"\n证据 P(检测阳性) = {result['证据 P(阳性)']:.4f}")
    print(f"后验 P(患病|检测阳性) = {result['后验 P(患病|阳性)']:.1%}")

    print("\n🔍 关键洞察：")
    print(f"   即使检测灵敏度高达 {sensitivity:.1%}，")
    print(f"   阳性结果时真正患病的概率只有 {result['后验 P(患病|阳性)']:.1%}！")
    print(f"   原因：假阳性太多（健康人群中 5% 误报）")

    # 示例 2：常见误解（反例）
    print("\n" + "=" * 60)
    print("示例 2：常见误解（反例）")
    print("=" * 60)

    misunderstanding = common_misunderstanding(prior_disease, sensitivity, false_positive)

    print(f"\n错误： 错误理解：\"检测阳性，所以我有 {sensitivity:.1%} 的概率患病\"")
    print(f"   问题：混淆了 P(阳性|患病) 和 P(患病|阳性)")

    print(f"\n通过：正确理解：")
    print(f"   P(患病|阳性) = {misunderstanding['正确理解（贝叶斯后验）']:.1%}")
    print(f"   （用贝叶斯定理正确计算）")

    print(f"\n差异：{misunderstanding['差异']:.1%} —— 这是一个巨大的差距！")

    # 示例 3：流失率场景（贯穿案例）
    churn_result = churn_rate_bayes_example()

    # 可视化
    plot_prior_posterior(churn_result['先验 P(患病)'],
                        churn_result['后验 P(患病|阳性)'],
                        output_dir)


if __name__ == "__main__":
    main()
