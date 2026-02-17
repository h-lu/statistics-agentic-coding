"""
示例：逻辑回归与概率预测——从 Sigmoid 函数到对数几率

运行方式：python3 chapters/week_10/examples/02_logistic_regression.py
预期输出：Sigmoid 函数可视化、逻辑回归训练与概率预测、系数解读
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 配置中文字体
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid 函数：把任何数压缩到 [0, 1]"""
    return 1 / (1 + np.exp(-z))


def visualize_sigmoid() -> None:
    """可视化 Sigmoid 函数"""
    print("=" * 60)
    print("Sigmoid 函数可视化")
    print("=" * 60)

    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Sigmoid 曲线
    z = np.linspace(-7, 7, 200)
    sigma = sigmoid(z)

    axes[0].plot(z, sigma, 'b-', linewidth=2, label='σ(z) = 1 / (1 + e^(-z))')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y = 0.5 (决策边界)')
    axes[0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('z (线性组合: a + b₁x₁ + b₂x₂ + ...)')
    axes[0].set_ylabel('σ(z) (概率)')
    axes[0].set_title('Sigmoid 函数：把任何数压缩到 [0, 1]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # 右图：输入输出对照表
    z_values = [-5, -2, 0, 2, 5]
    sigma_values = sigmoid(np.array(z_values))

    table_data = []
    for zv, sv in zip(z_values, sigma_values):
        table_data.append([zv, f"{sv:.4f}"])

    axes[1].axis('off')
    table = axes[1].table(cellText=table_data,
                          colLabels=['输入 z', '输出 σ(z)'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    axes[1].set_title('Sigmoid 函数输入输出对照', pad=20)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'sigmoid_function.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / 'sigmoid_function.png'}")

    # 打印对照表
    print("\nSigmoid 函数输入输出对照：")
    print("-" * 40)
    print(f"{'输入 z':>10} | {'输出 σ(z)':>15}")
    print("-" * 40)
    for zv, sv in zip(z_values, sigma_values):
        print(f"{zv:>10} | {sv:>15.4f}")
    print("-" * 40)


def demonstrate_odds_and_logit() -> None:
    """演示几率与对数几率"""
    print("\n" + "=" * 60)
    print("几率与对数几率")
    print("=" * 60)

    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    odds = probabilities / (1 - probabilities)
    log_odds = np.log(odds)

    print(f"\n{'概率 p':>10} | {'几率 (p/1-p)':>15} | {'对数几率 log(odds)':>20}")
    print("-" * 60)
    for p, o, lo in zip(probabilities, odds, log_odds):
        print(f"{p:>10.2f} | {o:>15.4f} | {lo:>20.4f}")
    print("-" * 60)

    print("\n关键洞察：")
    print("  - 概率 = 0.5 时，几率 = 1，对数几率 = 0")
    print("  - 概率 > 0.5 时，几率 > 1，对数几率 > 0")
    print("  - 概率 < 0.5 时，几率 < 1，对数几率 < 0")
    print("  - 逻辑回归假设：特征与对数几率之间是线性关系")


def train_logistic_regression() -> dict:
    """训练逻辑回归模型"""
    print("\n" + "=" * 60)
    print("逻辑回归实战：预测客户流失")
    print("=" * 60)

    # 使用 seaborn 的 titanic 数据集
    titanic = sns.load_dataset("titanic")
    # 简化数据：只保留数值特征，删除缺失值
    titanic_clean = titanic[['pclass', 'age', 'sibsp', 'parch', 'fare', 'survived']].dropna()

    X = titanic_clean[['pclass', 'age', 'sibsp', 'parch', 'fare']].values
    y = titanic_clean['survived'].values

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n模型性能：")
    print(f"  准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(f"    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    # 系数解读
    feature_names = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    print(f"\n逻辑回归系数解读：")
    print(f"{'特征':<10} | {'系数':>10} | {'exp(系数)':>12} | {'解读'}")
    print("-" * 70)
    for name, coef in zip(feature_names, model.coef_[0]):
        exp_coef = np.exp(coef)
        direction = "提高" if coef > 0 else "降低"
        print(f"{name:<10} | {coef:>10.4f} | {exp_coef:>12.4f} | {direction}生存概率")
    print("-" * 70)
    print(f"截距: {model.intercept_[0]:.4f}")

    # 预测概率示例
    print(f"\n预测概率示例（前 5 个测试样本）：")
    print(f"{'实际':<6} | {'预测概率':>10} | {'预测类别':>10}")
    print("-" * 40)
    for i in range(5):
        print(f"{y_test[i]:<6} | {y_prob[i]:>10.4f} | {y_pred[i]:>10}")

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_names': feature_names
    }


def visualize_probability_predictions(results: dict) -> None:
    """可视化概率预测"""
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    y_prob = results['y_prob']
    y_test = results['y_test']

    # 分别画出正类和负类的概率分布
    prob_neg = y_prob[y_test == 0]  # 实际未生存的预测概率
    prob_pos = y_prob[y_test == 1]  # 实际生存的预测概率

    ax.hist(prob_neg, bins=20, alpha=0.5, label='实际未生存', color='red', edgecolor='black')
    ax.hist(prob_pos, bins=20, alpha=0.5, label='实际生存', color='green', edgecolor='black')

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='决策边界 (0.5)')
    ax.set_xlabel('预测生存概率')
    ax.set_ylabel('样本数量')
    ax.set_title('逻辑回归概率预测分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'logistic_probability_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'logistic_probability_distribution.png'}")


def main() -> None:
    """主函数"""
    from pathlib import Path

    visualize_sigmoid()
    demonstrate_odds_and_logit()
    results = train_logistic_regression()
    visualize_probability_predictions(results)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    逻辑回归核心要点：
    1. Sigmoid 函数：把任何数压缩到 [0, 1]，输出是概率
    2. 对数几率 = a + b₁x₁ + b₂x₂ + ... （线性关系）
    3. 系数解读：
       - b > 0：特征增加会提高正类概率
       - b < 0：特征增加会降低正类概率
       - exp(b)：几率的变化倍数
    4. 预测的是概率，不是硬分类
    5. 通过阈值（默认 0.5）将概率转换为类别
    """)


if __name__ == "__main__":
    main()
