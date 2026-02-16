"""
示例：逻辑回归完整实践——从 sigmoid 到概率预测。

本例演示逻辑回归的完整流程：
1. Sigmoid 函数可视化：理解如何将线性输出映射到概率
2. 数据准备与预处理：特征缩放、分层抽样
3. 模型拟合：使用 scikit-learn 1.8+ 最佳实践
4. 概率预测与类别预测：理解阈值的作用
5. 系数解释：对数几率的含义
6. 统计显著性检验：使用 statsmodels

运行方式：python3 chapters/week_10/examples/03_logistic_regression.py
预期输出：
  - stdout 输出逻辑回归结果
  - 展示 sigmoid 函数可视化
  - 保存图表到 images/03_logistic_regression.png

核心概念：
  - Sigmoid: p = 1 / (1 + e^(-z))
  - 决策边界：p >= 0.5 预测为正类
  - 对数几率：ln(p/(1-p)) = a + b1*x1 + b2*x2 + ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid 函数：将任何实数映射到 (0, 1)。
    
    公式: p = 1 / (1 + e^(-z))
    
    参数:
        z: 输入值（可以是数组）
        
    返回:
        Sigmoid 转换后的值
    """
    return 1 / (1 + np.exp(-z))


def visualize_sigmoid() -> None:
    """可视化 sigmoid 函数"""
    print("=" * 70)
    print("Sigmoid 函数可视化")
    print("=" * 70)
    
    z = np.linspace(-10, 10, 100)
    p = sigmoid(z)
    
    print("\nSigmoid 函数特性：")
    print(f"  z = 0 时, p = {sigmoid(0):.4f} (中间点)")
    print(f"  z → +∞ 时, p → {sigmoid(10):.4f}")
    print(f"  z → -∞ 时, p → {sigmoid(-10):.4f}")
    print(f"  输出范围: ({p.min():.4f}, {p.max():.4f})")
    
    print("\nSigmoid 的作用：")
    print("  - 把线性组合 z = a + b₁x₁ + b₂x₂ + ... 压缩到 (0, 1)")
    print("  - 输出可解释为'属于正类的概率'")
    print("  - 平滑可导，便于优化")


def generate_customer_data(n: int = 800, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟电商客户数据。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征和高价值标签
    """
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        '历史消费金额': np.random.exponential(scale=100, size=n),
        '注册月数': np.random.randint(1, 48, n),
        '月均浏览次数': np.random.poisson(40, n),
        '购物车添加次数': np.random.poisson(8, n)
    })
    
    # 高价值客户的逻辑：历史消费高、注册久、浏览多、购物车添加多
    score = (
        0.01 * data['历史消费金额'] +
        0.1 * data['注册月数'] +
        0.05 * data['月均浏览次数'] +
        0.2 * data['购物车添加次数'] -
        5 +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > 0).astype(int)
    
    return data


def fit_logistic_regression(df: pd.DataFrame) -> dict:
    """
    拟合逻辑回归模型并返回结果。
    
    返回:
        dict 包含模型和评估结果
    """
    print("\n" + "=" * 70)
    print("逻辑回归模型拟合")
    print("=" * 70)
    
    # 特征和目标
    feature_cols = ['历史消费金额', '注册月数', '月均浏览次数', '购物车添加次数']
    X = df[feature_cols]
    y = df['是否高价值']
    
    print(f"\n数据集信息：")
    print(f"  样本量: {len(df)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  类别分布: 低价值={(y==0).sum()}, 高价值={(y==1).sum()}")
    
    # 划分训练集和测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n数据划分：")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 特征缩放（逻辑回归对特征尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n预处理：")
    print(f"  ✅ 使用 StandardScaler 进行特征缩放")
    
    # 拟合逻辑回归模型（scikit-learn 1.8+ 最佳实践）
    # 使用 'lbfgs' solver（默认推荐，适用范围广）
    model = LogisticRegression(
        solver='lbfgs',      # 拟牛顿法，适合小到中等数据集
        max_iter=1000,       # 避免 ConvergenceWarning
        C=1.0,               # 正则化强度的倒数（默认值）
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"\n模型参数：")
    print(f"  solver: lbfgs (拟牛顿法)")
    print(f"  max_iter: 1000")
    print(f"  C: 1.0 (正则化强度)")
    
    # 预测
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # 正类概率
    y_pred = model.predict(X_test_scaled)  # 预测类别
    
    # 评估
    accuracy = (y_pred == y_test).mean()
    
    print(f"\n模型评估：")
    print(f"  准确率: {accuracy:.2%}")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'feature_cols': feature_cols
    }


def interpret_coefficients(result: dict) -> None:
    """
    解释逻辑回归系数。
    """
    print("\n" + "=" * 70)
    print("系数解释：对数几率")
    print("=" * 70)
    
    model = result['model']
    feature_cols = result['feature_cols']
    
    print("\n模型系数：")
    print(f"{'特征':<15} {'系数':<12} {'影响方向':<10}")
    print("-" * 45)
    
    for name, coef in zip(feature_cols, model.coef_[0]):
        direction = "正向 ⬆️" if coef > 0 else "负向 ⬇️"
        print(f"{name:<15} {coef:<12.4f} {direction:<10}")
    
    print(f"\n截距 (Intercept): {model.intercept_[0]:.4f}")
    
    print("\n系数解读：")
    print("  逻辑回归系数解释的是对数几率（log odds）的变化：")
    print("  - 系数为正：该特征增加时，成为高价值客户的对数几率增加")
    print("  - 系数为负：该特征增加时，成为高价值客户的对数几率减少")
    print("  - 系数的绝对值越大，影响越大")
    
    # 计算优势比（Odds Ratio）
    print("\n优势比 (Odds Ratio) 解读：")
    print("  OR = e^coef，表示特征每增加 1 单位，优势比的变化倍数")
    print(f"{'特征':<15} {'优势比 (OR)':<15} {'解读':<30}")
    print("-" * 70)
    
    for name, coef in zip(feature_cols, model.coef_[0]):
        or_value = np.exp(coef)
        if or_value > 1:
            interpretation = f"增加 {(or_value-1)*100:.1f}% 的优势"
        else:
            interpretation = f"减少 {(1-or_value)*100:.1f}% 的优势"
        print(f"{name:<15} {or_value:<15.4f} {interpretation:<30}")
    
    print("\n⚠️ 注意：由于特征已标准化，这里的优势比是'每增加 1 个标准差'的变化")


def demonstrate_prediction(result: dict) -> None:
    """
    演示概率预测和类别预测。
    """
    print("\n" + "=" * 70)
    print("预测：概率 vs 类别")
    print("=" * 70)
    
    y_test = result['y_test']
    y_prob = result['y_prob']
    y_pred = result['y_pred']
    
    print("\n预测结果示例（前 15 个测试样本）：")
    print(f"{'样本':<6} {'实际':<8} {'预测概率':<12} {'预测类别':<10} {'结果':<8}")
    print("-" * 60)
    
    for i in range(15):
        status = "✅" if y_pred[i] == y_test.iloc[i] else "❌"
        print(f"{i:<6} {y_test.iloc[i]:<8} {y_prob[i]:<12.4f} {y_pred[i]:<10} {status:<8}")
    
    print("\n概率解释：")
    print(f"  预测概率 = 0.85: 模型认为该客户有 85% 的概率是高价值客户")
    print(f"  预测概率 = 0.12: 模型认为该客户只有 12% 的概率是高价值客户")
    
    print("\n决策阈值（默认 0.5）：")
    print(f"  p >= 0.5 → 预测为正类（高价值客户）")
    print(f"  p < 0.5  → 预测为负类（低价值客户）")
    
    # 展示不同阈值的预测
    thresholds = [0.3, 0.5, 0.7]
    print(f"\n不同阈值下的预测对比（前 5 个样本）：")
    print(f"{'概率':<10} " + " ".join([f"t={t}" for t in thresholds]))
    print("-" * 50)
    for i in range(5):
        preds = [1 if y_prob[i] >= t else 0 for t in thresholds]
        print(f"{y_prob[i]:<10.4f} " + " ".join([f"{p:>5}" for p in preds]))


def statistical_significance(result: dict) -> None:
    """
    使用 statsmodels 进行统计显著性检验。
    """
    print("\n" + "=" * 70)
    print("统计显著性检验（statsmodels）")
    print("=" * 70)
    
    try:
        import statsmodels.api as sm
        
        # 准备数据
        X_train_scaled = result['scaler'].transform(result['X_train'])
        y_train = result['y_train']
        
        # 添加常数项
        X_with_const = sm.add_constant(X_train_scaled)
        
        # 拟合模型
        logit_model = sm.Logit(y_train, X_with_const)
        result_sm = logit_model.fit(disp=0)  # disp=0 抑制迭代输出
        
        print("\n统计摘要（部分）：")
        print(f"{'变量':<15} {'系数':<10} {'标准误':<10} {'z 值':<10} {'p 值':<10} {'显著性':<8}")
        print("-" * 75)
        
        params = result_sm.params
        std_err = result_sm.bse
        z_values = result_sm.tvalues
        p_values = result_sm.pvalues
        
        var_names = ['截距'] + result['feature_cols']
        
        for i, name in enumerate(var_names):
            sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
            print(f"{name:<15} {params[i]:<10.4f} {std_err[i]:<10.4f} {z_values[i]:<10.4f} {p_values[i]:<10.4f} {sig:<8}")
        
        print("\n显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
        
        print("\n模型整体评估：")
        print(f"  Log-Likelihood: {result_sm.llf:.4f}")
        print(f"  AIC: {result_sm.aic:.4f}")
        print(f"  BIC: {result_sm.bic:.4f}")
        
    except ImportError:
        print("\n⚠️  statsmodels 未安装，跳过统计显著性检验")
        print("   安装命令: pip install statsmodels")


def plot_logistic_regression(result: dict) -> None:
    """绘制逻辑回归相关图表"""
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：Sigmoid 函数
    ax1 = axes[0]
    z = np.linspace(-10, 10, 100)
    p = sigmoid(z)
    
    ax1.plot(z, p, 'b-', linewidth=2.5, label='Sigmoid: p = 1/(1+e^(-z))')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='阈值 = 0.5')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(z, 0, p, where=(p >= 0.5), alpha=0.2, color='green', label='预测正类 (p≥0.5)')
    ax1.fill_between(z, 0, p, where=(p < 0.5), alpha=0.2, color='red', label='预测负类 (p<0.5)')
    ax1.set_xlabel('z (线性组合)', fontsize=12)
    ax1.set_ylabel('p (概率)', fontsize=12)
    ax1.set_title('Sigmoid 函数\n将线性输出压缩到 (0,1)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 右图：特征重要性（系数绝对值）
    ax2 = axes[1]
    model = result['model']
    feature_cols = result['feature_cols']
    
    coef_abs = np.abs(model.coef_[0])
    sorted_idx = np.argsort(coef_abs)[::-1]
    
    colors = ['#2E86AB' if model.coef_[0][i] > 0 else '#A23B72' for i in sorted_idx]
    
    ax2.barh([feature_cols[i] for i in sorted_idx], coef_abs[sorted_idx], color=colors)
    ax2.set_xlabel('|系数| (标准化后)', fontsize=12)
    ax2.set_title('特征重要性\n(蓝色=正向影响，紫色=负向影响)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_logistic_regression.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/03_logistic_regression.png")


def main() -> None:
    """主函数"""
    print("逻辑回归完整实践：从 sigmoid 到概率预测\n")
    
    # 可视化 sigmoid
    visualize_sigmoid()
    
    # 生成数据
    df = generate_customer_data(n=800, random_state=42)
    
    # 拟合逻辑回归
    result = fit_logistic_regression(df)
    
    # 解释系数
    interpret_coefficients(result)
    
    # 演示预测
    demonstrate_prediction(result)
    
    # 统计显著性检验
    statistical_significance(result)
    
    # 绘图
    plot_logistic_regression(result)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. Sigmoid 函数: p = 1/(1+e^(-z))，将线性输出映射到 (0,1)")
    print("  2. 逻辑回归输出的是概率，通过阈值（通常 0.5）转换为类别")
    print("  3. 系数解释：对数几率的变化，正负号表示影响方向")
    print("  4. 特征缩放对逻辑回归很重要（solver='lbfgs'）")
    print("  5. 使用 max_iter=1000 避免收敛警告")
    print("\n最佳实践（scikit-learn 1.8+）：")
    print("  - 小数据集: solver='lbfgs'（默认）")
    print("  - 大数据集: solver='sag' 或 'saga'")
    print("  - 需要 L1 正则化: solver='saga' 或 'liblinear'")
    print("  - 多分类问题: 避免使用 'liblinear'")
    print()


if __name__ == "__main__":
    main()
