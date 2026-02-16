"""
示例：ROC 曲线与 AUC——评估模型在所有阈值下的表现。

本例演示：
1. ROC 曲线的绘制：展示不同阈值下的 TPR 和 FPR
2. AUC 的计算与解释：曲线下面积的含义
3. 不同阈值下的表现对比
4. 最优阈值的选择方法
5. Bootstrap 估计 AUC 的置信区间

运行方式：python3 chapters/week_10/examples/05_roc_auc.py
预期输出：
  - stdout 输出 ROC 分析和 AUC 值
  - 展示 ROC 曲线可视化
  - 保存图表到 images/05_roc_auc.png

核心概念：
  - TPR (True Positive Rate) = TP / (TP + FN) = 查全率
  - FPR (False Positive Rate) = FP / (FP + TN)
  - AUC = 0.5: 随机猜测，AUC = 1.0: 完美分类
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.utils import resample
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


def generate_ecommerce_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟电商客户数据。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征
    """
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        '注册月数': np.random.randint(1, 48, n),
        '月均浏览次数': np.random.poisson(35, n),
        '购物车添加次数': np.random.poisson(6, n),
        '历史消费金额': np.random.exponential(scale=80, size=n)
    })
    
    score = (
        0.08 * data['注册月数'] +
        0.04 * data['月均浏览次数'] +
        0.15 * data['购物车添加次数'] +
        0.008 * data['历史消费金额'] -
        4 +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > np.percentile(score, 85)).astype(int)
    
    return data


def fit_model_and_get_probabilities(df: pd.DataFrame) -> dict:
    """
    拟合模型并返回预测概率。
    
    返回:
        dict 包含真实标签和预测概率
    """
    X = df[['注册月数', '月均浏览次数', '购物车添加次数', '历史消费金额']]
    y = df['是否高价值']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'y_test': y_test,
        'y_prob': y_prob
    }


def explain_roc_concepts() -> None:
    """解释 ROC 曲线的核心概念。"""
    print("=" * 70)
    print("ROC 曲线核心概念")
    print("=" * 70)
    
    print("\nROC 曲线是什么？")
    print("  ROC (Receiver Operating Characteristic) 曲线展示了模型在")
    print("  所有可能阈值下的权衡关系。")
    
    print("\n两个核心指标：")
    print("  - TPR (True Positive Rate，真正率) = TP / (TP + FN)")
    print("    又称：查全率 (Recall) 或敏感度 (Sensitivity)")
    print("    含义：真正类中被正确预测的比例")
    print()
    print("  - FPR (False Positive Rate，假正率) = FP / (FP + TN)")
    print("    又称：1 - 特异度 (Specificity)")
    print("    含义：负类中被错误预测为正类的比例")
    
    print("\nROC 曲线的解读：")
    print("  - 横轴：FPR（假正率）")
    print("  - 纵轴：TPR（真正率）")
    print("  - 曲线上的每个点对应一个阈值")
    print("  - 对角线（从 (0,0) 到 (1,1)）：随机分类器")
    print("  - 曲线越靠近左上角，模型越好")


def calculate_and_plot_roc(y_test: pd.Series, y_prob: np.ndarray) -> tuple:
    """
    计算 ROC 曲线并返回 FPR、TPR 和 AUC。
    
    返回:
        tuple (fpr, tpr, thresholds, auc_score)
    """
    print("\n" + "=" * 70)
    print("ROC 曲线计算")
    print("=" * 70)
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nAUC (Area Under Curve): {roc_auc:.4f}")
    
    print("\nAUC 解读：")
    print("  ┌─────────────┬─────────────────────────────────────┐")
    print("  │   AUC 范围  │              模型表现               │")
    print("  ├─────────────┼─────────────────────────────────────┤")
    print("  │    0.5      │ 随机猜测（无区分能力）              │")
    print("  │  0.5-0.7    │ 较差                                │")
    print("  │  0.7-0.85   │ 尚可                                │")
    print("  │  0.85-0.95  │ 良好                                │")
    print("  │  0.95-1.0   │ 优秀（但需检查是否过拟合）          │")
    print("  └─────────────┴─────────────────────────────────────┘")
    
    if roc_auc < 0.7:
        print(f"\n⚠️  AUC = {roc_auc:.4f} < 0.7，模型区分能力较弱")
    elif roc_auc > 0.95:
        print(f"\n⚠️  AUC = {roc_auc:.4f} > 0.95，需检查是否过拟合")
    else:
        print(f"\n✅ AUC = {roc_auc:.4f}，模型区分能力尚可")
    
    print("\n关键阈值点的表现：")
    print(f"{'阈值':<10} {'TPR':<10} {'FPR':<10} {'解读':<30}")
    print("-" * 65)
    
    # 找到一些关键阈值点
    key_thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    for thresh in key_thresholds:
        # 找到最接近的阈值索引
        idx = np.argmin(np.abs(thresholds - thresh))
        if idx < len(tpr):
            print(f"{thresholds[idx]:<10.4f} {tpr[idx]:<10.4f} {fpr[idx]:<10.4f} ", end="")
            if tpr[idx] > 0.8 and fpr[idx] < 0.2:
                print("较好（高TPR，低FPR）")
            elif tpr[idx] > 0.8:
                print("高查全率但误报较多")
            elif fpr[idx] < 0.2:
                print("低误报但漏报较多")
            else:
                print("一般")
    
    return fpr, tpr, thresholds, roc_auc


def find_optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> None:
    """
    寻找最优阈值。
    """
    print("\n" + "=" * 70)
    print("最优阈值选择")
    print("=" * 70)
    
    # 方法 1：最大化 TPR - FPR（最接近左上角）
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n方法 1：最大化 (TPR - FPR)")
    print(f"  最优阈值: {optimal_threshold:.4f}")
    print(f"  该阈值下: TPR = {tpr[optimal_idx]:.2%}, FPR = {fpr[optimal_idx]:.2%}")
    print(f"  适用场景: 平衡精确率和查全率")
    
    # 方法 2：高查全率阈值（TPR >= 90%）
    high_recall_indices = np.where(tpr >= 0.9)[0]
    if len(high_recall_indices) > 0:
        high_recall_idx = high_recall_indices[0]
        high_recall_threshold = thresholds[high_recall_idx]
        print(f"\n方法 2：高查全率 (TPR >= 90%)")
        print(f"  建议阈值: {high_recall_threshold:.4f}")
        print(f"  该阈值下: TPR = {tpr[high_recall_idx]:.2%}, FPR = {fpr[high_recall_idx]:.2%}")
        print(f"  适用场景: 漏报代价高（如疾病筛查、流失预警）")
    
    # 方法 3：低 FPR 阈值（FPR <= 10%）
    low_fpr_indices = np.where(fpr <= 0.1)[0]
    if len(low_fpr_indices) > 0:
        low_fpr_idx = low_fpr_indices[-1]  # 取最大的 FPR <= 0.1 的索引
        low_fpr_threshold = thresholds[low_fpr_idx]
        print(f"\n方法 3：低假正率 (FPR <= 10%)")
        print(f"  建议阈值: {low_fpr_threshold:.4f}")
        print(f"  该阈值下: TPR = {tpr[low_fpr_idx]:.2%}, FPR = {fpr[low_fpr_idx]:.2%}")
        print(f"  适用场景: 误报代价高（如垃圾邮件过滤、广告推荐）")


def bootstrap_auc_ci(y_test: pd.Series, y_prob: np.ndarray, n_bootstrap: int = 1000) -> None:
    """
    使用 Bootstrap 估计 AUC 的置信区间。
    """
    print("\n" + "=" * 70)
    print("Bootstrap 估计 AUC 置信区间")
    print("=" * 70)
    
    print(f"\n进行 {n_bootstrap} 次 Bootstrap 重采样...")
    
    auc_scores = []
    y_test_array = np.array(y_test)
    
    for i in range(n_bootstrap):
        # 重采样
        indices = resample(range(len(y_test)), random_state=i)
        y_test_boot = y_test_array[indices]
        y_prob_boot = y_prob[indices]
        
        # 计算 AUC（确保两类都存在）
        if len(np.unique(y_test_boot)) > 1:
            auc_boot = roc_auc_score(y_test_boot, y_prob_boot)
            auc_scores.append(auc_boot)
    
    # 计算 95% 置信区间
    auc_ci_lower = np.percentile(auc_scores, 2.5)
    auc_ci_upper = np.percentile(auc_scores, 97.5)
    auc_mean = np.mean(auc_scores)
    
    print(f"\nBootstrap 结果：")
    print(f"  AUC 均值: {auc_mean:.4f}")
    print(f"  95% 置信区间: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
    
    ci_width = auc_ci_upper - auc_ci_lower
    print(f"\n置信区间宽度: {ci_width:.4f}")
    if ci_width < 0.1:
        print("  ✅ 置信区间较窄，AUC 估计稳定")
    elif ci_width < 0.2:
        print("  ⚠️  置信区间中等，AUC 估计尚可接受")
    else:
        print("  ⚠️  置信区间较宽，可能需要更多数据")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, thresholds: np.ndarray) -> None:
    """绘制 ROC 曲线和相关图表"""
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：ROC 曲线
    ax1 = axes[0]
    ax1.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机分类器 (AUC = 0.5)')
    
    # 标记一些关键阈值点
    key_indices = [0, len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4, len(thresholds)-1]
    for idx in key_indices:
        if idx < len(thresholds):
            ax1.plot(fpr[idx], tpr[idx], 'ro', markersize=6)
            if idx == len(thresholds)//2:  # 在中间点附近标注
                ax1.annotate(f't={thresholds[idx]:.2f}', 
                           xy=(fpr[idx], tpr[idx]),
                           xytext=(fpr[idx]+0.1, tpr[idx]-0.1),
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # 标记最优阈值点（最接近左上角）
    optimal_idx = np.argmax(tpr - fpr)
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'g*', markersize=15, label='最优阈值')
    
    ax1.set_xlabel('假正率 (FPR)', fontsize=12)
    ax1.set_ylabel('真正率 (TPR)', fontsize=12)
    ax1.set_title('ROC 曲线\nReceiver Operating Characteristic', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(fpr, 0, tpr, alpha=0.2, color='blue', label=f'AUC = {roc_auc:.3f}')
    
    # 右图：阈值 vs TPR/FPR
    ax2 = axes[1]
    # 只取合理的阈值范围进行绘图
    valid_idx = (thresholds >= 0) & (thresholds <= 1)
    ax2.plot(thresholds[valid_idx], tpr[valid_idx], 'g-', linewidth=2, label='TPR (查全率)')
    ax2.plot(thresholds[valid_idx], fpr[valid_idx], 'r-', linewidth=2, label='FPR (假正率)')
    
    # 标记默认阈值 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='默认阈值 = 0.5')
    ax2.plot(0.5, tpr[idx_05], 'go', markersize=8)
    ax2.plot(0.5, fpr[idx_05], 'ro', markersize=8)
    
    ax2.set_xlabel('阈值', fontsize=12)
    ax2.set_ylabel('比率', fontsize=12)
    ax2.set_title('阈值 vs TPR/FPR\nThreshold Analysis', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '05_roc_auc.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/05_roc_auc.png")


def main() -> None:
    """主函数"""
    print("ROC 曲线与 AUC：评估模型在所有阈值下的表现\n")
    
    # 生成数据
    df = generate_ecommerce_data(n=1000, random_state=42)
    
    # 拟合模型
    result = fit_model_and_get_probabilities(df)
    y_test = result['y_test']
    y_prob = result['y_prob']
    
    # 解释 ROC 概念
    explain_roc_concepts()
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds, roc_auc = calculate_and_plot_roc(y_test, y_prob)
    
    # 寻找最优阈值
    find_optimal_threshold(fpr, tpr, thresholds)
    
    # Bootstrap 置信区间
    bootstrap_auc_ci(y_test, y_prob, n_bootstrap=1000)
    
    # 绘图
    plot_roc_curve(fpr, tpr, roc_auc, thresholds)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\nROC 曲线的价值：")
    print("  1. 评估模型在所有可能阈值下的表现")
    print("  2. 帮助选择适合业务目标的阈值")
    print("  3. AUC 不受类别比例影响，是稳定的评估指标")
    print("\nAUC 解读：")
    print("  - AUC = 0.5: 随机猜测，模型无区分能力")
    print("  - AUC < 0.7: 模型需要改进")
    print("  - AUC 0.7-0.85: 尚可接受")
    print("  - AUC 0.85-0.95: 表现良好")
    print("  - AUC > 0.95: 需检查是否过拟合")
    print("\n阈值选择：")
    print("  - 漏报代价高 → 选择高 TPR 的阈值")
    print("  - 误报代价高 → 选择低 FPR 的阈值")
    print("  - 平衡两者 → 选择 TPR - FPR 最大的阈值")
    print()


if __name__ == "__main__":
    main()
