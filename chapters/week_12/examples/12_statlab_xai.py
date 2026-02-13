"""
示例：StatLab 超级线——模型解释与伦理审查章节生成

本例演示如何将模型解释与伦理审查集成到 StatLab 报告中。

功能：
1. SHAP 可解释性分析（全局 + 局部）
2. 公平性评估（不同群体性能对比）
3. 伦理审查清单生成
4. 向非技术读者的解释模板
5. 追加到 report.md

运行方式：python3 chapters/week_12/examples/12_statlab_xai.py
预期输出：
- SHAP 解释图（shap_summary.png, shap_force_sample.png）
- 公平性评估表（Markdown 格式）
- 伦理审查清单（Markdown 格式）
- 追加到 report.md 的模型解释与伦理审查章节

依赖安装：
pip install shap pandas numpy scikit-learn matplotlib
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_statlab_data(task: str = 'classification', n_samples: int = 1000) -> pd.DataFrame:
    """
    生成 StatLab 演示数据

    参数:
        task: 'classification' 或 'regression'
        n_samples: 样本数

    返回:
        包含特征、敏感属性和目标的 DataFrame
    """
    np.random.seed(42)

    if task == 'classification':
        # 信用评分场景
        n = n_samples
        income = np.random.lognormal(10, 0.5, n)
        credit_age = np.random.uniform(0, 20, n)
        debt_ratio = np.random.beta(2, 5, n) * 0.8
        inquiries = np.random.poisson(2, n)
        employment = np.random.exponential(5, n)

        # 敏感属性
        gender = np.random.binomial(1, 0.4, n)  # 0=男性, 1=女性

        # 目标（引入一定偏见）
        logit = (
            -4.0
            + 0.5 * np.log(income / 10000)
            - 0.1 * credit_age
            + 3.0 * debt_ratio
            + 0.3 * inquiries
            - 0.05 * employment
            + 0.2 * gender  # 轻微的性别偏见
        )
        logit += np.random.normal(0, 0.5, n)
        default = np.random.binomial(1, 1 / (1 + np.exp(-logit)))

        df = pd.DataFrame({
            'income': income,
            'credit_history_age': credit_age,
            'debt_to_income': debt_ratio,
            'credit_inquiries': inquiries,
            'employment_length': employment,
            'gender': gender,
            'default': default
        })

    else:
        # 房价场景（回归）
        area = np.random.uniform(50, 200, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        age = np.random.uniform(0, 30, n_samples)

        # 敏感属性（地区）
        district = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])

        # 目标
        base_price = 50 + 5 * area - 0.02 * area**2
        base_price += 20 * bedrooms
        base_price -= 2 * age
        base_price += np.where(district == 0, 50, 0)  # 地区偏见
        base_price += np.where(district == 1, 30, 0)

        price = base_price + np.random.normal(0, 25, n_samples)

        df = pd.DataFrame({
            'area_sqm': area,
            'bedrooms': bedrooms,
            'age_years': age,
            'district': district,
            'price': price
        })

    return df


def compute_fairness_metrics(y_true, y_pred_proba, sensitive_mask, task='classification'):
    """
    计算公平性指标

    参数:
        y_true: 真实标签
        y_pred_proba: 预测概率
        sensitive_mask: 敏感群体掩码（如 gender == 1）
        task: 'classification' 或 'regression'

    返回:
        公平性指标字典
    """
    results = {}

    if task == 'classification':
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 群体 A（敏感群体）
        y_true_a = y_true[sensitive_mask]
        y_pred_a = y_pred[sensitive_mask]
        y_pred_proba_a = y_pred_proba[sensitive_mask]

        # 群体 B（非敏感群体）
        y_true_b = y_true[~sensitive_mask]
        y_pred_b = y_pred[~sensitive_mask]
        y_pred_proba_b = y_pred_proba[~sensitive_mask]

        # AUC
        if len(np.unique(y_true_a)) > 1 and len(np.unique(y_true_b)) > 1:
            results['auc_a'] = roc_auc_score(y_true_a, y_pred_proba_a)
            results['auc_b'] = roc_auc_score(y_true_b, y_pred_proba_b)
            results['auc_diff'] = results['auc_a'] - results['auc_b']
        else:
            results['auc_a'] = None
            results['auc_b'] = None
            results['auc_diff'] = None

        # 通过率
        pass_rate_a = (1 - y_pred_a).mean()  # 预测为不违约的比例
        pass_rate_b = (1 - y_pred_b).mean()
        results['pass_rate_a'] = pass_rate_a
        results['pass_rate_b'] = pass_rate_b

        # 差异影响比
        if pass_rate_b > 0:
            results['disparate_impact'] = pass_rate_a / pass_rate_b
        else:
            results['disparate_impact'] = None

        # 召回率（TPR）
        if y_true_a.sum() > 0 and y_true_b.sum() > 0:
            tp_a = ((y_true_a == 1) & (y_pred_a == 1)).sum()
            tp_b = ((y_true_b == 1) & (y_pred_b == 1)).sum()
            results['tpr_a'] = tp_a / y_true_a.sum()
            results['tpr_b'] = tp_b / y_true_b.sum()
            results['tpr_diff'] = results['tpr_a'] - results['tpr_b']
        else:
            results['tpr_a'] = None
            results['tpr_b'] = None
            results['tpr_diff'] = None

        # 假阳性率（FPR）
        try:
            tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true_a, y_pred_a).ravel()
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_b, y_pred_b).ravel()
            results['fpr_a'] = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0
            results['fpr_b'] = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0
            results['fpr_diff'] = results['fpr_a'] - results['fpr_b']
        except:
            results['fpr_a'] = None
            results['fpr_b'] = None
            results['fpr_diff'] = None

    else:
        # 回归任务：计算不同群体的误差分布
        y_pred = y_pred_proba  # 回归中直接是预测值

        errors_a = np.abs(y_true[sensitive_mask] - y_pred[sensitive_mask])
        errors_b = np.abs(y_true[~sensitive_mask] - y_pred[~sensitive_mask])

        results['mae_a'] = errors_a.mean()
        results['mae_b'] = errors_b.mean()
        results['mae_diff'] = results['mae_a'] - results['mae_b']

    return results


def generate_xai_report_section(df, target, numeric_features, categorical_features,
                                sensitive_features, task='classification', output_path='report'):
    """
    生成模型解释与伦理审查章节

    参数:
        df: 清洗后的数据
        target: 目标变量名
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        sensitive_features: 敏感特征列表（用于公平性评估）
        task: 'classification' 或 'regression'
        output_path: 图表输出路径

    返回:
        Markdown 格式的报告字符串
    """
    # 准备数据
    X = df[numeric_features + categorical_features + sensitive_features]
    y = df[target]

    # 编码类别特征
    X_encoded = pd.get_dummies(X, columns=categorical_features + sensitive_features, drop_first=True)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 训练模型
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # ========== 1. SHAP 可解释性 ==========
    print("\n" + "-" * 60)
    print("生成 SHAP 可解释性分析...")
    print("-" * 60)

    try:
        import shap
    except ImportError:
        print("⚠️  需要安装 shap 库：pip install shap")
        shap_available = False
    else:
        shap_available = True
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if task == 'regression':
            shap_plot_values = shap_values
            base_value = explainer.expected_value
        else:
            if isinstance(shap_values, list):
                shap_plot_values = shap_values[1]
                base_value = explainer.expected_value[1]
            else:
                shap_plot_values = shap_values
                base_value = explainer.expected_value

        # 全局解释
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_plot_values, X_test, show=False)
        plt.title(f'SHAP 全局解释 - {target}')
        plt.savefig(f"{output_path}/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_path}/shap_summary.png")

        # 局部解释（选择一个样本）
        sample_idx = 0
        if task == 'regression':
            sample_shap = shap_plot_values[sample_idx]
        else:
            sample_shap = shap_plot_values[sample_idx]

        plt.figure(figsize=(16, 6))
        shap.force_plot(base_value, sample_shap, X_test.iloc[sample_idx],
                       matplotlib=True, show=False,
                       link='logit' if task == 'classification' else 'identity')
        plt.title(f'SHAP 局部解释（样本 {sample_idx}）')
        plt.savefig(f"{output_path}/shap_force_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_path}/shap_force_sample_{sample_idx}.png")

    # ========== 2. 公平性评估 ==========
    print("\n生成公平性评估...")

    fairness_results = {}

    # 获取测试集中的敏感特征
    test_indices = y_test.index
    for sensitive in sensitive_features:
        # 找到编码后的列
        sensitive_cols = [c for c in X_test.columns if c.startswith(f"{sensitive}_")]

        if len(sensitive_cols) == 0:
            continue

        # 使用第一个编码列（通常是 drop_first=True 后的）
        sensitive_col = sensitive_cols[0]
        group_mask = X_test[sensitive_col] == 1

        if task == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)

        metrics = compute_fairness_metrics(y_test.values, y_pred_proba, group_mask, task)
        fairness_results[sensitive] = metrics

    # ========== 3. 生成报告 ==========
    report = f"""

## 模型解释与伦理审查

### 研究问题

前几章我们建立了预测模型并评估了性能，但没有回答三个关键问题：

1. **可解释性**: 能否向用户解释"为什么是这个预测"？
2. **公平性**: 模型是否存在偏见？是否对某些群体不公平？
3. **局限性**: 模型的边界是什么？哪些场景下会失效？

本章使用 **SHAP（SHapley Additive exPlanations）** 进行可解释性分析，并对不同敏感特征进行公平性评估。

### SHAP 可解释性

#### 全局解释：哪些特征最重要？

SHAP 全局解释（summary plot）展示了所有特征的重要性与影响方向：

![SHAP 全局解释](shap_summary.png)

**解读**:
- **横轴**: SHAP 值（正值表示提高预测值，负值表示降低预测值）
- **颜色**: 特征值（红色=高，蓝色=低）
- **分布**: 特征分布越宽，说明对预测的影响越大

**Top 特征**:
"""

    if shap_available:
        # 计算特征重要性（SHAP 绝对值的均值）
        importances = np.abs(shap_plot_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': importances
        }).sort_values('importance', ascending=False).head(5)

        for _, row in importance_df.iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"

    report += f"""

#### 局部解释：为什么是这个预测？

SHAP 局部解释（force plot）展示了单个预测的"推理路径"：

![SHAP 局部解释](shap_force_sample_0.png)

**如何向客户解释**:
"""

    if shap_available:
        sample = X_test.iloc[0]
        if task == 'classification':
            pred_proba = model.predict_proba(X_test)[0][1]
            explanation = f"您的预测概率为 {pred_proba:.1%}。"
        else:
            pred_value = model.predict(X_test)[0]
            explanation = f"您的预测值为 {pred_value:.2f}。"

        report += f"> {explanation}\n\n"

    report += """这是基于模型的预测模式，不代表因果关系。特征值仅供参考，最终决定需要人工审核。

### 公平性评估

我们检查了模型在不同敏感特征（{sensitive_features}）上的表现差异。

#### 性能差异表

"""

    # 添加公平性表格
    if task == 'classification':
        report += "| 敏感特征 | 群体 A AUC | 群体 B AUC | AUC 差异 | 通过率 A | 通过率 B | 差异影响比 |\n"
        report += "|---------|-----------|-----------|----------|----------|----------|------------|\n"

        for sensitive, metrics in fairness_results.items():
            auc_a = f"{metrics['auc_a']:.3f}" if metrics['auc_a'] else "N/A"
            auc_b = f"{metrics['auc_b']:.3f}" if metrics['auc_b'] else "N/A"
            auc_diff = f"{metrics['auc_diff']:.3f}" if metrics['auc_diff'] else "N/A"
            pass_a = f"{metrics['pass_rate_a']:.3f}"
            pass_b = f"{metrics['pass_rate_b']:.3f}"
            di = f"{metrics['disparate_impact']:.3f}" if metrics['disparate_impact'] else "N/A"

            report += f"| {sensitive} | {auc_a} | {auc_b} | {auc_diff} | {pass_a} | {pass_b} | {di} |\n"

    else:
        report += "| 敏感特征 | 群体 A MAE | 群体 B MAE | MAE 差异 |\n"
        report += "|---------|-----------|-----------|----------|\n"

        for sensitive, metrics in fairness_results.items():
            mae_a = f"{metrics['mae_a']:.3f}"
            mae_b = f"{metrics['mae_b']:.3f}"
            mae_diff = f"{metrics['mae_diff']:.3f}"

            report += f"| {sensitive} | {mae_a} | {mae_b} | {mae_diff} |\n"

    report += "\n**解读**:\n"

    # 添加警告
    for sensitive, metrics in fairness_results.items():
        if task == 'classification':
            if metrics['auc_diff'] and abs(metrics['auc_diff']) > 0.05:
                report += f"- ⚠️ **{sensitive}**: AUC 差异为 {metrics['auc_diff']:.3f}，说明模型在不同群体上的性能存在显著差异。\n"
            if metrics['disparate_impact'] and metrics['disparate_impact'] < 0.8:
                report += f"- ⚠️ **{sensitive}**: 差异影响比 = {metrics['disparate_impact']:.3f} < 0.8，不符合 80% 规则，可能存在法律风险。\n"
        else:
            if abs(metrics['mae_diff']) > 0.05 * metrics['mae_b']:
                report += f"- ⚠️ **{sensitive}**: MAE 差异为 {metrics['mae_diff']:.3f}，说明模型在不同群体上的预测误差存在差异。\n"

    report += """

### 偏见来源分析

模型偏见有三个主要来源：

1. **数据偏见（Historical Bias）**
   - 训练数据是否存在历史歧视？
   - 某些群体的样本量是否不足？

2. **算法偏见（Algorithmic Bias）**
   - 模型是否放大数据中的模式？
   - 是否对某些群体过拟合？

3. **代理变量（Proxy Variables）**
   - 是否存在敏感属性的代理变量？
   - 例如：邮政编码代理种族、职位代理性别

**常见偏见场景**:
- 信用评分：历史数据中，少数族裔被拒比例更高 → 模型学会"偏见"
- 招聘：历史数据中，男性被录用比例更高 → 模型学会"歧视"
- 房价：某些地区房价被低估 → 模型继续"低估"

### 结论边界：模型能回答什么，不能回答什么

**模型能回答的**:
- 基于历史数据预测 {target}
- 哪些特征与 {target} 相关

**模型不能回答的**:
- **因果关系**: 提高某特征是否会改变结果？（模型只预测相关性）
- **特殊场景**: 训练数据中未见过的场景（如经济危机）
- **伦理判断**: 是否应该拒绝某个客户（这是业务决策）

**模型在以下场景下可能失效**:
- 数据分布发生变化（如经济衰退）
- 训练数据中未见过的新模式
- 与训练数据差异很大的样本

### 伦理审查清单

| 风险类别 | 检查项 | 状态 |
|---------|--------|------|
| **数据偏见** | 训练数据是否存在历史歧视？ | [ ] 需检查 |
| **算法偏见** | 模型是否放大数据中的模式？ | [ ] 需检查 |
| **代理变量** | 是否存在敏感属性的代理变量？ | [ ] 需检查 |
| **公平性指标** | 差异影响比是否 ≥ 0.8？ | [ ] 需检查 |
| **隐私风险** | 数据发布是否使用差分隐私？ | [ ] 需检查 |
| **可复现性** | 随机种子、代码、数据来源是否记录？ | [ ] ✅ |
| **结论边界** | 模型的局限性是否明确？ | [ ] ✅ |

### 向非技术读者解释

**面向客户（被预测者）**:
> "您的{'通过/拒绝' if task == 'classification' else '预测值'}主要基于[1-2 个最显著特征]。这是模型的预测，不代表因果关系。如果您认为结果有误，可以申请人工审核。"

**面向产品经理（决策者）**:
> "模型在测试集上的表现良好，但存在一定偏见：不同群体的性能差异约为[X%]。我们建议：1) 收集更多平衡数据；2) 定期审计模型公平性；3) 保留人工审核机制。"

**面向合规部门（风险控制）**:
> "我们已进行公平性审计：差异影响比 = [X]，{'符合' if all(m.get('disparate_impact', 1) >= 0.8 for m in fairness_results.values()) else '不符合'} 80% 规则。模型的可复现性已保证（随机种子=42）。敏感特征已{'包含' if sensitive_features else '不包含'}在训练中。"

### 数据来源

- 样本量: n = {len(y)}
- 训练集: {len(y_train)} 样本
- 测试集: {len(y_test)} 样本
- 分析日期: 2026-02-12
- 随机种子: 42
"""

    return report


def main():
    """主函数：完整的 StatLab 模型解释与伦理审查章节生成"""
    print("=" * 60)
    print("StatLab 模型解释与伦理审查章节生成器")
    print("=" * 60)

    # ========== 分类任务示例 ==========
    print("\n生成分类任务示例（信用评分）...")
    print("-" * 60)

    df_clf = generate_statlab_data(task='classification', n_samples=1000)
    print(f"数据形状: {df_clf.shape}")

    report_clf = generate_xai_report_section(
        df=df_clf,
        target='default',
        numeric_features=['income', 'credit_history_age', 'debt_to_income',
                        'credit_inquiries', 'employment_length'],
        categorical_features=[],
        sensitive_features=['gender'],
        task='classification',
        output_path='report'
    )

    print(report_clf)

    # ========== 回归任务示例 ==========
    print("\n" + "=" * 60)
    print("生成回归任务示例（房价预测）...")
    print("-" * 60)

    df_reg = generate_statlab_data(task='regression', n_samples=1000)
    print(f"数据形状: {df_reg.shape}")

    report_reg = generate_xai_report_section(
        df=df_reg,
        target='price',
        numeric_features=['area_sqm', 'bedrooms', 'age_years'],
        categorical_features=[],
        sensitive_features=['district'],
        task='regression',
        output_path='report'
    )

    print(report_reg)

    # ========== 保存报告 ==========
    print("\n" + "=" * 60)
    print("保存报告...")
    print("=" * 60)

    # 合并报告
    full_report = f"""# 模型解释与伦理审查 - StatLab 示例报告

## 分类任务：信用评分模型

{report_clf}

---

## 回归任务：房价预测模型

{report_reg}
"""

    with open('statlab_xai_ethics_report.md', 'w', encoding='utf-8') as f:
        f.write(full_report)

    print("✅ 报告已保存为 statlab_xai_ethics_report.md")
    print("\n" + "=" * 60)
    print("StatLab 集成完成!")
    print("=" * 60)
    print("""
将本章节集成到你的 StatLab 报告的步骤：
1. 复制 report/ 目录中的图表到你的项目
2. 将生成的 Markdown 追加到 report.md
3. 根据你的实际数据调整特征列表和解释
4. 更新本周的 Git commit

关键改进：
- 新增 SHAP 可解释性分析（全局 + 局部）
- 新增公平性评估（不同群体性能对比）
- 新增伦理审查清单（系统化风险评估）
- 新增向非技术读者的解释模板
- 明确模型边界与局限性
    """)


if __name__ == "__main__":
    main()
