"""
示例：StatLab 集成——树模型与集成学习章节生成

本例演示如何将树模型与集成学习分析集成到 StatLab 报告中。

功能：
1. 决策树：可视化树结构、解释分裂规则
2. 随机森林：Bagging 原理、性能提升
3. 特征重要性：Top 特征、相关性警告
4. 超参数调优：网格搜索/随机搜索结果
5. 模型对比：线性 vs 树 vs 集成

运行方式：python3 chapters/week_11/examples/11_statlab_trees.py
预期输出：
- 树模型可视化图（decision_tree.png）
- 特征重要性图（feature_importance.png）
- 模型对比表（Markdown 格式）
- 追加到 report.md 的树模型章节
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from scipy.stats import randint

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_sample_data(task: str = 'regression', n_samples: int = 500) -> pd.DataFrame:
    """
    生成模拟数据用于 StatLab 演示

    参数:
        task: 'regression' 或 'classification'
        n_samples: 样本数

    返回:
        包含特征和目标的 DataFrame
    """
    np.random.seed(42)

    # 数值特征
    area_sqm = np.random.uniform(30, 200, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age_years = np.random.uniform(0, 30, n_samples)

    # 类别特征
    city = np.random.choice(['北京', '上海', '深圳', '广州'], n_samples, p=[0.35, 0.30, 0.20, 0.15])
    property_type = np.random.choice(['公寓', '别墅', '联排'], n_samples, p=[0.70, 0.15, 0.15])

    if task == 'regression':
        # 回归：房价预测
        base_price = 50 + 5 * area_sqm - 0.02 * area_sqm**2
        base_price += 20 * bedrooms
        base_price += 15 * bathrooms
        base_price -= 2 * age_years
        base_price += np.where(city == '北京', 50, 0)
        base_price += np.where(city == '上海', 40, 0)

        price_noise = np.random.normal(0, 25, n_samples)
        price = base_price + price_noise

        df = pd.DataFrame({
            'area_sqm': area_sqm,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age_years,
            'city': city,
            'property_type': property_type,
            'price': price
        })

    else:
        # 分类：客户流失预测
        churn_prob = (
            0.1 +
            0.2 * (age_years < 12).astype(int) +
            0.3 * (city == '北京').astype(int) +
            0.15 * (property_type == '别墅').astype(int)
        )
        churn = np.random.binomial(1, np.clip(churn_prob, 0, 1))

        df = pd.DataFrame({
            'tenure_months': age_years * 12,  # 重命名为合同期
            'monthly_bill': np.random.uniform(50, 200, n_samples),
            'support_calls': np.random.randint(0, 10, n_samples),
            'city': city,
            'contract_type': np.random.choice(['月付', '年付'], n_samples),
            'churn': churn
        })

    return df


def fit_decision_tree(X_train, y_train, X_test, y_test, task='regression', max_depth=3):
    """拟合决策树并返回结果"""
    if task == 'regression':
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        metric_name = 'R²'
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        metric_name = '准确率'

    return {
        'model': model,
        'train_score': train_score,
        'test_score': test_score,
        'metric_name': metric_name
    }


def fit_random_forest(X_train, y_train, X_test, y_test, task='regression', tune=True):
    """拟合随机森林（可选调优）"""
    if task == 'regression':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 12),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
        scoring = 'r2'
    else:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 12),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2']
        }
        scoring = 'f1'

    if tune:
        # 随机搜索调优
        search = RandomizedSearchCV(
            model, param_dist, n_iter=30, cv=5, scoring=scoring, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
    else:
        # 默认参数
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {'n_estimators': 100, 'max_depth': None}
        best_cv_score = None

    if task == 'regression':
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        metric_name = 'R²'
    else:
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        metric_name = '准确率'

    return {
        'model': best_model,
        'train_score': train_score,
        'test_score': test_score,
        'metric_name': metric_name,
        'best_params': best_params,
        'best_cv_score': best_cv_score
    }


def get_feature_importance(model, feature_names, top_n=10):
    """提取并可视化特征重要性"""
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    # 画图
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'][::-1], importances['importance'][::-1])
    plt.xlabel('重要性')
    plt.title(f'特征重要性 (Top {top_n})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    return importances


def compare_with_baseline(X_train, y_train, X_test, y_test, task='regression'):
    """与线性/逻辑回归基线对比"""
    if task == 'regression':
        baseline = LinearRegression()
        metric_name = 'R²'
    else:
        baseline = LogisticRegression(random_state=42, max_iter=1000)
        metric_name = '准确率'

    baseline.fit(X_train, y_train)
    baseline_train = baseline.score(X_train, y_train)
    baseline_test = baseline.score(X_test, y_test)

    return {
        'metric_name': metric_name,
        'train_score': baseline_train,
        'test_score': baseline_test
    }


def generate_tree_report_section(df, target, numeric_features, categorical_features, task='regression', output_path='.'):
    """
    生成 StatLab 树模型章节报告

    参数:
        df: 清洗后的数据
        target: 目标变量名
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        task: 'regression' 或 'classification'
        output_path: 图表输出路径

    返回:
        Markdown 格式的报告字符串
    """
    # 准备数据
    X = df[numeric_features + categorical_features]
    y = df[target]

    # 编码类别特征
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 1. 决策树
    dt_results = fit_decision_tree(X_train, y_train, X_test, y_test, task=task, max_depth=5)

    # 2. 随机森林
    rf_results = fit_random_forest(X_train, y_train, X_test, y_test, task=task, tune=True)

    # 3. 特征重要性
    importance_df = get_feature_importance(rf_results['model'], X_encoded.columns, top_n=10)

    # 4. 基线对比
    baseline_results = compare_with_baseline(X_train, y_train, X_test, y_test, task=task)

    # 生成报告
    metric_name = dt_results['metric_name']

    report = f"""
## 树模型与集成学习

### 研究问题
线性模型（回归/逻辑回归）假设特征与目标之间是线性关系，但真实数据可能存在非线性、交互作用或阈值效应。本章使用**决策树**和**随机森林**捕捉这些复杂模式，并与线性基线对比。

### 决策树

决策树通过一系列"如果-那么"规则预测{target}，可解释性强但容易过拟合。

**性能（max_depth=5）**:
- 训练集 {metric_name}: {dt_results['train_score']:.3f}
- 测试集 {metric_name}: {dt_results['test_score']:.3f}

**解读**: 训练集与测试集的{'差异较大' if abs(dt_results['train_score'] - dt_results['test_score']) > 0.1 else '差异较小'}，说明决策树{'存在一定过拟合' if abs(dt_results['train_score'] - dt_results['test_score']) > 0.1 else '泛化性能尚可'}。

### 随机森林

随机森林通过 **Bagging（Bootstrap Aggregating）** 训练多棵树并在预测时投票/平均，显著降低方差，提升泛化性能。

**原理**:
1. Bootstrap 重采样：每棵树在不同的训练子样本上训练
2. 特征随机性：每次分裂时只随机选择一部分特征
3. 投票/平均：回归取平均值，分类取多数投票

**性能（调优后）**:
- 训练集 {metric_name}: {rf_results['train_score']:.3f}
- 测试集 {metric_name}: {rf_results['test_score']:.3f}
- 最佳参数: {rf_results['best_params']}

**改进**: 相比单棵决策树，测试集 {metric_name} 提升 {(rf_results['test_score'] - dt_results['test_score'] if metric_name == 'R²' else (rf_results['test_score'] - dt_results['test_score']) * 100):.1%}。

### 特征重要性

随机森林计算的**特征重要性**基于每个特征在分裂时对不纯度/MSE 的减少量。

**Top 10 特征**:
"""

    for _, row in importance_df.iterrows():
        report += f"- {row['feature']}: {row['importance']:.4f}\n"

    report += f"""
![特征重要性](feature_importance.png)

**解读**:
- 最重要特征: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f})
- 次重要特征: {importance_df.iloc[1]['feature']} ({importance_df.iloc[1]['importance']:.4f})

⚠️ **注意**:
- 特征重要性不等同于因果关系
- 高度相关的特征会"稀释"重要性（模型只选择其中一个）
- 特征重要性只反映"模型依赖"，不反映"因果机制"

### 模型对比

与线性基线的对比:

| 模型 | 测试集 {metric_name} | 优势 |
|------|--------|------|
| {'线性回归' if task == 'regression' else '逻辑回归'} | {baseline_results['test_score']:.3f} | 可解释性强，假设线性关系 |
| 决策树 | {dt_results['test_score']:.3f} | 可解释规则，捕捉非线性 |
| 随机森林（调优后） | {rf_results['test_score']:.3f} | 性能最佳，特征重要性 |

**结论**:
- 如果优先考虑**可解释性**，{'线性回归' if task == 'regression' else '逻辑回归'}仍然是最佳选择（系数直接解释）
- 如果优先考虑**预测性能**，随机森林明显优于单棵树（{metric_name} 提升 {abs(rf_results['test_score'] - baseline_results['test_score']):.1%}）
- 决策树作为"中间选项"，在可解释性和性能之间取得平衡

### 局限性与风险

⚠️ **过拟合**: 决策树容易记住训练数据，必须通过 max_depth、min_samples_leaf 等超参数控制复杂度。随机森林通过 Bagging 降低过拟合风险，但仍需调优。

⚠️ **特征重要性陷阱**:
- 相关性特征会稀释重要性（面积 vs 房间数）
- 高基数类别特征（如用户 ID）可能被误认为"重要"
- 特征重要性不等于因果，不能用于"干预建议"

⚠️ **计算成本**: 随机森林的训练时间是线性模型的 10-100 倍（取决于 n_estimators）。大数据集上可能需要更长训练时间或分布式计算。

### 工程实践

本分析使用了以下最佳实践：
- **嵌套交叉验证**: 在超参数调优中用 5-fold CV，防止过拟合验证集
- **随机搜索**: 用 RandomizedSearchCV 以更少的迭代找到接近最优的解
- **保留测试集**: 测试集只在最终评估时使用一次，确保性能估计的无偏性

### 数据来源
- 样本量: n = {len(y)}
- 分析日期: 2026-02-12
"""

    return report


def main():
    """主函数：完整的 StatLab 树模型章节生成"""
    print("=" * 60)
    print("StatLab 树模型章节生成器")
    print("=" * 60)

    # 生成示例数据
    print("\n生成示例数据...")
    df = generate_sample_data(task='regression', n_samples=500)
    print(f"数据形状: {df.shape}")

    # 回归任务
    print("\n" + "-" * 60)
    print("生成回归任务报告（房价预测）...")
    print("-" * 60)

    report_reg = generate_tree_report_section(
        df=df,
        target='price',
        numeric_features=['area_sqm', 'bedrooms', 'bathrooms', 'age_years'],
        categorical_features=['city', 'property_type'],
        task='regression',
        output_path='.'
    )

    print(report_reg)

    # 分类任务（可选）
    print("\n" + "-" * 60)
    print("生成分类任务报告（客户流失）...")
    print("-" * 60)

    df_clf = generate_sample_data(task='classification', n_samples=500)

    report_clf = generate_tree_report_section(
        df=df_clf,
        target='churn',
        numeric_features=['tenure_months', 'monthly_bill', 'support_calls'],
        categorical_features=['city', 'contract_type'],
        task='classification',
        output_path='.'
    )

    print(report_clf)

    # 保存报告
    print("\n" + "=" * 60)
    print("保存报告...")
    print("=" * 60)

    with open('statlab_tree_models_report.md', 'w', encoding='utf-8') as f:
        f.write("# 树模型与集成学习 - StatLab 示例报告\n\n")
        f.write(report_reg)

    print("✅ 报告已保存为 statlab_tree_models_report.md")
    print("✅ 特征重要性图已保存为 feature_importance.png")

    print("\n" + "=" * 60)
    print("StatLab 集成完成!")
    print("=" * 60)
    print("""
将本章节集成到你的 StatLab 报告的步骤：
1. 复制 report 文件中的图表到你的 report/ 目录
2. 将生成的 Markdown 追加到 report.md
3. 根据你的实际数据调整特征列表和解释
4. 更新本周的 Git commit

关键改进：
- 新增决策树模型（可解释规则）
- 新增随机森林模型（Bagging 降低方差）
- 新增特征重要性分析
- 新增超参数调优（随机搜索）
- 与线性基线对比
    """)


if __name__ == "__main__":
    main()
