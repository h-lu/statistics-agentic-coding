"""
示例：Pipeline 防止数据泄漏——正确使用交叉验证

运行方式：python3 chapters/week_10/examples/05_pipeline_data_leakage.py
预期输出：对比有无 Pipeline 的交叉验证结果，展示数据泄漏的影响
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
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


def create_data_with_missing() -> tuple:
    """创建带缺失值的数据集"""
    # 使用 titanic 数据集
    titanic = sns.load_dataset("titanic")

    # 选择特征并添加一些缺失值
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
    df = titanic[features].copy()

    # 随机添加一些缺失值（模拟真实场景）
    np.random.seed(42)
    for col in ['age', 'fare']:
        mask = np.random.random(len(df)) < 0.1  # 10% 缺失
        df.loc[mask, col] = np.nan

    X = df.drop('survived', axis=1)
    y = df['survived']

    return X, y


def bad_preprocessing_before_split() -> None:
    """
    错误示范：在划分之前做预处理（数据泄漏！）

    问题：
    1. 填充缺失值时用了全部数据的均值（包括测试集）
    2. 标准化时用了全部数据的均值和标准差（包括测试集）
    3. 交叉验证时，验证集的信息泄漏到训练集
    """
    print("=" * 60)
    print("错误示范：在划分之前做预处理（数据泄漏！）")
    print("=" * 60)

    X, y = create_data_with_missing()

    print("\n原始数据缺失值情况：")
    print(X.isnull().sum())

    # 定义数值型和分类型特征
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    # 错误做法：在划分之前填充缺失值
    print("\n错误操作：在 train_test_split 之前填充缺失值")
    X_filled = X.copy()
    for col in numeric_features:
        X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
    for col in categorical_features:
        X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])

    # 编码分类特征
    X_encoded = pd.get_dummies(X_filled, columns=categorical_features, drop_first=True)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n测试集 AUC: {test_auc:.4f}")

    # 交叉验证（错误：数据已经用全部数据预处理过）
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\n问题分析：")
    print("  - 填充缺失值时用了测试集的统计量（数据泄漏）")
    print("  - 交叉验证的每一折中，验证集的信息参与了均值计算")
    print("  - 导致模型表现虚高，上线后崩溃")


def good_pipeline_approach() -> None:
    """
    正确做法：使用 Pipeline 防止数据泄漏

    优势：
    1. 预处理只在训练集上 fit
    2. 预处理应用到测试集时，只用训练集的统计量
    3. 交叉验证时，每一折的预处理都是独立的
    """
    print("\n" + "=" * 60)
    print("正确做法：使用 Pipeline 防止数据泄漏")
    print("=" * 60)

    X, y = create_data_with_missing()

    # 定义数值型和分类型特征
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    # 定义预处理器
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 创建完整 Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练
    pipeline.fit(X_train, y_train)

    # 预测
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n测试集 AUC: {test_auc:.4f}")

    # 交叉验证（正确：每一折的预处理都是独立的）
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nPipeline 的工作原理：")
    print("  - fit() 时：每个步骤依次 fit，后面步骤用前面步骤的输出")
    print("  - predict() 时：前几步只 transform，最后一步 predict")
    print("  - 交叉验证时：每一折的预处理都是独立的（无泄漏）")


def compare_with_without_pipeline() -> None:
    """对比有无 Pipeline 的交叉验证结果"""
    print("\n" + "=" * 60)
    print("对比：有无 Pipeline 的交叉验证结果")
    print("=" * 60)

    # 创建更极端的数据泄漏场景
    np.random.seed(42)
    n_samples = 500
    n_features = 50

    # 生成随机特征（与目标无关）
    X = np.random.randn(n_samples, n_features)
    # 目标变量
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # 添加一些缺失值
    mask = np.random.random(X.shape) < 0.2
    X[mask] = np.nan

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    print("\n数据集信息：")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features} (大部分与目标无关)")
    print(f"  缺失值比例: 约 20%")

    # 错误做法：先填充再交叉验证
    print("\n错误做法：先填充缺失值再交叉验证")
    X_filled = X_df.fillna(X_df.mean())
    model = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores_bad = cross_val_score(model, X_filled, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {cv_scores_bad.mean():.4f} (+/- {cv_scores_bad.std():.4f})")

    # 正确做法：用 Pipeline
    print("\n正确做法：用 Pipeline 进行交叉验证")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv_scores_good = cross_val_score(pipeline, X_df, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {cv_scores_good.mean():.4f} (+/- {cv_scores_good.std():.4f})")

    print("\n结果对比：")
    print(f"  错误做法 AUC: {cv_scores_bad.mean():.4f} (虚高！)")
    print(f"  正确做法 AUC: {cv_scores_good.mean():.4f} (真实水平)")
    print(f"  差异: {cv_scores_bad.mean() - cv_scores_good.mean():.4f}")

    # 可视化对比
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：各折 AUC 对比
    x = np.arange(5)
    width = 0.35

    axes[0].bar(x - width/2, cv_scores_bad, width, label='错误做法（先填充）', color='red', alpha=0.7)
    axes[0].bar(x + width/2, cv_scores_good, width, label='正确做法（Pipeline）', color='green', alpha=0.7)
    axes[0].set_xlabel('交叉验证折数')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('各折 AUC 对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'折 {i+1}' for i in range(5)])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：均值对比
    means = [cv_scores_bad.mean(), cv_scores_good.mean()]
    stds = [cv_scores_bad.std(), cv_scores_good.std()]
    labels = ['错误做法\n（先填充）', '正确做法\n（Pipeline）']
    colors = ['red', 'green']

    bars = axes[1].bar(labels, means, yerr=stds, color=colors, alpha=0.7, capsize=10)
    axes[1].set_ylabel('平均 AUC')
    axes[1].set_title('平均 AUC 对比')
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.4f}±{std:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'pipeline_leakage_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'pipeline_leakage_comparison.png'}")


def demonstrate_data_leakage_checklist() -> None:
    """演示数据泄漏检查清单"""
    print("\n" + "=" * 60)
    print("数据泄漏检查清单")
    print("=" * 60)

    print("""
    数据泄漏常见场景与防御：

    | 场景 | 错误做法 | 正确做法 |
    |------|---------|---------|
    | 预处理 | 在划分之前填充/标准化 | 先划分，再用训练集统计量 |
    | 特征选择 | 用全部数据做特征选择 | 在 Pipeline 内做特征选择 |
    | 交叉验证 | 先预处理再做交叉验证 | Pipeline + 交叉验证 |

    黄金法则：
    ✅ 预处理在划分之后
    ✅ 使用 Pipeline
    ✅ 交叉验证用 Pipeline
    ✅ 检查特征定义（确保特征不包含目标信息）
    ✅ 时间序列注意时序（不要用未来预测过去）

    如何检查数据泄漏？
    1. 用"傻瓜基线"对比：如果模型 AUC 和随机猜测差不多，可能有泄漏
    2. 检查交叉验证得分：如果训练集得分很高但交叉验证得分低，可能是过拟合或泄漏
    3. 审查特征：确认特征没有包含目标信息（如用"退款金额"预测"是否退货"）

    示例代码：傻瓜基线对比
    """)

    # 演示傻瓜基线
    from sklearn.dummy import DummyClassifier

    X, y = create_data_with_missing()
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 你的模型
    your_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    # 傻瓜基线
    dummy = DummyClassifier(strategy='most_frequent')
    dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring='roc_auc')

    print(f"\n你的模型 AUC:   {your_scores.mean():.4f} (+/- {your_scores.std():.4f})")
    print(f"傻瓜基线 AUC:   {dummy_scores.mean():.4f} (+/- {dummy_scores.std():.4f})")
    print(f"提升: {your_scores.mean() - dummy_scores.mean():.4f}")

    if your_scores.mean() - dummy_scores.mean() < 0.05:
        print("\n警告：你的模型 AUC 和傻瓜基线接近，可能有数据泄漏或模型无效！")
    else:
        print("\n模型有效：AUC 显著高于傻瓜基线。")


def main() -> None:
    """主函数"""
    from pathlib import Path

    bad_preprocessing_before_split()
    good_pipeline_approach()
    compare_with_without_pipeline()
    demonstrate_data_leakage_checklist()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    数据泄漏与防御核心要点：

    1. 数据泄漏的定义：
       - 测试集的信息（直接或间接）进入训练集
       - 导致模型在测试集上表现虚高，上线后崩溃

    2. 常见泄漏场景：
       - 预处理在划分之前（用全部数据的统计量）
       - 交叉验证中的泄漏（验证集信息参与训练）
       - 目标泄漏（特征包含目标信息）

    3. 防御方法：
       - 使用 Pipeline 把预处理和模型绑定
       - 交叉验证时用 Pipeline
       - 先划分数据，再做预处理

    4. Pipeline 的工作原理：
       - fit()：每个步骤依次 fit
       - predict()：前几步 transform，最后一步 predict
       - 确保测试集只用训练集的统计量

    老潘的话："没有 Pipeline 的交叉验证，不是评估，是自欺欺人。"
    """)


if __name__ == "__main__":
    main()
