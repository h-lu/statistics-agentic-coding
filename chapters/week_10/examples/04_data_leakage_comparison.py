"""
示例：数据泄漏——评估中最常见的陷阱

本例演示：
1. 错误做法：全局 StandardScaler（数据泄漏）
2. 正确做法：Pipeline 内 StandardScaler
3. 对比两种做法的交叉验证结果

运行方式：python3 chapters/week_10/examples/04_data_leakage_comparison.py
预期输出：
- 对比两种做法的交叉验证分数
- 控制台输出详细解释
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification

# 设置随机种子
np.random.seed(42)


def generate_data_with_shift(n_samples: int = 1000) -> tuple:
    """
    生成有分布偏移的数据（模拟训练集和测试集分布不同）

    这种场景下，数据泄漏的影响会更明显
    """
    # 训练集：均值较小
    X_train = np.random.randn(int(n_samples * 0.7), 5) * 2
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    # 测试集：均值较大（模拟分布偏移）
    X_test = np.random.randn(int(n_samples * 0.3), 5) * 2 + 1
    y_test = (X_test[:, 0] + X_test[:, 1] > 1).astype(int)

    # 合并（用于交叉验证）
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    return X, y


def wrong_approach_global_scaling(X, y) -> float:
    """
    错误做法：全局 StandardScaler（数据泄漏）

    问题：
    1. 在整个数据集上计算均值和方差
    2. 交叉验证的每个折都能"看到"其他折的统计量
    3. 测试集信息泄漏到训练过程
    """
    print("\n" + "=" * 60)
    print("❌ 错误做法：全局 StandardScaler")
    print("=" * 60)

    # 步骤1：在整个数据集上 fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n步骤1：在整个数据集上 fit StandardScaler")
    print(f"  计算的均值: {scaler.mean_}")
    print(f"  计算的方差: {scaler.var_}")

    # 步骤2：交叉验证
    model = LogisticRegression(random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    print(f"\n步骤2：对标准化后的数据做交叉验证")
    print(f"  CV 准确率: {scores}")
    print(f"  平均: {scores.mean():.3f} ± {scores.std():.3f}")

    # 解释问题
    print(f"\n⚠️  问题所在：")
    print(f"  1. 全局 fit 时，测试集的均值/方差信息被'教给'了 scaler")
    print(f"  2. 交叉验证的每个折在训练时已经'知道'其他折的统计量")
    print(f"  3. 评估结果虚高，但模型在生产环境会表现很差")

    return scores.mean()


def correct_approach_pipeline(X, y) -> float:
    """
    正确做法：Pipeline 内 StandardScaler

    优势：
    1. 每个折内独立拟合 scaler
    2. 测试集信息不会泄漏
    3. 评估结果更真实
    """
    print("\n" + "=" * 60)
    print("✅ 正确做法：Pipeline 内 StandardScaler")
    print("=" * 60)

    # 构建 Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print(f"\n步骤1：构建 Pipeline")
    print(f"  Pipeline(steps=[('scaler', StandardScaler()),")
    print(f"                   ('log_reg', LogisticRegression())])")

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    print(f"\n步骤2：对 Pipeline 做交叉验证")
    print(f"  每个 CV 折内：")
    print(f"    1. 在训练集上 fit scaler")
    print(f"    2. transform 训练集和测试集")
    print(f"    3. 在训练集上 fit 模型")
    print(f"    4. 在测试集上评估")

    print(f"\n  CV 准确率: {scores}")
    print(f"  平均: {scores.mean():.3f} ± {scores.std():.3f}")

    # 解释优势
    print(f"\n✅ 优势：")
    print(f"  1. 每个 CV 折独立计算均值/方差（无泄漏）")
    print(f"  2. 测试集信息永远不会用于 fit scaler")
    print(f"  3. 评估结果更接近真实性能")

    return scores.mean()


def demonstrate_leakage_mechanism() -> None:
    """
    演示数据泄漏的机制
    """
    print("\n" + "=" * 60)
    print("数据泄漏机制演示")
    print("=" * 60)

    # 模拟数据
    print("\n假设有 10 个样本，分为 2 折（每折 5 个样本）：")
    print()

    data = pd.DataFrame({
        '样本': list(range(10)),
        '特征值': [1, 2, 3, 4, 5, 10, 12, 14, 16, 18],
        '折': ['折1'] * 5 + ['折2'] * 5
    })

    print(data.to_string(index=False))

    # 全局标准化
    print(f"\n{'='*60}")
    print("❌ 全局 StandardScaler：")
    print("=" * 60)
    global_mean = data['特征值'].mean()
    print(f"  全局均值 = {global_mean:.1f}")

    print(f"\n  折1 训练时：")
    print(f"    使用均值 {global_mean:.1f} 标准化（包含了折2的信息！）")

    print(f"\n  折2 训练时：")
    print(f"    使用均值 {global_mean:.1f} 标准化（包含了折1的信息！）")

    print(f"\n  💀 结果：每个折在训练时都'看到'了其他折的信息")

    # Pipeline 标准化
    print(f"\n{'='*60}")
    print("✅ Pipeline 内 StandardScaler：")
    print("=" * 60)

    fold1_mean = data[data['折'] == '折1']['特征值'].mean()
    fold2_mean = data[data['折'] == '折2']['特征值'].mean()

    print(f"  折1 均值 = {fold1_mean:.1f}")
    print(f"  折2 均值 = {fold2_mean:.1f}")

    print(f"\n  折1 训练时：")
    print(f"    只在折1上 fit，使用均值 {fold1_mean:.1f}")

    print(f"\n  折2 训练时：")
    print(f"    只在折2上 fit，使用均值 {fold2_mean:.1f}")

    print(f"\n  ✅ 结果：每个折独立计算统计量，无信息泄漏")


def compare_results(wrong_score: float, correct_score: float) -> None:
    """对比两种结果"""
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    print(f"\n错误做法（全局标准化）：CV 准确率 = {wrong_score:.3f}")
    print(f"正确做法（Pipeline）：     CV 准确率 = {correct_score:.3f}")

    leakage = wrong_score - correct_score

    print(f"\n虚高幅度：{leakage:.1%}")
    print(f"\n💡 结论：")
    print(f"  数据泄漏导致评估结果虚高 {leakage:.1%}")
    print(f"  如果根据错误做法的结果上线，实际性能会大打折扣")

    if leakage > 0.05:
        print(f"\n⚠️  警告：泄漏幅度超过 5%，这是严重的工程问题！")
    elif leakage > 0.02:
        print(f"\n⚠️  注意：泄漏幅度在 2%-5%，需要修复")
    else:
        print(f"\n✅ 泄漏幅度较小（< 2%），但最佳实践仍是使用 Pipeline")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例4: 数据泄漏——评估中最常见的陷阱")
    print("=" * 60)

    # 1. 生成数据
    X, y = generate_data_with_shift(n_samples=1000)

    print(f"\n📊 数据概览:")
    print(f"  总样本数: {len(X)}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  正类比例: {y.mean():.1%}")

    # 2. 演示泄漏机制
    demonstrate_leakage_mechanism()

    # 3. 错误做法
    wrong_score = wrong_approach_global_scaling(X, y)

    # 4. 正确做法
    correct_score = correct_approach_pipeline(X, y)

    # 5. 对比结果
    compare_results(wrong_score, correct_score)

    # 6. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
数据泄漏是机器学习中最常见、最隐蔽的错误：

问题根源：
1. 在 train-test split 之前做预处理
2. 在交叉验证之前做全局预处理
3. 特征选择使用了测试集信息
4. 数据增强使用了测试集统计量

后果：
1. 评估结果虚高（误导决策）
2. 生产环境性能大幅下降
3. 论文/报告结论不可复现

最佳实践：
1. ✅ 用 Pipeline + ColumnTransformer
2. ✅ 每个折内独立 fit 预处理
3. ✅ 测试集只用于 transform，不用于 fit
4. ✅ 数据来源、版本、预处理步骤都记录下来

记住：
"考试前偷看答案，考得再好也没用。"
    """)

    print("\n" + "=" * 60)
    print("✅ 示例4完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
