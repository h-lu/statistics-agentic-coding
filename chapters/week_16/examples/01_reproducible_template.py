"""
示例：可复现分析模板

本例演示如何让分析从"能跑"升级到"可复现、可审计"。

运行方式：python3 chapters/week_16/examples/01_reproducible_template.py
预期输出：stdout 输出依赖版本、数据版本、分析流程、结果元数据
"""
from __future__ import annotations

import datetime
import subprocess
import sys

# 尝试导入分析依赖
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==================== 1. 依赖版本 ====================
def print_dependency_versions():
    """打印所有依赖的版本信息"""
    print("=" * 50)
    print("依赖版本：")
    print("=" * 50)
    print(f"  Python: {sys.version.split()[0]}")

    if SKLEARN_AVAILABLE:
        print(f"  numpy: {np.__version__}")
        print(f"  pandas: {pd.__version__}")

        import sklearn
        print(f"  sklearn: {sklearn.__version__}")
    else:
        print("  ⚠️  sklearn 未安装，部分功能不可用")

    print(f"  运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ==================== 2. 固定随机性 ====================
def setup_reproducibility(random_seed: int = 42):
    """设置所有随机种子，确保结果可复现"""
    print("=" * 50)
    print("可复现性设置：")
    print("=" * 50)
    print(f"  随机种子: {random_seed}")

    if SKLEARN_AVAILABLE:
        np.random.seed(random_seed)
        print("  ✓ numpy 随机种子已固定")

        # 设置 Python hash seed（Python 3.3+）
        os_env = f"PYTHONHASHSEED={random_seed}"
        print(f"  建议: export {os_env}")
    else:
        print("  ⚠️  无法固定随机种子（numpy 未安装）")
    print()


# ==================== 3. 数据版本记录 ====================
def log_data_version(data_source: str, data_version: str):
    """记录数据来源和版本"""
    print("=" * 50)
    print("数据版本：")
    print("=" * 50)
    print(f"  数据来源: {data_source}")
    print(f"  数据版本: {data_version}")

    # 记录文件哈希（如果数据文件存在）
    try:
        import hashlib
        with open(data_source, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        print(f"  文件哈希: {file_hash}")
    except (FileNotFoundError, OSError):
        print(f"  ⚠️  数据文件不存在（仅记录元数据）")
    print()


# ==================== 4. 决策日志 ====================
def log_analysis_decisions():
    """记录分析流程中的所有决策（供审计）"""
    print("=" * 50)
    print("分析决策日志：")
    print("=" * 50)

    decisions = [
        ("数据预处理", [
            "缺失值：删除行（MCAR 假设，基于 Week 03 诊断）",
            "异常值：保留（基于 Week 03 的业务规则检查，无系统偏差）"
        ]),
        ("特征工程", [
            "数值特征：标准化（基于 Week 03 的分布检查，近似正态）",
            "类别特征：One-hot 编码（名义变量，无顺序关系）"
        ]),
        ("模型训练", [
            "模型：线性回归（基于 Week 09 的残差诊断，假设满足）",
            "评估：5 折交叉验证（避免数据泄漏，基于 Week 10 的 Pipeline 实践）"
        ])
    ]

    for category, items in decisions:
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")

    print()


# ==================== 5. 结果输出（包含元数据） ====================
def run_mock_analysis():
    """运行模拟分析，输出带元数据的结果"""
    if not SKLEARN_AVAILABLE:
        print("=" * 50)
        print("⚠️  警告：sklearn 未安装，跳过模拟分析")
        print("=" * 50)
        return

    print("=" * 50)
    print("模拟分析：")
    print("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)

    # 评估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"  训练集 R²: {train_score:.4f}")
    print(f"  测试集 R²: {test_score:.4f}")
    print()

    # 输出元数据
    print("=" * 50)
    print("结果元数据：")
    print("=" * 50)
    print(f"  分析完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  随机种子: 42")
    print(f"  训练样本量: {len(X_train)}")
    print(f"  测试样本量: {len(X_test)}")
    print(f"  结果: R² = {test_score:.4f}")
    print()


# ==================== 6. 生成 requirements.txt ====================
def generate_requirements():
    """生成 requirements.txt 内容（供复制）"""
    print("=" * 50)
    print("建议的 requirements.txt 内容：")
    print("=" * 50)
    requirements = """numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
scikit-learn==1.5.0
statsmodels==0.14.0
matplotlib==3.9.0
seaborn==0.13.0
"""
    print(requirements)


# ==================== 主函数 ====================
def main():
    """执行完整的可复现分析流程"""
    print("\n" + "=" * 50)
    print("StatLab 可复现分析模板")
    print("=" * 50 + "\n")

    # 1. 打印依赖版本
    print_dependency_versions()

    # 2. 设置可复现性
    setup_reproducibility(random_seed=42)

    # 3. 记录数据版本
    log_data_version(
        data_source="data/user_behavior_20250315.csv",
        data_version="2025-03-15"
    )

    # 4. 记录分析决策
    log_analysis_decisions()

    # 5. 运行模拟分析
    run_mock_analysis()

    # 6. 生成 requirements.txt
    generate_requirements()

    print("=" * 50)
    print("✅ 可复现分析模板执行完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
