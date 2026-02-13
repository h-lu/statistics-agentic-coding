# Week 10 测试用例矩阵

## 测试文件结构

```
tests/
├── __init__.py                  # 模块初始化
├── conftest.py                  # 共享 fixtures
├── test_smoke.py               # 烟雾测试（基础功能）
├── test_classification.py      # 主要测试（核心概念）
└── test_edge_cases.py          # 边界测试（极端场景）
```

## 测试覆盖统计

| 文件 | 测试类数 | 测试函数数 |
|------|----------|-----------|
| test_smoke.py | 2 | 13 |
| test_classification.py | 7 | 38 |
| test_edge_cases.py | 7 | 32 |
| **总计** | **16** | **83** |

## 核心概念覆盖

### 1. 逻辑回归 (Logistic Regression)

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 4 | sklearn/statsmodels 拟合、Sigmoid 输出范围、优势比解释、决策边界 |
| 边界 (Edge Cases) | 3 | 空数据、单观测、完全可分 |
| **小计** | **7** | |

**核心测试**：
- `test_logistic_regression_sklearn_fits_correctly`: sklearn 逻辑回归拟合
- `test_sigmoid_output_range`: Sigmoid 输出在 [0, 1] 范围
- `test_sigmoid_function_properties`: sigmoid(0) = 0.5, 对称性
- `test_odds_ratio_interpretation`: 优势比 = exp(系数)
- `test_decision_boundary_threshold`: predict() 等价于 (proba >= 0.5)
- `test_logistic_regression_statsmodels`: statsmodels 详细输出

### 2. 混淆矩阵 (Confusion Matrix)

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 6 | 混淆矩阵计算、精确率/召回率/F1 公式、准确率悖论 |
| 边界 (Edge Cases) | 5 | 完美预测、全错、分母为 0、F1=0 |
| **小计** | **11** | |

**核心测试**：
- `test_confusion_matrix_calculation`: TP/TN/FP/FN 正确计算
- `test_precision_calculation`: P = TP / (TP + FP)
- `test_recall_calculation`: R = TP / (TP + FN)
- `test_f1_harmonic_mean`: F1 = 2PR / (P + R)
- `test_imbalanced_data_accuracy_paradox`: 85% 不流失时预测"全不流失"准确率也是 85%
- `test_perfect_predictions_confusion_matrix`: 完美预测 FN=0, FP=0

### 3. ROC-AUC

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 6 | ROC 曲线结构、AUC 范围、完美/随机分类器、排序解释 |
| 边界 (Edge Cases) | 6 | AUC=1/0/0.5、全相同标签、阈值无关性、单调性 |
| **小计** | **12** | |

**核心测试**：
- `test_roc_curve_structure`: 从 (0,0) 到 (1,1)，FPR/TPR 在 [0,1]
- `test_auc_range`: AUC ∈ [0, 1]
- `test_auc_perfect_classifier`: 完美分类器 AUC = 1.0
- `test_auc_random_classifier`: 随机分类器 AUC ≈ 0.5
- `test_auc_ranking_interpretation`: AUC = P(正类概率 > 负类概率)
- `test_auc_threshold_independence`: AUC 不依赖分类阈值

### 4. 交叉验证 (Cross-Validation)

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 5 | K-fold/StratifiedKFold 创建、分数统计、多指标评估 |
| 边界 (Edge Cases) | 4 | 折数>样本数、单折、单类样本、shuffle 一致性 |
| **小计** | **9** | |

**核心测试**：
- `test_kfold_creates_correct_splits`: 5-fold 创建 5 个折
- `test_stratified_kfold_preserves_class_ratio`: 每个折类别比例与整体一致
- `test_cross_val_score_returns_correct_length`: 分数数量 = cv 折数
- `test_cv_score_mean_and_std`: 均值表示性能，标准差表示稳定性
- `test_cv_multiple_metrics`: 同时评估 accuracy/f1/roc_auc/recall

### 5. 数据泄漏 (Data Leakage)

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 2 | Pipeline 防止泄漏、ColumnTransformer 混合类型 |
| 反例 (Error Cases) | 1 | 全局 StandardScaler 泄漏测试集信息 |
| 边界 (Edge Cases) | 3 | 特征选择泄漏、split 顺序、trivial case |
| **小计** | **6** | |

**核心测试**：
- `test_global_scaler_leaks_test_info`: 全局 fit scaler 会泄漏
- `test_pipeline_prevents_leakage`: Pipeline 内每个折独立拟合预处理
- `test_column_transformer_with_mixed_types`: 数值 StandardScaler + 类别 OneHotEncoder
- `test_train_test_split_order_matters`: 必须先 split，再预处理

### 6. AI 报告审查

| 测试类别 | 测试数量 | 覆盖场景 |
|---------|---------|----------|
| 正例 (Happy Path) | 1 | 识别合格的分类报告 |
| 反例 (Error Cases) | 3 | 只报告准确率、缺少阈值讨论、数据泄漏模式 |
| **小计** | **4** | |

**核心测试**：
- `test_check_good_report_has_all_elements`: 包含混淆矩阵、精确率/召回率/F1、AUC、阈值、局限性
- `test_detect_only_accuracy_report`: 识别缺少混淆矩阵、精确率/召回率、AUC
- `test_detect_no_threshold_discussion`: 识别缺少阈值讨论
- `test_detect_data_leakage_report`: 识别"全局标准化"泄漏模式

### 7. 边界场景汇总

| 边界类别 | 测试数量 |
|---------|----------|
| 逻辑回归边界 | 3（空数据、单观测、完全可分） |
| 混淆矩阵边界 | 5（完美/最差预测、分母为 0） |
| ROC-AUC 边界 | 6（AUC=1/0/0.5、全相同标签、单调性） |
| 交叉验证边界 | 4（折数>样本、单折、单类样本） |
| 数据泄漏边界 | 3（特征选择、trivial case） |
| 阈值边界 | 4（阈值=0/1、最优阈值、权衡） |
| 不平衡数据 | 3（极端不平衡 99:1、单少数类、平衡数据） |
| **小计** | **28** |

## 测试命名规范

遵循格式：`test_<功能>_<场景>_<预期结果>`

示例：
- `test_logistic_regression_sklearn_fits_correctly`: 逻辑回归用 sklearn 正确拟合
- `test_sigmoid_output_range`: Sigmoid 输出范围检查
- `test_precision_zero_denominator`: 精确率分母为 0 的边界情况
- `test_global_scaler_leaks_test_info`: 全局 scaler 泄漏测试集信息

## Fixtures 总览

### 数据 Fixtures

| Fixture | 用途 | 样本量 | 特征 |
|---------|------|--------|------|
| `churn_data` | 贯穿案例（流失数据） | 500 | tenure_months, monthly_charges, total_charges |
| `churn_data_imbalanced` | 不平衡数据（85% vs 15%） | 500 | tenure_months, monthly_charges |
| `churn_data_with_categories` | 混合类型数据 | 500 | 数值 + 类别特征 |
| `binary_classification_data` | 线性可分二分类 | 400 | feature_1, feature_2 |
| `perfect_classifier_data` | 完美分类器（AUC=1） | 200 | feature, true_label, predicted_proba |
| `random_classifier_data` | 随机分类器（AUC≈0.5） | 200 | feature, true_label, predicted_proba |

### 边界 Fixtures

| Fixture | 用途 | 特点 |
|---------|------|------|
| `single_class_data` | 单类别数据 | 全是类别 0 |
| `very_small_dataset` | 极小数据集 | 10 个样本 |
| `perfect_separation_data` | 完全可分数据 | 可能导致系数无穷大 |
| `data_for_leakage_test` | 泄漏测试 | 300 样本，3 特征 |

### 报告 Fixtures

| Fixture | 用途 | 特点 |
|---------|------|------|
| `good_classification_report` | 合格报告示例 | 包含所有关键要素 |
| `bad_classification_report_only_accuracy` | 只报告准确率 | 缺少混淆矩阵等 |
| `bad_classification_report_no_threshold` | 缺少阈值讨论 | 无阈值业务权衡 |
| `bad_classification_report_leakage` | 存在数据泄漏 | "全局标准化"模式 |

## 学习目标映射

### 学习目标 1: 理解逻辑回归的动机

- ✅ 为什么不能用线性回归做分类（预测值超出 [0,1]）
- ✅ Sigmoid 函数把线性得分映射到概率
- ✅ 优势比（odds ratio）解释

**对应测试**：`TestLogisticRegression` 全部

### 学习目标 2: 正确解释逻辑回归系数

- ✅ 系数是对数优势比的变化
- ✅ exp(系数) = 优势比
- ✅ 决策阈值的选择

**对应测试**：`test_odds_ratio_interpretation`, `test_decision_boundary_threshold`

### 学习目标 3: 从准确率升级到混淆矩阵

- ✅ 混淆矩阵（TP/TN/FP/FN）
- ✅ 精确率、召回率、F1
- ✅ 准确率悖论

**对应测试**：`TestConfusionMatrix` 全部

### 学习目标 4: 掌握 ROC-AUC

- ✅ ROC 曲线（FPR vs TPR）
- ✅ AUC 在 [0, 1] 范围
- ✅ 阈值无关评估

**对应测试**：`TestROCAUC` 全部

### 学习目标 5: 正确使用 K-fold 交叉验证

- ✅ K-fold 折数正确
- ✅ StratifiedKFold 保持类别比例
- ✅ 交叉验证估计泛化性能

**对应测试**：`TestCrossValidation` 全部

### 学习目标 6: 识别并防御数据泄漏

- ✅ 识别全局 StandardScaler 的泄漏模式
- ✅ Pipeline + ColumnTransformer 防止泄漏
- ✅ train_test_split 顺序重要性

**对应测试**：`TestDataLeakage` 全部

### 学习目标 7: AI 报告审查

- ✅ 识别缺少混淆矩阵、精确率/召回率
- ✅ 识别缺少阈值讨论
- ✅ 识别数据泄漏模式

**对应测试**：`TestAIClassificationReportReview` 全部

## 运行测试

```bash
# 运行所有测试
python3 -m pytest chapters/week_10/tests -v

# 运行特定测试类
python3 -m pytest chapters/week_10/tests/test_classification.py::TestLogisticRegression -v

# 运行边界测试
python3 -m pytest chapters/week_10/tests/test_edge_cases.py -v

# 运行烟雾测试（需要 starter_code/solution.py）
python3 -m pytest chapters/week_10/tests/test_smoke.py -v

# 查看测试覆盖率
python3 -m pytest chapters/week_10/tests --cov=. --cov-report=html
```

## 注意事项

1. **Smoke tests** (`test_smoke.py`) 依赖 `starter_code/solution.py`，当该文件不存在时会跳过
2. **边界测试** (`test_edge_cases.py`) 独立运行，不依赖 `solution.py`
3. **主要测试** (`test_classification.py`) 使用 scikit-learn/statsmodels 验证统计概念
4. 所有测试使用固定随机种子 (`random_state=42`) 确保可复现
5. 测试命名清晰，失败时能直接看出哪里坏了

## 后续工作

当 `starter_code/solution.py` 实现后，测试需要验证：
- `fit_logistic_regression(X, y)` 返回正确拟合的模型
- `calculate_confusion_matrix(y_true, y_pred)` 返回 TP/TN/FP/FN
- `calculate_precision_recall_f1(y_true, y_pred)` 返回精确率/召回率/F1
- `calculate_roc_auc(y_true, y_proba)` 返回 AUC 和 ROC 曲线数据
- `cross_validate_model(X, y, cv)` 返回交叉验证分数
- `detect_data_leakage(X, y)` 识别泄漏模式
- `review_classification_report(report)` 审查报告质量
