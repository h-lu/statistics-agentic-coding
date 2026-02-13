# Week 11 测试矩阵 (Test Matrix)

## 测试概览

本周测试覆盖 **树模型与集成学习** 的核心功能：

| 模块 | 测试文件 | 测试数量 | 覆盖场景 |
|------|---------|---------|---------|
| 烟雾测试 | `test_smoke.py` | 15+ | 基本功能可用性 |
| 决策树 | `test_decision_tree.py` | 30+ | 初始化、拟合、预测、过拟合检测 |
| 随机森林 | `test_random_forest.py` | 35+ | 初始化、拟合、OOB、方差降低 |
| 特征重要性 | `test_feature_importance.py` | 40+ | 内置、置换、相关性陷阱 |
| 超参数调优 | `test_hyperparameter_tuning.py` | 35+ | 网格搜索、随机搜索、对比 |
| 集成测试 | `test_integration.py` | 30+ | 完整流水线、模型对比、报告生成 |
| **总计** | - | **185+** | - |

---

## 1. 决策树测试 (`test_decision_tree.py`)

### 1.1 初始化测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_decision_tree_with_max_depth` | 带 `max_depth` 参数的树 | 深度不超过指定值 |
| `test_decision_tree_with_min_samples_split` | 带 `min_samples_split` 参数 | 参数生效 |
| `test_decision_tree_with_criterion_mse` | 回归树使用 MSE 准则 | 准则设置为 MSE |
| `test_decision_tree_with_criterion_gini` | 分类树使用 Gini 准则 | 准则设置为 Gini |

### 1.2 拟合与预测测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_decision_tree_fit_regression` | 回归树拟合 | R² > 0 |
| `test_decision_tree_fit_classification` | 分类树拟合 | 准确率 > 0.5 |
| `test_decision_tree_predict_single_sample` | 单样本预测 | 返回有效预测值 |

### 1.3 特征重要性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_feature_importances_exists` | 特征重要性属性存在 | 返回重要性数组 |
| `test_feature_importances_ordering` | 重要性排序合理 | 重要特征排在前面 |
| `test_feature_importances_zero_for_unused` | 未使用特征重要性为 0 | 噪声特征重要性 < 0.1 |

### 1.4 树导出测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_export_tree_text_not_empty` | 导出文本不为空 | 返回非空字符串 |
| `test_export_tree_text_contains_depth` | 文本包含深度信息 | 包含分裂关键词 |

### 1.5 过拟合检测测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_overfitting_deep_tree` | 深树过拟合检测 | `is_overfitting = True` |
| `test_no_overfitting_shallow_tree` | 浅树不过拟合 | `is_overfitting = False` 或小 gap |
| `test_overfitting_threshold_sensitivity` | 阈值敏感性 | 不同阈值不同结果 |

### 1.6 边界测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_decision_tree_single_feature` | 单特征树 | 成功拟合 |
| `test_decision_tree_small_dataset` | 小数据集 | 成功拟合 |
| `test_decision_tree_constant_target` | 常数目标 | 预测接近均值 |
| `test_decision_tree_perfect_fit` | 完美拟合 | 准确率 > 0.9 |

---

## 2. 随机森林测试 (`test_random_forest.py`)

### 2.1 初始化测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_random_forest_with_n_estimators` | 树的数量参数 | n_estimators 正确设置 |
| `test_random_forest_with_max_features` | 最大特征数参数 | 参数生效 |
| `test_random_forest_with_max_depth` | 深度参数 | 每棵树深度受限 |
| `test_random_forest_classification` | 分类随机森林 | 成功拟合 |

### 2.2 拟合与预测测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_random_forest_fit_regression` | 回归森林拟合 | R² > 0 |
| `test_random_forest_fit_classification` | 分类森林拟合 | 准确率 > 0.5 |
| `test_random_forest_predict_proba` | 概率预测 | 返回有效概率 |

### 2.3 特征重要性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_feature_importance_exists` | 特征重要性存在 | 返回重要性数组 |
| `test_extract_feature_importance_returns_dataframe` | 返回 DataFrame | 包含特征和重要性列 |
| `test_feature_importance_ranking` | 重要性排序 | 降序排列 |
| `test_feature_importance_consistency` | 相同种子一致性 | 结果相似 |

### 2.4 OOB 分数测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_rf_with_oob_score` | OOB 分数存在 | 返回有效 OOB 分数 |
| `test_compare_rf_oob_score_vs_test_score` | OOB vs 测试分数 | 差异 < 20% |
| `test_compare_rf_oob_score_function` | OOB 分数对比函数 | 返回对比结果 |

### 2.5 方差降低测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_variance_reduction_vs_single_tree` | 相比单棵树方差降低 | 森林方差更小 |
| `test_measure_variance_reduction` | 方差降低测量 | 返回测量报告 |
| `test_rf_more_stable_than_tree` | 稳定性更好 | 标准差更小 |

### 2.6 树多样性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_get_rf_tree_diversity` | 树多样性测量 | 返回多样性指标 |
| `test_rf_trees_are_different` | 树之间不同 | 预测有差异 |
| `test_max_features_impacts_diversity` | max_features 影响 | 不同设置不同多样性 |

---

## 3. 特征重要性测试 (`test_feature_importance.py`)

### 3.1 内置特征重要性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_builtin_importance_calculation` | 内置重要性计算 | 返回重要性 |
| `test_builtin_importance_sum_to_one` | 重要性之和为 1 | sum ≈ 1.0 |
| `test_builtin_importance_non_negative` | 非负性 | 所有 ≥ 0 |
| `test_builtin_importance_known_truth` | 识别真值 | 正确识别重要特征 |

### 3.2 置换重要性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_permutation_importance_calculation` | 置换重要性计算 | 返回重要性 |
| `test_permutation_importance_reduces_score` | 置换降低分数 | 置换后分数下降 |
| `test_permutation_importance_known_truth` | 识别真值 | 正确排序 |
| `test_permutation_importance_n_repeats` | 重复次数参数 | 不同 n_repeats 都成功 |

### 3.3 内置 vs 置换对比测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_compare_builtin_vs_permutation` | 对比函数 | 返回两种重要性 |
| `test_builtin_and_permutation_correlation` | 相关性 | 有一定相关性 |
| `test_permutation_more_reliable_for_correlated` | 对相关特征更可靠 | 正确处理相关特征 |

### 3.4 相关性稀释测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_detect_correlation_dilution` | 检测相关性稀释 | 识别高相关对 |
| `test_correlation_dilution_reduces_importance` | 降低单个重要性 | 验证稀释效应 |
| `test_correlation_matrix_detection` | 相关性矩阵检测 | 返回相关矩阵 |

### 3.5 解释与可视化测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_interpret_feature_importance_returns_text` | 返回解释文本 | 包含关键信息 |
| `test_interpret_includes_warning` | 包含警告 | 有相关性/因果警告 |
| `test_interpret_no_causality_claim` | 不做因果声称 | 避免因果语言 |
| `test_plot_feature_importance` | 绘制重要性图 | 成功生成 |
| `test_plot_importance_top_k` | 绘制 Top K | 只显示 K 个 |

### 3.6 边界与陷阱测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_importance_with_single_feature` | 单特征 | 重要性 ≈ 1.0 |
| `test_importance_with_unimportant_feature` | 不重要特征 | 重要性接近 0 |
| `test_importance_with_constant_feature` | 常数特征 | 重要性 ≈ 0 |
| `test_high_cardinality_overestimates_importance` | 高基数陷阱 | 识别或警告 |
| `test_permutation_importance_reveals_trap` | 置换揭示陷阱 | 正确识别 |

---

## 4. 超参数调优测试 (`test_hyperparameter_tuning.py`)

### 4.1 网格搜索测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_grid_search_basic` | 基本网格搜索 | 返回最佳参数 |
| `test_grid_search_returns_best_params` | 返回最佳参数 | 参数在网格中 |
| `test_grid_search_cv_results` | CV 结果 | 返回详细结果 |
| `test_grid_search_with_scoring_metric` | 不同评分指标 | 支持多指标 |
| `test_grid_search_computationally_expensive` | 大参数网格 | 能处理（慢） |

### 4.2 随机搜索测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_random_search_basic` | 基本随机搜索 | 返回最佳参数 |
| `test_random_search_n_iter_parameter` | n_iter 参数 | 不同 n_iter 都成功 |
| `test_random_search_with_distributions` | 使用分布 | 支持 scipy 分布 |

### 4.3 网格 vs 随机对比测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_compare_grid_vs_random` | 对比函数 | 返回两种结果 |
| `test_random_search_faster_than_grid` | 速度对比 | 随机更快（通常） |
| `test_grid_vs_random_score_quality` | 得分质量 | 随机接近网格 |

### 4.4 敏感性分析测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_analyze_hyperparameter_sensitivity` | 敏感性分析 | 返回得分曲线 |
| `test_max_depth_sensitivity` | max_depth 敏感性 | 有最佳深度 |
| `test_n_estimators_sensitivity` | n_estimators 敏感性 | 收益递减 |

### 4.5 验证测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_validate_hyperparameters_valid` | 有效参数 | valid = True |
| `test_validate_hyperparameters_invalid` | 无效参数 | valid = False |
| `test_validate_hyperparameters_warning` | 过拟合风险警告 | 有警告 |

### 4.6 边界测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_tuning_with_small_dataset` | 小数据集调优 | 能处理（CV 受限）|
| `test_tuning_with_single_param` | 单参数调优 | 成功 |
| `test_tuning_with_extreme_values` | 极端值调优 | 找到合理参数 |

### 4.7 嵌套 CV 测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_nested_cv_prevents_overfitting` | 嵌套 CV 防止过拟合 | 外层得分更保守 |

---

## 5. 集成测试 (`test_integration.py`)

### 5.1 完整流水线测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_complete_regression_pipeline` | 完整回归流水线 | 包含模型、评估、重要性 |
| `test_complete_classification_pipeline` | 完整分类流水线 | 包含模型、评估 |
| `test_pipeline_with_hyperparameter_tuning` | 带调优流水线 | 包含最佳参数 |
| `test_pipeline_with_feature_importance` | 带重要性流水线 | 包含两种重要性 |

### 5.2 模型对比测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_compare_linear_vs_tree_vs_forest` | 三模型对比 | 都包含在结果中 |
| `test_compare_models_returns_r2` | 返回 R² | R² > 0 |
| `test_compare_classification_models` | 分类模型对比 | 准确率 > 0.5 |
| `test_forest_outperforms_tree` | 森林优于树 | 森林不比树差 |

### 5.3 报告生成测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_generate_tree_report_regression` | 回归报告 | 包含 R²/MSE |
| `test_generate_tree_report_classification` | 分类报告 | 包含准确率/召回率 |
| `test_report_includes_feature_importance` | 包含重要性 | 有重要性信息 |
| `test_report_includes_limitations` | 包含局限性 | 有局限性讨论 |

### 5.4 AI 代码审查测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_review_good_tree_code` | 审查好代码 | 无严重问题 |
| `test_review_overfitting_code` | 审查过拟合代码 | 检测到过拟合 |
| `test_review_no_tuning_code` | 审查无调优代码 | 检测到缺失 |
| `test_review_misinterpretation_code` | 审查错误解释 | 检测到因果问题 |

### 5.5 端到端工作流测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_complete_analysis_workflow` | 完整分析 | 所有步骤成功 |
| `test_workflow_with_categorical_features` | 带类别特征 | 能处理编码 |
| `test_workflow_detects_and_warns_overfitting` | 检测过拟合 | 包含警告 |

### 5.6 稳健性与可复现性测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_model_robustness_to_noise` | 噪声稳健性 | 都能学到一些 |
| `test_model_stability_across_splits` | 划分稳定性 | 森林更稳定 |
| `test_same_random_seed_same_results` | 相同种子相同结果 | 预测完全相同 |
| `test_different_random_seed_different_results` | 不同种子不同结果 | 预测不同 |

---

## 6. 烟雾测试 (`test_smoke.py`)

### 6.1 基本功能测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_smoke_fit_decision_tree` | 决策树拟合 | 返回模型 |
| `test_smoke_fit_random_forest` | 随机森林拟合 | 返回模型 |
| `test_smoke_calculate_feature_importance` | 特征重要性计算 | 返回重要性 |
| `test_smoke_calculate_permutation_importance` | 置换重要性计算 | 返回重要性 |
| `test_smoke_tune_hyperparameters_grid` | 网格搜索 | 返回最佳参数 |
| `test_smoke_tune_hyperparameters_random` | 随机搜索 | 返回最佳参数 |
| `test_smoke_detect_overfitting` | 过拟合检测 | 返回检测报告 |
| `test_smoke_compare_tree_models` | 模型对比 | 返回对比结果 |
| `test_smoke_review_tree_model_code` | 代码审查 | 返回审查结果 |

### 6.2 端到端测试

| 测试名 | 测试内容 | 预期结果 |
|--------|---------|---------|
| `test_complete_tree_workflow` | 完整树工作流 | 所有步骤成功 |
| `test_complete_review_workflow` | 完整审查工作流 | 检测到问题 |

---

## 测试覆盖的场景

### 正例（Happy Path）
- ✅ 正常大小的数据集（100-1000 样本）
- ✅ 合理的超参数组合
- ✅ 正确的特征类型（数值、类别编码）
- ✅ 完整的数据（无缺失值）

### 边界案例（Edge Cases）
- 🔲 单特征数据
- 🔲 极小数据集（10-20 样本）
- 🔲 常数目标变量
- 🔲 高基数类别特征
- 🔲 高度相关特征
- 🔲 噪声特征
- 🔲 完美可分数据

### 反例（Negative Cases）
- ❌ 无效的超参数（负数、0）
- ❌ 容易过拟合的配置（无深度限制）
- ❌ 缺少超参数调优
- ❌ 错误的特征重要性解释（因果声称）

---

## 运行测试

```bash
# 运行所有测试
pytest chapters/week_11/tests/ -v

# 运行特定测试文件
pytest chapters/week_11/tests/test_decision_tree.py -v

# 运行特定测试类
pytest chapters/week_11/tests/test_decision_tree.py::TestDecisionTreeInitialization -v

# 运行特定测试
pytest chapters/week_11/tests/test_decision_tree.py::TestDecisionTreeInitialization::test_decision_tree_with_max_depth -v

# 运行烟雾测试
pytest chapters/week_11/tests/test_smoke.py -v

# 运行集成测试
pytest chapters/week_11/tests/test_integration.py -v

# 查看测试覆盖率（需要安装 pytest-cov）
pytest chapters/week_11/tests/ --cov=chapters/week_11/starter_code --cov-report=html
```

---

## 预期结果

当 `starter_code/solution.py` 实现完成后：

1. **烟雾测试**: 所有 15+ 测试应该通过
2. **决策树测试**: 所有 30+ 测试应该通过
3. **随机森林测试**: 所有 35+ 测试应该通过
4. **特征重要性测试**: 所有 40+ 测试应该通过
5. **超参数调优测试**: 所有 35+ 测试应该通过
6. **集成测试**: 所有 30+ 测试应该通过

总体：**185+ 测试应该通过**
