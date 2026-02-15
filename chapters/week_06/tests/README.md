# Week 06 测试用例说明

本目录包含 Week 06（假设检验、效应量与 AI 审查训练）的 pytest 测试用例。

## 测试文件结构

```
tests/
├── __init__.py           # 测试包初始化
├── conftest.py           # 共享 fixtures
├── test_smoke.py         # 冒烟测试（基础功能验证）
├── test_solution.py      # 主要测试用例
└── test_edge_cases.py    # 边界情况测试
```

## 测试用例矩阵

### 1. p 值理解测试 (test_solution.py:TestPValueInterpretation)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_interpret_p_value_significant | p < 0.05 时正确拒绝原假设 |
| 正例 | test_interpret_p_value_not_significant | p >= 0.05 时无法拒绝原假设 |
| 边界 | test_interpret_p_value_boundary | p = 0.05 时的边界判断 |
| 正例 | test_interpret_p_value_multiple_alpha | 不同显著性水平下的解释 |

### 2. t 检验测试 (test_solution.py:TestTwoSampleTTest, TestProportionTest, TestPairedTTest)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_t_test_significant_difference | 两组均值显著差异 |
| 正例 | test_t_test_no_difference | 无差异情况 |
| 边界 | test_t_test_small_sample | 小样本情况 |
| 反例 | test_t_test_empty_data | 空数据应报错 |
| 反例 | test_t_test_single_value | 单值数据无法计算标准差 |
| 正例 | test_proportion_test_significant | 比例检验检测显著差异 |
| 正例 | test_proportion_test_same_rate | 相同比例 |
| 正例 | test_paired_t_test_significant | 配对检验检测显著变化 |
| 反例 | test_paired_t_test_mismatched_lengths | 长度不匹配应报错 |

### 3. 卡方检验测试 (test_solution.py:TestChiSquareTest)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_chi_square_independent | 独立变量的卡方检验 |
| 正例 | test_chi_square_dependent | 有关联变量的卡方检验 |
| 正例 | test_chi_square_expected_frequencies | 期望频数计算 |

### 4. 效应量测试 (test_solution.py:TestCohensD, TestRiskDifference)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_cohens_d_large_effect | Cohen's d > 0.8 大效应 |
| 正例 | test_cohens_d_medium_effect | 0.2 <= d < 0.8 中等效应 |
| 正例 | test_cohens_d_small_effect | d < 0.2 小效应 |
| 边界 | test_cohens_d_no_effect | d ≈ 0 无效应 |
| 正例 | test_cohens_d_direction | 效应量的方向 |
| 反例 | test_cohens_d_empty_data | 空数据应报错 |
| 正例 | test_risk_difference_calculation | 风险差计算 |
| 正例 | test_risk_ratio_calculation | 风险比计算 |

### 5. 前提假设检查测试 (test_solution.py:TestNormalityCheck, TestVarianceHomogeneityCheck)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_shapiro_wilk_normal_data | 正态数据的 Shapiro-Wilk 检验 |
| 正例 | test_shapiro_wilk_skewed_data | 偏态数据的 Shapiro-Wilk 检验 |
| 边界 | test_shapiro_wilk_constant_data | 常数数据的正态性检验 |
| 边界 | test_shapiro_wilk_small_sample | 极小样本的正态性检验 |
| 反例 | test_shapiro_wilk_empty_data | 空数据应报错 |
| 正例 | test_levene_equal_variance | 方差相等数据的 Levene 检验 |
| 正例 | test_levene_unequal_variance | 方差不等数据的 Levene 检验 |
| 边界 | test_levene_constant_data | 常数数据的方差齐性检验 |
| 正例 | test_choose_test_based_on_assumptions | 基于假设选择合适的检验 |
| 正例 | test_assumption_report_includes_warnings | 假设检查报告应包含警告 |

### 6. AI 结论审查测试 (test_solution.py:TestAIReportReview)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_review_good_report | 审查合格的 AI 报告 |
| 反例 | test_review_bad_report_missing_ci | 缺少置信区间应标记为问题 |
| 反例 | test_review_bad_report_missing_effect_size | 缺少效应量应标记为问题 |
| 反例 | test_review_bad_report_no_assumption_check | 未检查假设应标记为严重问题 |
| 反例 | test_review_bad_report_overinterpretation | 过度解读应标记 |
| 反例 | test_review_wrong_test_for_data_type | 对二元数据使用 t 检验应警告 |

### 7. 多重比较校正测试 (test_solution.py:TestMultipleComparisonCorrection)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_bonferroni_correction | Bonferroni 校正 |
| 正例 | test_false_discovery_rate | FDR (Benjamini-Hochberg) 校正 |
| 正例 | test_calculate_family_wise_error_rate | 计算家族错误率 |

### 8. 综合测试 (test_solution.py:TestCompleteHypothesisTestWorkflow)

| 类型 | 测试用例 | 描述 |
|------|---------|------|
| 正例 | test_complete_two_group_test | 完整的两组比较流程 |
| 正例 | test_generate_test_report | 生成 Markdown 格式的检验报告 |

### 9. 边界情况测试 (test_edge_cases.py)

| 类别 | 测试用例 | 描述 |
|------|---------|------|
| 空输入 | test_p_value_interpretation_empty_input | 空 p 值应报错 |
| 空输入 | test_t_test_empty_group | 空组应报错 |
| 空输入 | test_cohens_d_empty_data | 空数据应报错 |
| 空输入 | test_normality_test_empty_data | 空数据应报错 |
| 极小样本 | test_t_test_n_equals_2 | 每组 2 个样本 |
| 极小样本 | test_t_test_n_equals_1 | 每组 1 个样本（无法计算标准差） |
| 极小样本 | test_shapiro_wilk_min_sample | Shapiro-Wilk 最小需要 3 个样本 |
| 极小样本 | test_bootstrap_n_equals_2 | Bootstrap 用 2 个样本 |
| 极端值 | test_t_test_with_extreme_outlier | 单个极端异常值的影响 |
| 极端值 | test_cohens_d_with_outlier | 异常值对效应量的影响 |
| 极端值 | test_proportion_test_all_success | 全部成功（100% 转化率） |
| 极端值 | test_proportion_test_all_failure | 全部失败（0% 转化率） |
| 特殊类型 | test_all_same_values | 全部相同值（方差为0） |
| 特殊类型 | test_binary_with_rare_event | 稀有事件（转化率 < 1%） |
| 特殊类型 | test_very_large_sample | 极大样本（n > 100000） |
| 特殊类型 | test_nan_values | 包含 NaN 值的数据 |
| 特殊类型 | test_inf_values | 包含 Inf 值的数据 |
| 参数边界 | test_alpha_boundary_zero | alpha = 0（无效） |
| 参数边界 | test_alpha_boundary_one | alpha = 1（无效） |
| 参数边界 | test_p_value_boundary_zero | p 值 = 0（极端罕见） |
| 参数边界 | test_p_value_boundary_one | p 值 = 1（完全符合原假设） |
| 参数边界 | test_negative_p_value | 负 p 值（无效） |
| 参数边界 | test_p_value_greater_than_one | p 值 > 1（无效） |
| 多重比较 | test_single_hypothesis | 单个假设（无需校正） |
| 多重比较 | test_zero_hypotheses | 零个假设（无效输入） |
| 多重比较 | test_all_significant_hypotheses | 所有假设都显著 |
| 多重比较 | test_all_non_significant_hypotheses | 所有假设都不显著 |
| 多重比较 | test_very_many_hypotheses | 极多假设（alpha 变得很小） |
| 数值稳定性 | test_very_small_numbers | 极小数值 |
| 数值稳定性 | test_very_large_numbers | 极大数值 |
| 数值稳定性 | test_mixed_scale_numbers | 混合数量级的数值 |
| 列联表边界 | test_2x2_table_with_zero | 包含 0 的 2x2 列联表 |
| 列联表边界 | test_very_small_counts | 极小计数的列联表 |
| 列联表边界 | test_single_row_table | 单行列联表（无效） |
| 列联表边界 | test_single_column_table | 单列列联表（无效） |

## 共享 Fixtures (conftest.py)

### 数据生成 Fixtures

- `null_hypothesis_data`: 符合原假设的数据（无真实差异）
- `alternative_hypothesis_data`: 符合备择假设的数据（有真实差异）
- `normal_two_groups`: 正态分布的两组数据
- `small_sample_groups`: 小样本数据
- `binary_conversion_data`: 二元转化数据（0/1）
- `paired_data`: 配对数据

### 效应量 Fixtures

- `large_effect_data`: 大效应量数据 (Cohen's d > 0.8)
- `medium_effect_data`: 中等效应量数据 (0.2 <= d < 0.8)
- `small_effect_data`: 小效应量数据 (d < 0.2)
- `no_effect_data`: 无效应数据 (d ≈ 0)

### 前提假设 Fixtures

- `normal_data`: 正态分布数据
- `skewed_data`: 偏态分布数据（指数分布）
- `equal_variance_groups`: 方差相等的数据
- `unequal_variance_groups`: 方差不等的数据
- `constant_data`: 常数数据

### 边界情况 Fixtures

- `empty_data`: 空数组
- `single_value_data`: 单值数组
- `two_values_data`: 两值数组（最小可计算标准差）

### AI 审查 Fixtures

- `ai_good_report`: 合格的 AI 生成的检验报告
- `ai_bad_report`: 有问题的 AI 生成的检验报告
- `multiple_hypotheses_results`: 多重检验结果

## 运行测试

```bash
# 运行所有测试
python3 -m pytest chapters/week_06/tests -q

# 运行特定测试文件
python3 -m pytest chapters/week_06/tests/test_solution.py -q

# 运行特定测试类
python3 -m pytest chapters/week_06/tests/test_solution.py::TestCohensD -q

# 运行特定测试用例
python3 -m pytest chapters/week_06/tests/test_solution.py::TestCohensD::test_cohens_d_large_effect -v

# 显示详细输出
python3 -m pytest chapters/week_06/tests -v

# 只运行失败的测试
python3 -m pytest chapters/week_06/tests --lf

# 运行并显示覆盖率
python3 -m pytest chapters/week_06/tests --cov=chapters/week_06/starter_code/solution
```

## 预期接口规范

测试用例定义了 `starter_code/solution.py` 应该实现的函数接口：

### p 值理解
- `interpret_p_value(p_value, alpha=0.05)`: 解释 p 值

### t 检验
- `two_sample_t_test(group_a, group_b, alpha=0.05)`: 双样本 t 检验
- `proportion_test(conversions_a, conversions_b, alpha=0.05)`: 比例检验
- `paired_t_test(before, after, alpha=0.05)`: 配对 t 检验

### 卡方检验
- `chi_square_test(contingency_table, alpha=0.05)`: 卡方检验

### 效应量
- `cohens_d(group1, group2)`: 计算 Cohen's d
- `interpret_cohens_d(d)`: 解释 Cohen's d
- `risk_difference(conversions_a, conversions_b)`: 风险差
- `risk_ratio(conversions_a, conversions_b)`: 风险比

### 前提假设检查
- `check_normality(data, alpha=0.05)`: 正态性检验
- `check_variance_homogeneity(group1, group2, alpha=0.05)`: 方差齐性检验
- `choose_test_auto(group_a, group_b, alpha=0.05)`: 自动选择检验方法

### AI 结论审查
- `review_ai_report(report)`: 审查 AI 报告
- `bonferroni_correction(results, alpha=0.05)`: Bonferroni 校正
- `fdr_correction(results, q=0.05)`: FDR 校正
- `calculate_family_wise_error_rate(n_hypotheses, alpha=0.05)`: 计算家族错误率

### 综合流程
- `complete_two_group_test(group_a, group_b, alpha=0.05)`: 完整的两组比较
- `generate_hypothesis_test_report(group_a, group_b, group_names, value_name)`: 生成 Markdown 报告
