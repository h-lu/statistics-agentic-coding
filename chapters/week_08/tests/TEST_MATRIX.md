# Week 08 测试矩阵

## 概览

本周测试覆盖区间估计与重采样方法的三大主题：
1. **置信区间 (Confidence Interval)**
2. **Bootstrap 重采样**
3. **置换检验 (Permutation Test)**

**总测试数**: 100 个

---

## 测试文件结构

```
tests/
├── __init__.py                    # 包初始化
├── conftest.py                    # 共享 fixtures (33 个)
├── test_smoke.py                  # 基线测试 (9 个)
├── test_confidence_interval.py    # CI 专项测试 (27 个)
├── test_bootstrap.py              # Bootstrap 专项测试 (26 个)
├── test_permutation.py            # 置换检验专项测试 (26 个)
└── test_solution.py               # 综合测试 (12 个)
```

---

## 测试矩阵

### 1. 置信区间测试 (27 个)

| 类别 | 测试名 | 场景 | 预期结果 |
|------|--------|------|----------|
| **正例** | `test_ci_normal_data_large_sample` | 大样本正态数据 | CI 计算正确，包含均值 |
| **正例** | `test_ci_normal_data_small_sample` | 小样本正态数据 | 使用 t 分布，CI 更宽 |
| **正例** | `test_ci_skewed_data` | 偏态数据 | CI 仍有效，非负下界 |
| **正例** | `test_ci_width_decreases_with_sample_size` | 样本量递增 | CI 宽度减小 |
| **正例** | `test_ci_proportion_data` | 二元比例数据 | CI 在 [0,1] 范围内 |
| **边界** | `test_ci_different_confidence_levels` | 不同置信水平 (90% vs 95%) | 更高置信 → 更宽 CI |
| **边界** | `test_ci_minimal_sample` | 两值数据 (n=2) | 仍可计算 CI |
| **边界** | `test_ci_constant_data` | 常量数据 (方差=0) | CI 宽度 ≈ 0 |
| **边界** | `test_ci_single_value` | 单值数据 | CI 退化为该值 |
| **边界** | `test_ci_nan_handling` | 包含 NaN | 报错或处理 |
| **反例** | `test_ci_empty_data` | 空数据 | 报错 |
| **反例** | `test_ci_negative_values` | 负值数据 | CI 仍有效 |
| **验证** | `test_ci_matches_theoretical_for_normal_data` | 与理论值比较 | 误差 < 10% |
| **验证** | `test_ci_result_structure` | 返回格式一致性 | dict 或 tuple，包含必要字段 |
| **解释** | `test_ci_interpretation_correct_format` | CI 解释函数 | 输出有意义解释 |
| **解释** | `test_ci_interpretation_avoids_common_misconception` | 避免误读 | 不说"参数有 95% 概率" |

### 2. Bootstrap 测试 (26 个)

| 类别 | 测试名 | 场景 | 预期结果 |
|------|--------|------|----------|
| **正例** | `test_bootstrap_mean_normal_data` | 正态数据 Bootstrap 均值 | 接近原始均值 (误差 < 5%) |
| **正例** | `test_bootstrap_mean_with_standard_error` | Bootstrap 标准误 | 接近理论值 (误差 < 20%) |
| **正例** | `test_bootstrap_mean_convergence` | 收敛性 | 更多 Bootstrap → 更稳定 |
| **正例** | `test_bootstrap_mean_skewed_data` | 偏态数据 | Bootstrap 仍准确 |
| **正例** | `test_bootstrap_ci_percentile_method` | Percentile 方法 | CI 有效，包含均值 |
| **正例** | `test_bootstrap_ci_width` | CI 宽度 | 95% 比 90% 更宽 |
| **正例** | `test_bootstrap_ci_vs_theoretical_ci` | 与理论 CI 比较 | 误差 < 15% |
| **边界** | `test_bootstrap_mean_small_sample` | 小样本 (n=20) | 仍有效 |
| **边界** | `test_bootstrap_mean_minimal_sample` | 极小样本 (n=5) | 仍可运行 |
| **边界** | `test_bootstrap_mean_constant_data` | 常量数据 | SE = 0 |
| **边界** | `test_bootstrap_mean_single_value` | 单值数据 | SE = 0 |
| **边界** | `test_bootstrap_ci_small_sample` | 小样本 CI | CI 更宽 |
| **边界** | `test_bootstrap_ci_constant_data` | 常量数据 CI | 宽度 ≈ 0 |
| **边界** | `test_bootstrap_different_n_bootstrap` | 不同重采样次数 | CI 宽度相似 (误差 < 20%) |
| **反例** | `test_bootstrap_mean_empty_data` | 空数据 | 报错 |
| **BCa** | `test_bootstrap_ci_bca_method` | BCa 方法 | 对偏态数据更准确 |
| **中位数** | `test_bootstrap_median` | Bootstrap 中位数 | 接近原始中位数 |
| **中位数** | `test_bootstrap_ci_for_median` | 中位数 CI | 对偏态数据有效 |
| **离群点** | `test_bootstrap_with_outliers` | 离群点数据 | 反映数据包含离群点 |
| **可复现** | `test_bootstrap_mean_reproducibility` | 固定随机种子 | 相同种子 → 相同结果 |
| **负值** | `test_bootstrap_negative_values` | 负值数据 | 正常工作 |
| **零方差** | `test_bootstrap_zero_variance_data` | 零方差数据 | SE = 0 |
| **大样本** | `test_bootstrap_very_large_sample` | 大样本 (n=10000) | 高精度估计 |

### 3. 置换检验测试 (26 个)

| 类别 | 测试名 | 场景 | 预期结果 |
|------|--------|------|----------|
| **正例** | `test_permutation_test_equal_groups` | 无差异组 | p 值通常较大 |
| **正例** | `test_permutation_test_different_groups` | 有差异组 | p 值较小 |
| **正例** | `test_permutation_test_observed_statistic` | 观测统计量 | 等于均值差 |
| **正例** | `test_permutation_test_direction` | 单尾 vs 双尾 | 双尾 p ≥ 单尾 p |
| **正例** | `test_permutation_test_skewed_data` | 偏态数据 | p 值仍有效 |
| **正例** | `test_permutation_vs_t_test_normal_data` | 与 t 检验比较 (正态) | p 值接近 |
| **正例** | `test_permutation_statistic_distribution` | 置换分布 | 关于 0 对称 |
| **边界** | `test_permutation_test_unequal_sample_sizes` | 不等样本量 | 仍有效 |
| **边界** | `test_permutation_test_small_samples` | 小样本 | 仍有效 |
| **边界** | `test_permutation_test_different_n_permutations` | 不同置换次数 | p 值都有效 |
| **边界** | `test_permutation_identical_groups` | 完全相同的组 | p 值 ≈ 1 |
| **边界** | `test_permutation_constant_groups` | 常量组 (不同值) | p 值有效 |
| **边界** | `test_permutation_single_value_groups` | 单值组 | p 值有效 |
| **边界** | `test_permutation_very_large_difference` | 极大差异 | p 值极小 |
| **边界** | `test_permutation_binary_data` | 二元数据 | 正常工作 |
| **边界** | `test_permutation_negative_values` | 负值数据 | 正常工作 |
| **反例** | `test_permutation_test_empty_data` | 空数据 | 报错 |
| **可复现** | `test_permutation_test_reproducibility` | 固定随机种子 | 相同种子 → 相同 p |
| **比较** | `test_permutation_vs_t_test_skewed_data` | 与 t 检验比较 (偏态) | 置换检验更可靠 |
| **集成** | `test_permutation_with_ci` | 置换 + Bootstrap CI | 返回 p 值和 CI |
| **StatLab** | `test_statlab_conversion_rate_test` | A/B 测试转化率 | p 值有效 |
| **StatLab** | `test_statlab_spending_comparison` | 用户群组消费 | p 值极小 (大差异) |

### 4. 综合测试 (12 个)

| 类别 | 测试名 | 场景 | 预期结果 |
|------|--------|------|----------|
| **接口** | `test_has_confidence_interval_functions` | CI 函数存在性 | 至少 1 个 CI 函数 |
| **接口** | `test_has_bootstrap_functions` | Bootstrap 函数存在性 | 至少 1 个 Bootstrap 函数 |
| **接口** | `test_has_permutation_functions` | 置换函数存在性 | 至少 1 个置换函数 |
| **工作流** | `test_complete_workflow_single_group` | 单组完整工作流 | 理论 CI ≈ Bootstrap CI |
| **工作流** | `test_complete_workflow_two_groups` | 两组比较工作流 | p 值 + CI 一致性 |
| **工作流** | `test_ci_bootstrap_agreement_on_skewed_data` | 偏态数据方法比较 | Percentile vs BCa |
| **StatLab** | `test_add_ci_to_estimate` | 给点估计加 CI | 返回点估计 + CI |
| **StatLab** | `test_compare_groups_with_uncertainty` | 组间比较带不确定性 | 返回差异 + CI + p |
| **StatLab** | `test_statlab_user_spending_workflow` | 用户消费分析 | 完整工作流 |
| **StatLab** | `test_statlab_conversion_workflow` | A/B 测试转化率 | CI 在 [-1, 1] |
| **数值** | `test_bootstrap_stability_across_runs` | 可复现性 | 相同种子 → 相同结果 |
| **文档** | `test_functions_have_usage_examples` | 文档完整性 | 关键函数有文档 |

---

## Fixtures 概览 (conftest.py)

### 置信区间 Fixtures
- `normal_data_small`: 小样本正态数据 (n=30)
- `normal_data_large`: 大样本正态数据 (n=200)
- `skewed_data`: 偏态数据 (指数分布)
- `bimodal_data`: 双峰分布数据
- `binary_proportion_data`: 二元比例数据

### Bootstrap Fixtures
- `bootstrap_test_data`: 标准 Bootstrap 测试数据
- `bootstrap_small_sample`: 小样本 (n=20)
- `bootstrap_minimal_sample`: 极小样本 (n=5)
- `bootstrap_outlier_data`: 包含离群点数据

### 置换检验 Fixtures
- `permutation_equal_groups`: 无差异组
- `permutation_different_groups`: 有差异组 (均值差=15)
- `permutation_small_difference`: 小差异组 (均值差=5)
- `permutation_unequal_sizes`: 不等样本量 (30 vs 70)
- `permutation_skewed_groups`: 偏态分布组

### 边界情况 Fixtures
- `empty_data`: 空数据
- `single_value_data`: 单值数据
- `two_values_data`: 两值数据
- `constant_data`: 常量数据
- `nan_data`: 包含 NaN 的数据

### StatLab Fixtures
- `statlab_user_spending`: 用户群组消费数据
- `statlab_conversion_rates`: A/B 测试转化率数据

---

## 运行测试

```bash
# 运行所有测试
python3 -m pytest chapters/week_08/tests -q

# 运行特定测试文件
python3 -m pytest chapters/week_08/tests/test_confidence_interval.py -q
python3 -m pytest chapters/week_08/tests/test_bootstrap.py -q
python3 -m pytest chapters/week_08/tests/test_permutation.py -q

# 运行带详细输出
python3 -m pytest chapters/week_08/tests -v

# 运行特定测试
python3 -m pytest chapters/week_08/tests/test_bootstrap.py::TestBootstrapMean::test_bootstrap_mean_normal_data -v
```

---

## 测试命名规范

测试命名遵循格式：`test_<功能>_<场景>_<预期结果>`

示例：
- `test_ci_normal_data_large_sample`: CI + 正态数据 + 大样本
- `test_bootstrap_mean_convergence`: Bootstrap 均值 + 收敛性
- `test_permutation_test_skewed_data`: 置换检验 + 偏态数据

这种命名确保测试失败时能直接看出问题所在。
