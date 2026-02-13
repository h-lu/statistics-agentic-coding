# Week 15 测试用例矩阵

## 测试文件概览

| 文件 | 测试数量 | 测试内容 |
|------|----------|----------|
| `test_smoke.py` | 32 | 基础导入、简单功能测试 |
| `test_pca.py` | 30 | PCA 降维、方差解释、特征载荷 |
| `test_clustering.py` | 33 | K-means 聚类、肘部法则、轮廓系数 |
| `test_streaming.py` | 28 | 流式统计（在线均值、方差、分位数） |
| `test_ab_testing.py` | 30 | A/B 测试工程化、SRM 检测、决策规则 |
| `test_statlab.py` | 12 | StatLab 集成测试 |

**总计: 165 个测试用例**

---

## 测试分类矩阵

### 1. PCA 降维测试

#### 正例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_pca_fits_on_simple_2d_data` | PCA 在二维数据上拟合 | 成功降维到 1 维 |
| `test_explained_variance_ratio_decreases` | 方差解释比例递减 | PC1 > PC2 > PC3 ... |
| `test_explained_variance_sum_equals_one` | 方差解释比例和为 1 | Σ = 1.0 |
| `test_cumulative_variance_is_monotonic` | 累积方差单调递增 | cumsum[i] <= cumsum[i+1] |
| `test_select_components_by_variance_threshold` | 按阈值选择主成分 | 达到 85% 方差阈值 |
| `test_feature_loadings_shape` | 特征载荷矩阵形状 | (n_components, n_features) |
| `test_feature_loadings_unit_length` | 主成分单位长度 | ||loading|| = 1 |
| `test_feature_loadings_orthogonal` | 主成分相互正交 | dot(PCi, PCj) ≈ 0 |
| `test_feature_loadings_interpretation` | 特征载荷解释 | 相关特征载荷同符号 |

#### 边界
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_pca_with_constant_data` | 常数数据（零方差） | 方差为 0 或警告 |
| `test_pca_with_single_feature` | 单特征数据 | 无法降维或解释 100% |
| `test_pca_with_curse_of_dimensionality` | p >> n (维度灾难) | 主成分数 ≤ n |
| `test_pca_minimum_sample_size` | 最小样本量 | n=2 可拟合 |
| `test_pca_with_empty_data` | 空数据 | 抛出 ValueError |

#### 反例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_pca_creates_new_features` | PCA 是特征提取不是特征选择 | 主成分是线性组合 |
| `test_detect_missing_variance_threshold_check` | AI 报告缺少方差阈值检查 | 检测到缺失 |

---

### 2. 聚类分析测试

#### 正例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_kmeans_fits_well_separated_clusters` | 分离良好的簇 | 正确识别 3 个簇 |
| `test_kmeans_cluster_centers` | 簇中心接近真实中心 | 在合理范围内 |
| `test_kmeans_inertia_decreases_with_k` | WCSS 随 K 递减 | inertia[K] > inertia[K+1] |
| `test_kmeans_predict_new_data` | 预测新数据点 | 返回 0-K-1 标签 |
| `test_elbow_method_visual_inspection` | 肘部法则可见 | 可检测到肘部 |
| `test_silhouette_score_range` | 轮廓系数范围 | -1 ≤ score ≤ 1 |
| `test_silhouette_score_decreases_with_bad_k` | 错误 K 轮廓系数更低 | 最优 K 得分最高 |
| `test_cluster_size_distribution` | 簇大小分布 | 无极端不平衡 |
| `test_cluster_centers_match_sample_means` | 簇中心 = 样本均值 | 数值匹配 |

#### 边界
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_kmeans_with_single_cluster` | K=1 (所有点一簇) | 中心 = 全局均值 |
| `test_kmeans_minimum_samples` | 最小样本量 | n >= K 才能拟合 |
| `test_kmeans_with_empty_data` | 空数据 | 抛出 ValueError |
| `test_silhouette_score_requires_min_clusters` | 轮廓系数最少 2 簇 | K=1 时报错 |
| `test_elbow_method_with_overlapping_clusters` | 重叠簇肘部不明显 | 肘部难识别 |

#### 反例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_clustering_has_no_ground_truth` | 聚类无需标签 | 无监督学习 |
| `test_clustering_evaluation_is_harder` | 聚类评估更难 | 需内部指标 |
| `test_detect_missing_k_selection_rationale` | 报告缺少 K 值选择理由 | 检测到缺失 |

---

### 3. 流式统计测试

#### 正例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_online_mean_converges_to_batch_mean` | 在线均值收敛到批量均值 | 误差 < 1e-10 |
| `test_online_mean_returns_intermediate_values` | 返回中间结果 | 每次更新后可调用 |
| `test_online_variance_converges_to_batch_variance` | 在线方差收敛到批量方差 | 误差 < 1e-10 |
| `test_welford_algorithm_numerical_stability` | Welford 算法数值稳定 | 大幅值数据相对误差小 |
| `test_online_quantile_approximation_error` | 在线分位数近似误差 | 误差 < 15% |
| `test_online_quantile_converges_with_more_bins` | 更多箱更精确 | 箱多误差更小 |

#### 边界
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_online_mean_with_single_value` | 单值均值 | mean = x |
| `test_online_mean_with_constant_data` | 常数数据 | mean = 常数值 |
| `test_online_mean_with_drift` | 均值漂移适应 | 均值随数据变化 |
| `test_online_variance_with_constant_data` | 常数方差 | variance = 0 |
| `test_online_variance_with_single_value` | 单值方差 | variance = 0 |
| `test_online_statistics_empty_initially` | 空状态 | 返回 0 |

#### 反例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_computational_complexity` | 流式 O(1) vs 批量 O(n) | 流式更快 |
| `test_memory_efficiency` | 流式 O(1) 内存 vs 批量 O(n) | 流式更省内存 |
| `test_bootstrap_estimates_uncertainty` | Bootstrap 估计不确定性 | CI 包含均值 |
| `test_streaming_does_not_replace_bootstrap` | 流式不替代 Bootstrap | 互补关系 |

---

### 4. A/B 测试工程化测试

#### 正例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_t_test_detects_significant_difference` | t 检测显著差异 | p < 0.05 |
| `test_t_test_no_significant_difference` | 无差异时 p > 0.05 | p > 0.05 |
| `test_t_test_effect_size_calculation` | 效应量计算 | effect = mean_B - mean_A |
| `test_t_test_confidence_interval` | 95% CI 计算 | effect ± 1.96*SE |
| `test_detect_srm_with_chi_square` | SRM 检测 | 卡方 p < 0.05 |
| `test_launch_b_when_significant_and_large_effect` | 决策规则 | launch_B |
| `test_precommitted_sample_size_controls_error` | 预设样本量控制 FPR | FPR ≈ 5% |

#### 边界
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_t_test_requires_sufficient_sample_size` | 小样本低检验力 | p 可能不显著 |
| `test_t_test_with_minimum_sample_size` | 最小样本 | 可运行但低力 |
| `test_srm_extreme_imbalance` | 极端不平衡 (90:10) | p 非常小 |
| `test_srm_minimum_sample_detection` | 最小样本 SRM 检测 | 可运行但不可靠 |
| `test_ab_test_with_constant_values` | 常数值 | 方差为 0 或警告 |
| `test_ab_test_identical_groups` | 相同组 | p = 1.0, t = 0.0 |

#### 反例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_early_stopping_increases_false_positive_rate` | 早期停止增加 FPR | FPR > 5% |
| `test_automated_decision_requires_srm_check` | 自动决策需 SRM 检查 | SRM 异常时警告 |
| `test_detect_missing_srm_check` | 报告缺少 SRM 检查 | 检测到缺失 |
| `test_detect_causal_claim` | 因果语言检测 | 检测到"导致"等词 |

---

### 5. StatLab 集成测试

#### 正例
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_pca_then_clustering_workflow` | PCA + 聚类流程 | 完整运行成功 |
| `test_pca_clustering_reduces_computation_cost` | 降维减少计算成本 | 特征数减少 |
| `test_pca_clustering_preserves_structure` | 降维保留结构 | 簇数接近真实值 |
| `test_streaming_stats_per_cluster` | 每簇流式统计 | 每簇有独立状态 |
| `test_streaming_vs_batch_cluster_stats` | 流式=批量 | 统计量匹配 |
| `test_ab_test_per_cluster` | 分层 A/B 测试 | 每簇有结果 |
| `test_complete_pipeline_pca_cluster_stream_ab` | 完整流程 | 所有步骤成功 |
| `test_workflow_output_structure` | 输出结构 | 包含 PCA 和聚类信息 |

#### 边界
| 测试名称 | 测试内容 | 预期结果 |
|---------|---------|----------|
| `test_workflow_handles_edge_cases` | 小数据等边界 | 仍能运行 |
| `test_ab_test_detects_segment_specific_effects` | 分层效应检测 | 不同簇不同结果 |

---

## 测试命名规范

遵循以下命名规范：
- `test_<功能>_<场景>_<预期结果>`
- 示例：
  - `test_pca_fits_on_simple_2d_data` (正例)
  - `test_pca_with_empty_data` (边界)
  - `test_detect_missing_variance_threshold_check` (反例/AI审查)

---

## 运行测试

```bash
# 运行所有测试
python3 -m pytest chapters/week_15/tests -q

# 运行特定文件
python3 -m pytest chapters/week_15/tests/test_pca.py -q

# 运行特定类
python3 -m pytest chapters/week_15/tests/test_pca.py::TestExplainedVariance -q

# 运行特定测试
python3 -m pytest chapters/week_15/tests/test_pca.py::TestExplainedVariance::test_explained_variance_sum_equals_one -v

# 显示详细输出
python3 -m pytest chapters/week_15/tests -v

# 显示 print 输出
python3 -m pytest chapters/week_15/tests -v -s
```

---

## 测试覆盖要点

### PCA 降维
- [x] n_components 选择（按方差阈值）
- [x] explained_variance_ratio 计算
- [x] 累积方差解释比例
- [x] 特征载荷解释
- [x] 降维前后维度对比
- [x] 重构误差

### K-means 聚类
- [x] K 值选择（肘部法则）
- [x] 轮廓系数计算
- [x] inertia_ (WCSS) 变化
- [x] 簇中心计算
- [x] 新数据预测
- [x] 簇大小分布

### 流式统计
- [x] 在线均值增量更新
- [x] 在线方差 (Welford 算法)
- [x] 在线分位数近似
- [x] 与批量统计等价性
- [x] 计算复杂度 O(1)

### A/B 测试工程化
- [x] t 检验流程
- [x] SRM 检测（卡方）
- [x] 效应量计算
- [x] 置信区间计算
- [x] 决策规则（launch/continue/reject）
- [x] 早期停止问题识别

### StatLab 集成
- [x] PCA + 聚类流程
- [x] 流式统计监控
- [x] 分层 A/B 测试
- [x] 完整分析输出结构

---

## 数据驱动测试 (pytest.mark.parametrize)

以下测试使用 parametrize 实现数据驱动：

```python
@pytest.mark.parametrize("k,expected_k", [
    (2, 2),
    (3, 3),
    (5, 5),
])
def test_kmeans_with_different_k(k, expected_k):
    # 测试不同 K 值
    pass
```

---

## Fixtures 说明

主要 fixtures 定义在 `conftest.py`：

| Fixture | 数据形状 | 用途 |
|---------|----------|------|
| `simple_2d_data` | (200, 2) | PCA 基础测试 |
| `high_dim_data` | (100, 50) | 高维 PCA 测试 |
| `well_separated_clusters` | (150, 2) + labels | 聚类正例 |
| `overlapping_clusters` | (150, 2) | 聚类边界 |
| `streaming_data` | (1000,) | 流式统计测试 |
| `streaming_data_with_drift` | (1000,) | 均值漂移测试 |
| `ab_test_data_significant` | 400 rows, 2 groups | A/B 测试正例 |
| `ab_test_data_no_effect` | 400 rows, 2 groups | A/B 测试无效应 |
| `ab_test_data_with_srm` | 500 rows, imbalanced | SRM 检测 |
| `good_*_report` | 文本 | AI 报告审查正例 |
| `bad_*_report` | 文本 | AI 报告审查反例 |

---

## 已知问题和限制

1. **概率性测试**：部分测试依赖随机数，可能偶尔失败（如 FPR 测试）
2. **数值精度**：浮点比较使用 1e-10 容差
3. **scipy 版本**：空组 t 检验行为在不同 scipy 版本可能不同
4. **sklearn 警告**：常数数据会产生警告（正常）

---

## 后续改进方向

1. 添加更多真实数据集测试
2. 增加性能基准测试
3. 添加可视化输出测试（检查图表生成）
4. 增加更多 AI 报告审查案例
