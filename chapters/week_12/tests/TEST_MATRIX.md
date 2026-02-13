# Week 12 测试用例矩阵

## 概览

Week 12 测试框架覆盖可解释 AI（SHAP）、公平性指标、差分隐私和伦理审查四大主题。

```
总测试文件数: 6
总测试用例数: ~120+
```

---

## 测试文件结构

| 文件 | 测试类数 | 测试用例数 | 覆盖主题 |
|------|----------|-----------|---------|
| `test_smoke.py` | 2 | ~15 | 基本功能验证 |
| `test_shap_explainability.py` | 4 | ~12 | SHAP 值计算与解释 |
| `test_fairness_metrics.py` | 6 | ~25 | 公平性指标计算 |
| `test_edge_cases.py` | 10 | ~30 | 边界情况处理 |
| `test_differential_privacy.py` | 6 | ~20 | 差分隐私与伦理审查 |
| `test_week12_integration.py` | 5 | ~18 | 端到端集成测试 |

---

## 按功能分类的测试覆盖

### 1. SHAP 可解释性（XAI）

#### 正例（Happy Path）
| 测试用例 | 描述 | 预期结果 |
|---------|------|---------|
| `test_shap_values_returns_array` | SHAP 值返回正确格式 | 返回 numpy 数组或列表 |
| `test_shap_values_global_importance_ranking` | 正确排序特征重要性 | 重要特征排在前 |
| `test_shap_values_additivity_property` | 验证可加性 | SHAP 值之和接近预测值 |
| `test_explain_single_prediction_returns_dict` | 单个预测解释返回字典 | 包含 base_value、shap_values 等 |
| `test_credit_scoring_shap_values` | 真实信用评分场景 | 成功计算 SHAP 值 |

#### 边界用例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_shap_with_empty_dataframe` | 空数据框 | 优雅处理或报错 |
| `test_shap_with_single_sample` | 单个样本 | 能计算或优雅失败 |
| `test_shap_with_single_feature` | 单个特征 | 能计算 SHAP 值 |
| `test_shap_with_negative_feature_values` | 包含负值的特征 | 能正常处理 |
| `test_shap_with_outliers` | 包含离群点 | 能正常处理 |
| `test_shap_with_constant_feature` | 常数特征 | SHAP 值接近 0 |
| `test_shap_with_imbalanced_data` | 极度不平衡数据 | 能正常处理 |

#### 反例（错误场景）
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_review_bad_shap_wrong_explainer` | 使用错误的 Explainer | 检测到问题 |
| `test_review_bad_shap_global_only` | 只有全局解释 | 检测到缺少局部解释 |

---

### 2. 公平性指标

#### 正例
| 测试用例 | 描述 | 预期结果 |
|---------|------|---------|
| `test_disparate_impact_perfect_equality` | 完全相等 | DI ratio = 1.0 |
| `test_disparate_impact_80_percent_rule` | 80% 规则场景 | DI ratio ≈ 0.8 |
| `test_equal_opportunity_perfect_equality` | 完全平等的召回率 | 召回率差异 = 0 |
| `test_equalized_odds_perfect` | 完美的均等几率 | TPR 和 FPR 都相等 |
| `test_detect_strong_proxy` | 强代理变量检测 | 成功检测 |

#### 边界用例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_disparate_impact_edge_cases` | 全通过/全不通过 | 返回 NaN 或 0/0 处理 |
| `test_equal_opportunity_with_single_class` | 只有一个类别 | 优雅处理 |
| `test_fairness_metrics_with_small_group` | 极小样本组 | 能计算（虽不稳定）|
| `test_detect_proxy_with_no_sensitive_col` | 缺少敏感列 | 优雅处理或报错 |

#### 反例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_review_bad_fairness_no_group_analysis` | 缺少群体分析 | 检测到问题 |

---

### 3. 差分隐私

#### 正例
| 测试用例 | 描述 | 预期结果 |
|---------|------|---------|
| `test_add_noise_returns_array` | 加噪返回数组 | 返回 numpy 数组 |
| `test_add_noise_preserves_length` | 保持数据长度 | 长度不变 |
| `test_add_noise_laplace_distribution` | 噪声来自拉普拉斯分布 | 噪声均值接近 0 |
| `test_add_noise_epsilon_impact` | epsilon 影响噪声大小 | 小 epsilon → 大噪声 |
| `test_add_noise_sensitivity_impact` | sensitivity 影响噪声大小 | 大 sensitivity → 大噪声 |

#### 边界用例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_add_noise_with_negative_values` | 包含负值 | 能正常处理 |
| `test_differential_privacy_zero_epsilon` | epsilon = 0 | 噪声最大 |
| `test_differential_privacy_single_value` | 单个值 | 能计算 |
| `test_differential_privacy_all_same_values` | 所有值相同 | 加噪后不再相同 |

#### 反例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_check_privacy_budget_exceeded` | 超过预算 | 检测到超预算 |

---

### 4. 伦理审查

#### 正例
| 测试用例 | 描述 | 预期结果 |
|---------|------|---------|
| `test_create_checklist_returns_dict` | 创建审查清单 | 返回字典或列表 |
| `test_create_checklist_has_all_categories` | 包含所有风险类别 | 7 大类别齐全 |
| `test_explain_to_customer` | 向客户解释 | 避免技术术语 |
| `test_explain_to_product_manager` | 向产品经理解释 | 清晰易懂 |
| `test_explain_to_compliance` | 向合规部门解释 | 包含公平性/隐私指标 |

#### 边界用例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_explain_with_recommendations` | 包含建议 | 提供建设性建议 |
| `test_handle_invalid_audience_in_explanation` | 无效受众 | 使用默认受众 |

#### 反例
| 测试用例 | 描述 | 预期行为 |
|---------|------|---------|
| `test_review_bad_explanation_code` | 技术术语过多 | 检测到问题 |

---

## 按测试类型分类

### 烟雾测试（Smoke Tests）
快速验证核心功能是否能运行。

- 位置：`test_smoke.py`
- 数量：~15 个测试
- 覆盖：所有主要函数的基本调用

### 单元测试（Unit Tests）
测试单个功能点。

- 位置：所有测试文件
- 数量：~80 个测试
- 覆盖：SHAP、公平性、差分隐私、伦理审查

### 边界测试（Edge Case Tests）
测试极端情况和异常输入。

- 位置：`test_edge_cases.py`
- 数量：~30 个测试
- 覆盖：空数据、单样本、负值、离群点、不平衡数据

### 集成测试（Integration Tests）
测试端到端工作流。

- 位置：`test_week12_integration.py`
- 数量：~18 个测试
- 覆盖：完整 XAI 工作流、公平性审计、伦理审查

---

## 测试命名规范

遵循格式：`test_<功能>_<场景>_<预期结果>`

示例：
- `test_shap_values_returns_array` → 测试 SHAP 值返回数组
- `test_disparate_impact_80_percent_rule` → 测试 80% 规则场景
- `test_explain_to_customer` → 测试向客户解释
- `test_shap_with_empty_dataframe` → 测试空数据框场景

---

## 测试数据 Fixtures

### 核心数据集
| Fixture | 描述 | 行数 | 用途 |
|---------|------|------|------|
| `credit_scoring_data` | 信用评分数据 | 1000 | 贯穿案例 |
| `credit_scoring_data_with_proxy` | 包含代理变量 | 1000 | 代理变量检测 |
| `biased_credit_data` | 有偏见的数据 | 1000 | 公平性测试 |
| `fair_classification_data` | 相对公平的数据 | 1000 | 验证公平性指标 |
| `unfair_classification_data` | 不公平的数据 | 1000 | 检测偏见 |

### 边界测试数据
| Fixture | 描述 | 用途 |
|---------|------|------|
| `empty_dataframe` | 空数据框 | 空数据测试 |
| `single_sample_data` | 单个样本 | 极小数据测试 |
| `single_feature_data` | 单特征 | 特征数量边界 |
| `data_with_negative_values` | 包含负值 | 负值处理 |
| `data_with_outliers` | 包含离群点 | 离群点处理 |
| `constant_feature_data` | 常数特征 | 无信息特征 |
| `highly_imbalanced_data` | 极度不平衡 | 类别不平衡 |

### 代码示例 Fixtures
| Fixture | 描述 | 用途 |
|---------|------|------|
| `good_shap_code_example` | 合格的 SHAP 代码 | 代码审查 |
| `bad_shap_code_wrong_explainer` | 错误的 Explainer | 问题检测 |
| `bad_shap_code_global_only` | 只有全局解释 | 问题检测 |
| `good_fairness_code_example` | 合格的公平性代码 | 代码审查 |
| `bad_fairness_code_no_group_analysis` | 缺少群体分析 | 问题检测 |
| `good_explanation_for_nontechnical` | 好的非技术解释 | 解释质量 |
| `bad_explanation_technical_jargon` | 技术术语过多 | 问题检测 |

---

## 待实现函数清单

测试框架期望 `solution.py` 实现以下函数：

### SHAP 相关
```python
def calculate_shap_values(model, X):
    """计算 SHAP 值"""

def explain_single_prediction(model, sample):
    """解释单个预测"""

def calculate_feature_importance_shap(model, X):
    """基于 SHAP 计算特征重要性"""
```

### 公平性相关
```python
def calculate_disparate_impact(y_pred, group_labels):
    """计算差异影响比"""

def calculate_equal_opportunity(y_true, y_pred, group_labels):
    """计算平等机会差异"""

def calculate_equalized_odds(y_true, y_pred, group_labels):
    """计算均等几率"""

def detect_proxy_variables(df, sensitive_col, method='correlation'):
    """检测代理变量"""
```

### 差分隐私相关
```python
def add_differential_privacy_noise(data, epsilon, sensitivity):
    """添加差分隐私噪声"""

def check_privacy_budget(epsilon_or_epsilons, total_budget=1.0):
    """检查隐私预算"""
```

### 伦理审查相关
```python
def create_ethics_checklist(**kwargs):
    """创建伦理审查清单"""

def explain_to_nontechnical(explanation, audience='customer', **kwargs):
    """向非技术人员解释"""

def review_xai_code(code):
    """审查 XAI 代码"""
```

---

## 运行测试

```bash
# 运行所有 Week 12 测试
python3 -m pytest chapters/week_12/tests -v

# 运行特定测试文件
python3 -m pytest chapters/week_12/tests/test_shap_explainability.py -v

# 运行特定测试类
python3 -m pytest chapters/week_12/tests/test_edge_cases.py::TestEmptyAndSmallData -v

# 运行特定测试用例
python3 -m pytest chapters/week_12/tests/test_fairness_metrics.py::TestDisparateImpact::test_disparate_impact_perfect_equality -v

# 显示详细输出
python3 -m pytest chapters/week_12/tests -vv -s
```

---

## 下一步

1. 实现 `starter_code/solution.py` 中的函数
2. 运行测试验证实现正确性
3. 根据测试失败情况调整实现或测试
4. 确保所有测试通过后，Week 12 章包可发布
