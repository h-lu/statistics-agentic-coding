# Week 12 测试框架

## 文件清单

| 文件 | 行数 | 描述 |
|------|------|------|
| `__init__.py` | 7 | 测试包初始化 |
| `conftest.py` | ~450 | 共享 fixtures（数据、代码示例） |
| `test_smoke.py` | ~230 | 基本功能验证 |
| `test_shap_explainability.py` | ~290 | SHAP 可解释性测试 |
| `test_fairness_metrics.py` | ~380 | 公平性指标测试 |
| `test_edge_cases.py` | ~480 | 边界用例测试 |
| `test_differential_privacy.py` | ~470 | 差分隐私与伦理审查 |
| `test_week12_integration.py` | ~500 | 端到端集成测试 |

**总计**: ~2800 行测试代码

## 测试覆盖矩阵

| 主题 | 正例 | 边界 | 反例 | 集成 |
|------|------|------|------|------|
| SHAP 可解释性 | ✓ | ✓ | ✓ | ✓ |
| 公平性指标 | ✓ | ✓ | ✓ | ✓ |
| 差分隐私 | ✓ | ✓ | ✓ | ✓ |
| 伦理审查 | ✓ | ✓ | ✓ | ✓ |

## 核心功能测试

### 1. SHAP 值计算与解释
- 全局解释（特征重要性排序）
- 局部解释（单个样本的推理路径）
- 可加性验证
- 分类 vs 回归任务

### 2. 公平性指标
- 差异影响比（Disparate Impact, 80% 规则）
- 平等机会（Equal Opportunity, 召回率差异）
- 均等几率（Equalized Odds, TPR + FPR）
- 代理变量检测（相关性分析）

### 3. 差分隐私
- 拉普拉斯噪声添加
- epsilon 和 sensitivity 影响
- 隐私预算跟踪
- 边界情况（负值、零 epsilon）

### 4. 伦理审查
- 审查清单生成
- 向不同受众解释（客户/产品经理/合规）
- 代码质量审查
- 完整审计流程

## 运行测试

```bash
# 所有测试
python3 -m pytest chapters/week_12/tests -v

# 特定类别
python3 -m pytest chapters/week_12/tests/test_shap_explainability.py -v

# 带输出
python3 -m pytest chapters/week_12/tests -vv -s
```

## 依赖

测试框架依赖 `solution.py` 中的以下函数：

```python
# SHAP
calculate_shap_values(model, X)
explain_single_prediction(model, sample)
calculate_feature_importance_shap(model, X)

# 公平性
calculate_disparate_impact(y_pred, group_labels)
calculate_equal_opportunity(y_true, y_pred, group_labels)
calculate_equalized_odds(y_true, y_pred, group_labels)
detect_proxy_variables(df, sensitive_col, method='correlation')

# 差分隐私
add_differential_privacy_noise(data, epsilon, sensitivity)
check_privacy_budget(epsilon_or_epsilons, total_budget=1.0)

# 伦理审查
create_ethics_checklist(**kwargs)
explain_to_nontechnical(explanation, audience='customer', **kwargs)
review_xai_code(code)
```

详细测试用例说明见 `TEST_MATRIX.md`
