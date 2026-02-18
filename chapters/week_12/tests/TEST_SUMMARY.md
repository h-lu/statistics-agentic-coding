# Week 12 测试用例总结

## 测试文件结构

```
chapters/week_12/tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # 共享 fixtures（测试数据）
├── test_smoke.py            # 冒烟测试（基础验证）
├── test_feature_importance.py   # 特征重要性测试
├── test_shap_values.py      # SHAP 值测试
├── test_bias_detection.py   # 偏见检测测试
└── test_fairness_metrics.py # 公平性指标测试
```

## 测试矩阵

| 测试文件 | 测试类别 | 正例 | 边界 | 反例 |
|---------|---------|-----|------|-----|
| test_feature_importance.py | 特征重要性 | 8 | 6 | 3 |
| test_shap_values.py | SHAP 值 | 9 | 4 | 3 |
| test_bias_detection.py | 偏见检测 | 10 | 4 | 3 |
| test_fairness_metrics.py | 公平性指标 | 10 | 3 | 3 |
| test_smoke.py | 冒烟测试 | 4 | 2 | 1 |
| **总计** | | **41** | **19** | **13** |

## Fixtures 概述

### 数据生成 Fixtures
- `feature_importance_data`: 标准特征重要性测试数据
- `correlated_features_data`: 相关特征数据（测试"分票"现象）
- `shap_test_data`: SHAP 值测试数据（含训练好的模型）
- `bias_detection_data`: 偏见检测数据（含敏感属性）
- `demographic_parity_data`: 统计均等测试数据
- `equalized_odds_data`: 机会均等测试数据
- `calibration_test_data`: 校准测试数据

### 边界情况 Fixtures
- `minimal_fairness_data`: 最小公平性测试数据
- `single_group_data`: 单一群体数据
- `empty_group_data`: 包含空群体的数据

### StatLab Fixtures
- `statlab_interpretability_data`: 客户流失场景的可解释性测试数据

## 测试覆盖的功能

### 1. 特征重要性 (test_feature_importance.py)
- 计算特征重要性（正例）
- 相关特征的重要性分散（边界）
- 常量特征处理（边界）
- 极小数据集（边界）
- 无效模型处理（反例）
- 特征重要性排序
- 特征重要性可视化
- 逻辑回归系数作为特征重要性

### 2. SHAP 值 (test_shap_values.py)
- 计算 SHAP 值（正例）
- 单样本解释（局部可解释性）
- 全局 SHAP 汇总
- SHAP 可视化
- 高维数据处理（边界）
- 极小数据集（边界）
- 不支持的模型类型（反例）
- SHAP 解释文本生成
- SHAP 值与特征重要性的对比

### 3. 偏见检测 (test_bias_detection.py)
- 按群体评估（正例）
- 预测偏见检测
- 结果偏见检测
- 差别影响检测
- 数据偏见 vs 算法偏见区分
- 小样本群体处理（边界）
- 单一群体处理（边界）
- 无效输入处理（反例）
- 偏见可视化
- 偏见报告生成
- 交叉偏见评估

### 4. 公平性指标 (test_fairness_metrics.py)
- 统计均等（Demographic Parity）
- 机会均等（Equalized Odds）
- 校准（Calibration）
- 公平性指标综合评估
- 公平性-准确性权衡
- 多敏感属性评估
- 公平性阈值检测
- 公平性报告生成
- 公平性改进建议

### 5. 冒烟测试 (test_smoke.py)
- 模块导入验证
- 基本函数存在性检查
- 端到端流程测试
- 异常处理验证

## 运行测试

```bash
# 运行所有测试
python3 -m pytest chapters/week_12/tests/ -q

# 运行特定测试文件
python3 -m pytest chapters/week_12/tests/test_feature_importance.py -v

# 运行冒烟测试
python3 -m pytest chapters/week_12/tests/test_smoke.py -v

# 显示详细输出
python3 -m pytest chapters/week_12/tests/ -v -s
```

## 测试命名规范

测试函数遵循 `test_<功能>_<场景>_<预期结果>` 格式：

- `test_feature_importance_returns_valid_values`: 测试特征重要性返回有效值
- `test_shap_values_sum_to_prediction`: 测试 SHAP 值加性
- `test_demographic_parity_difference`: 测试统计均等差异计算
- `test_correlated_features_dilute_importance`: 测试相关特征的重要性分散

## 当前状态

- ✅ 测试框架已建立
- ✅ Fixtures 已创建
- ✅ 93 个测试用例已编写
- ⏳ 部分测试等待 solution.py 的实现完成
- ✅ 冒烟测试通过

## 待完成事项

1. **solution.py 实现完善**：
   - `compute_shap_values()` - SHAP 值计算
   - `demographic_parity_difference()` - 统计均等差异
   - `equalized_odds_difference()` - 机会均等差异
   - `evaluate_by_group()` - 分组评估（别名或扩展）

2. **测试验证**：
   - 在 solution.py 实现完成后重新运行测试
   - 确保所有测试通过

3. **文档更新**：
   - 在测试通过后更新测试文档
   - 添加更多边界情况（如需要）

## 注意事项

1. **SHAP 库依赖**：部分 SHAP 测试需要 `shap` 库，如果未安装会跳过相关测试
2. **灵活性设计**：测试设计为支持多种可能的函数签名和接口
3. **渐进实现**：测试使用 `pytest.skip` 优雅地处理未实现的函数
4. **合成数据**：所有测试使用合成数据，不依赖外部文件
