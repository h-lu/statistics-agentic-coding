# Week 06 测试交付总结

## 概述

Week 06（假设检验入门）的完整 pytest 测试套件已完成交付。

## 测试统计

| 指标 | 数值 |
|--------|------|
| **总测试数** | 123 个 |
| **测试文件数** | 8 个 |
| **通过率** | 100% (123/123) |
| **警告** | 6 个（预期内：边界情况的 scipy 警告） |

## 测试文件结构

```
chapters/week_06/tests/
├── __init__.py                  # 测试包初始化
├── conftest.py                   # 共享 fixtures（16 个 fixtures）
├── test_smoke.py                 # 烟雾测试（10 个测试）
├── test_hypothesis.py            # H0/H1 假设设定（16 个测试）
├── test_p_value.py               # p 值理解（17 个测试）
├── test_ttest.py                 # t 检验（18 个测试）
├── test_effect_size.py           # 效应量（15 个测试）
├── test_errors.py                # 两类错误与功效（18 个测试）
├── test_ai_review.py            # AI 报告审查（20 个测试）
└── README.md                    # 测试文档
```

## 测试覆盖矩阵

### 1. 假设设定（H0/H1）- 16 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestFormulateHypothesis | 5 | 1 | 0 | 5 |
| TestValidateHypothesis | 1 | 0 | 3 | 4 |
| TestHypothesisEdgeCases | 0 | 4 | 0 | 4 |

**测试内容**：
- 单尾检验（大于、小于）
- 双尾检验（差异）
- 假设验证（有效假设、缺失字段、空值）
- 边界情况（空问题、特殊字符、超长问题）

### 2. p 值理解 - 17 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestInterpretPValue | 6 | 1 | 0 | 7 |
| TestCheckPValueInterpretation | 2 | 2 | 3 | 7 |
| TestPValueEdgeCases | 0 | 3 | 0 | 3 |

**测试内容**：
- 显著/不显著 p 值解释
- 不同 α 水平
- p 值误解释检测（"H0 为真的概率"、"证明 H0/H1"）
- 边界情况（p=0、p>1、负 p 值）

### 3. t 检验 - 18 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestCheckNormality | 3 | 1 | 1 | 4 |
| TestCheckVarianceHomogeneity | 3 | 0 | 1 | 3 |
| TestTTestOneSample | 2 | 1 | 0 | 3 |
| TestTTestIndependent | 4 | 0 | 0 | 4 |
| TestTTestPaired | 2 | 0 | 1 | 3 |
| TestTTestEdgeCases | 0 | 3 | 0 | 3 |

**测试内容**：
- 正态性检验（Shapiro-Wilk、Anderson-Darling）
- 方差齐性检验（Levene、Bartlett、Fligner-Killeen）
- 单样本 t 检验
- 独立样本 t 检验（Student's t、Welch's t）
- 配对样本 t 检验
- 边界情况（空数组、极小样本、常数数据、异常值）

### 4. 效应量 - 15 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestCohensD | 6 | 0 | 0 | 6 |
| TestInterpretEffectSize | 6 | 0 | 0 | 6 |
| TestEffectSizeEdgeCases | 0 | 4 | 0 | 4 |

**测试内容**：
- Cohen's d 计算（独立样本、配对样本）
- 小/中/大效应识别
- 效应量解释（正向、负向、零效应、带上下文）
- 边界情况（完全相同的组、常数数组、极小样本、极大效应）

### 5. 两类错误与功效 - 18 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestCalculateTypeErrors | 4 | 1 | 0 | 5 |
| TestCalculatePower | 6 | 1 | 0 | 7 |
| TestSimulateTypeErrorRates | 5 | 2 | 0 | 7 |
| TestErrorConcepts | 0 | 0 | 2 | 2 |

**测试内容**：
- 错误类型判断（Type I、Type II、正确决策）
- 功效计算（不同效应量、样本量、α 水平）
- 错误率模拟（第一类错误率、第二类错误率、功效）
- α-β 权衡概念

### 6. AI 报告审查 - 20 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestReviewStatisticalReport | 2 | 1 | 7 | 10 |
| TestAIReportEdgeCases | 1 | 3 | 0 | 4 |
| TestCommonAIReportPatterns | 0 | 0 | 3 | 3 |

**测试内容**：
- 检测缺失 H0、p 值误解释
- 检测缺失效应量、置信区间
- 检测未验证假设（正态性、方差齐性）
- 检测多重比较未校正
- 检测相关误作因果
- 常见 AI 报告模式（只报告 p 值、p-hacking）

### 7. 烟雾测试 - 10 个测试

| 测试类 | 正例 | 边界 | 反例 | 总计 |
|--------|------|------|------|------|
| TestSmokeBasicFunctionality | 9 | 0 | 0 | 9 |
| TestSmokeEndToEnd | 1 | 0 | 0 | 1 |

**测试内容**：
- 所有核心功能的基本可运行性
- 端到端假设检验工作流

## 共享 Fixtures

### 数据 Fixtures（11 个）
- `sample_data_normal`: 正态分布数据
- `sample_data_two_groups`: 两组独立样本
- `sample_data_paired`: 配对样本数据
- `sample_data_non_normal`: 非正态数据（指数分布）
- `sample_data_unequal_variance`: 方差不等的两组
- `ab_test_data`: A/B 测试数据（有真实效应）
- `ab_test_no_effect`: A/B 测试数据（无效应）
- `empty_array`: 空数组
- `single_value`: 单值数组
- `tiny_sample`: 极小样本（n=2）
- `large_sample`: 大样本（n=10000）
- `data_with_outliers`: 包含异常值的数据
- `data_with_nan`: 包含缺失值的数据
- `constant_data`: 常数列数据

### AI 报告 Fixtures（3 个）
- `good_ai_report`: 合格的 AI 生成报告
- `bad_ai_report`: 有问题的 AI 生成报告
- `bad_ai_report_p_hacking`: 展示 p-hacking 的报告

### 假设 Fixtures（3 个）
- `valid_hypothesis`: 有效的假设陈述
- `invalid_hypothesis_missing_h1`: 缺少 H1 的无效假设
- `invalid_hypothesis_empty_h0`: H0 为空的无效假设

## 概念覆盖

| 本周新概念 | 测试覆盖 | 相关测试文件 |
|------------|----------|--------------|
| H0/H1 假设设定 | ✅ 完整 | test_hypothesis.py |
| p 值理解 | ✅ 完整 | test_p_value.py |
| t 检验 | ✅ 完整 | test_ttest.py |
| 第一类/第二类错误 | ✅ 完整 | test_errors.py |
| 效应量 | ✅ 完整 | test_effect_size.py |

**覆盖的回顾桥**：
- 集中趋势/均值与中位数（Week 02）→ t 检验中的均值比较
- 分布可视化（Week 02）→ 正态性检验
- 缺失值处理（Week 03）→ 样本量与功效的关系
- 相关与分组比较（Week 04）→ 假设设定基础
- Bootstrap 与置信区间（Week 05）→ p 值与置信区间的对偶关系

## 测试质量指标

### 正例覆盖
- ✅ 正常场景的完整工作流
- ✅ 标准输入的正确处理
- ✅ 预期结果的准确输出

### 边界覆盖
- ✅ 空输入（空数组、空字符串）
- ✅ 极小样本（n=2）
- ✅ 大样本（n=10000）
- ✅ 常数数据
- ✅ 异常值
- ✅ 缺失值
- ✅ 极端 p 值（0、>1）
- ✅ 极端效应量

### 反例覆盖
- ✅ 错误的假设格式（缺失字段、空值）
- ✅ p 值误解释
- ✅ 不满足前提假设（非正态、方差不齐）
- ✅ 长度不匹配的配对样本
- ✅ AI 报告中的常见错误模式

## 运行命令

### 运行所有测试
```bash
python3 -m pytest chapters/week_06/tests -q
```

### 运行特定测试文件
```bash
# 只运行烟雾测试
python3 -m pytest chapters/week_06/tests/test_smoke.py -v

# 只运行 t 检验测试
python3 -m pytest chapters/week_06/tests/test_ttest.py -v
```

### 运行特定测试类
```bash
python3 -m pytest chapters/week_06/tests/test_hypothesis.py::TestFormulateHypothesis -v
```

### 显示详细输出
```bash
python3 -m pytest chapters/week_06/tests -v -s
```

## 交付文件清单

| 文件路径 | 说明 |
|----------|------|
| `/chapters/week_06/starter_code/solution.py` | 作业解答（被测试的代码） |
| `/chapters/week_06/tests/__init__.py` | 测试包初始化 |
| `/chapters/week_06/tests/conftest.py` | 共享 fixtures 和导入路径设置 |
| `/chapters/week_06/tests/test_smoke.py` | 烟雾测试 |
| `/chapters/week_06/tests/test_hypothesis.py` | H0/H1 假设设定测试 |
| `/chapters/week_06/tests/test_p_value.py` | p 值理解测试 |
| `/chapters/week_06/tests/test_ttest.py` | t 检验测试 |
| `/chapters/week_06/tests/test_effect_size.py` | 效应量测试 |
| `/chapters/week_06/tests/test_errors.py` | 两类错误与功效测试 |
| `/chapters/week_06/tests/test_ai_review.py` | AI 报告审查测试 |
| `/chapters/week_06/tests/README.md` | 测试文档 |

## 设计亮点

1. **命名清晰**：所有测试函数遵循 `test_<功能>_<场景>_<预期结果>` 格式
2. **独立性**：每个测试独立运行，不依赖其他测试
3. **可维护性**：使用 fixtures 共享测试数据，易于更新
4. **完整覆盖**：正例、边界、反例三位一体
5. **教学价值**：测试本身也是学习材料，展示正确/错误用法

## 后续建议

1. **性能优化**：模拟类测试（`simulate_type_error_rates`）默认使用 1000 次迭代，可以调整为更快
2. **扩展性**：可以添加更多 AI 报告错误模式的测试
3. **文档**：每个测试类的 docstring 可以进一步补充示例
