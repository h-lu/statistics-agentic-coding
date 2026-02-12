# Week 06 测试文档

## 概述

本目录包含 Week 06（假设检验入门）的所有测试用例。

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # 共享 fixtures
├── test_smoke.py            # 烟雾测试（快速验证基本功能）
├── test_hypothesis.py       # H0/H1 假设设定测试
├── test_p_value.py          # p 值理解测试
├── test_ttest.py            # t 检验测试
├── test_effect_size.py      # 效应量测试
├── test_errors.py           # 两类错误与功效测试
├── test_ai_review.py       # AI 报告审查测试
└── README.md               # 本文档
```

## 运行测试

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

### 运行特定测试类或函数

```bash
# 运行单个测试类
python3 -m pytest chapters/week_06/tests/test_hypothesis.py::TestFormulateHypothesis -v

# 运行单个测试
python3 -m pytest chapters/week_06/tests/test_p_value.py::TestInterpretPValue::test_interpret_significant_p_value -v
```

### 显示详细输出

```bash
# 显示打印输出
python3 -m pytest chapters/week_06/tests -v -s

# 显示更详细的错误信息
python3 -m pytest chapters/week_06/tests -vv
```

### 运行标记的测试

```bash
# 运行快速测试（如果有标记）
python3 -m pytest chapters/week_06/tests -m "not slow"
```

## 测试覆盖范围

### 1. test_hypothesis.py - 假设设定

- `TestFormulateHypothesis`: 测试将研究问题转化为 H0/H1
  - 单尾检验（大于、小于）
  - 双尾检验（差异）
  - 默认情况

- `TestValidateHypothesis`: 测试假设验证
  - 有效假设
  - 缺少 H1
  - 空 H0
  - H0 不包含"相等"概念

- `TestHypothesisEdgeCases`: 边界情况
  - 空问题
  - 特殊字符
  - 超长问题

### 2. test_p_value.py - p 值理解

- `TestInterpretPValue`: p 值解释
  - 显著 p 值
  - 不显著 p 值
  - 不同 α 水平
  - 返回值结构

- `TestCheckPValueInterpretation`: 识别误解释
  - 正确解释
  - "H0 为真的概率"误解释
  - "证明 H0/H1"误解释
  - 积极信号检测

- `TestPValueEdgeCases`: 边界情况
  - p=0
  - p>1
  - 负 p 值

### 3. test_ttest.py - t 检验

- `TestCheckNormality`: 正态性检验
  - Shapiro-Wilk 检验
  - Anderson-Darling 检验
  - 大样本处理

- `TestCheckVarianceHomogeneity`: 方差齐性检验
  - Levene 检验
  - Bartlett 检验
  - Fligner-Killeen 检验

- `TestTTestOneSample`: 单样本 t 检验
  - 基本功能
  - 包含正态性检查

- `TestTTestIndependent`: 独立样本 t 检验
  - 等方差（Student's t）
  - 不等方差（Welch's t）
  - 包含假设检查

- `TestTTestPaired`: 配对样本 t 检验
  - 基本功能
  - 长度不匹配错误

- `TestTTestEdgeCases`: 边界情况
  - 极小样本
  - 常数数据
  - 异常值

### 4. test_effect_size.py - 效应量

- `TestCohensD`: Cohen's d 计算
  - 独立样本
  - 配对样本
  - 小/中/大效应
  - 零效应

- `TestInterpretEffectSize`: 效应量解释
  - 不同大小的效应
  - 负向效应
  - 带上下文解释

- `TestEffectSizeEdgeCases`: 边界情况
  - 完全相同的组
  - 常数数组
  - 极小样本

### 5. test_errors.py - 两类错误与功效

- `TestCalculateTypeErrors`: 错误类型判断
  - 第一类错误（假阳性）
  - 第二类错误（假阴性）
  - 正确决策

- `TestCalculatePower`: 功效计算
  - 不同效应量
  - 样本量影响
  - α 水平影响

- `TestSimulateTypeErrorRates`: 错误率模拟
  - 第一类错误率
  - 第二类错误率
  - 功效随样本量/效应量变化

- `TestErrorConcepts`: 错误概念理解
  - α-β 权衡
  - 功效定义

### 6. test_ai_review.py - AI 报告审查

- `TestReviewStatisticalReport`: 报告审查
  - 检测缺失 H0
  - 检测 p 值误解释
  - 检测缺失效应量
  - 检测缺失置信区间
  - 检测未验证假设
  - 检测多重比较未校正
  - 检测相关误作因果

- `TestAIReportEdgeCases`: 边界情况
  - 空报告
  - 完美报告
  - p-hacking 证据

- `TestCommonAIReportPatterns`: 常见模式
  - 只报告 p 值
  - 统计显著但实际意义不大
  - p-hacking 痕迹

### 7. test_smoke.py - 烟雾测试

- `TestSmokeBasicFunctionality`: 基本功能测试
- `TestSmokeEndToEnd`: 端到端工作流测试

## 共享 Fixtures（conftest.py）

### 数据 Fixtures

- `sample_data_normal`: 正态分布数据
- `sample_data_two_groups`: 两组独立样本
- `sample_data_paired`: 配对样本数据
- `sample_data_non_normal`: 非正态数据（指数分布）
- `sample_data_unequal_variance`: 方差不等的两组

### A/B 测试 Fixtures

- `ab_test_data`: A/B 测试数据（有真实效应）
- `ab_test_no_effect`: A/B 测试数据（无效应）

### 边界情况 Fixtures

- `empty_array`: 空数组
- `single_value`: 单值数组
- `tiny_sample`: 极小样本（n=2）
- `large_sample`: 大样本（n=10000）
- `data_with_outliers`: 包含异常值的数据
- `data_with_nan`: 包含缺失值的数据
- `constant_data`: 常数列数据

### AI 报告 Fixtures

- `good_ai_report`: 合格的 AI 生成报告
- `bad_ai_report`: 有问题的 AI 生成报告
- `bad_ai_report_p_hacking`: 展示 p-hacking 的报告

### 假设 Fixtures

- `valid_hypothesis`: 有效的假设陈述
- `invalid_hypothesis_missing_h1`: 缺少 H1 的无效假设
- `invalid_hypothesis_empty_h0`: H0 为空的无效假设

## 测试统计

- **总测试数**: 约 90+ 个测试用例
- **覆盖概念**: 5 个（H0/H1、p 值、t 检验、效应量、两类错误）
- **测试类别**: 正例、边界、反例

## 开发说明

### 添加新测试

1. 在相应的测试文件中添加新的测试类或测试函数
2. 使用 `conftest.py` 中的现有 fixtures 或创建新的
3. 遵循命名约定：`test_<功能>_<场景>_<预期结果>`
4. 运行测试确保通过

### 测试命名约定

- 类名：`Test<功能名>`（如 `TestTTestIndependent`）
- 方法名：`test_<场景>_<条件>_<预期结果>`
  - `test_t_test_equal_variance_passes`
  - `test_cohens_d_small_effect`
  - `test_detect_missing_effect_size`

## 常见问题

### Q: 测试失败怎么办？

1. 检查是测试代码问题还是 solution.py 实现问题
2. 如果 solution.py 未实现，在测试中添加注释说明
3. 修复问题后重新运行测试

### Q: 如何调试特定测试？

```bash
# 在失败时进入 pdb 调试器
python3 -m pytest chapters/week_06/tests/test_ttest.py::TestTTestOneSample::test_one_sample_with_true_mean --pdb
```

### Q: 测试太慢怎么办？

只运行烟雾测试：
```bash
python3 -m pytest chapters/week_06/tests/test_smoke.py
```

## 更新日志

- 2026-02-12: 初始版本，包含完整的 Week 06 测试套件
