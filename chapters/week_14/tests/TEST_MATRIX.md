# Week 14 测试矩阵

本文档说明 Week 14（贝叶斯推断）的测试设计。

## 测试文件结构

```
tests/
├── conftest.py                 # fixtures 和共享配置
├── test_prior.py                # 先验相关测试
├── test_posterior.py            # 后验计算测试
├── test_mcmc.py                # MCMC 采样测试
├── test_hierarchical.py          # 层次模型测试
├── test_statlab.py              # StatLab 集成测试
└── test_smoke.py               # 基础冒烟测试
```

## 核心概念测试矩阵

### 1. 先验（Prior）

| 测试类型 | 测试数量 | 测试内容 |
|---------|---------|---------|
| 正例 | 3 | 无信息先验、弱信息先验、共轭先验 |
| 边界 | 2 | 极端弱信息、Jeffreys 先验 |
| 反例 | 2 | 负参数、零方差退化 |

**关键测试**：
- `test_create_uniform_prior_beta_1_1`: 验证 Beta(1,1) 均匀分布
- `test_create_conjugate_prior_beta_binomial`: 验证共轭先验
- `test_beta_prior_with_negative_parameters_raises_error`: 验证负参数报错

### 2. 后验（Posterior）

| 测试类型 | 测试数量 | 测试内容 |
|---------|---------|---------|
| 正例 | 3 | Beta-Binomial 后验、参数合理性、统计量计算 |
| 边界 | 5 | 零数据、全成功/全失败、单次观测、不同数据情况 |
| 反例 | 1 | 无效数据（成功次数 > 试验次数） |

**关键测试**：
- `test_beta_binomial_conjugate_posterior`: 验证后验公式正确
- `test_zero_data_posterior_equals_prior`: 验证零数据时后验=先验
- `test_probability_b_better_than_a`: 验证 A/B 测试概率计算

### 3. MCMC 采样

| 测试类型 | 测试数量 | 测试内容 |
|---------|---------|---------|
| 正例 | 5 | 成功采样、R-hat 收敛、trace shape、多变量、确定性变量 |
| 边界 | 3 | 少样本、极端数据、不同采样器 |
| 反例 | 1 | chains=1 无法计算 r_hat |

**关键测试**：
- `test_simple_ab_test_mcmc`: 验证 PyMC 基本采样
- `test_mcmc_convergence_r_hat`: 验证 R-hat < 1.05
- `test_ess_sufficiently_large`: 验证 ESS > 400

**注意**：所有 MCMC 测试使用 `@pytest.mark.skipif` 跳过 PyMC 未安装的情况

### 4. 层次模型（Hierarchical Model）

| 测试类型 | 测试数量 | 测试内容 |
|---------|---------|---------|
| 正例 | 3 | 成功采样、shrinkage 效应、超先验合理 |
| 边界 | 4 | 单组退化、两组最小、相同数据、样本不平衡 |
| 反例 | 0 | （无典型反例，更多是边界情况） |

**关键测试**：
- `test_hierarchical_model_sampling`: 验证基本层次模型
- `test_shrinkage_effect`: 验证小样本组向全局均值收缩
- `test_hierarchical_vs_pooled`: 对比层次模型 vs 合并模型

### 5. StatLab 集成

| 测试类型 | 测试数量 | 测试内容 |
|---------|---------|---------|
| 正例 | 3 | 报告结构、概率陈述、可信区间 |
| 边界 | 3 | 空数据、单次观测、零方差 |
| 反例 | 0 | （更多是边界情况） |

**关键测试**：
- `test_bayesian_report_structure`: 验证报告包含 mean, HDI
- `test_prior_sensitivity_consistent_results`: 验证不同先验结果一致
- `test_full_bayesian_analysis_workflow`: 端到端工作流

## 测试覆盖率

### 代码维度
- **先验类型**：无信息、弱信息、强信息、Jeffreys、共轭先验
- **后验计算**：解析解（Beta-Binomial）、数值解（MCMC）
- **模型类型**：A/B 测试、线性回归、层次模型
- **诊断指标**：R-hat、ESS、trace plot

### 场景维度
- **数据规模**：小样本（n=10）、中等样本（n=100-1000）、大样本（n=10000）
- **数据质量**：正常数据、极端数据（全成功/全失败）、空数据
- **先验强度**：无信息、弱信息、强信息、极端先验

### 概念维度
- **频率 vs 贝叶斯**：置信区间 vs 可信区间
- **先验敏感性**：不同先验下结论稳定性
- **决策友好性**：概率陈述、预期损失
- **信息共享**：层次模型的 shrinkage

## 运行测试

### 运行所有测试
```bash
cd /Users/wangxq/Documents/statistics-agentic-coding
python3 -m pytest chapters/week_14/tests -v
```

### 只运行特定文件
```bash
python3 -m pytest chapters/week_14/tests/test_prior.py -v
python3 -m pytest chapters/week_14/tests/test_mcmc.py -v
```

### 跳过需要 PyMC 的测试
```bash
python3 -m pytest chapters/week_14/tests -v -m "not pymc"
```

### 只运行冒烟测试
```bash
python3 -m pytest chapters/week_14/tests/test_smoke.py -v
```

## Fixtures 说明

### 数据 Fixtures
- `simple_ab_data`: 标准 A/B 测试数据（52/1000 vs 58/1000）
- `small_ab_data`: 小样本 A/B 测试（5/100 vs 8/100）
- `extreme_ab_data`: 极端数据（全成功/全失败）
- `hierarchical_data`: 4 国 A/B 测试（大样本 + 小样本）
- `regression_data`: 线性回归数据（y = 50 + 1.5*x1 + 0.3*x2）

### 配置 Fixtures
- `random_seed`: 固定随机种子（42）
- `priors`: 常用先验配置
- `convergence_thresholds`: R-hat 和 ESS 阈值
- `skip_pymc`: PyMC 可用性检查

## 依赖

### 必需依赖
- `pytest`: 测试框架
- `numpy`: 数值计算
- `scipy`: 统计分布
- `pandas`: 数据处理

### 可选依赖
- `pymc`: MCMC 采样（如未安装，相关测试会被跳过）
- `arviz`: 贝叶斯诊断（如未安装，相关测试会被跳过）

## 测试设计原则

### 1. 正确性优先
- 所有测试都应该验证贝叶斯方法的正确实现
- 使用解析解（如 Beta-Binomial）验证数值解

### 2. 清晰的错误信息
- 测试失败时，断言信息应清晰指出问题
- 使用 `pytest.approx` 处理浮点精度

### 3. 独立性
- 每个测试应该独立运行
- 使用 fixtures 共享数据，但不依赖执行顺序

### 4. 适当的边界情况
- 覆盖常见边界（零数据、极端数据）
- 不测试过于罕见的情况（除非章节特别说明）

### 5. PyMC 可选
- 所有 MCMC 相关测试使用 `@pytest.mark.skipif`
- 允许在没有 PyMC 的环境中运行基础测试

## 测试通过标准

### 基础标准
- 所有 test_prior.py 测试通过
- 所有 test_posterior.py 测试通过
- test_smoke.py 的基础导入测试通过

### 完整标准（需要 PyMC）
- 所有 test_mcmc.py 测试通过
- 所有 test_hierarchical.py 测试通过
- 所有 test_statlab.py 测试通过

### 收敛标准
- R-hat < 1.05（大部分测试要求 < 1.01）
- ESS > 400（最低），> 1000（优秀）

## 更新日志

- 2026-02-13: 初始版本，包含所有核心测试
