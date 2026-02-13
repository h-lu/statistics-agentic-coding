# Week 14 测试使用说明

本目录包含 Week 14（贝叶斯推断）的 pytest 测试用例。

## 快速开始

### 安装依赖

```bash
# 基础依赖（必需）
pip install numpy scipy pandas pytest

# 可选依赖（用于 MCMC 测试）
pip install pymc arviz
```

### 运行所有测试

```bash
# 从项目根目录运行
cd /Users/wangxq/Documents/statistics-agentic-coding
python3 -m pytest chapters/week_14/tests -v
```

## 测试文件说明

| 文件 | 描述 | 需要 PyMC |
|------|------|----------|
| `test_smoke.py` | 基础冒烟测试 | 否 |
| `test_prior.py` | 先验相关测试 | 否 |
| `test_posterior.py` | 后验计算测试 | 否 |
| `test_mcmc.py` | MCMC 采样测试 | 是 |
| `test_hierarchical.py` | 层次模型测试 | 是 |
| `test_statlab.py` | StatLab 集成测试 | 部分需要 |

## 运行特定测试

### 只运行基础测试（不需要 PyMC）

```bash
python3 -m pytest chapters/week_14/tests/test_prior.py -v
python3 -m pytest chapters/week_14/tests/test_posterior.py -v
```

### 跳过需要 PyMC 的测试

```bash
python3 -m pytest chapters/week_14/tests -v -m "not pymc"
```

### 只运行特定测试文件

```bash
# 测试先验
python3 -m pytest chapters/week_14/tests/test_prior.py -v

# 测试后验
python3 -m pytest chapters/week_14/tests/test_posterior.py -v

# 测试 MCMC（需要 PyMC）
python3 -m pytest chapters/week_14/tests/test_mcmc.py -v

# 测试层次模型（需要 PyMC）
python3 -m pytest chapters/week_14/tests/test_hierarchical.py -v
```

### 运行特定测试函数

```bash
# 运行单个测试
python3 -m pytest chapters/week_14/tests/test_prior.py::TestPriorCreation::test_create_uniform_prior_beta_1_1 -v

# 使用关键字匹配
python3 -m pytest chapters/week_14/tests/test_prior.py -k "uniform" -v
```

## 测试输出

### 正常输出

```
tests/test_prior.py::TestPriorCreation::test_create_uniform_prior_beta_1_1 PASSED
tests/test_prior.py::TestPriorCreation::test_create_weakly_informative_prior PASSED
tests/test_prior.py::TestPriorCreation::test_create_conjugate_prior_beta_binomial PASSED
...
======================== 42 passed in 2.34s =========================
```

### PyMC 未安装时的输出

```
tests/test_mcmc.py::TestMCMCSampling::test_simple_ab_test_mcmc SKIPPED (PyMC not installed)
tests/test_mcmc.py::TestMCMCSampling::test_mcmc_convergence_r_hat SKIPPED (PyMC not installed)
...
======================== 20 passed, 22 skipped in 1.12s =====================
```

## Fixtures

测试使用以下 fixtures（定义在 `conftest.py`）：

### 数据 Fixtures
- `simple_ab_data`: 标准 A/B 测试数据
- `small_ab_data`: 小样本 A/B 测试数据
- `extreme_ab_data`: 极端情况数据
- `hierarchical_data`: 层次模型数据（4 国）
- `regression_data`: 线性回归数据

### 配置 Fixtures
- `random_seed`: 固定随机种子（42）
- `priors`: 常用先验配置
- `convergence_thresholds`: R-hat 和 ESS 阈值

## 调试建议

### 查看详细输出

```bash
# 显示 print 输出
python3 -m pytest chapters/week_14/tests/test_prior.py -v -s

# 显示更短的追溯
python3 -m pytest chapters/week_14/tests/test_prior.py --tb=short
```

### 运行到第一个失败

```bash
python3 -m pytest chapters/week_14/tests -v -x
```

### 运行上次失败的测试

```bash
python3 -m pytest chapters/week_14/tests --lf
```

## 常见问题

### Q: 为什么很多测试被跳过？

A: 如果 PyMC 未安装，所有 MCMC 相关测试会被跳过。安装 PyMC：
```bash
pip install pymc arviz
```

### Q: 测试运行很慢？

A: MCMC 测试涉及采样，可能需要较长时间。可以：
1. 只运行特定测试文件
2. 使用 `-m "not pymc"` 跳过 MCMC 测试
3. 减少采样数量（修改测试中的 `draws` 参数）

### Q: 如何添加新测试？

A: 在相应文件中添加测试函数：
1. `test_prior.py`: 先验相关测试
2. `test_posterior.py`: 后验计算测试
3. `test_mcmc.py`: MCMC 采样测试
4. `test_hierarchical.py`: 层次模型测试
5. `test_statlab.py`: StatLab 集成测试

测试命名格式：`test_<功能>_<场景>_<预期结果>`

示例：
```python
def test_conjugate_posterior_updates_correctly():
    """
    测试共轭后验正确更新

    Beta 先验 + Binomial 似然 = Beta 后验
    """
    prior = stats.beta(2, 40)
    successes = 58
    trials = 1000

    posterior = stats.beta(prior.args[0] + successes,
                        prior.args[1] + trials - successes)

    assert posterior.args == (2 + 58, 40 + 1000 - 58)
```

## 测试覆盖范围

详细测试矩阵见 [TEST_MATRIX.md](TEST_MATRIX.md)

## 贡献指南

1. 保持测试独立（每个测试应该能单独运行）
2. 使用清晰的测试名称
3. 添加适当的文档字符串
4. 使用 `pytest.approx` 处理浮点精度
5. 对于需要 PyMC 的测试，使用 `@pytest.mark.skipif`
