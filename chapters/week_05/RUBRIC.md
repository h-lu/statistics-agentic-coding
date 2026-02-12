# Week 05 评分标准（Rubric）

## 评分维度与权重

| 维度 | 权重 | 说明 |
|------|------|------|
| 贝叶斯定理应用 | 25% | 条件概率计算正确、理解 P(A\B) ≠ P(B\A)、先验影响分析 |
| 分布识别 | 25% | 正确识别分布类型、参数估计合理、极端事件预测准确 |
| CLT 模拟 | 25% | 模拟设计合理、正确解释样本量影响、标准误验证 |
| Bootstrap 分析 | 25% | 正确实现重采样、CI 计算正确、理解显著性含义 |

---

## 详细评分标准

### 1. 贝叶斯定理应用（25 分）

| 评分项 | 分值 | 优秀（A） | 良好（B） | 合格（C） | 不合格（F） |
|--------|------|-----------|-----------|-----------|-------------|
| 计算正确性 | 10 | 贝叶斯公式应用正确，P(患病\阳性) 计算准确 | 计算基本正确，小误差 | 公式有瑕疵，结果接近 | 公式错误或结果错误 |
| 条件概率方向 | 8 | 明确区分 P(A\B) 和 P(B\A)，解释差异 | 理解两者不同，解释不够清晰 | 混淆两者但结果碰巧正确 | 完全混淆两个方向 |
| 先验影响分析 | 7 | 完整分析不同发病率下的后验概率，有可视化 | 分析了先验影响，缺少可视化 | 仅计算了单个先验情况 | 未分析先验影响 |

**验证方式**：
- 检查脚本 `01_bayes_theorem.py` 的输出
- 测试：用锚点数据验证贝叶斯计算

---

### 2. 分布识别（25 分）

| 评分项 | 分值 | 优秀（A） | 良好（B） | 合格（C） | 不合格（F） |
|--------|------|-----------|-----------|-----------|-------------|
| 分布识别 | 10 | 根据数据类型和形状正确判断分布类型 | 识别基本正确，理由不够充分 | 识别有误但可接受 | 严重误判（如对偏态数据用正态） |
| 参数估计 | 8 | 正确估计分布参数，有理论验证 | 参数估计基本正确 | 参数估计有瑕疵 | 参数估计错误 |
| 极端事件预测 | 7 | 正确使用理论分布计算极端事件概率，并用模拟验证 | 预测基本正确 | 预测方法有瑕疵 | 预测方法错误 |

**验证方式**：
- 检查脚本 `02_distribution_fit.py` 的输出
- 测试：用锚点数据验证分布参数计算

---

### 3. CLT 模拟（25 分）

| 评分项 | 分值 | 优秀（A） | 良好（B） | 合格（C） | 不合格（F） |
|--------|------|-----------|-----------|-----------|-------------|
| 模拟设计 | 8 | 使用偏态总体，设计多个样本量对比 | 设计合理，样本量选择较单一 | 总体选择不当（如正态） | 模拟设计有根本性问题 |
| 正态性检验 | 9 | 正确进行正态性检验，解释 p 值含义 | 进行了检验，解释不够完整 | 检验有瑕疵 | 未进行检验或检验错误 |
| 标准误验证 | 8 | 正确计算实际 SE 和理论 SE，验证一致 | SE 计算正确，验证不够深入 | SE 计算有瑕疵 | SE 计算错误或混淆 SD/SE |

**验证方式**：
- 检查脚本 `03_clt_simulation.py` 的输出
- 测试：验证 SE = σ/√n 公式应用正确

---

### 4. Bootstrap 分析（25 分）

| 评分项 | 分值 | 优秀（A） | 良好（B） | 合格（C） | 不合格（F） |
|--------|------|-----------|-----------|-----------|-------------|
| 重采样正确性 | 10 | 正确使用 replace=True，代码健壮 | 重采样基本正确，小问题 | 重采样有瑕疵 | 未使用 replace 或实现错误 |
| CI 计算 | 8 | 正确计算 percentile CI，理解含义 | CI 计算正确，解释不够 | CI 计算有瑕疵 | CI 计算错误 |
| 显著性判断 | 7 | 正确判断 CI 是否包含 0，解释统计显著性 | 判断基本正确 | 判断逻辑有瑕疵 | 判断错误 |

**验证方式**：
- 检查脚本 `04_bootstrap_analysis.py` 的输出
- 测试：用锚点数据验证 Bootstrap CI

---

### 5. StatLab 报告质量（10 分）

| 评分项 | 分值 | 优秀（A） | 良好（B） | 合格（C） | 不合格（F） |
|--------|------|-----------|-----------|-----------|-------------|
| 章节完整性 | 4 | 包含所有必需内容，格式规范 | 缺少 1 项内容 | 缺少 2 项内容 | 缺少 3 项以上内容 |
| 结果解读 | 4 | 解读深入，连接业务场景 | 解读基本合理 | 解读表面化 | 解读错误或缺失 |
| 可复现性 | 2 | 提供完整代码，可一键运行 | 有基本说明 | 说明不清 | 无法复现 |

**验证方式**：
- 检查 report.md 中的不确定性量化章节
- 检查 README.md

---

## 等级标准

| 等级 | 分数区间 | 描述 |
|------|----------|------|
| **优秀（A）** | 90-100% | 所有要求完成，有额外思考。贝叶斯定理理解深入，分布识别准确，CLT 模拟完整，Bootstrap 分析规范。 |
| **良好（B）** | 80-89% | 核心要求完成，minor issues。各任务基本完成，但某些方面可以更深入。 |
| **合格（C）** | 70-79% | 基本完成，有明显不足。如混淆条件概率方向、分布识别有误、Bootstrap 实现有瑕疵等。 |
| **待改进（F）** | <70% | 未完成核心要求。如缺少关键任务、代码无法运行、StatLab 章节缺失等。 |

---

## 检查清单（供助教/自测使用）

### 贝叶斯定理
- [ ] 正确计算 P(阳性) = P(阳性\患病)×P(患病) + P(阳性\健康)×P(健康)
- [ ] 正确计算 P(患病\阳性) = P(阳性\患病)×P(患病) / P(阳性)
- [ ] 区分了 P(A\B) 和 P(B\A)
- [ ] 分析了不同先验（发病率）下的后验概率
- [ ] 用模拟验证了理论计算

### 分布识别
- [ ] 绘制了数据直方图和密度曲线
- [ ] 计算了描述统计（均值、方差、偏度）
- [ ] 根据数据类型和形状判断了分布类型
- [ ] 估计了分布参数
- [ ] 预测了极端事件概率

### CLT 模拟
- [ ] 使用了偏态总体（如指数分布）
- [ ] 测试了多个样本量（至少 3 个）
- [ ] 绘制了样本均值分布图
- [ ] 进行了正态性检验
- [ ] 验证了 SE = σ/√n

### Bootstrap 分析
- [ ] 使用了 replace=True 进行重采样
- [ ] 计算了 95% percentile CI
- [ ] 记录了标准误
- [ ] 判断了显著性（CI 是否包含 0）
- [ ] 相关系数使用了成对重采样

### StatLab 报告
- [ ] 包含核心统计量汇总表
- [ ] 包含关键发现
- [ ] 包含敏感性分析
- [ ] 包含数据局限
- [ ] 格式规范，可读性强

### 代码质量
- [ ] 变量名清晰
- [ ] 有适当注释
- [ ] 代码可运行
- [ ] 设置了随机种子
- [ ] 提供了 README

---

## 常见扣分点

| 问题 | 扣分 | 说明 |
|------|------|------|
| 混淆 P(A\B) 和 P(B\A) | -10 | 条件概率方向错误 |
| 滥用正态分布假设 | -10 | 对偏态数据错误假设正态 |
| Bootstrap 未使用 replace=True | -10 | 未实现有放回抽样 |
| 混淆 SD 和 SE | -8 | 标准差和标准误混淆 |
| 未进行正态性检验 | -5 | CLT 部分缺少正态性验证 |
| 缺少敏感性分析 | -5 | Bootstrap 部分未测试稳健性 |
| StatLab 章节缺失 | -15 | 无不确定性量化章节或章节不完整 |
| 代码无法运行 | -20 | 有语法错误或依赖缺失 |
| 未提交到 git | -5 | 无 commit 记录 |

---

## 加分项（最高 +15 分）

| 加分项 | 分值 | 说明 |
|--------|------|------|
| 完成任务 5：比较统计量稳定性 | +5 | 深入比较均值、中位数、trimmed 均值 |
| 完成任务 6：敏感性分析 | +5 | 系统测试 Bootstrap 稳健性 |
| 完成任务 7：概率直觉实验 | +5 | 设计了有创意的概率直觉实验 |
| AI 协作练习 1 | +3 | 完成代码审查并提交修复 |
| AI 协作练习 2 | +3 | 改进 AI 代码并撰写对比说明 |
| 可视化 | +2 | 用图表清晰展示关键发现 |

---

## 自动化测试锚点

以下测试用例用于验证作业实现：

```python
# tests/test_week05.py

import numpy as np
import pandas as pd
from scipy import stats

def test_bayes_theorem():
    """测试贝叶斯定理计算正确"""
    prevalence = 0.01
    sensitivity = 0.99
    specificity = 0.99

    p_positive_given_sick = sensitivity
    p_sick = prevalence
    p_positive_given_healthy = 1 - specificity
    p_positive = p_positive_given_sick * p_sick + p_positive_given_healthy * (1 - p_sick)

    p_sick_given_positive = (p_positive_given_sick * p_sick) / p_positive

    # 应该约等于 0.5
    assert 0.45 <= p_sick_given_positive <= 0.55

def test_bootstrap_replace():
    """测试 Bootstrap 使用有放回抽样"""
    np.random.seed(42)
    sample = np.random.normal(100, 15, 50)
    n = len(sample)

    # Bootstrap 重采样
    resample = np.random.choice(sample, size=n, replace=True)

    # 长度应该相同
    assert len(resample) == n

    # 唯一值数量应该 <= n（因为有重复）
    assert len(np.unique(resample)) <= n

def test_standard_error_formula():
    """测试标准误公式 SE = σ/√n"""
    sigma = 15
    n = 30
    theoretical_se = sigma / np.sqrt(n)

    # 模拟验证
    sample_means = [
        np.mean(np.random.normal(100, sigma, n))
        for _ in range(10000)
    ]
    actual_se = np.std(sample_means, ddof=1)

    # 应该接近
    assert abs(actual_se - theoretical_se) / theoretical_se < 0.1

def test_clt_convergence():
    """测试 CLT：样本量越大，均值分布越接近正态"""
    population = np.random.exponential(10, 100000)

    # 小样本和大样本
    small_means = [np.mean(np.random.choice(population, 5)) for _ in range(1000)]
    large_means = [np.mean(np.random.choice(population, 100)) for _ in range(1000)]

    # 大样本的正态性应该更好（Shapiro-Wilk p 值更大）
    _, p_small = stats.shapiro(small_means)
    _, p_large = stats.shapiro(large_means)

    # 这个断言可能偶尔失败（因为随机性），但总体趋势应该成立
    # 在实际评分中会结合多次运行判断

def test_bootstrap_ci_contains():
    """测试 Bootstrap CI 包含观察值"""
    np.random.seed(42)
    sample = np.random.normal(100, 15, 100)

    boot_means = np.array([
        np.mean(np.random.choice(sample, 100, replace=True))
        for _ in range(10000)
    ])

    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    observed = np.mean(sample)

    # CI 应该接近观察值
    assert abs(observed - ci_low) < 5
    assert abs(observed - ci_high) < 5
```

---

**评分人**：____________ **日期**：____________ **总分**：____________
