# Week 12 作业：解释与伦理——从"能预测"到"能负责"

> "所有模型都是错的，但有些是有用的。问题在于：你能解释它为什么有用吗？以及它对所有人都公平吗？"
> — 改编自 George Box

## 作业概述

本周作业将带你完成一个完整的**模型可解释性与公平性评估流程**：从特征重要性到 SHAP 值的局部可解释性，从检测模型偏见到评估分组公平性，最后用业务语言向非技术读者解释模型结论。

你将使用上周的**电商客户流失数据**（新增敏感属性：性别、年龄段），回答业务问题：**这个模型不仅预测准确，而且负责任吗？**

---

## 基础层（必做）

### 任务 1：从特征重要性到 SHAP 值——理解全局与局部可解释性

小北上周用随机森林训练了流失预测模型，AUC 达到了 0.89。业务方很满意，但问了一个问题："这个模型到底在看什么？"

小北立刻输出了特征重要性：`days_since_last_purchase` 排第一（0.42），`purchase_count` 第二（0.28）。

老潘看完说："**你只回答了一半**。特征重要性告诉你'模型整体上看什么'，但如果业务方问'为什么这个 VIP 客户被预测为流失'，你答不上来。"

**你的任务**：
1. 加载你上周训练的随机森林模型（或重新训练一个）
2. 计算并展示特征重要性（全局可解释性）
3. 使用 SHAP 值解释一个单个预测（局部可解释性）
4. 回答：**同一个特征在不同样本上的 SHAP 值为什么可以正负不同？**

**输入示例**：
```python
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据（假设有敏感属性：gender, age）
df = pd.read_csv("data/customer_churn.csv")

# 特征和目标
X = df[['purchase_count', 'avg_spend', 'days_since_last_purchase',
        'membership_days', 'gender', 'age']]
y = df['is_churned']

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 1. 特征重要性（全局可解释性）
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("特征重要性：")
print(importance_df)

# 2. SHAP 值（局部可解释性）
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 选择一个被预测为流失的样本
churn_samples = X_test[y_test == 1]
if len(churn_samples) > 0:
    sample_idx = churn_samples.index[0]
    # 解释这个样本
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][sample_idx],
        X_test.loc[sample_idx],
        matplotlib=True
    )
```

**输出示例**：
```
特征重要性：
                    feature  importance
2  days_since_last_purchase    0.4123
0              purchase_count    0.2856
1                  avg_spend    0.1532
5                        age    0.0824
3              membership_days    0.0489
4                     gender    0.0176
```

**分析要点**：
- 特征重要性是**全局的、聚合的**：告诉你平均而言什么重要
- SHAP 值是**局部的、样本级别的**：告诉你每个特征对**这个**预测的贡献
- 同一个特征（如 `avg_spend`）对样本 A 可能贡献 +0.2（增加流失风险），对样本 B 可能贡献 -0.1（降低流失风险）

**提交物**：
- 代码文件（如 `week12_basic_shap.py`）
- 特征重要性表和 SHAP 瀑布图
- 一段分析（3-4 句话）解释全局与局部可解释性的区别

**评分点**：
- [ ] 正确计算并展示了特征重要性
- [ ] 正确使用了 SHAP 库（TreeExplainer）
- [ ] 解释了一个单样本的预测
- [ ] 用自己的话解释了全局 vs 局部可解释性的区别

**常见错误**：
- 只输出特征重要性，没有计算 SHAP 值
- 误以为 SHAP 值对所有样本都相同
- 没有理解 SHAP 值的正负含义

---

### 任务 2：解读 SHAP 汇总图——从分布到洞察

阿码看到了 SHAP 汇总图，很困惑："这个图到底在说什么？"

老潘说："**这张图同时回答两个问题：哪些特征重要，以及它们怎么影响预测**。"

**你的任务**：
1. 绘制 SHAP 汇总图
2. 写一段文字描述（3-5 句话），解释：
   - 哪些特征最重要？
   - 高特征值（红色点）和低特征值（蓝色点）对流失概率的影响方向是什么？
3. 找出一个"反直觉"的发现（如果有）：有没有某个特征的方向与你预期的不同？

**输入示例**：
```python
# SHAP 汇总图
shap.summary_plot(shap_values[1], X_test, plot_type="dot")
```

**分析示例**（仅供参考格式）：
```
SHAP 汇总图解读：
- days_since_last_purchase 是最重要的特征（点分布最宽）
- 高值（红色）的 SHAP 值为正，意味着：最近购买天数越长，流失风险越高
- purchase_count 次重要，但高值（红色）的 SHAP 值为负，意味着：购买次数越多，流失风险越低
反直觉发现：gender 的特征重要性很低（0.0176），在 SHAP 图上几乎看不到影响——这与我预期不同，我原以为性别会有影响
```

**提交物**：
- SHAP 汇总图
- 一段文字描述（3-5 句话）

**评分点**：
- [ ] 正确绘制了 SHAP 汇总图
- [ ] 正确解读了特征重要性（点分布宽度）
- [ ] 正确解读了方向（红色 vs 蓝色的位置）
- [ ] 给出了自己的观察或反直觉发现

---

### 任务 3：检测模型偏见——分组公平性评估

老潘收到了一个警告："模型对某些地区的拒绝率是 45%，但整体拒绝率只有 20%。这是偏见吗？"

小北说："拒绝率高不等于偏见，可能那个地区的客户真的信用风险更高。"

"好回答。"老潘说，"**但你需要验证：预测风险和真实风险是否匹配**。"

**你的任务**：
1. 按敏感属性（性别、年龄段）分组评估模型性能
2. 计算每个群体的：预测正率、真阳性率、假阳性率、准确率
3. 回答：**模型对不同群体公平吗？**

**输入示例**：
```python
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def fairness_evaluation(y_true, y_pred, y_prob, sensitive_attr):
    """
    按敏感属性分组评估模型
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'sensitive': sensitive_attr
    })

    results = {}
    for group in df['sensitive'].unique():
        group_df = df[df['sensitive'] == group]
        if len(group_df) < 10:  # 样本太少，跳过
            continue

        cm = confusion_matrix(group_df['y_true'], group_df['y_pred'])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        results[group] = {
            'count': len(group_df),
            'positive_rate': group_df['y_pred'].mean(),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'accuracy': accuracy_score(group_df['y_true'], group_df['y_pred']),
            'avg_predicted_prob': group_df['y_prob'].mean()
        }

    return pd.DataFrame(results).T

# 示例：按性别分组评估
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

gender_results = fairness_evaluation(y_test, y_pred, y_prob, X_test['gender'])
print("按性别分组评估：")
print(gender_results)
```

**输出示例**：
```
按性别分组评估：
       count  positive_rate  true_positive_rate  false_positive_rate  accuracy  avg_predicted_prob
F       180          0.166               0.650                0.082     0.856              0.192
M       220          0.218               0.780                0.105     0.864              0.221

指标差异：
- 真阳性率差异: 0.130
- 假阳性率差异: 0.023
```

**分析要点**：
- 真阳性率差异 0.13：模型对男性流失客户的抓取率比女性高 13%
- 这可能意味着女性流失客户更容易被漏掉
- 但也要看真实流失率：如果女性真实流失率更低，这个差异可能是合理的

**提交物**：
- 代码
- 分组评估结果
- 一段分析（3-4 句话）解释公平性

**评分点**：
- [ ] 正确实现了分组评估函数
- [ ] 计算了完整的公平性指标（TPR, FPR, etc.）
- [ ] 计算了组间差异
- [ ] 给出了合理的公平性分析

**常见错误**：
- 只看整体准确率，没有分组评估
- 只看差异数字，没有考虑样本量和真实风险差异
- 没有理解 TPR/FPR 的业务含义

---

### 任务 4：向非技术读者解释——从 AUC 到业务语言

小北写了一份模型报告给业务方，里面全是"SHAP 值 0.31"、"AUC 0.89"、"p < 0.05"。

业务方的反应是："……你说人话？"

老潘看了报告后笑了："**你不是在写论文，是在写一份业务方会看的报告**。"

**你的任务**：
1. 把以下统计术语翻译成业务语言：
   - AUC = 0.89
   - SHAP 值 = 0.31（days_since_last_purchase）
   - 召回率 = 72%
   - 假阳性率 = 15%
2. 写一段"模型结论与行动建议"（5-8 句话），让非技术读者能看懂

**翻译示例**（仅供参考格式）：
```
统计术语 → 业务语言：
- "AUC = 0.89" → "模型区分流失和非流失客户的能力很强（满分1.0，0.89表示接近完美）"
- "SHAP 值 = 0.31" → "最近购买天数是流失风险的最大来源（贡献约31%的预测信号）"
- "召回率 = 72%" → "模型能抓到 72% 的真实流失客户"
- "假阳性率 = 15%" → "每 100 个被预测为流失的客户中，有 15 个实际上不会流失"
```

**报告示例**（仅供参考格式）：
```
客户流失预测模型——结论与建议

模型性能：
模型区分流失和非流失客户的能力很强（AUC = 0.89，满分1.0）。在 100 个真实会流失的客户中，模型能抓到 72 个。

主要风险信号：
- 最近购买天数是流失风险的最大来源：超过 30 天未购买的客户，流失风险显著增加
- 购买次数少（< 3 次）的客户流失风险是购买次数多（> 10 次）的客户的 2.5 倍

行动建议：
- 优先联系：超过 30 天未购买 + 购买次数 < 3 次的客户
- 考虑激励：向非会员客户推送会员优惠
```

**提交物**：
- 统计术语翻译表
- 一段面向非技术读者的模型结论（5-8 句话）

**评分点**：
- [ ] 正确翻译了所有统计术语
- [ ] 报告使用了业务语言（不是技术黑话）
- [ ] 包含了具体的行动建议
- [ ] 语言简洁、易懂

---

## 进阶层（推荐完成）

### 任务 5：完整的可解释性报告——从技术指标到风险管理

老潘说："**完整的模型报告不只是 AUC 和特征重要性**，还要包括公平性评估、伦理风险清单和模型边界说明。"

**你的任务**：
1. 创建一个完整的可解释性报告 Markdown 文件，包含：
   - 模型性能概述（用业务语言）
   - SHAP 汇总图和解读
   - 单样本预测解释示例
   - 公平性评估（分组指标）
   - 伦理风险清单
   - 模型边界说明
2. 报告应该让非技术读者能看懂核心结论

**报告结构示例**：
```markdown
# 客户流失预测模型——可解释性与伦理报告

## 1. 模型性能概述（面向非技术读者）
...

## 2. 模型可解释性
### 2.1 SHAP 汇总图
![SHAP汇总图](output/shap_summary.png)

### 2.2 单样本预测解释
...

## 3. 公平性评估
### 3.1 按性别分组
...

### 3.2 按年龄段分组
...

## 4. 伦理风险与缓解措施
| 风险类型 | 风险等级 | 缓解措施 |
|---------|---------|---------|
| 数据偏见 | 中 | ... |
| 分配不公 | 低 | ... |

## 5. 模型边界
- 模型基于历史数据训练，可能无法预测新业务场景
- 对于样本量较少的客户群体，预测不确定性较高
```

**提交物**：
- 完整的可解释性报告 Markdown 文件
- 生成报告的代码

**评分点**：
- [ ] 报告结构完整（包含所有必要部分）
- [ ] 使用了业务语言（不是纯技术术语）
- [ ] 包含了图表和可视化
- [ ] 伦理风险清单合理

---

### 任务 6：公平性-准确性权衡分析——有完美模型吗？

阿码问："能不能让模型既准确又公平？"

老潘说："**没有完美的模型，只有权衡**。你可以牺牲一些准确性来换取更公平的分配，但业务方需要决定这个代价值不值得。"

**你的任务**：
1. 计算当前模型的公平性-准确性基线
2. （可选）尝试一种简单的偏见缓解策略：调整不同群体的分类阈值
3. 分析：**牺牲多少准确性可以换来多少公平性改善？**

**输入示例**：
```python
import numpy as np

def analyze_fairness_accuracy_tradeoff(y_true, y_prob, sensitive_attr, thresholds):
    """
    分析不同阈值下的公平性-准确性权衡
    """
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # 整体指标
        acc = accuracy_score(y_true, y_pred)

        # 分组指标
        group_metrics = {}
        for group in sensitive_attr.unique():
            mask = sensitive_attr == group
            if mask.sum() < 10:
                continue
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            cm = confusion_matrix(group_y_true, group_y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            group_metrics[group] = {
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
            }

        # 计算公平性差异
        tpr_diff = max(m['tpr'] for m in group_metrics.values()) - \
                   min(m['tpr'] for m in group_metrics.values())

        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'tpr_diff': tpr_diff
        })

    return pd.DataFrame(results)

# 分析不同阈值下的权衡
thresholds = np.linspace(0.2, 0.6, 9)
tradeoff_df = analyze_fairness_accuracy_tradeoff(
    y_test, y_prob, X_test['gender'], thresholds
)

print("公平性-准确性权衡分析：")
print(tradeoff_df)
```

**输出示例**：
```
公平性-准确性权衡分析：
   threshold  accuracy  tpr_diff
0      0.20     0.812     0.082
1      0.25     0.835     0.095
2      0.30     0.855     0.115
3      0.35     0.868     0.130
4      0.40     0.872     0.145
5      0.45     0.869     0.158
6      0.50     0.860     0.172
```

**分析要点**：
- 阈值从 0.30 提高到 0.40：准确率提升 1.7%，但 TPR 差异增加 3%
- 这意味着：更准确的模型对女性群体更不公平
- 业务方需要决定：是否愿意牺牲 1.7% 的准确率来换取更公平的 TPR？

**提交物**：
- 代码
- 权衡分析结果
- 一段分析（3-4 句话）解释权衡

**评分点**：
- [ ] 实现了公平性-准确性权衡分析
- [ ] 计算了不同阈值/配置下的指标
- [ ] 给出了明确的权衡结论
- [ ] 用业务语言解释了权衡的含义

---

## 挑战层（可选）

### 任务 7：综合项目——为你的流失预测模型添加完整的可解释性和公平性评估

老潘说："**这才是完整的模型交付**：不仅有预测力，还有可解释性、公平性和风险管理。"

**你的任务**：
1. 整合本周所有内容，为你的流失预测模型创建一个完整的"模型卡片"（Model Card）
2. 模型卡片应该包含：
   - 模型用途和预期使用者
   - 模型性能（整体和分组）
   - 可解释性分析（SHAP）
   - 公平性评估
   - 伦理风险和缓解措施
   - 模型边界和维护计划
3. 写一页"执行摘要"给高管（非技术背景）

**模型卡片结构示例**：
```markdown
# 客户流失预测模型 - 模型卡片

## 模型概述
- **用途**：预测客户在未来 30 天内的流失风险
- **预期使用者**：营销团队、客户服务团队
- **训练数据**：2023-2025 年客户交易数据（N=10,000）
- **最后更新**：2026-02-18

## 模型性能
- **整体 AUC**：0.89
- **整体准确率**：0.87
- **分组性能**：见公平性评估部分

## 可解释性
- **最重要特征**：days_since_last_purchase（SHAP 值 0.31）
- **SHAP 汇总图**：见附件

## 公平性评估
| 分组 | 真阳性率 | 假阳性率 | 准确率 |
|------|---------|---------|--------|
| 女性 | 0.65 | 0.08 | 0.86 |
| 男性 | 0.78 | 0.11 | 0.86 |
- **公平性差异**：TPR 差异 0.13，FPR 差异 0.03

## 伦理风险
| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| ... | ... | ... |

## 模型边界
- 模型未见过疫情期间的数据
- 对于新注册客户（< 7 天）预测不确定性较高

## 维护计划
- 建议每季度重新训练
- 每月审计分组公平性指标
```

**提交物**：
- 完整的模型卡片 Markdown 文件
- 执行摘要（一页）
- 生成模型卡片的代码

**评分点**：
- [ ] 模型卡片结构完整
- [ ] 包含了所有必要部分（性能、可解释性、公平性、风险、边界）
- [ ] 执行摘要简洁、易懂
- [ ] 代码可复现

---

## AI 协作练习（可选）

下面这段文字是某个 AI 工具生成的"模型解释"：

> "本模型使用随机森林，AUC 为 0.89，表明模型性能优秀。SHAP 分析显示 days_since_last_purchase 是最重要的特征（SHAP 值 0.31），其次是 purchase_count（SHAP 值 0.18）。模型对男性和女性的准确率都是 0.86，说明模型不存在性别偏见。"

**审查清单**：
- [ ] 它解释了 SHAP 值的含义吗？（0.31 代表什么？）
- [ ] 它正确判断了公平性吗？（准确率相同 ≠ 公平）
- [ ] 它提到了模型的边界和风险吗？
- [ ] 它使用了业务语言还是技术黑话？
- [ ] 它给出了具体的行动建议吗？

**你的修订版**（3-4 句话）：
```
（用你自己的话写，修正上述问题）
例如："AI 的解释有两个问题：第一，SHAP 值 0.31 没有解释单位——应该是'贡献约 31% 的预测信号'。第二，准确率相同不代表公平，应该看真阳性率和假阳性率的差异。"
```

**提交物**：
- 审查清单（勾选哪些问题存在）
- 你的修订版（3-4 句话）

---

## 提交检查清单

在提交作业前，请确认：

- [ ] 代码可以运行（或注明哪些部分是伪代码）
- [ ] 输出结果包含在提交中（或截图）
- [ ] 分析部分用你自己的话写（不是复制粘贴）
- [ ] 向非技术读者的解释使用了业务语言
- [ ] 如果遇到困难，参考了 `starter_code/solution.py`，请说明参考了哪些部分

---

## 提示与帮助

如果你在完成作业时遇到困难：
1. 回顾 CHAPTER.md 中的示例代码
2. 参考本周的 StatLab 示例（`examples/12_statlab_interpretability.py`）
3. 查阅 shap 库官方文档：https://shap.readthedocs.io/
4. 如果你遇到困难，可以参考 `starter_code/solution.py`（但不要直接复制）

**记住**：作业的目的是巩固理解，不是完美复制代码。即使遇到困难，也要尝试用自己的话解释问题和思路。

---

祝你本周学习愉快！记住老潘的话：**"模型能预测只是第一步，模型能解释、能负责才是部署的前提。"**
