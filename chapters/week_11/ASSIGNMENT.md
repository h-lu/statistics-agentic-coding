# Week 11 作业：树模型与基线对比——更复杂不等于更好

> "所有模型都是错的，但有些是有用的。问题在于：你怎么知道'有用'？"
> — George Box

## 作业概述

本周作业将带你完成一个完整的**树模型建模与基线对比流程**：从单棵决策树的训练与可视化，到随机森林的集成学习，再到"与基线对比"的模型选择框架。

你将使用一份**电商客户流失数据**，回答业务问题：**决策树和随机森林比逻辑回归好吗？这种提升值得牺牲可解释性吗？**

---

## 基础层（必做）

### 任务 1：训练你的第一棵决策树

小北上周用逻辑回归预测流失，AUC 0.87。他很好奇：如果用决策树，会更好吗？

**你的任务**：
1. 加载 `data/customer_churn.csv`（或使用 `starter_code/week_11.py` 中的示例数据）
2. 复用 Week 10 的特征选择（`purchase_count`, `avg_spend`, `days_since_last_purchase` 等）
3. 划分训练集和测试集（test_size=0.3, random_state=42）
4. 训练一棵决策树，**限制深度为 3**（防止过拟合）
5. 计算测试集的准确率、精确率、召回率、F1、AUC
6. 可视化决策树结构

**输入示例**：
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("data/customer_churn.csv")

# 特征和目标
X = df[['purchase_count', 'avg_spend', 'days_since_last_purchase', 'membership_days']]
y = df['is_churned']

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树（限制深度）
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 预测
y_pred = tree.predict(X_test)
y_prob = tree.predict_proba(X_test)[:, 1]

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化树
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, class_names=['Not Churn', 'Churn'],
          filled=True, rounded=True, fontsize=10)
plt.show()
```

**输出示例**：
```
准确率: 0.8600
AUC: 0.8450

分类报告:
              precision    recall  f1-score   support

       0       0.88      0.96      0.92       240
       1       0.67      0.35      0.46        60

accuracy                           0.86       300
macro avg       0.77      0.66      0.69       300
weighted avg       0.84      0.86      0.84       300
```

**提交物**：
- 代码文件（如 `week11_basic.py`）
- 决策树可视化图
- 评估指标输出

**评分点**：
- [ ] 正确使用了 DecisionTreeClassifier
- [ ] 设置了 max_depth=3（或其他合理的深度限制）
- [ ] 计算了完整的评估指标（准确率、AUC、分类报告）
- [ ] 可视化了决策树结构

**常见错误**：
- 没有限制树的深度，导致严重过拟合
- 只报告准确率，没有报告 AUC
- 忘记可视化决策树，失去可解释性

---

### 任务 2：解读决策树——从根节点到叶节点

小北看到决策树的可视化图后，很兴奋："这个树好直观！我能看懂它是怎么做的决策。"

阿码问了一个好问题："你能用文字描述一下这棵树的决策规则吗？"

**你的任务**：
1. 观察你训练的决策树可视化图
2. 写一段文字描述（3-5 句话），解释：
   - 根节点用什么特征分裂？阈值是多少？
   - 哪条路径的流失率最高？
   - 哪条路径的流失率最低？
3. 回答：**决策树相比逻辑回归，在可解释性上有什么优势？**

**分析示例**（仅供参考格式）：
```
决策树解读：
- 根节点：首先判断 'days_since_last_purchase < 45.5'
  - 如果是（最近购买过的客户）：
    - 再判断 'purchase_count < 3.5'...
  - 如果否（很久没购买的客户）：
    - 流失率较高（约 70%），因为...

最高流失率路径：
days_since_last_purchase >= 45.5 AND avg_spend < 87.3
这条路径的叶节点流失率约 72%，包含 40 个样本。

可解释性优势：
决策树可以用 if-else 规则直观展示，业务方不需要懂统计学就能理解。逻辑回归的系数需要解释"对数几率"的变化，不如决策树直观。
```

**提交物**：
- 一段文字描述（3-5 句话）
- 对可解释性优势的分析（2-3 句话）

**评分点**：
- [ ] 正确解读了根节点的分裂规则
- [ ] 正确识别了最高/最低流失率的路径
- [ ] 给出了合理的可解释性优势分析
- [ ] 用自己的话描述，不是复制粘贴

---

## 进阶层（推荐完成）

### 任务 3：训练随机森林——群体智慧提升预测力

老潘看了小北的决策树，说："单棵树有个问题：不稳定。如果你重新训练一次，树结构可能完全不同。"

"那怎么办？"小北问。

"用随机森林。"老潘说，"训练很多棵树，让它们投票。"

**你的任务**：
1. 训练一个随机森林（n_estimators=100, max_depth=5）
2. 计算测试集的评估指标（准确率、AUC、分类报告）
3. 与决策树对比：**随机森林比决策树好了多少？**
4. 回答：**随机森林的可解释性比决策树差在哪里？**

**输入示例**：
```python
from sklearn.ensemble import RandomForestClassifier

# 随机森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 预测
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 评估
print(f"随机森林准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"随机森林 AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"决策树 AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"提升量: {roc_auc_score(y_test, y_prob_rf) - roc_auc_score(y_test, y_prob):.4f}")
```

**输出示例**：
```
随机森林准确率: 0.8733
随机森林 AUC: 0.8900
决策树 AUC: 0.8450
提升量: 0.0450
```

**分析要点**：
- 随机森林通过"投票"降低了单棵树的不稳定性
- 可解释性下降：100 棵树无法全部可视化，只能通过特征重要性间接解释

**提交物**：
- 代码
- 随机森林与决策树的对比结果
- 一段分析（2-3 句话）解释可解释性的差异

**评分点**：
- [ ] 正确使用了 RandomForestClassifier
- [ ] 设置了合理的超参数（max_depth, n_estimators 等）
- [ ] 计算了提升量（随机森林 vs 决策树）
- [ ] 解释了可解释性的差异

**常见错误**：
- 没有限制 max_depth，可能导致过拟合
- 只看 AUC 数字，没有计算提升量
- 没有意识到可解释性的下降

---

### 任务 4：特征重要性——随机森林"告诉"你哪些特征重要

阿码问："随机森林是黑盒，我怎么知道它在用什么特征做决策？"

"用特征重要性。"老潘说，"虽然你不能画出整个森林，但你能知道哪些特征最重要。"

**你的任务**：
1. 提取随机森林的特征重要性
2. 画出特征重要性条形图
3. 回答：**最重要的 3 个特征是什么？这与业务直觉一致吗？**

**输入示例**：
```python
import pandas as pd
import matplotlib.pyplot as plt

# 获取特征重要性
importances = rf.feature_importances_
feature_names = X.columns

# 创建 DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df)

# 画图
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()
```

**输出示例**：
```
                    feature  importance
2  days_since_last_purchase    0.5123
0              purchase_count    0.2856
1                  avg_spend    0.1532
3            membership_days    0.0489
```

**分析要点**：
- 最重要特征是距上次购买天数，这与业务直觉一致（很久没买的客户更可能流失）
- 特征重要性是相对值，所有特征重要性之和为 1

**提交物**：
- 代码
- 特征重要性表和条形图
- 一段分析（2-3 句话）解释特征重要性的业务含义

**评分点**：
- [ ] 正确提取了特征重要性
- [ ] 画出了特征重要性条形图
- [ ] 正确解读了最重要的特征
- [ ] 给出了业务解释（而不是只给数字）

---

## 挑战层（可选）

### 任务 5：基线对比框架——更复杂一定更好吗？

老潘看了小北的随机森林结果，问了一个关键问题："你的随机森林 AUC 0.89，确实不错。但你和基线比了吗？"

"基线？"小北困惑了。

"最简单的模型。"老潘说，"比如逻辑回归、甚至'总是预测不流失'。如果随机森林比逻辑回归只好了 0.02 AUC，但训练时间慢了 50 倍，你觉得值得吗？"

**你的任务**：
1. 训练三个基线模型：
   - 傻瓜基线（DummyClassifier，strategy='most_frequent'）
   - 逻辑回归基线
   - 决策树基线
2. 训练随机森林
3. 制作一个对比表，包含：AUC、训练时间（用 time 模块测量）
4. 回答：**随机森林比逻辑回归的提升量，值得牺牲可解释性吗？**

**输入示例**：
```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

# 定义模型
models = {
    'dummy': DummyClassifier(strategy='most_frequent'),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
}

# 训练并对比
results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'auc': auc, 'train_time': train_time}

# 打印对比表
print("| 模型 | AUC | 训练时间 |")
print("|------|-----|---------|")
for name, res in results.items():
    print(f"| {name} | {res['auc']:.4f} | {res['train_time']:.4f}s |")
```

**输出示例**：
```
| 模型 | AUC | 训练时间 |
|------|-----|---------|
| dummy | 0.5000 | 0.0012s |
| logistic_regression | 0.8700 | 0.0156s |
| decision_tree | 0.8450 | 0.0089s |
| random_forest | 0.8900 | 0.5234s |
```

**分析要点**：
- 随机森林比逻辑回归提升 0.02 AUC（约 2.3%）
- 但训练时间慢了约 33 倍
- 是否值得取决于业务场景：如果追求最高预测力，选随机森林；如果需要可解释性或快速训练，选逻辑回归

**提交物**：
- 代码
- 模型对比表
- 一段分析（3-4 句话）回答"值得吗？"

**评分点**：
- [ ] 训练了至少 3 个基线模型
- [ ] 制作了完整的对比表
- [ ] 计算了提升量（相对和绝对）
- [ ] 给出了明确的模型选择理由（不能只说"随机森林最好"）

**常见错误**：
- 没有训练傻瓜基线，不知道模型是否比"瞎猜"好
- 只看 AUC 绝对值，不看提升量
- 没有考虑训练时间和可解释性
- 没有给出明确的模型选择建议

---

### 任务 6（加分）：Bootstrap 估计 AUC 置信区间——提升量是真实的吗？

阿码问了一个很统计的问题："随机森林比逻辑回归好了 0.02 AUC，这是真实的提升，还是运气？"

"好问题。"老潘说，"你可以用 Bootstrap 估计 AUC 的置信区间，看看两个模型的置信区间是否重叠。"

**你的任务**：
1. 用 Bootstrap 估计随机森林和逻辑回归的 AUC 分布（n_bootstrap=1000）
2. 计算 95% 置信区间
3. 回答：**提升量是否显著？（置信区间是否重叠）**

**输入示例**：
```python
import numpy as np

def bootstrap_auc(model, X, y, n_bootstrap=1000):
    """Bootstrap 估计 AUC 的分布"""
    np.random.seed(42)
    auc_scores = []
    n = len(X)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]
        y_prob = model.predict_proba(X_boot)[:, 1]
        auc_scores.append(roc_auc_score(y_boot, y_prob))
    return np.array(auc_scores)

# 获取 AUC 分布
auc_lr_dist = bootstrap_auc(log_reg_model, X_test, y_test)
auc_rf_dist = bootstrap_auc(rf_model, X_test, y_test)

# 计算置信区间
def ci_interval(dist, alpha=0.05):
    return np.percentile(dist, [alpha/2*100, (1-alpha/2)*100])

ci_lr = ci_interval(auc_lr_dist)
ci_rf = ci_interval(auc_rf_dist)

print(f"逻辑回归 AUC: {auc_lr_dist.mean():.4f}, 95% CI: [{ci_lr[0]:.4f}, {ci_lr[1]:.4f}]")
print(f"随机森林 AUC: {auc_rf_dist.mean():.4f}, 95% CI: [{ci_rf[0]:.4f}, {ci_rf[1]:.4f}]")

# 判断是否显著
if ci_lr[1] < ci_rf[0] or ci_rf[1] < ci_lr[0]:
    print("提升量显著（置信区间不重叠）")
else:
    print("提升量可能不显著（置信区间重叠）")
```

**输出示例**：
```
逻辑回归 AUC: 0.8700, 95% CI: [0.8234, 0.9156]
随机森林 AUC: 0.8900, 95% CI: [0.8498, 0.9302]
提升量可能不显著（置信区间重叠）
```

**分析要点**：
- 虽然随机森林 AUC 更高，但置信区间重叠，说明提升量可能不显著
- 这意味着随机森林的优势可能只是运气，不是真实差异
- 在这种情况下，逻辑回归可能更合适（更简单、更快、更可解释）

**提交物**：
- 代码
- Bootstrap 置信区间结果
- 一段分析（3-4 句话）解释提升量是否显著

**评分点**：
- [ ] 正确实现了 Bootstrap
- [ ] 计算了 95% 置信区间
- [ ] 正确判断了提升量是否显著
- [ ] 给出了模型选择建议

---

## AI 协作练习（可选）

下面这段代码是某个 AI 工具生成的"超参数优化"代码：

```python
from sklearn.model_selection import GridSearchCV

# AI 生成的代码
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc'
)

grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳 AUC: {grid_search.best_score_:.4f}")
```

**审查清单**：
- [ ] 它有没有和基线对比？（你能看出这个模型比逻辑 regression 好多少吗？）
- [ ] 它有没有检查过拟合？（训练集和测试集的表现差距？）
- [ ] 它有没有考虑训练时间和预测时间？
- [ ] 它有没有用 Pipeline 防止数据泄漏？
- [ ] 它有没有说明模型选择的理由？

**你的修订版**（3-4 句话）：
```
（用你自己的话写，修正上述问题）
例如："AI 推荐的模型 AUC 0.91，比逻辑回归提升 0.04。但这个提升量是否显著？建议先用 Bootstrap 估计置信区间，再考虑是否值得牺牲可解释性。此外，AI 没有使用 Pipeline，可能存在数据泄漏风险。"
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
- [ ] 如果遇到困难，参考了 `starter_code/solution.py`，请说明参考了哪些部分

---

## 提示与帮助

如果你在完成作业时遇到困难：
1. 回顾 CHAPTER.md 中的示例代码
2. 参考本周的 StatLab 示例（`examples/11_statlab_tree_models.py`）
3. 查阅 scikit-learn 官方文档（不要害怕查文档！）
4. 如果你遇到困难，可以参考 `starter_code/solution.py`（但不要直接复制）

**记住**：作业的目的是巩固理解，不是完美复制代码。即使遇到困难，也要尝试用自己的话解释问题和思路。

---

祝你本周学习愉快！记住老潘的话：**"没有基线对比的模型选择，不是分析，是炫技。"**
