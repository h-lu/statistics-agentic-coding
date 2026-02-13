# Week 13 作业标准答案要点

> **注意**：这是给教师和助教参考的评分标准，不是给学生看的"参考答案"。
> 数据分析题可能有多个正确答案，重点是逻辑是否自洽、方法是否正确。

---

## 任务 1：因果三层级识别

### 标准答案

| 问题 | 层级 | 能回答的方法 | 理由 |
|------|------|-------------|------|
| A：用券用户消费高多少？ | 关联 | 相关分析、均值差、t 检验 | 只是描述两组差异，不回答干预效果 |
| B：如果发券，消费会提高多少？ | 干预 | RCT、因果推断（匹配、IV、DID） | 需要控制混杂，识别因果效应 |
| C：如果张三没用券，他会消费多少？ | 反事实 | 反事实推理、个体因果效应模型 | 无法直接观测，需要强假设 |
| D：教育和收入相关吗？ | 关联 | 相关系数、回归 | 只回答相关性，不说明因果方向 |
| E：如果培训，绩效会提高吗？ | 干预 | RCT、倾向评分匹配、工具变量 | 需要识别因果效应 |
| F：如果患者没服药，血压会是多少？ | 反事实 | 反事实推理 | 个体反事实，无法直接观测 |

### 关键区分

**问题 A vs 问题 B**：
- A：观察性问题（Observational）——描述现状
- B：干预性问题（Interventional）——预测干预效果
- 为什么"相关方法"不能回答"干预问题"？
  - 混杂变量（如活跃度）同时影响用券和消费
  - 直接比较会夸大或低估因果效应
  - 需要调整混杂才能识别因果效应

**把相关当成因果的真实场景**：
- 新闻："喝咖啡的人更长寿" → 可能是"高收入的人既喝咖啡也长寿"（混杂）
- 产品："推送通知后活跃度提高" → 可能是"活跃用户更易看到通知"（反向因果）
- 医疗："服药后康复率提高" → 可能是"轻症患者更易服药"（选择偏差）

---

## 任务 2：画因果图（DAG）

### 标准因果图

```
用户活跃度 (Activity)
    ↓         ↓
历史消费 (History_Spend) → 优惠券 (Coupon) → 消费 (Spending)
    ↑                                    ↑
    └──────────────────────────────────────┘
                    ↓
              使用频率 (Frequency)
                    ↓
              消费 (Spending)
```

或更清晰的版本：

```
Activity → Coupon
   ↓         ↓
History_Spend → Spending
   ↑         ↑
   └─────────┘

Coupon → Frequency → Spending
```

### 三种结构识别

**1. 链式（中介）**：
- `Coupon → Frequency → Spending`
- Frequency 是**中介变量**（Mediator）
- 如果调整 Frequency，会切断因果路径，低估优惠券效应

**2. 叉式（混杂）**：
- `Activity → Coupon` 和 `Activity → Spending`
- `History_Spend → Coupon` 和 `History_Spend → Spending`
- Activity 和 History_Spend 是**混杂变量**（Confounder）
- 必须调整它们才能识别因果效应

**3. 对撞（选择偏差）**：
- 本例中无明显对撞结构
- 如果加入"用户注册"（只有注册用户才有消费数据），则可能形成对撞：
  - `Coupon → 注册 ← Spending`（假设用券和高消费都促进注册）
  - 如果只看注册用户，调整"注册"会制造虚假关联

### 问题回答

**Q1：直接比较会被哪些混杂影响？**
- A：Activity（活跃用户既爱领券也爱消费）
- A：History_Spend（高消费用户既关注优惠也继续高消费）
- 结果：50 元的差异被夸大，部分是混杂效应

**Q2：Frequency 是中介还是混杂？**
- A：中介变量
- 理由：`Coupon → Frequency → Spending`，优惠券通过提高使用频率影响消费
- 如果调整 Frequency：会切断因果路径，低估真实效应

**Q3：如果调整 Frequency 会怎样？**
- A：低估优惠券效应
- 原因：调整中介 = 问"保持使用频率不变，优惠券是否有效"
- 但我们真正想知道的是：优惠券的**总效应**（包括通过 Frequency 的间接效应）

---

## 任务 3：后门准则应用

### 标准答案

**因果路径**：
- `在线课程 → 考试成绩`（因果路径，想估计的效应）

**后门路径**：
1. `在线课程 ← 学习动机 → 考试成绩`
2. `在线课程 ← 学习动机 ← 家庭收入 → 考试成绩`（如果收入影响动机）
3. `在线课程 ← 家庭收入 → 考试成绩`（如果收入直接影响课程使用和成绩）

**后门准则三条件**：
1. 调整集 Z 不包含 Treatment 的后代（不调整中介）
2. Z 阻断所有后门路径
3. Z 不打开新的虚假路径（不调整对撞）

**正确的调整集**：
- `{学习动机}` 或 `{家庭收入}` 或 `{学习动机, 家庭收入}`
- 最小调整集：`{学习动机}`

**不能调整的变量**（如果有）：
- 学习时长（中介）：`在线课程 → 学习时长 → 考试成绩`
- 调整中介会切断因果路径

### 对比分析表（示例数字）

| 调整策略 | 估计值 | 是否有偏？ | 原因 |
|---------|--------|-----------|------|
| 未调整   | +15 分 | 有偏 | 学习动机混杂未控制（动机高的学生既用课程也考得好） |
| 错误调整（学习时长） | +5 分 | 有偏 | 中介被切断，低估总效应 |
| 正确调整（学习动机） | +10 分 | 无偏 | 后门路径被阻断，因果路径保留 |

### DoWhy 验证

```python
from dowhy import CausalModel

causal_graph = """digraph {
    Motivation -> Online_Course;
    Motivation -> Exam_Score;
    Online_Course -> Exam_Score;
}"""

model = CausalModel(
    data=df,
    treatment="Online_Course",
    outcome="Exam_Score",
    graph=causal_graph
)

identified_estimand = model.identify_effect()
print(identified_estimand)
```

**预期输出**：
```
Estimand type: nonparametric-ate
Estimand expression:
  d
------(Expectation(Exam_Score|Online_Course,Motivation))
dOnline_Course
```

DoWhy 自动识别出需要调整 `Motivation`。

---

## 任务 4：倾向评分匹配实现

### 标准代码框架

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 1. 估计倾向评分
ps_model = LogisticRegression()
ps_model.fit(df[['activity', 'history_spend']], df['coupon'])
df['propensity_score'] = ps_model.predict_proba(df[['activity', 'history_spend']])[:, 1]

# 2. 匹配
treated = df[df['coupon'] == 1]
control = df[df['coupon'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])
matched_control = control.iloc[indices.flatten()]

# 3. 计算 ATT
att = (treated['spending'].values - matched_control['spending'].values).mean()

# 4. Bootstrap 置信区间
n_boot = 500
att_samples = []
for i in range(n_boot):
    treated_boot = treated.sample(n=len(treated), replace=True, random_state=i)
    control_boot = control.sample(n=len(control), replace=True, random_state=i)
    # ... 匹配和计算 ATT ...
    att_samples.append(att_boot)

ci_low = np.percentile(att_samples, 2.5)
ci_high = np.percentile(att_samples, 97.5)
```

### 标准输出示例

**匹配质量检查**：

| 协变量 | 匹配前 SMD | 匹配后 SMD | 改善 |
|--------|-----------|-----------|------|
| activity | 0.85 | 0.05 | ✓ 平衡 |
| history_spend | 0.92 | 0.03 | ✓ 平衡 |

**因果效应估计**：
- ATT：28.5 元
- 95% CI：[20.3, 36.7] 元
- 结论：在控制了混杂后，优惠券使消费提高约 29 元（置信区间不包含 0，效应显著）

### 关键检查点

**SMD 计算**：
```python
def calculate_smd(treated, control, var):
    treated_mean = treated[var].mean()
    control_mean = control[var].mean()
    treated_var = treated[var].var()
    control_var = control[var].var()
    pooled_std = np.sqrt((treated_var + control_var) / 2)
    smd = (treated_mean - control_mean) / pooled_std
    return smd
```

**平衡性标准**：
- SMD < 0.1：平衡良好
- 0.1 ≤ SMD < 0.2：可接受
- SMD ≥ 0.2：不平衡，需改进匹配方法

**可视化要求**：
- 匹配前：两张直方图（倾向评分分布），明显分离
- 匹配后：两张直方图基本重叠

---

## AI 协作练习：常见 AI 错误示例

### AI 可能的错误输出（示例）

> "优惠券确实有效！使用优惠券的用户平均消费 150 元，未使用的用户平均消费 100 元，差异为 50 元（p < 0.001）。这个差异非常显著，说明优惠券使消费提高了 50 元。"

### 问题识别

**高危错误**：
1. **混淆相关与因果**：50 元是均值差（关联），不是因果效应
2. **未调整混杂**：没有提到活跃度和历史消费的混杂
3. **夸大结论**：没有说明限制（如只适用于短期效应）

**中危错误**：
4. **未报告置信区间**：只有 p 值和点估计，没有不确定性量化
5. **未识别策略**：没有说明"为什么不用调整"（隐含假设随机分配）

**低危问题**：
6. **缺少可视化**：没有倾向评分分布图或平衡性检查

### 修订版结论（示例）

> "观察数据显示，使用优惠券的用户平均消费比未使用的高 50 元（p < 0.001），但这是**关联**不是**因果**。活跃用户和高消费用户更可能领取优惠券，这会夸大优惠券的真实效应。
>
> 在控制了用户活跃度和历史消费后（倾向评分匹配），优惠券的**因果效应**估计为 29 元（95% CI [20, 38] 元）。
>
> **限制**：这个结论假设没有未观察混杂（如用户收入），且只适用于观察期内的短期效应。我们估计的是处理组平均效应（ATT），不代表对所有用户的效果相同。"

---

## StatLab 集成检查点

### 报告应包含的内容

```markdown
## 因果推断

### 研究问题

本章回答的因果问题是：

**"如果给用户发放优惠券，他的消费金额会提高多少？"**

注意：这与关联问题不同。关联问题是"用券用户和未用券用户的消费差异"，而因果问题是"发券这个行为的因果效应"。

### 因果假设

我们用因果图（DAG）表达因果假设：

![因果图](causal_dag.png)

**图解**：
- **处理变量（X）**：优惠券使用（0=未使用，1=使用）
- **结果变量（Y）**：消费金额
- **混杂变量（Z）**：用户活跃度、历史消费（同时影响用券和消费）
- **中介变量（M）**：使用频率（优惠券通过提高使用频率影响消费）

### 识别策略

根据**后门准则（Backdoor Criterion）**，我们需要调整以下混杂变量：

- 用户活跃度：同时影响用券和消费
- 历史消费：同时影响用券和消费

**不调整的变量**：
- 使用频率：中介变量，调整会切断因果路径

### 因果效应估计

我们用两种方法估计因果效应，以检查稳健性。

#### 方法 1：带调整集的回归

| 变量 | 系数 | 95% CI | p 值 |
|------|------|--------|------|
| 优惠券使用 | **30.2** 元 | [20.5, 39.9] | <0.001 |
| 用户活跃度 | 5.1 元 | [3.2, 7.0] | <0.001 |
| 历史消费 | 0.3 元 | [0.2, 0.4] | <0.001 |

**解读**：在控制了用户活跃度和历史消费后，优惠券使消费金额提高 **30.2 元**（95% CI [20.5, 39.9] 元）。

#### 方法 2：倾向评分匹配

匹配质量检查：

![倾向评分分布（匹配前后）](propensity_score_matching.png)

匹配后的因果效应：

| 指标 | 估计值 | 95% CI |
|------|--------|--------|
| **ATT（处理组平均处理效应）** | **28.5** 元 | [20.3, 36.7] |

**解读**：倾向评分匹配估计的因果效应为 **28.5 元**（95% CI [20.3, 36.7] 元），与回归结果接近，结论稳健。

### 结论边界

**我们能回答的（因果结论）**：
- 给用户发放优惠券，会使他的消费金额提高约 **29 ± 8 元**（回归和匹配的平均）
- 这个结论在调整了混杂变量（活跃度、历史消费）后成立

**我们不能回答的（只是相关或未知）**：
- 优惠券对所有用户的效果相同（我们估计的是平均效应，异质性未知）
- 长期效应（数据只有 3 个月，无法回答 1 年后的效应）
- 个体因果效应（反事实："如果这个用户没用券，他会消费多少"）

**限制**：
- 存在未观察混杂的可能（如用户收入，如果数据中没有）
- 匹配会丢弃无法匹配的样本（可能影响外推性）
- 效应估计的是 ATT（对用券用户的效果），不是 ATE（对全人群的效果）
```

---

## 验收代码示例（供助教使用）

### 自动检查脚本（伪代码）

```python
# check_week13_submission.py

def check_causal_levels(submission):
    """检查任务 1：因果三层级"""
    correct_answers = {
        'A': 'association',
        'B': 'intervention',
        'C': 'counterfactual',
        'D': 'association',
        'E': 'intervention',
        'F': 'counterfactual'
    }
    # 解析学生答案，计算匹配度
    matches = check_answer_key(submission, correct_answers)
    return matches >= 5, f"匹配 {matches}/6"

def check_dag_structure(submission):
    """检查任务 2：因果图结构"""
    # 检查是否有处理变量、结果变量、混杂变量
    has_treatment = 'coupon' in submission.variables.lower()
    has_outcome = 'spending' in submission.variables.lower()
    has_confounders = any(c in submission.variables.lower()
                         for c in ['activity', 'history'])
    # 检查边是否正确
    required_edges = [
        ('activity', 'coupon'),
        ('activity', 'spending'),
        ('history_spend', 'coupon'),
        ('history_spend', 'spending'),
        ('coupon', 'spending')
    ]
    edges_correct = check_edges(submission.edges, required_edges)
    return has_treatment and has_outcome and has_confounders and edges_correct

def check_backdoor_application(submission):
    """检查任务 3：后门准则"""
    # 检查是否正确识别混杂
    correct_confounders = {'activity', 'history_spend', 'motivation'}
    student_adjustment_set = set(submission.adjustment_set)
    # 检查调整集是否包含关键混杂
    adjusts_correct = correct_confounders.intersection(student_adjustment_set)
    # 检查是否调整了中介（不应调整）
    does_not_adjust_mediator = 'frequency' not in student_adjustment_set
    return len(adjusts_correct) >= 2 and does_not_adjust_mediator

def check_psm_implementation(submission):
    """检查任务 4：倾向评分匹配"""
    # 检查是否有倾向评分
    has_propensity_score = 'propensity_score' in submission.columns
    # 检查是否有 ATT 估计
    has_att = submission.att is not None
    # 检查是否有置信区间
    has_ci = submission.ci_low is not None and submission.ci_high is not None
    # 检查 ATT 是否在合理范围（10-50 元）
    att_reasonable = 10 <= submission.att <= 50 if has_att else False
    # 检查是否有 SMD
    has_smd = 'smd' in submission.report.lower()
    return all([has_propensity_score, has_att, has_ci, att_reasonable, has_smd])

def check_ai_review(submission):
    """检查 AI 协作练习"""
    # 检查是否识别出"混淆相关与因果"
    has_confusion_error = '混淆' in submission.review or \
                         'correlation' in submission.review.lower()
    # 检查修订版是否区分关联和因果
    has_distinction = '关联' in submission.revision and \
                     '因果' in submission.revision
    # 检查是否有置信区间
    has_ci = '置信区间' in submission.revision or 'ci' in submission.revision.lower()
    return has_confusion_error and has_distinction and has_ci

# 综合评分
def grade_submission(submission_path):
    results = {
        'task1': check_causal_levels(submission),
        'task2': check_dag_structure(submission),
        'task3': check_backdoor_application(submission),
        'task4': check_psm_implementation(submission),
        'task_ai': check_ai_review(submission)
    }
    # 计算总分
    total_score = calculate_weighted_score(results)
    return total_score, results
```

---

## 常见错误与评分建议

### 错误 1：混淆相关与因果

**表现**：
- 直接比较两组均值差异就下结论"优惠券有效"
- 没有提到混杂变量
- 没有画因果图

**扣分**：
- 核心任务（任务 2-4）：-10 分
- 如果多处出现，累加扣分

### 错误 2：盲目调整一切变量

**表现**：
- 调整了中介变量（如使用频率）
- 没有用后门准则科学选择调整集
- 认为"多调整总没错"

**扣分**：
- 任务 3（后门准则）：-8 分
- 任务 4（匹配）：如果调整了中介，ATT 估计有偏，-5 分

### 错误 3：未检查匹配质量

**表现**：
- 倾向评分匹配后没有计算 SMD
- 没有画匹配前后的分布图
- 无法判断匹配是否有效

**扣分**：
- 任务 4：-10 分（质量检查占 8 分）

### 错误 4：夸大因果结论

**表现**：
- 说"优惠券有效"，没有说明限制
- 没有报告置信区间
- 没有区分 ATT 和 ATE

**扣分**：
- 每处：-3 分
- 如果多处夸大，-5 到 -10 分

### 错误 5：AI 审查不完整

**表现**：
- 只识别了 AI 的小错误（如术语不严谨）
- 遗漏了核心错误（如混淆相关与因果）
- 没有修订版结论

**扣分**：
- AI 协作练习：-5 到 -10 分（取决于遗漏的严重性）
