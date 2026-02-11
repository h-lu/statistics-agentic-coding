# Week 03 作业：数据清洗与准备

> "数据清洗不是技术问题，而是决策问题。"

本周作业要求你为一组真实数据完成从"原始"到"可分析"的完整清洗流程，并生成一份可追溯的清洗决策日志。

---

## 作业结构

| 层级 | 内容 | 建议时间 |
|------|------|----------|
| 基础作业（必做） | 缺失值分析、异常值检测、特征变换、清洗日志 | 3-4 小时 |
| 进阶作业（选做） | 缺失机制启发式判断、填充策略对比 | 1-2 小时 |
| 挑战作业（选做） | 清洗函数封装与测试 | 1-2 小时 |

---

## 基础作业（必做）

### 任务 1：缺失值分析（30 分）

**目标**：识别缺失模式，判断缺失机制，选择合适的处理策略。

**步骤**：

1. **缺失概览**
   - 计算数据集中每列的缺失数量和缺失率
   - 按缺失率降序排列，找出"问题列"

2. **缺失模式分析**
   - 对缺失率 > 5% 的列，分析缺失是否与某些观测值相关
   - 示例：高收入用户的收入缺失率是否更高？（判断是否为 MAR）
   - 至少分析 2 个字段的缺失相关性

3. **机制判断与策略选择**
   - 为每个有缺失的字段判断：MCAR / MAR / MNAR / 无法判断
   - 选择处理策略：删除 / 常数填充 / 均值填充 / 中位数填充 / 分组填充
   - 记录选择理由

**预期输出示例**：

```text
=== 缺失值分析报告 ===

1. 缺失概览
| 字段 | 缺失数 | 缺失率 |
|------|--------|--------|
| income | 123 | 12.3% |
| age | 15 | 1.5% |

2. 缺失模式分析
- income 字段：高收入用户（消费前 20%）的缺失率 25.0%，低收入用户缺失率 5.2%
  → 判断为 MAR（缺失与观测到的消费行为相关）

3. 处理策略
| 字段 | 机制判断 | 处理策略 | 理由 |
|------|----------|----------|------|
| income | MAR | 按用户等级分组，组内中位数填充 | 利用 MAR 信息，中位数对极端值稳健 |
| age | MCAR | 中位数填充 | 缺失率低，MCAR 假设下填充影响小 |
```

**常见错误**：
- 不分析缺失模式，直接删除所有含缺失的行
- 把 MAR 当成 MCAR 处理，导致系统性偏差
- 用 0 填充所有缺失值（尤其是收入、年龄这类字段）

---

### 任务 2：异常值检测与处理（25 分）

**目标**：用统计方法发现异常点，用业务规则分类处理。

**步骤**：

1. **统计检测**
   - 选择至少 2 个数值列，用 IQR 方法检测异常值
   - 记录每列的 Q1、Q3、IQR、上下界
   - 统计检测到的异常点数量

2. **业务规则分类**
   - 对检测到的异常点，结合其他字段设计业务规则进行分类
   - 至少区分 3 类：错误（需删除）、VIP（需保留）、待确认（需进一步分析）

3. **处理决策**
   - 对"错误"类：删除或修正
   - 对"VIP"类：保留，可考虑对数变换
   - 对"待确认"类：标记但不删除

**预期输出示例**：

```text
=== 异常值处理报告 ===

1. 统计检测（monthly_spend 列）
- Q1 = 120, Q3 = 450, IQR = 330
- 下界 = -375（实际最小值 50，无异常）
- 上界 = 945
- 检测到 23 个异常点（> 945）

2. 业务规则分类
| 规则 | 分类 | 数量 |
|------|------|------|
| 消费 > 10000 且 注册天数 < 7 | suspicious（疑似刷单） | 3 |
| 消费 > 10000 且 用户等级 = 钻石 | VIP | 15 |
| 消费 > 945 且 注册天数 > 365 | high_spend（正常高消费） | 5 |

3. 处理决策
- 删除 3 条 suspicious 记录
- 保留 VIP 和 high_spend，对 monthly_spend 做对数变换以减小极端值影响
```

**常见错误**：
- 直接用 Z-score 检测严重偏斜的数据（Z-score 假设近似正态）
- 不查看异常点的具体特征，直接全部删除
- 把 VIP 用户当成"错误"删除

---

### 任务 3：特征变换（25 分）

**目标**：对数值特征进行缩放，对类别特征进行编码。

**步骤**：

1. **数值特征缩放**
   - 选择至少 2 个数值特征
   - 根据分布形状选择：StandardScaler（近似正态）或 MinMaxScaler（有界数据）
   - 记录变换前后的描述统计对比

2. **类别特征编码**
   - 选择至少 1 个 nominal 类别（如城市、颜色），使用 OneHotEncoder
   - 选择至少 1 个 ordinal 类别（如等级、评分），使用 LabelEncoder 或手动映射
   - 解释为什么选择这种编码方式

3. **变换函数封装**
   - 把变换步骤写成可复用的函数
   - 函数返回变换后的 DataFrame 和变换器（用于新数据）

**预期输出示例**：

```text
=== 特征变换报告 ===

1. 数值特征缩放
| 特征 | 方法 | 变换前范围 | 变换后范围 |
|------|------|------------|------------|
| age | StandardScaler | 18-80 | mean=0, std=1 |
| monthly_income | StandardScaler | 3000-50000 | mean=0, std=1 |

理由：两特征均近似正态分布，StandardScaler 使它们处于可比较尺度

2. 类别特征编码
| 特征 | 类型 | 方法 | 输出示例 |
|------|------|------|----------|
| city | nominal | OneHotEncoder(drop='first') | city_Beijing, city_Shanghai... |
| user_level | ordinal | 手动映射 | bronze=0, silver=1, gold=2... |

理由：city 无顺序关系，用 one-hot；user_level 有明确顺序，用 label 编码
```

**常见错误**：
- 对 nominal 类别用 label 编码（模型会误以为有顺序关系）
- 对严重偏斜的数据用 StandardScaler（应考虑 RobustScaler 或对数变换）
- 不保存变换器，无法应用到新数据

---

### 任务 4：清洗决策日志（20 分）

**目标**：生成完整的清洗决策日志，追加到 report.md。

**要求**：

1. 日志必须包含以下字段的决策记录：
   - 问题描述（发现了什么）
   - 处理策略（做了什么）
   - 选择理由（为什么这么做）
   - 替代方案（考虑过什么其他方法）
   - 影响评估（处理前后数据的变化）

2. 日志格式参考（markdown 表格或列表）

3. 将日志追加到你的 `report.md` 文件中

**预期输出示例**：

```markdown
## 数据清洗与预处理

> 本章记录所有数据清洗决策，确保分析过程可复现、可审计。

### 缺失值处理

| 字段 | 缺失率 | 机制判断 | 处理策略 | 理由 |
|------|--------|----------|----------|------|
| income | 12.3% | MAR | 按用户等级分组，组内中位数填充 | 利用 MAR 信息，中位数稳健 |
| age | 1.5% | MCAR | 全局中位数填充 | 缺失率低，简单策略足够 |

### 异常值处理

| 字段 | 检测方法 | 异常数 | 处理策略 |
|------|----------|--------|----------|
| monthly_spend | IQR | 23 | 分类处理：删除 3 条疑似刷单，保留 20 条 VIP |

### 特征变换

| 字段 | 变换类型 | 方法 | 理由 |
|------|----------|------|------|
| age, monthly_income | 缩放 | StandardScaler | 特征尺度差异大，标准化后公平比较 |
| city | 编码 | OneHotEncoder(drop='first') | Nominal 类别，避免多重共线性 |

---
*清洗日志生成时间：2026-02-11*
```

---

## 进阶作业（选做）

### 任务 5：缺失机制启发式判断（+10 分）

实现一个函数 `detect_missingness_mechanism(df, target_col, correlate_cols)`，该函数：

1. 计算目标列的缺失率
2. 对每个 correlate_col，比较"目标列缺失"和"目标列不缺失"时该列的分布差异
3. 如果某 correlate_col 在两种情况下分布显著不同，提示可能是 MAR
4. 如果缺失与任何观测值都无关，提示可能是 MCAR
5. 返回一个字典，包含判断结果和依据

**提示**：
- 可以用 t 检验比较均值差异
- 可以用卡方检验比较类别分布差异
- 注意：这只是一个启发式判断，不能 100% 确定机制

---

### 任务 6：填充策略对比（+10 分）

对同一个有缺失的数值列，尝试 3 种不同的填充策略：

1. 删除缺失行
2. 全局中位数填充
3. 分组中位数填充（利用 MAR 信息）

然后计算并对比以下指标：
- 填充后的均值、标准差
- 与"完整案例"（仅使用无缺失的数据）的均值差异
- 填充后分布的形状变化（可画直方图对比）

**思考问题**：
- 哪种策略最接近"真实情况"？（假设完整案例代表真实分布）
- 填充是否压缩了方差？压缩了多少？
- 如果缺失是 MNAR，以上策略会有什么偏差？

---

## 挑战作业（选做）

### 任务 7：清洗流水线封装（+10 分）

将你的整个清洗流程封装成一个可复用的模块：

```python
class DataCleaner:
    """数据清洗流水线，支持 fit/transform 模式。"""

    def __init__(self):
        self.missing_strategies = {}
        self.outlier_bounds = {}
        self.scalers = {}
        self.encoders = {}
        self.decisions = []

    def fit(self, df):
        """学习清洗参数（如 scaler 的均值/方差）。"""
        pass

    def transform(self, df):
        """应用清洗变换。"""
        pass

    def get_cleaning_log(self):
        """返回清洗决策日志。"""
        pass
```

要求：
- 支持 `fit` 学习参数、`transform` 应用变换的模式
- 所有决策自动记录到 `decisions` 列表
- 提供 `get_cleaning_log()` 方法生成 markdown 格式日志
- 包含至少 2 个单元测试

---

## 提交要求

### 文件列表

```
week_03/
├── data/
│   ├── raw/              # 原始数据（不要修改）
│   └── processed/        # 清洗后的数据
├── scripts/
│   ├── 01_missing_analysis.py    # 任务 1
│   ├── 02_outlier_detection.py   # 任务 2
│   ├── 03_feature_transform.py   # 任务 3
│   └── 04_generate_log.py        # 任务 4
├── report.md             # 包含清洗决策日志
└── README.md             # 运行说明
```

### Git 提交

至少 3 次 commit，建议：

```bash
# 第一次：完成缺失值分析
git add scripts/01_missing_analysis.py
git commit -m "feat: add missing value analysis"

# 第二次：完成异常值检测
git add scripts/02_outlier_detection.py
git commit -m "feat: add outlier detection with business rules"

# 第三次：完成特征变换和日志
git add scripts/03_feature_transform.py scripts/04_generate_log.py report.md
git commit -m "feat: add feature transformation and cleaning log"
```

### 提交内容

1. 清洗后的数据文件（CSV 格式）
2. 所有脚本文件
3. 包含清洗决策日志的 report.md
4. README.md（说明如何运行脚本复现清洗过程）

---

## 评分标准

详见 `RUBRIC.md`。

---

## 提示与资源

### 推荐数据集

如果你还没有自己的数据集，可以从以下选择：

- **Titanic**：经典的缺失值处理案例
- **UCI Adult**：收入预测，有类别特征编码需求
- **Kaggle House Prices**：数值特征缩放、缺失值处理

### 关键代码片段

**缺失模式分析**：
```python
# 比较缺失组和非缺失组的分布
df_missing = df[df['income'].isna()]
df_not_missing = df[df['income'].notna()]

print("缺失组的平均消费:", df_missing['spend'].mean())
print("非缺失组的平均消费:", df_not_missing['spend'].mean())
```

**IQR 异常值检测**：
```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
```

**StandardScaler**：
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['col1_std', 'col2_std']] = scaler.fit_transform(df[['col1', 'col2']])
```

**OneHotEncoder**：
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['city']])
```

### 遇到困难？

- 参考 `starter_code/solution.py` 中的示例实现
- 回顾 CHAPTER.md 中的代码示例
- 在讨论区提问，附上你的数据和具体错误信息

---

**截止日期**：本周日 23:59
