# Week 15 作业：高级统计计算——当数据太多、太快、太复杂时

> "高维不等于高信息。有时候，10 个精心设计的特征比 50 个冗余特征更有价值。"
> — 老潘

## 作业概述

本周作业将带你完成一个完整的**降维与聚类分析流程**：从理解维度灾难，到用 PCA 压缩高维数据，再到用 K-means 发现隐藏分组，最后将数学输出翻译成业务含义。

你将使用**电商客户行为数据**（50 个特征），回答核心问题：**"高维数据中有哪些隐藏结构？如何用降维和聚类发现这些结构？"**——这是无监督学习的核心问题。

---

## 基础层（必做）

### 任务 1：维度灾难——高维数据的陷阱

小北本周遇到了一个真实的业务问题：公司给了他一份包含 50 个特征的数据集——访问频次、购买频次、客单价、各个品类的偏好分数、时间模式指标……

他把所有特征都塞进了逻辑回归模型，结果：训练集准确率 98%，测试集准确率 62%。

老潘看了一眼，说："**你遇到的不只是过拟合，还有维度灾难**。"

**你的任务**：

1. 回答以下问题：
   - 什么是维度灾难？高维数据会给统计分析带来哪些问题？
   - 什么时候需要降维？给三个信号
   - 特征选择和降维有什么本质区别？

2. 用代码验证维度灾难：计算不同维度下，随机点之间的平均距离

**输入示例**（仅供参考格式）：

```
问题 1：什么是维度灾难？

维度灾难是指：当特征数量（维度）增加时，数据在高维空间中会变得极其稀疏，
导致许多统计方法失效。

具体表现：
- 数据密度：低维时点之间很近，高维时点之间距离几乎相等
- 样本需求：需要指数级增长的样本才能"填满"空间
- 过拟合风险：模型会记住噪声，泛化能力差
- 可视化：无法直接可视化高维数据

问题 2：什么时候需要降维？

信号 1：特征数 >> 样本数（如 50 个特征但只有 200 个样本）
信号 2：特征高度相关（相关矩阵显示大量相关系数 > 0.7）
信号 3：无法可视化（想"看"数据结构，但 50 个特征画不出来）

问题 3：特征选择 vs 降维

特征选择：从 50 个特征中挑选 5 个，丢弃 45 个
降维：把 50 个特征"压缩"成 5 个新特征，每个新特征是原特征的线性组合

区别：特征选择保留原始特征的含义，降维创建抽象的新特征
```

**提交物**：
- 一段文字（400-600 字）回答上述问题
- 用自己的话解释，不是复制粘贴

**评分点**：
- [ ] 正确解释了维度灾难的概念和表现
- [ ] 给出了三个需要降维的信号
- [ ] 正确区分了特征选择和降维的本质差异
- [ ] 用自己的话，不是复制粘贴

**常见错误**：
- 认为"特征越多越好"
- 混淆特征选择和降维（认为降维就是"自动选特征"）
- 没有理解高维空间中距离度量失效的问题

---

### 任务 2：PCA 降维——从 50 维到 2 维

阿码本周最大的困惑是："**PCA 降维后，那些主成分到底是什么？怎么解释？**"

老潘的答案是："**看载荷（loading）**。载荷是每个原始特征在主成分中的权重。"

**你的任务**：

1. 对给定的数据做 PCA 降维：
   - 先标准化数据（PCA 对尺度敏感）
   - 保留所有主成分，查看累积解释方差
   - 选择合适的主成分数量

2. 解释主成分：
   - 查看前 3 个主成分的载荷
   - 将主成分翻译成业务语言

3. 可视化降维结果：
   - 画累积方差图
   - 画 2D 散点图（前 2 个主成分）

**输入示例**（仅供参考格式）：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数据说明：
# - X 是 (n_samples, 50) 的 NumPy 数组，代表电商客户行为特征
# - feature_names 是 50 个特征的名称列表（如 total_spend, visit_freq 等）
# - 数据来源：模拟生成的电商客户数据，包含消费、访问、品类偏好等维度
# - y（可选）是客户流失标签（0=留存，1=流失）

# 1. 标准化（PCA 对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA：保留所有主成分
pca = PCA()
pca.fit(X_scaled)

# 3. 查看累积解释方差
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
print(f"前 3 个主成分解释方差: {cumulative_var[2]:.2%}")
print(f"前 10 个主成分解释方差: {cumulative_var[9]:.2%}")

# 4. 选择主成分数量（比如保留 80% 方差）
n_components_80 = np.argmax(cumulative_var >= 0.8) + 1
print(f"保留 80% 方差需要 {n_components_80} 个主成分")

# 5. 查看主成分载荷
loadings_pc1 = pd.DataFrame({
    'feature': feature_names,
    'loading': pca.components_[0]
}).sort_values('loading', ascending=False)

print("第一主成分载荷（前 10 个特征）:")
print(loadings_pc1.head(10))
```

**输出示例**（仅供参考格式）：

```
前 3 个主成分解释方差: 45.2%
前 10 个主成分解释方差: 78.5%
保留 80% 方差需要 12 个主成分

第一主成分载荷（前 10 个特征）:
            feature  loading
total_spend 总消费   0.42
visit_freq  访问频次   0.38
purchase_count 购买次数 0.35
avg_cart_value 平均购物车金额 0.31
...

解释：第一主成分主要受"消费和访问行为"影响 → 可以命名为"综合活跃度"
```

**提交物**：
- 代码
- 累积方差图
- 2D 散点图
- 一段文字解释前 2-3 个主成分的业务含义（200-300 字）

**评分点**：
- [ ] 代码可以运行
- [ ] 正确标准化了数据
- [ ] 计算了累积解释方差
- [ ] 查看了主成分载荷
- [ ] 将主成分翻译成了业务语言
- [ ] 图表清晰、有标注

**常见错误**：
- 忘记标准化数据（PCA 对尺度敏感）
- 没有解释主成分的含义（只给数字不给解释）
- 累积方差图没有标注阈值线（如 80%、90%）
- 2D 散点图没有标注坐标轴的含义

---

### 任务 3：K-means 聚类——发现隐藏分组

小北本周的问题是："**如果我根本没有标签，怎么把数据分组？**"

老潘的答案是："**聚类分析**。但记住：聚类没有'正确答案'，只有'有用的分组'。"

**你的任务**：

1. 对标准化后的数据做 K-means 聚类：
   - 尝试不同的 K 值（2-10）
   - 用轮廓系数和肘部法则选择最优 K

2. 评估聚类质量：
   - 计算轮廓系数
   - 解释轮廓系数的含义

3. 解释聚类结果：
   - 查看每个簇的中心点
   - 将簇翻译成业务语言

**输入示例**（仅供参考格式）：

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. 尝试不同的 K 值
silhouette_scores = []
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 轮廓系数：范围 [-1, 1]，越大越好
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

    # 簇内平方和
    inertias.append(kmeans.inertia_)

# 2. 选择最优 K（轮廓系数最大的 K）
best_K = K_range[np.argmax(silhouette_scores)]
print(f"最优 K 值（轮廓系数）: {best_K}")
print(f"最大轮廓系数: {max(silhouette_scores):.3f}")

# 3. 用最优 K 做最终聚类
final_kmeans = KMeans(n_clusters=best_K, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# 4. 查看簇中心（反标准化回原始尺度）
centroids_original = scaler.inverse_transform(final_kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=feature_names)

print("\n各簇在关键特征上的均值:")
print(centroids_df[['total_spend', 'visit_freq', 'discount_usage']])
```

**输出示例**（仅供参考格式）：

```
最优 K 值（轮廓系数）: 3
最大轮廓系数: 0.523

各簇在关键特征上的均值:
    total_spend  visit_freq  discount_usage
0       3200.00        45.00           0.85
1       8500.00       120.00           0.20
2        800.00         8.00           0.60

业务解释：
- 簇 0：中等消费、中等频次、高优惠券使用 → 价格敏感型客户
- 簇 1：高消费、高频次、低优惠券使用 → 高价值活跃客户
- 簇 2：低消费、低频次、中等优惠券使用 → 流失风险客户
```

**提交物**：
- 代码
- 肘部法则图和轮廓系数图
- 一段文字解释聚类结果和业务含义（200-300 字）

**评分点**：
- [ ] 代码可以运行
- [ ] 尝试了多个 K 值
- [ ] 用轮廓系数和肘部法则选择了 K
- [ ] 解释了轮廓系数的含义
- [ ] 将簇翻译成了业务语言
- [ ] 图表清晰、有标注

**常见错误**：
- 没有标准化数据（K-means 对尺度敏感）
- 只用了一个 K 值（没有尝试多个 K）
- 没有评估聚类质量（没有看轮廓系数）
- 把聚类当成分类（误以为"簇 2 就是流失客户"）

---

## 进阶层（推荐完成）

### 任务 4：降维 + 聚类组合——先用 PCA 降维，再聚类

小北本周学到新技巧："**能不能先用 PCA 降维，再在低维空间做聚类？**"

老潘的答案是："**可以，而且有时候效果更好**。PCA 降噪后再聚类，可以减少噪声的干扰。"

**你的任务**：

1. 比较两种方法：
   - 方法 1：在原始 50 维数据上做 K-means
   - 方法 2：先用 PCA 降到 10 维，再做 K-means

2. 评估两种方法：
   - 比较轮廓系数
   - 比较运行时间

3. 回答：什么时候应该先降维再聚类？

**输入示例**（仅供参考格式）：

```python
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 方法 1：在原始数据上聚类
start_time = time.time()
kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_original = kmeans_original.fit_predict(X_scaled)
time_original = time.time() - start_time
silhouette_original = silhouette_score(X_scaled, labels_original)

# 方法 2：PCA 降维后再聚类
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

start_time = time.time()
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)
time_pca = time.time() - start_time
silhouette_pca = silhouette_score(X_pca, labels_pca)

print(f"原始 50 维: 轮廓系数 = {silhouette_original:.3f}, 时间 = {time_original:.2f}s")
print(f"PCA 10 维: 轮廓系数 = {silhouette_pca:.3f}, 时间 = {time_pca:.2f}s")
```

**提交物**：
- 代码
- 两种方法的比较结果
- 一段文字解释：什么时候应该先降维再聚类？（200-300 字）

**评分点**：
- [ ] 正确实现了两种方法
- [ ] 比较了轮廓系数和运行时间
- [ ] 解释了两种方法的优劣
- [ ] 给出了使用建议

---

### 任务 5：PCA vs 特征选择——什么时候用哪个？

阿码本周的问题是："**什么时候用 PCA，什么时候用特征选择？**"

老潘的答案是："**取决于你是否需要解释**。如果你必须向产品经理解释'哪些特征影响流失'，用特征选择。如果你只是想做可视化或建模，用 PCA。"

**你的任务**：

1. 比较两种方法：
   - 方法 1：PCA 降维到 10 维
   - 方法 2：SelectKBest 选择 10 个最重要的特征

2. 用逻辑回归评估：
   - 比较交叉验证准确率
   - 比较可解释性

3. 回答：什么时候用 PCA，什么时候用特征选择？

**输入示例**（仅供参考格式）：

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 方法 1：PCA 降维
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# 方法 2：特征选择
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)

# 用逻辑回归评估
model = LogisticRegression(max_iter=1000)

scores_pca = cross_val_score(model, X_pca, y, cv=5)
scores_selected = cross_val_score(model, X_selected, y, cv=5)

print(f"PCA 降维: 平均准确率 = {scores_pca.mean():.3f}")
print(f"特征选择: 平均准确率 = {scores_selected.mean():.3f}")

# 查看特征选择选择了哪些特征
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"\n选择的特征: {selected_features}")
```

**提交物**：
- 代码
- 两种方法的比较结果
- 一段文字解释：什么时候用 PCA，什么时候用特征选择？（200-300 字）

**评分点**：
- [ ] 正确实现了两种方法
- [ ] 比较了模型性能
- [ ] 讨论了可解释性的差异
- [ ] 给出了决策建议

---

## 挑战层（可选）

### 任务 6：层次聚类分析——用树状图发现分组结构

老潘在公司经常用层次聚类："**当你不知道该分几个簇时，层次聚类的树状图会给你启发**。"

**你的任务**：

1. 用 scipy 做层次聚类：
   - 计算链接矩阵（用 'ward' 方法）
   - 画树状图

2. 从树状图选择 K 值：
   - 观察树状图的结构
   - 在合适的高度"切"出簇

3. 比较 K-means 和层次聚类的结果

**输入示例**（仅供参考格式）：

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 1. 计算层次聚类的链接矩阵
Z = linkage(X_scaled, method='ward')  # ward 方法最小化簇内方差

# 2. 画树状图
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='lastp', p=30)  # 只显示最后 30 个簇
plt.xlabel('样本数量')
plt.ylabel('距离')
plt.title('层次聚类树状图')
plt.savefig('output/15_dendrogram.png', dpi=150)

# 3. 从树状图选择 K（比如观察到明显的 3 个分支）
n_clusters = 3
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X_scaled)

# 4. 比较 K-means 和层次聚类的轮廓系数
from sklearn.metrics import silhouette_score
silhouette_hierarchical = silhouette_score(X_scaled, labels_hierarchical)
print(f"层次聚类轮廓系数: {silhouette_hierarchical:.3f}")
```

**提交物**：
- 代码
- 树状图
- 一段文字解释：从树状图中观察到了什么？K-means 和层次聚类的结果有什么差异？（200-300 字）

**评分点**：
- [ ] 正确使用 scipy 做了层次聚类
- [ ] 绘制了树状图
- [ ] 从树状图中选择了合理的 K 值
- [ ] 比较了 K-means 和层次聚类的结果

---

### 任务 7：聚类稳定性分析——用 Bootstrap 评估

阿码本周最后的问题是："**我今天跑一次 K-means，明天再跑一次，结果可能不一样。怎么知道聚类结果是否稳定？**"

老潘的答案是："**用 Bootstrap 验证稳定性**。多次重采样数据，看聚类结果是否一致。"

**你的任务**：

1. 实现 Bootstrap 聚类稳定性分析：
   - 重采样数据 100 次
   - 每次做 K-means
   - 计算聚类结果的一致性（用 adjusted_rand_score）

2. 回答：聚类结果稳定吗？

**输入示例**（仅供参考格式）：

```python
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
import numpy as np

n_bootstrap = 100
ari_scores = []

for i in range(n_bootstrap):
    # 重采样数据
    X_resampled, y_resampled = resample(X_scaled, labels,
                                         replace=True, random_state=i)

    # 做聚类
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_resampled = kmeans.fit_predict(X_resampled)

    # 计算与原始聚类的一致性（只看重采样部分的样本）
    # 注意：这里简化处理，实际需要匹配样本索引
    ari = adjusted_rand_score(labels[:len(labels_resampled)], labels_resampled)
    ari_scores.append(ari)

print(f"平均 ARI: {np.mean(ari_scores):.3f}")
print(f"ARI 标准差: {np.std(ari_scores):.3f}")

if np.mean(ari_scores) > 0.7:
    print("聚类结果稳定（ARI > 0.7）")
else:
    print("聚类结果不稳定，需要调整参数或方法")
```

**提交物**：
- 代码
- Bootstrap 分析结果
- 一段文字解释：聚类结果是否稳定？为什么？（200-300 字）

**评分点**：
- [ ] 正确实现了 Bootstrap 分析
- [ ] 计算了聚类一致性（ARI）
- [ ] 解释了稳定性结果
- [ ] 给出了改进建议（如果不稳定）

---

## AI 协作练习（可选，主导期）

下面这段文字是某个 AI 工具生成的"客户分群分析结论"：

> "我们对客户数据进行了 K-means 聚类分析，发现数据可以分成 3 个簇。第一簇的平均消费是 3200 元，第二簇是 8500 元，第三簇是 800 元。轮廓系数是 0.52，说明聚类效果很好。建议公司对高消费的第二簇客户投入更多资源。"

**审查清单**：
- [ ] 它说明了 K 值是怎么选择的吗？（是肘部法则？轮廓系数？还是业务需求？）
- [ ] 它做了数据标准化吗？（K-means 对尺度敏感，没有标准化会导致方差大的特征主导距离）
- [ ] 它比较了不同 K 值的结果吗？（有没有尝试 K=2, K=4？）
- [ ] 它解释了每个簇的业务含义吗？（不只是"第一簇、第二簇"，而是"价格敏感型、高价值型"）
- [ ] 它验证了聚类稳定性吗？（重采样数据后，聚类结果还一致吗？）
- [ ] "对第二簇投入更多资源"——这个建议有成本收益分析吗？（第二簇有多少客户？投入产出比是多少？）

**你的修订版**（用你自己的话写，修正上述问题）：

```
例如："AI 的结论有几个问题：第一，没有说明 K 值是怎么选择的（为什么是 3 而不是 4？）。第二，没有说明是否标准化了数据（K-means 对尺度敏感）。第三，没有解释每个簇的业务含义（不只是给数字）。第四，没有验证聚类稳定性（不同初始化会得到相同结果吗？）。第五，建议'对第二簇投入更多资源'没有考虑成本（第二簇有多少客户？）。修订版应该明确：K 值选择方法、标准化步骤、簇的业务解释、稳定性验证、成本收益分析。"
```

**提交物**：
- 审查清单（勾选哪些问题存在）
- 你的修订版（3-5 句话）

---

## StatLab 本周任务

老潘说："**StatLab 报告需要升级——从'单一特征分析'到'结构化客户分群'**。"

**你的任务**：

1. **PCA 降维可视化**：
   - 画累积方差图
   - 画 2D 散点图（前 2 个主成分）
   - 解释前 2-3 个主成分的业务含义

2. **K-means 聚类分析**：
   - 用轮廓系数选择 K 值
   - 计算每个簇的中心点
   - 解释每个簇的业务含义

3. **降维 + 聚类组合**（可选）：
   - 先用 PCA 降维
   - 再在低维空间做聚类
   - 比较与直接聚类的差异

4. **生成报告**：
   - 汇总降维和聚类结果
   - 将数学输出翻译成业务语言

**StatLab 模板**（仅供参考格式）：

```markdown
## 客户分群分析

### PCA 降维结果
- 原始特征数: 50
- 前 10 个主成分解释方差: 78.5%
- 前 2 个主成分解释方差: 32.1%

主成分解释:
- PC1（综合活跃度，18.5%）: 主要受总消费、访问频次、购买次数影响
- PC2（价格敏感度，13.6%）: 主要受优惠券使用、价格敏感度影响
- PC3（品类多样性，8.2%）: 主要受品类偏好、多样性指标影响

可视化: 见 `output/pca_2d_scatter.png`

### K-means 聚类结果
- 簇数量: 3
- 轮廓系数: 0.52

各簇特征:
| 簇 | 样本数 | 总消费 | 访问频次 | 优惠券使用 | 业务解释 |
|----|--------|--------|----------|------------|----------|
| 0 | 450 | 8500 | 120 | 20% | 高价值活跃客户 |
| 1 | 320 | 3200 | 45 | 85% | 价格敏感型客户 |
| 2 | 230 | 800 | 8 | 60% | 流失风险客户 |

业务建议:
- 高价值活跃客户: 提供专属服务、忠诚度计划
- 价格敏感型客户: 针对性促销、个性化折扣
- 流失风险客户: 调研流失原因、激活活动

### 可视化
- PCA 累积方差图: `output/pca_cumulative_variance.png`
- PCA 2D 散点图: `output/pca_2d_scatter.png`
- 聚类评估图: `output/clustering_evaluation.png`
```

**提交物**：
- 更新后的 StatLab 报告
- PCA 和聚类分析的可视化图

---

## 提交检查清单

在提交作业前，请确认：

- [ ] 代码可以运行（或注明哪些部分是伪代码）
- [ ] 输出结果包含在提交中（或截图）
- [ ] 分析部分用你自己的话写（不是复制粘贴）
- [ ] PCA 前标准化了数据
- [ ] K-means 前标准化了数据
- [ ] 查看了主成分载荷并解释了业务含义
- [ ] 用轮廓系数选择了 K 值
- [ ] 将簇标签翻译成了业务语言
- [ ] 如果遇到困难，参考了 `starter_code/solution.py`，请说明参考了哪些部分

---

## 提示与帮助

如果你在完成作业时遇到困难：

1. 回顾 CHAPTER.md 中的示例代码
2. 参考本周的 StatLab 示例（`examples/15_dimensionality_reduction.py`、`examples/15_clustering.py`）
3. 查阅 scikit-learn 官方文档：https://scikit-learn.org/stable/modules/decomposition.html
4. 如果你对 PCA 或 K-means 的参数不熟悉，可以参考 `starter_code/solution.py`（但不要直接复制）

**记住**：作业的目的是巩固理解，不是完美复制代码。即使遇到困难，也要尝试用自己的话解释问题和思路。

---

祝你本周学习愉快！记住老潘的话：**"降维和聚类的核心是'发现结构，不是创造结构'。你是在揭示数据中已有的模式，不是强行把数据分组。如果聚类结果没有业务解释，要么 K 值不对，要么数据本身就没有明显分组。"**
