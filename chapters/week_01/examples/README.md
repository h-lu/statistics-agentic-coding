# Week 01 示例代码

本目录包含 Week 01 的所有示例代码，用于演示统计三问、数据类型判断、pandas 基础操作和数据卡生成。

## 示例文件说明

### 01_three_questions.py
**演示**：统计三问（描述/推断/预测）的区别

运行方式：
```bash
python3 chapters/week_01/examples/01_three_questions.py
```

预期输出：
- 打印三种企鹅的平均嘴峰长度（描述性统计）
- 说明三类问题的区别和适用场景

---

### 02_data_types.py
**演示**：数值型 vs 分类型数据的区别，以及常见错误

运行方式：
```bash
python3 chapters/week_01/examples/02_data_types.py
```

预期输出：
- 展示 pandas 自动推断的类型
- ❌ 错误示范：对分类型数据算均值（"平均物种"）
- ✅ 正确做法：使用 category 类型
- 对比数值型数据的描述统计

---

### 03_pandas_basics.py
**演示**：pandas 基础操作及常见错误恢复

运行方式：
```bash
python3 chapters/week_01/examples/03_pandas_basics.py
```

预期输出：
- 数据加载（shape、head、dtypes、缺失统计）
- 类型转换（object → category）
- 常见错误与恢复方式：
  - 路径问题（FileNotFoundError）
  - 编码问题（UnicodeDecodeError）
  - 日期解析问题

---

### 04_data_card.py
**演示**：生成数据卡（数据的"身份证"）

运行方式：
```bash
python3 chapters/week_01/examples/04_data_card.py
```

预期输出：
- 在 `examples/` 目录下生成 `data_card.md` 文件
- 数据卡包含：
  - 数据来源
  - 字典字典
  - 规模概览
  - 缺失概览
  - 数据类型分布

---

### 99_statlab.py
**演示**：StatLab 超级线入口脚本

运行方式：
```bash
python3 chapters/week_01/examples/99_statlab.py
```

预期输出：
- 在 `examples/output/` 目录下生成 `report.md`
- 这是本周的 StatLab 里程碑：最小可用报告

---

## 输出文件

运行示例后会生成以下文件：

```
examples/
├── data_card.md          # 由 04_data_card.py 生成
└── output/
    └── report.md         # 由 99_statlab.py 生成
```

---

## 数据集

所有示例都使用 seaborn 内置的 Palmer Penguins 数据集，无需额外下载。

数据集信息：
- **来源**：Palmer Station, Antarctica LTER
- **样本数**：344 只企鹅
- **特征**：7 个（物种、岛屿、嘴峰长度、嘴峰深度、鳍肢长度、体重、性别）
- **缺失值**：部分样本有缺失

---

## 依赖安装

```bash
pip install pandas seaborn
```

---

## 注意事项

1. **路径问题**：示例使用 seaborn 内置数据集，无路径问题。如果你从本地文件读取，请使用相对路径（如 `data/penguins.csv`）而非绝对路径。

2. **类型转换**：分类型数据（species、island、sex）应该转成 `category` 类型，而非 `object` 或数值型。

3. **可复现性**：StatLab 报告（99_statlab.py）是可复现分析的基础，下周会在此基础上增量修改。

---

## 与正文的对应关系

| 示例文件 | 对应章节 |
|---------|---------|
| 01_three_questions.py | 第 1 节：统计三问 |
| 02_data_types.py | 第 2 节：数据类型判断 |
| 03_pandas_basics.py | 第 3 节：pandas 基础 |
| 04_data_card.py | 第 4 节：数据卡生成 |
| 99_statlab.py | 第 5 节：StatLab 起步 |
