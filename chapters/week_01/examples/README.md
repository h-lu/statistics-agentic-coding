# Week 01 示例代码说明

本目录包含 Week 01 的所有示例代码，用于演示本周的核心概念。

## 文件列表

### 核心示例（按学习顺序）

1. **01_three_questions.py** - 统计三问分类器
   - 演示如何区分描述/推断/预测三类统计问题
   - 包含交互式命令行工具
   - 运行方式：`python3 01_three_questions.py`

2. **02_data_types.py** - 数据类型检测器
   - 演示数值型/分类型的识别
   - 包含"坏示例 vs 好示例"对比
   - 运行方式：`python3 02_data_types.py`

3. **03_pandas_basics.py** - pandas 基础操作
   - 演示 DataFrame 的索引机制（loc vs iloc）
   - 演示常见错误与正确用法
   - 运行方式：`python3 03_pandas_basics.py`

4. **04_data_card.py** - 数据卡生成器
   - 演示如何为数据集生成"身份证"
   - 输出 `data_card.md` 文件
   - 运行方式：`python3 04_data_card.py`

### StatLab 超级线

5. **99_statlab.py** - StatLab 初始化脚本
   - 生成第一版 `report.md`
   - 这是整个 16 周项目的起点
   - 运行方式：`python3 99_statlab.py`
   - 注意：需要先配置 `DATA_PATH` 和 `metadata`

## 测试

运行所有测试：
```bash
python -m pytest ../tests/ -v
```

运行特定测试：
```bash
python -m pytest ../tests/test_examples.py::test_three_questions_module -v
```

## 依赖

所有示例都需要以下依赖：
- pandas
- numpy

安装方式：
```bash
pip install pandas numpy
```

## 代码风格

所有示例遵循以下规范：
- 使用类型提示（type hints）
- 包含完整的 docstring
- 使用 `if __name__ == "__main__"` 保护主函数
- 代码可独立运行，不依赖其他示例

## 输出示例

运行 `04_data_card.py` 会生成类似以下的 Markdown 输出：

```markdown
# 电商用户行为分析数据卡

## 数据来源
- **来源**：公司内部用户行为数据库，SQL 导出
- **生成时间**：2026-02-11

## 数据描述
2025 年上半年（1-6 月）活跃用户的注册、浏览、购买行为数据。

## 样本规模
- **行数**：8
- **列数**：7
...
```
