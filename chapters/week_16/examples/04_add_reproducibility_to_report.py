"""
示例：添加可复现章节到报告

本例演示如何生成一个完整的"可复现分析"章节，
包含依赖、随机性、数据版本、运行命令。

运行方式：python3 chapters/week_16/examples/04_add_reproducibility_to_report.py
预期输出：在 report/report.md 末尾追加"可复现分析"章节
"""
from __future__ import annotations

from pathlib import Path


def generate_reproducibility_section() -> str:
    """
    生成可复现分析章节（Markdown 格式）

    返回：
        str: 可复现分析章节的 Markdown 内容
    """
    section = """

## 可复现分析

本报告的所有分析都可以在一台新机器上复现。以下是复现步骤：

### 依赖安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt**：
```text
numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
scikit-learn==1.5.0
statsmodels==0.14.0
matplotlib==3.9.0
seaborn==0.13.0
```

### 固定随机性

所有分析使用固定随机种子 `RANDOM_SEED = 42`，确保结果可复现。

关键位置：
- 数据划分：`train_test_split(..., random_state=42)`
- 采样：`np.random.seed(42)`
- 模型初始化：`Model(random_state=42)`

**为什么要固定随机性？**
- 数据划分的随机性会影响训练集/测试集分布
- 采样算法（如 Bootstrap）的随机性会影响结果
- 固定随机种子后，同样的代码和数据会得到完全相同的结果

### 数据版本

- 数据来源：`data/user_behavior_20250315.csv`
- 数据版本：2025-03-15
- 数据快照：`data/snapshots/user_behavior_20250315.csv.gz`（压缩备份）

**如何验证数据完整性？**
```bash
# 计算 MD5 哈希
md5sum data/user_behavior_20250315.csv

# 对比快照哈希（应该一致）
md5sum data/snapshots/user_behavior_20250315.csv.gz
```

### 运行分析

```bash
# 克隆仓库
git clone https://github.com/yourusername/statlab-project.git
cd statlab-project

# 切换到最终版本
git checkout final

# 运行完整分析
python scripts/run_full_analysis.py
```

**输出**：
- 分析报告：`report/report_final.md`
- 可视化图表：`report/figures/`
- 模型文件：`models/`

### 可复现检查清单

在重新运行分析前，请确认：

- [ ] Python 版本：3.10 或 3.11
- [ ] 依赖版本：与 requirements.txt 一致
- [ ] 随机种子：已设置为 42
- [ ] 数据文件：与快照哈希一致
- [ ] 运行环境：建议使用 Linux 或 macOS（Windows 可能需要调整路径）

### 常见问题

**Q1: 为什么我的 R² 与报告不一致？**
- 检查随机种子是否正确设置
- 检查数据文件哈希是否一致
- 检查依赖版本是否与 requirements.txt 一致

**Q2: 如何在不同的数据集上复现分析？**
- 替换 `data/user_behavior_20250315.csv`
- 更新 `DATA_VERSION` 和 `DATA_SOURCE` 常量
- 重新运行 `scripts/run_full_analysis.py`

**Q3: 可以使用不同的随机种子吗？**
- 可以，但结果会有细微差异
- 建议记录新种子，以便追溯

---
"""
    return section


def add_reproducibility_to_report(report_path: str = "report/report.md",
                                 output_path: str = "report/report_with_reproducibility.md"):
    """
    在报告末尾添加"可复现分析"章节

    参数：
        report_path: 原始报告路径
        output_path: 输出报告路径

    返回：
        bool: 是否成功添加
    """
    report_file = Path(report_path)

    # 检查文件是否存在
    if not report_file.exists():
        print(f"⚠️  报告文件不存在: {report_path}")
        return False

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report = f.read()
    except Exception as e:
        print(f"⚠️  读取报告失败: {e}")
        return False

    # 检查是否已经有"可复现分析"
    if "## 可复现分析" in report:
        print("⚠️  报告已经包含'可复现分析'，跳过")
        return False

    # 生成可复现分析章节
    reproducibility_section = generate_reproducibility_section()

    # 在报告末尾添加
    updated_report = report + reproducibility_section

    # 保存更新后的报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ 可复现分析章节已添加到: {output_path}")
    return True


def generate_requirements_txt(output_path: str = "report/requirements.txt"):
    """
    生成 requirements.txt 文件

    参数：
        output_path: 输出文件路径

    返回：
        bool: 是否成功生成
    """
    requirements = """numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
scikit-learn==1.5.0
statsmodels==0.14.0
matplotlib==3.9.0
seaborn==0.13.0
"""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(requirements)

    print(f"✅ requirements.txt 已生成: {output_path}")
    return True


def print_reproducibility_tips():
    """打印可复现性提示"""
    print("\n" + "=" * 50)
    print("可复现分析的三层标准：")
    print("=" * 50)
    print("""
1. 能跑（Runs）
   - 代码能执行，不报错
   - 最低要求，但不够

2. 可复现（Reproducible）
   - 换一台机器，用同样的数据和代码，能得到同样的结论
   - 需要：依赖明确、随机性固定、数据版本记录

3. 可审计（Auditable）
   - 任何人能检查每一步为什么这么做
   - 需要：清洗决策记录、模型选择理由、假设检查结果

本脚本帮助你实现第 2 层：可复现。
    """)


def main():
    """执行完整的流程"""
    print("\n" + "=" * 50)
    print("可复现分析章节生成工具")
    print("=" * 50 + "\n")

    # 1. 打印可复现性提示
    print_reproducibility_tips()

    # 2. 生成 requirements.txt
    print("\n" + "=" * 50)
    print("生成 requirements.txt...")
    print("=" * 50)
    generate_requirements_txt()

    # 3. 添加可复现分析到报告
    print("\n" + "=" * 50)
    print("添加可复现分析章节到报告...")
    print("=" * 50)

    success = add_reproducibility_to_report(
        report_path="report/report.md",
        output_path="report/report_with_reproducibility.md"
    )

    if success:
        print("\n" + "=" * 50)
        print("✅ 可复现分析章节生成完成")
        print("=" * 50)
        print("\n建议：")
        print("  1. 检查 report/report_with_reproducibility.md")
        print("  2. 根据实际依赖版本修改 requirements.txt")
        print("  3. 验证分析脚本是否能复现报告结果")
        print("  4. 将 report_with_reproducibility.md 重命名为 report.md")
    else:
        print("\n" + "=" * 50)
        print("⚠️  处理未完成，请检查报告文件")
        print("=" * 50)


if __name__ == "__main__":
    main()
