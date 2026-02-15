"""
示例：异常值处理决策（错误 vs 发现 vs 边界）。

本例演示如何判断和处理异常值：
1. 错误型异常值（Errors）：数据录入错误、测量误差 → 修正或删除
2. 发现型异常值（Discoveries）：真实但极端的观测 → 保留并单独分析
3. 边界型异常值（Edge cases）：合法但不常见 → 谨慎处理

运行方式：python3 chapters/week_03/examples/04_outlier_decision.py
预期输出：
- output/cleaning_log.md：清洗日志（Markdown 格式）
- 控制台输出：决策过程和理由

核心知识点：
- 统计规则（IQR/Z-score）只给出"候选异常值清单"
- 真正的问题：这个值到底是什么？错误？发现？边界？
- 老潘的黄金法则：删一个异常值，就要写一句为什么
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# 清洗日志类
# =============================================================================

class CleaningLog:
    """清洗日志：记录每一个数据决策"""

    def __init__(self):
        self.logs = []

    def add(self,
            variable: str,
            issue: str,
            original_value: any,
            action: Literal["remove", "replace", "keep", "correct"],
            reason: str,
            n_affected: int = None) -> None:
        """
        添加一条清洗记录

        Args:
            variable: 变量名
            issue: 问题类型（如 "outlier", "missing", "invalid"）
            original_value: 原始值
            action: 处理动作
            reason: 处理理由（老潘说：这是最重要的！）
            n_affected: 影响的行数
        """
        self.logs.append({
            "variable": variable,
            "issue": issue,
            "original_value": original_value,
            "action": action,
            "reason": reason,
            "n_affected": n_affected
        })

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame(self.logs)

    def to_markdown(self) -> str:
        """生成 Markdown 格式的清洗日志"""
        df = self.to_dataframe()
        if df.empty:
            return "## 数据清洗\n\n本数据集质量良好，未发现需要处理的问题。\n"

        md = ["## 数据清洗与决策记录\n\n"]
        md.append("以下记录了本报告对数据的所有处理决策及其理由。\n\n")

        # 按问题类型分组
        for issue in df["issue"].unique():
            issue_logs = df[df["issue"] == issue]
            issue_name = {
                "error_outlier": "错误型异常值",
                "discovery_outlier": "发现型异常值",
                "missing_values": "缺失值",
                "invalid_values": "无效值"
            }.get(issue, issue)

            md.append(f"### {issue_name}\n\n")

            for _, row in issue_logs.iterrows():
                md.append(f"- **{row['variable']}**：{row['action']}\n")
                md.append(f"  - 原始值：{row['original_value']}\n")
                md.append(f"  - 理由：{row['reason']}\n")
                if row["n_affected"]:
                    md.append(f"  - 影响：{row['n_affected']} 行\n")
                md.append("\n")

        return "".join(md)


# =============================================================================
# 类型1：错误型异常值
# =============================================================================

def demonstrate_error_outliers() -> tuple[pd.DataFrame, CleaningLog]:
    """
    演示错误型异常值的处理

    特征：明显不符合物理或业务约束
    示例：年龄 250 岁、收入 -1000 元、体重 10000g
    处理：修正（如果有把握）或标记为缺失
    """
    print("="*60)
    print("【类型1】错误型异常值（Errors）")
    print("="*60)

    # 创建带错误的数据
    data = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 250, 28, 35],  # 250 显然是错的（多打了一个 0？）
        "income": [50000, 60000, 55000, 70000, -1000]  # -1000 显然是错的
    })

    print("\n原始数据：")
    print(data)
    print()

    log = CleaningLog()

    # 检测年龄异常值（业务规则：年龄应该在 0-120 之间）
    age_outliers = data[(data["age"] < 0) | (data["age"] > 120)]
    print("检测到的年龄异常值：")
    print(age_outliers[["user_id", "age"]])
    print()

    # 处理：把明显错误的值标记为缺失
    data_cleaned = data.copy()
    for idx, row in age_outliers.iterrows():
        original_value = row["age"]
        data_cleaned.loc[idx, "age"] = np.nan

        # 判断：是多打了一个 0 吗？
        if original_value > 100:
            likely_correct = original_value // 10
            log.add(
                variable="age",
                issue="error_outlier",
                original_value=original_value,
                action="replace_with_nan",
                reason=f"超出合理范围（>120）。疑似录入错误（{original_value} → 可能应为 {likely_correct}）。因无法确认，标记为缺失。",
                n_affected=1
            )

    # 检测收入异常值（业务规则：收入不应为负数）
    income_outliers = data[data["income"] < 0]
    print("检测到的收入异常值：")
    print(income_outliers[["user_id", "income"]])
    print()

    for idx, row in income_outliers.iterrows():
        data_cleaned.loc[idx, "income"] = np.nan
        log.add(
            variable="income",
            issue="error_outlier",
            original_value=row["income"],
            action="replace_with_nan",
            reason="收入不应为负数，疑似数据录入错误。标记为缺失。",
            n_affected=1
        )

    print("清洗后数据：")
    print(data_cleaned)
    print()

    return data_cleaned, log


# =============================================================================
# 类型2：发现型异常值
# =============================================================================

def demonstrate_discovery_outliers() -> pd.DataFrame:
    """
    演示发现型异常值的处理

    特征：真实但极端的观测值
    示例：批发商客户的超高消费、超级用户的活跃度
    处理：保留并单独分析（不要删除！）

    老潘的故事：当年直接删掉了所有"看起来奇怪"的点，
                 结果删掉了最重要的信号——那是 B2B 客户！
    """
    print("\n" + "="*60)
    print("【类型2】发现型异常值（Discoveries）")
    print("="*60)
    print("\n老潘的故事：")
    print("  我当年分析电商用户数据，发现有一个用户的年消费额是 100 万美元，")
    print("  是平均值的 100 倍。IQR 和 Z-score 都会标记它为异常值。")
    print("  但我没有直接删掉，而是去查了一下——发现这是一个批发商客户。")
    print()
    print("  这种异常值不应该删除，而应该单独分析！")
    print()

    # 模拟数据：大部分是 B2C 客户，少数是 B2B 客户
    np.random.seed(42)
    n_b2c = 100
    n_b2b = 5

    b2c_spending = np.random.normal(loc=5000, scale=2000, size=n_b2c)
    b2c_spending = np.maximum(b2c_spending, 100)  # 确保非负
    b2b_spending = np.random.normal(loc=50000, scale=10000, size=n_b2b)

    data = pd.DataFrame({
        "user_id": range(1, n_b2c + n_b2b + 1),
        "annual_spending": np.concatenate([b2c_spending, b2b_spending]),
        "customer_type": ["B2C"] * n_b2c + ["B2B"] * n_b2b
    })

    print("模拟数据：前10行")
    print(data.head(10))
    print()
    print(f"B2C 客户平均消费：${data[data['customer_type']=='B2C']['annual_spending'].mean():.0f}")
    print(f"B2B 客户平均消费：${data[data['customer_type']=='B2B']['annual_spending'].mean():.0f}")
    print()

    # 用 IQR 检测异常值
    all_spending = data["annual_spending"]
    q25 = all_spending.quantile(0.25)
    q75 = all_spending.quantile(0.75)
    iqr = q75 - q25
    upper = q75 + 1.5 * iqr

    outliers = data[all_spending > upper]
    print(f"IQR 规则检测到 {len(outliers)} 个异常值：")
    print(outliers.head())
    print()

    print("⚠️  不要直接删除这些异常值！")
    print("  → 它们可能代表一个不同的群体（如 B2B vs B2C）")
    print("  → 应该单独分组分析，而不是混在一起")
    print("  → 或者用对数变换压缩尺度")
    print()

    return data


# =============================================================================
# 类型3：边界型异常值
# =============================================================================

def demonstrate_edge_cases() -> None:
    """
    演示边界型异常值的处理

    特征：合法但不常见的值
    示例：年龄 100 岁（不是不可能，但很罕见）
    处理：保留，但要标注

    老潘的建议：不要自动删掉边界值。
                 先问一句：这个值在我的业务场景下合理吗？
    """
    print("\n" + "="*60)
    print("【类型3】边界型异常值（Edge Cases）")
    print("="*60)

    # 创建数据
    data = pd.DataFrame({
        "person_id": [1, 2, 3, 4, 5],
        "age": [25, 30, 100, 28, 35],
        "context": ["大学生", "上班族", "退休人员", "研究生", "教师"]
    })

    print("\n示例数据：")
    print(data)
    print()

    print("问题：age=100 是异常值吗？")
    print()
    print("  - 如果数据集是'大学生'：100 岁可能是错误")
    print("  - 如果数据集是'退休金领取者'：100 岁完全合理")
    print()
    print("老潘的建议：")
    print("  → 不要自动删掉边界值")
    print("  → 结合业务场景判断")
    print("  → 如果保留，可以添加标注（如 'is_age_extreme' 列）")
    print()


# =============================================================================
# 演示完整的清洗流程
# =============================================================================

def demonstrate_full_cleaning_workflow() -> CleaningLog:
    """演示完整的数据清洗决策流程"""
    print("\n" + "="*60)
    print("【完整流程】从检测到决策到记录")
    print("="*60)

    # 创建模拟数据
    np.random.seed(42)
    data = pd.DataFrame({
        "id": range(1, 101),
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.randint(20000, 100000, size=100)
    })

    # 添加一些问题数据
    data.loc[5, "age"] = 250  # 错误
    data.loc[10, "income"] = -5000  # 错误
    data.loc[50, "income"] = 500000  # 可能是发现型异常值

    log = CleaningLog()

    print("\n步骤1：用统计规则标记候选异常值")
    print("─"*60)

    # 1. 年龄检查（业务规则）
    age_outliers = data[(data["age"] < 0) | (data["age"] > 120)]
    if len(age_outliers) > 0:
        print(f"\n检测到 {len(age_outliers)} 个年龄异常值")
        for idx, row in age_outliers.iterrows():
            print(f"  ID {row['id']}: age = {row['age']}")
            # 判断：是错误还是边界？
            if row["age"] > 120:
                log.add(
                    variable="age",
                    issue="error_outlier",
                    original_value=row["age"],
                    action="replace_with_nan",
                    reason=f"超出合理范围（>120），疑似录入错误",
                    n_affected=1
                )
                data.loc[idx, "age"] = np.nan

    # 2. 收入检查（业务规则 + 统计规则）
    income_outliers_negative = data[data["income"] < 0]
    if len(income_outliers_negative) > 0:
        print(f"\n检测到 {len(income_outliers_negative)} 个负数收入")
        for idx, row in income_outliers_negative.iterrows():
            print(f"  ID {row['id']}: income = ${row['income']}")
            log.add(
                variable="income",
                issue="error_outlier",
                original_value=row["income"],
                action="replace_with_nan",
                reason="收入不应为负数",
                n_affected=1
            )
            data.loc[idx, "income"] = np.nan

    # 3. 收入检查（IQR 规则）
    income_data = data["income"].dropna()
    q25 = income_data.quantile(0.25)
    q75 = income_data.quantile(0.75)
    iqr = q75 - q25
    upper = q75 + 1.5 * iqr

    income_outliers_high = data[data["income"] > upper]
    if len(income_outliers_high) > 0:
        print(f"\n检测到 {len(income_outliers_high)} 个高收入异常值（IQR 规则）")
        for idx, row in income_outliers_high.iterrows():
            print(f"  ID {row['id']}: income = ${row['income']:,.0f}")

            # 判断：是错误还是发现？
            # 这里需要人工判断：是数据录入错误（多打一个 0），
            #   还是真实的高收入用户？
            print(f"    → 需要人工判断：是录入错误还是真实的高收入用户？")
            print(f"    → 假设经核实为真实的高收入用户（如企业主）")
            log.add(
                variable="income",
                issue="discovery_outlier",
                original_value=f"${row['income']:,.0f}",
                action="keep",
                reason=f"IQR 规则标记为异常值，但经核实为真实的高收入用户。保留并单独分析。",
                n_affected=1
            )

    print("\n步骤2：生成清洗日志")
    print("─"*60)
    print("\n清洗日志（DataFrame 格式）：")
    print(log.to_dataframe())
    print()

    return log


# =============================================================================
# 保存清洗日志到文件
# =============================================================================

def save_cleaning_log(log: CleaningLog, output_dir: Path) -> None:
    """保存清洗日志到 Markdown 文件"""
    md_content = log.to_markdown()

    output_file = output_dir / "cleaning_log.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"清洗日志已保存到：{output_file}")


# =============================================================================
# 老潘的黄金法则
# =============================================================================

def print_laopan_golden_rule() -> None:
    """输出老潘的黄金法则"""
    print("\n" + "="*60)
    print("老潘的黄金法则")
    print("="*60)
    print()
    print("在公司里，你删一个异常值，就要写一句为什么。")
    print()
    print("我当年吃过亏。我做过一个分析，直接删掉了所有'看起来奇怪'的点，")
    print("结果结论完全变了——我删掉的是最重要的信号。")
    print()
    print("从那以后，我学会了：")
    print("  → 异常值不是敌人，而是需要理解的故事")
    print("  → 统计规则只给出'候选异常值清单'")
    print("  → 真正的工作是判断：这是错误，还是发现？")
    print()


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "="*60)
    print("异常值处理决策示例")
    print("="*60)

    output_dir = setup_output_dir()

    # 演示三种类型的异常值
    _, log1 = demonstrate_error_outliers()
    demonstrate_discovery_outliers()
    demonstrate_edge_cases()

    # 演示完整流程
    log = demonstrate_full_cleaning_workflow()

    # 保存清洗日志
    save_cleaning_log(log, output_dir)

    # 老潘的黄金法则
    print_laopan_golden_rule()

    print("="*60)
    print("核心结论")
    print("="*60)
    print("1. 错误型异常值：修正或标记为缺失")
    print("2. 发现型异常值：保留并单独分析（不要删除！）")
    print("3. 边界型异常值：结合业务场景判断")
    print("4. 清洗日志：记录每一个决策和理由")
    print("\n下一步：把清洗日志整合进 StatLab 报告（见示例99）")


if __name__ == "__main__":
    main()
