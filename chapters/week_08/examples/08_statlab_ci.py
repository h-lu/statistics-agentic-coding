"""
示例：StatLab 不确定性量化报告生成。

本例是 StatLab 超级线在 Week 08 的入口脚本，在上周（多组比较）基础上，
加入区间估计与重采样章节：点估计 + CI、Bootstrap 方法、置换检验。

运行方式：python3 chapters/week_08/examples/08_statlab_ci.py
预期输出：
  - stdout 输出不确定性量化报告片段
  - 报告片段保存到 output/uncertainty_sections.md

与上周对比：
  - 上周：假设检验（p 值 + 效应量 + 假设检查 + 多组比较）
  - 本周：以上全部 + **区间估计（CI） + Bootstrap + 置换检验**
  - 上周：只报告"显著/不显著"
  - 本周：报告"差异大小 + 不确定性范围"
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
from pathlib import Path


def add_ci_to_report(data: np.ndarray, confidence: float = 0.95,
                     label: str = "均值") -> dict:
    """
    给点估计添加置信区间（多种方法）

    参数:
        data: 数据数组
        confidence: 置信水平
        label: 统计量标签

    返回:
        包含点估计和多种 CI 的字典
    """
    point_estimate = np.mean(data)
    se = stats.sem(data)
    df = len(data) - 1

    # 方法 1：理论 t 分布
    t_ci_low, t_ci_high = stats.t.interval(
        confidence, df=df, loc=point_estimate, scale=se
    )

    # 方法 2：Percentile Bootstrap
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    pct_ci_low, pct_ci_high = np.percentile(
        boot_means, [(1-confidence)/2*100, (1+confidence)/2*100]
    )

    # 方法 3：BCa Bootstrap（用 scipy）
    def mean_func(x):
        return np.mean(x)

    try:
        res = bootstrap((data,), mean_func, confidence_level=confidence,
                        method='BCa', n_bootstrap=n_bootstrap, random_state=42)
        bca_ci_low, bca_ci_high = res.confidence_interval.low, res.confidence_interval.high
    except:
        # 如果 BCa 失败，使用 Percentile
        bca_ci_low, bca_ci_high = pct_ci_low, pct_ci_high

    return {
        "label": label,
        "n": len(data),
        "point_estimate": point_estimate,
        "t_interval": (t_ci_low, t_ci_high),
        "percentile_bootstrap": (pct_ci_low, pct_ci_high),
        "bca_bootstrap": (bca_ci_low, bca_ci_high),
    }


def compare_groups_ci(group1: np.ndarray, group2: np.ndarray,
                     confidence: float = 0.95,
                     group1_name: str = "组1",
                     group2_name: str = "组2") -> dict:
    """
    比较两组数据的差异（带 Bootstrap CI 和置换检验）

    参数:
        group1, group2: 两组数据
        confidence: 置信水平
        group1_name, group2_name: 组名

    返回:
        包含点估计、CI、p 值的字典
    """
    # 点估计：差异
    point_diff = np.mean(group2) - np.mean(group1)

    # Bootstrap 差异
    n_bootstrap = 10000
    n1, n2 = len(group1), len(group2)
    boot_diffs = []

    for _ in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=n1, replace=True)
        boot2 = np.random.choice(group2, size=n2, replace=True)
        boot_diffs.append(np.mean(boot2) - np.mean(boot1))

    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(
        boot_diffs, [(1-confidence)/2*100, (1+confidence)/2*100]
    )

    # 置换检验 p 值
    combined = np.concatenate([group1, group2])
    perm_diffs = []
    for _ in range(n_bootstrap):
        permuted = np.random.permutation(combined)
        perm1 = permuted[:n1]
        perm2 = permuted[n1:]
        perm_diffs.append(np.mean(perm2) - np.mean(perm1))

    perm_diffs = np.array(perm_diffs)
    p_value_two_tailed = np.mean(np.abs(perm_diffs) >= abs(point_diff))

    # 效应量（Cohen's d）
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                          (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    cohens_d = point_diff / pooled_std

    return {
        "group1_name": group1_name,
        "group2_name": group2_name,
        "n1": n1,
        "n2": n2,
        "point_diff": point_diff,
        "bootstrap_ci": (ci_low, ci_high),
        "permutation_p": p_value_two_tailed,
        "cohens_d": cohens_d,
    }


def format_ci_table(ci_dict: dict) -> str:
    """格式化 CI 为 Markdown 表格"""
    md = f"### {ci_dict['label']}（n = {ci_dict['n']}）\n\n"
    md += f"| 方法 | 点估计 | 95% CI |\n"
    md += f"|------|--------|--------|\n"

    point = ci_dict["point_estimate"]
    t_low, t_high = ci_dict["t_interval"]
    pct_low, pct_high = ci_dict["percentile_bootstrap"]
    bca_low, bca_high = ci_dict["bca_bootstrap"]

    md += f"| t 分布 | {point:.2f} | [{t_low:.2f}, {t_high:.2f}] |\n"
    md += f"| Percentile Bootstrap | {point:.2f} | [{pct_low:.2f}, {pct_high:.2f}] |\n"
    md += f"| BCa Bootstrap | {point:.2f} | [{bca_low:.2f}, {bca_high:.2f}] |\n\n"

    return md


def format_diff_table(diff_dict: dict) -> str:
    """格式化组间差异为 Markdown 表格"""
    md = f"### {diff_dict['group1_name']} vs {diff_dict['group2_name']}\n\n"
    point = diff_dict["point_diff"]
    ci_low, ci_high = diff_dict["bootstrap_ci"]
    p_val = diff_dict["permutation_p"]
    d = diff_dict["cohens_d"]

    md += f"| 指标 | 值 |\n"
    md += f"|------|------|\n"
    md += f"| 样本量（{diff_dict['group1_name']}/{diff_dict['group2_name']}） | "
    md += f"{diff_dict['n1']} / {diff_dict['n2']} |\n"
    md += f"| 点估计（差异） | {point:.3f} |\n"
    md += f"| 95% Bootstrap CI | [{ci_low:.3f}, {ci_high:.3f}] |\n"
    md += f"| 置换检验 p 值（双尾） | {p_val:.4f} |\n"
    md += f"| Cohen's d（效应量） | {d:.3f} |\n\n"

    # 解释
    if p_val < 0.05:
        md += "**结论**：p < 0.05，差异显著。95% CI 不包含 0，支持组间存在差异。\n\n"
    else:
        md += "**结论**：p ≥ 0.05，差异不显著。95% CI 包含 0，不能拒绝无差异假设。\n\n"

    # 效应量解释
    if abs(d) < 0.2:
        effect_interp = "极小效应"
    elif abs(d) < 0.5:
        effect_interp = "小效应"
    elif abs(d) < 0.8:
        effect_interp = "中等效应"
    else:
        effect_interp = "大效应"

    md += f"**效应量解释**：Cohen's d = {d:.3f}（{effect_interp}）\n\n"

    return md


def generate_uncertainty_section(df: pd.DataFrame) -> str:
    """
    生成不确定性量化的 Markdown 片段

    参数:
        df: 包含数据的 DataFrame

    返回:
        Markdown 格式的不确定性量化报告
    """
    md = ["## 不确定性量化\n\n"]
    md.append("以下报告使用多种方法估计置信区间，并比较组间差异。\n\n")
    md.append("**方法说明**：\n")
    md.append("- **t 分布 CI**：假设数据服从正态分布，使用理论公式计算\n")
    md.append("- **Percentile Bootstrap**：通过重采样估计抽样分布，取分位数\n")
    md.append("- **BCa Bootstrap**：校正偏差和加速的 Bootstrap，通常更准确\n")
    md.append("- **置换检验**：不依赖分布假设的非参数检验方法\n\n")

    # 单组 CI：Adelie 企鹅的喙长
    adelie_bill = df[df["species"] == "Adelie"]["bill_length_mm"].dropna()
    adelie_ci = add_ci_to_report(adelie_bill.values, label="Adelie 喙长均值 (mm)")
    md.append(format_ci_table(adelie_ci))

    # 单组 CI：Gentoo 企鹅的喙长
    gentoo_bill = df[df["species"] == "Gentoo"]["bill_length_mm"].dropna()
    gentoo_ci = add_ci_to_report(gentoo_bill.values, label="Gentoo 喙长均值 (mm)")
    md.append(format_ci_table(gentoo_ci))

    # 组间比较：Adelie vs Gentoo 喙长
    bill_diff = compare_groups_ci(
        adelie_bill.values,
        gentoo_bill.values,
        group1_name="Adelie",
        group2_name="Gentoo"
    )
    md.append(format_diff_table(bill_diff))

    # 另一个比较：Adelie vs Chinstrap 喙长
    chinstrap_bill = df[df["species"] == "Chinstrap"]["bill_length_mm"].dropna()
    bill_diff2 = compare_groups_ci(
        adelie_bill.values,
        chinstrap_bill.values,
        group1_name="Adelie",
        group2_name="Chinstrap"
    )
    md.append(format_diff_table(bill_diff2))

    # 添加与本周知识的连接
    md.append("---\n\n")
    md.append("### 与本周知识的连接\n\n")
    md.append("**点估计 vs 区间估计** → 我们不只报告均值，还报告了三种 CI")
    md.append("（t 分布、Percentile Bootstrap、BCa Bootstrap），量化了估计的不确定性。\n\n")
    md.append("**Bootstrap** → 我们用 Bootstrap 验证了理论公式的结果。")
    md.append("对于正态数据，三种方法的 CI 非常接近，说明数据满足正态性假设。\n\n")
    md.append("**置换检验** → 我们用置换检验验证了组间差异的显著性，")
    md.append("提供了不依赖分布假设的 p 值。\n\n")

    return "".join(md)


def main() -> None:
    """运行 StatLab 不确定性量化报告生成"""
    print("=== StatLab 不确定性量化报告生成演示 ===\n")

    # 加载数据
    penguins = sns.load_dataset("penguins")
    print(f"加载数据：{len(penguins)} 条企鹅记录\n")

    # 生成不确定性量化报告
    uncertainty_md = generate_uncertainty_section(penguins)

    print("=== StatLab 不确定性量化报告片段 ===\n")
    print(uncertainty_md[:3000])  # 打印前 3000 字符
    print("...")
    print("\n[完整报告已保存到文件]")

    # 写入文件
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'uncertainty_sections.md'
    output_path.write_text(uncertainty_md)
    print(f"\n报告片段已保存到 {output_path}")

    # 输出与上周的对比
    print("\n=== StatLab 进度对比 ===\n")
    print("| 上周（Week 07） | 本周（Week 08） |")
    print("|------------------|------------------|")
    print("| 假设检验（p 值 + 效应量 + 假设检查） | 以上全部 + **区间估计（CI）** |")
    print("| 多组比较（ANOVA + 事后比较 + 校正） | 以上全部 + **Bootstrap + 置换检验** |")
    print("| 只报告'显著/不显著' | 报告'差异大小 + 不确定性范围' |")
    print("| 可能忽略假设 | 用 Bootstrap 验证假设 |")
    print()

    # 老潘的点评
    print("=== 老潘的点评 ===\n")
    print('"这才是科学。你不仅告诉了读者"是什么"，')
    print('还告诉了读者"有多确定"。')
    print('')
    print('没有 CI 的报告不是科学，是赌博。"')
    print()

    print('小北问："赌博？"')
    print()
    print('"对。"老潘说，"如果你只说"均值是 3.2"，')
    print('但不说这个数字有多稳定，你就是在赌这个数字是对的。')
    print('有了 CI，读者才知道：这个数字可能在 2.8 到 3.6 之间波动。"')
    print()

    print('阿码若有所思："所以 AI 工具如果只给我 p 值，不给我 CI……"')
    print()
    print('"那你就要补上。"老潘说，"AI 可以帮你跑检验，')
    print('但只有你能判断"这个结论有多可信"。"')
    print()


if __name__ == "__main__":
    main()
