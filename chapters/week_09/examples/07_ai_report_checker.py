"""
示例：AI 生成的回归报告审查清单

本例演示如何审查 AI 生成的回归分析报告，识别常见问题：
1. 缺少残差诊断
2. 缺少多重共线性检查
3. 误解释系数为因果关系
4. 缺少置信区间

运行方式：python3 chapters/week_09/examples/07_ai_report_checker.py
预期输出：
- 审查清单结果（哪些项目缺失）
- 示例 AI 报告的问题标注
"""
from __future__ import annotations

import re
from typing import Dict, List


# ============================================================================
# 示例 AI 生成的报告（有问题的版本）
# ============================================================================

BAD_AI_REPORT = """
# 房价回归分析报告

## 模型摘要
- 模型类型: 线性回归
- R² = 0.82
- F 统计量 = 45.6, p < 0.001

## 系数结果
| 变量 | 系数 | p 值 |
|------|------|------|
| 面积 | 1.25 | <0.001 |
| 房龄 | -0.38 | 0.012 |

## 结论
面积和房龄显著影响房价。广告费每增加 1 万元，销售额涨 5 万元。
模型拟合良好，可以用于预测。
"""


# ============================================================================
# 示例人工修订后的报告（好的版本）
# ============================================================================

GOOD_REPORT = """
# 房价回归分析报告

## 模型摘要
- 方程: 房价 = 20.5 + 1.25×面积 - 0.38×房龄
- R² = 0.82 (调整 R² = 0.80)
- F(2, 97) = 45.6, p < 0.001

## 系数解释 (95% CI)
| 变量 | 系数 | 标准误 | 95% CI | p 值 |
|------|------|--------|---------|------|
| 截距 | 20.50 | 3.20 | [14.12, 26.88] | <0.001 |
| 面积 | 1.25 | 0.18 | [0.89, 1.61] | <0.001 |
| 房龄 | -0.38 | 0.12 | [-0.62, -0.14] | 0.002 |

**解释**: 在其他变量不变的情况下，面积每增加 1 平米，房价平均上涨 1.25 万元 (95% CI: [0.89, 1.61])。

## 残差诊断
- **线性假设**: 残差 vs 拟合值图显示残差随机散布，无线性模式 ✓
- **正态性**: QQ 图显示残差近似沿对角线分布，Shapiro-Wilk p = 0.08 ✓
- **等方差**: 残差散布在所有拟合值上大致均匀 ✓
- **独立性**: Durbin-Watson = 1.95 ✓

## 多重共线性检查
- 面积 VIF = 1.2, 房龄 VIF = 1.1
- 无严重共线性问题 ✓

## 异常点分析
- Cook's D > 1 的点: 2 个 (索引 #45, #128)
- 删除后系数变化 < 10%
- 模型对异常点稳健 ✓

## 局限性与因果警告
⚠️ **本分析仅描述关联，不能推断因果**。可能的混杂变量包括地段、装修、楼层等。

## 数据来源
- 样本量: n = 100
- 缺失值: 已删除 (3 个观测)
"""


# ============================================================================
# 审查工具函数
# ============================================================================

class RegressionReportChecker:
    """回归报告审查工具"""

    def __init__(self, report: str):
        self.report = report.lower()
        self.original_report = report

    def check_residual_plots(self) -> bool:
        """检查是否有残差图"""
        keywords = ['残差图', 'residual plot', '残差 vs 拟合',
                   'residuals vs fitted', 'qq图', 'qq plot']
        return any(kw in self.report for kw in keywords)

    def check_normality_test(self) -> bool:
        """检查是否有正态性检验"""
        keywords = ['shapiro', '正态性', 'normality',
                   'qq 图', 'qq plot', '正态']
        return any(kw in self.report for kw in keywords)

    def check_vif(self) -> bool:
        """检查是否有 VIF 检查"""
        keywords = ['vif', '方差膨胀', 'multicollinearity',
                   '共线性', '相关矩阵']
        return any(kw in self.report for kw in keywords)

    def check_cooks_distance(self) -> bool:
        """检查是否有 Cook's 距离"""
        keywords = ['cook', '影响点', 'influential',
                   '异常点', 'outlier', 'leverage']
        return any(kw in self.report for kw in keywords)

    def check_confidence_interval(self) -> bool:
        """检查是否有置信区间"""
        keywords = ['95% ci', '置信区间', 'confidence interval',
                   'ci:', '[']  # 简单检查是否有区间表示
        return any(kw in self.report for kw in keywords)

    def check_homoscedasticity(self) -> bool:
        """检查是否有同方差检验"""
        keywords = ['等方差', '同方差', 'heteroscedasticity',
                   'breusch', 'bp 检验', 'homoscedastic']
        return any(kw in self.report for kw in keywords)

    def check_causal_warning(self) -> bool:
        """检查是否有因果警告"""
        # 如果有"导致"这类因果词但没警告，则不合格
        causal_words = ['导致', 'cause', '使']
        warning_words = ['因果', 'causal', '关联', 'association',
                        '混杂', 'confound', '不能推断']
        has_causal = any(word in self.report for word in causal_words)
        has_warning = any(word in self.report for word in warning_words)
        # 如果提到因果，必须有警告
        return not has_causal or has_warning

    def check_all(self) -> Dict[str, bool]:
        """运行所有检查"""
        return {
            "残差图": self.check_residual_plots(),
            "正态性检验": self.check_normality_test(),
            "VIF 检查": self.check_vif(),
            "Cook's 距离": self.check_cooks_distance(),
            "置信区间": self.check_confidence_interval(),
            "同方差检验": self.check_homoscedasticity(),
            "因果警告": self.check_causal_warning(),
        }

    def print_report(self) -> None:
        """打印审查结果"""
        results = self.check_all()

        print("=" * 70)
        print("回归报告审查结果")
        print("=" * 70)

        for item, checked in results.items():
            status = "✅" if checked else "❌"
            print(f"{status} {item}")

        missing = [item for item, checked in results.items() if not checked]
        if missing:
            print(f"\n⚠️  缺失项目: {', '.join(missing)}")
        else:
            print(f"\n✓ 所有检查项都通过!")

        score = sum(results.values()) / len(results) * 100
        print(f"\n总体评分: {score:.0f}%")

        if score >= 80:
            print("评级: 优秀")
        elif score >= 60:
            print("评级: 良好（需改进缺失项）")
        else:
            print("评级: 不合格（需要重大修订）")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数：演示报告审查"""
    print("=" * 70)
    print("示例7: AI 生成的回归报告审查")
    print("=" * 70)

    # ========================================
    # 场景1: 审查有问题的 AI 报告
    # ========================================
    print("\n" + "=" * 70)
    print("场景1: 审查 AI 生成的报告（有问题的版本）")
    print("=" * 70)
    print("\n报告内容:")
    print("-" * 70)
    print(BAD_AI_REPORT)
    print("-" * 70)

    checker_bad = RegressionReportChecker(BAD_AI_REPORT)
    checker_bad.print_report()

    print("\n" + "=" * 70)
    print("问题分析:")
    print("=" * 70)
    print("""
    1. ❌ 缺少残差图: 无法判断线性、正态性、等方差假设是否满足
    2. ❌ 缺少置信区间: 只有点估计，没有表达不确定性
    3. ❌ 缺少 VIF 检查: 不知道是否存在多重共线性
    4. ❌ 缺少异常点分析: 未识别强影响点
    5. ❌ 误解释为因果: "导致"一词暗示因果，但观察数据不能直接推断
    6. ❌ 缺少局限性讨论: 未说明因果推断的限制
    """)

    # ========================================
    # 场景2: 审查修订后的报告
    # ========================================
    print("\n" + "=" * 70)
    print("场景2: 审查人工修订后的报告（好的版本）")
    print("=" * 70)
    print("\n报告内容:")
    print("-" * 70)
    print(GOOD_REPORT[:500] + "...\n[内容省略]")
    print("-" * 70)

    checker_good = RegressionReportChecker(GOOD_REPORT)
    checker_good.print_report()

    # ========================================
    # 使用建议
    # ========================================
    print("\n" + "=" * 70)
    print("使用建议: 如何修复 AI 报告")
    print("=" * 70)
    print("""
    1. 补充残差诊断图:
       - 残差 vs 拟合值图 (线性 + 等方差)
       - QQ 图 (正态性)
       - 解释图中观察到的模式

    2. 添加置信区间:
       - 系数: 1.25 [0.89, 1.61]
       - 解释: "我们有 95% 的把握认为真实系数在这个范围内"

    3. 检查多重共线性:
       - 计算 VIF
       - 如果 VIF > 10，考虑删除或合并变量

    4. 识别异常点:
       - 计算 Cook's 距离
       - 评估删除前后的模型变化

    5. 修正因果解释:
       - "相关" ≠ "因果"
       - 添加: "本分析仅描述关联，因果推断需额外假设"

    6. 讨论局限性:
       - 可能的混杂变量
       - 数据来源与样本代表性
       - 假设是否满足
    """)

    print("\n" + "=" * 70)
    print("✅ 示例7完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
