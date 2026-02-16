"""
Starter code for week_10: 预测的艺术——分类模型与逻辑回归入门

Contract:
- Implement solve(text: str) -> str
- Treat ``text`` as the raw input (e.g., CSV/TSV text).
- Return a Markdown classification report.
- tests assert against this file only
"""
from __future__ import annotations

import io
from typing import Dict, List, Tuple


def parse_csv(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse CSV text into headers and rows."""
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return [], []
    headers = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:]]
    return headers, rows


def calculate_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calculate confusion matrix components."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calculate_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    """Calculate precision, recall, F1 from confusion matrix."""
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def generate_report(headers: List[str], rows: List[List[str]]) -> str:
    """Generate a classification report from parsed CSV data."""
    if len(headers) < 2 or not rows:
        return "# Error\n\nInsufficient data for classification report."
    
    # Assume last column is prediction, second-to-last is actual
    try:
        y_true = [int(row[-2]) for row in rows]
        y_pred = [int(row[-1]) for row in rows]
    except (ValueError, IndexError):
        return "# Error\n\nCould not parse target columns as integers."
    
    cm = calculate_confusion_matrix(y_true, y_pred)
    metrics = calculate_metrics(cm)
    
    report = io.StringIO()
    report.write("# Week 10 分类评估报告\n\n")
    
    report.write("## 混淆矩阵\n\n")
    report.write("|  | 预测: 正类 | 预测: 负类 |\n")
    report.write("|--|-----------|-----------|\n")
    report.write(f"| **实际: 正类** | TP: {cm['TP']} | FN: {cm['FN']}\n")
    report.write(f"| **实际: 负类** | FP: {cm['FP']} | TN: {cm['TN']}\n")
    report.write("\n")
    
    report.write("## 分类指标\n\n")
    report.write(f"- **准确率 (Accuracy)**: {metrics['accuracy']:.4f}\n")
    report.write(f"- **精确率 (Precision)**: {metrics['precision']:.4f}\n")
    report.write(f"- **查全率 (Recall)**: {metrics['recall']:.4f}\n")
    report.write(f"- **F1 分数**: {metrics['f1']:.4f}\n")
    report.write("\n")
    
    report.write("## 指标解读\n\n")
    if metrics['precision'] > 0.8 and metrics['recall'] < 0.5:
        report.write("- 精确率高但查全率低：模型对预测为正的样本很有把握，但漏掉了很多真正的正样本。\n")
    elif metrics['precision'] < 0.5 and metrics['recall'] > 0.8:
        report.write("- 查全率高但精确率低：模型找出了大部分正样本，但有很多误报。\n")
    elif metrics['f1'] > 0.8:
        report.write("- F1 分数较高：模型在精确率和查全率之间取得了较好的平衡。\n")
    else:
        report.write("- 各项指标均较低：模型整体表现有待提升，建议检查数据质量或尝试其他模型。\n")
    
    return report.getvalue()


def solve(text: str) -> str:
    """Transform input CSV text into a classification report (Markdown)."""
    headers, rows = parse_csv(text)
    return generate_report(headers, rows)


def main() -> None:
    import sys
    data = sys.stdin.read()
    sys.stdout.write(solve(data))


if __name__ == "__main__":
    main()
