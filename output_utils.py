# 输出目录辅助模块
# 所有示例代码统一使用此模块获取输出路径

from pathlib import Path

def get_output_dir(week: str) -> Path:
    """
    获取指定周的输出目录，如不存在则创建。

    用法：
        from output_utils import get_output_dir
        output_dir = get_output_dir("week_13")
        plt.savefig(output_dir / "causal_dag.png")
    """
    # 获取项目根目录（向上两级到 statistics-agentic-coding）
    root = Path(__file__).parent
    output_dir = root / "output" / week
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_path(week: str, filename: str) -> Path:
    """
    获取指定周和文件名的完整输出路径。

    用法：
        from output_utils import get_output_path
        plt.savefig(get_output_path("week_13", "causal_dag.png"))
    """
    output_dir = get_output_dir(week)
    return output_dir / filename
