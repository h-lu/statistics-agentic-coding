"""
Smoke tests for week_05 examples.

These tests verify that examples can run without errors.
"""
import pytest
from pathlib import Path


def test_example_01_simulation_intro_runs():
    """Test that example 01 runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/01_simulation_intro.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "抛硬币模拟实验" in result.stdout


def test_example_02_sampling_distribution_runs():
    """Test that example 02 runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/02_sampling_distribution.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "抽样分布模拟" in result.stdout


def test_example_03_clt_simulation_runs():
    """Test that example 03 runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/03_clt_simulation.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "中心极限定理" in result.stdout


def test_example_04_bootstrap_intro_runs():
    """Test that example 04 runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/04_bootstrap_intro.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "Bootstrap" in result.stdout


def test_example_05_false_positive_simulation_runs():
    """Test that example 05 runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/05_false_positive_simulation.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    assert "假阳性模拟" in result.stdout


def test_example_99_statlab_runs():
    """Test that StatLab example runs without error."""
    import subprocess
    result = subprocess.run(
        ["python3", "chapters/week_05/examples/99_statlab.py"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0
    assert "Week 05 更新完成" in result.stdout


def test_output_files_created():
    """Test that example output files are created."""
    import subprocess

    # Run example 01
    subprocess.run(
        ["python3", "chapters/week_05/examples/01_simulation_intro.py"],
        capture_output=True,
        timeout=30
    )

    # Check output directory exists
    output_dir = Path("chapters/week_05/examples/output")
    assert output_dir.exists()

    # Check that at least one output file exists
    output_files = list(output_dir.glob("*.png"))
    assert len(output_files) > 0
