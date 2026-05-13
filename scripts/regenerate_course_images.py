#!/usr/bin/env python3
"""Regenerate course images with the shared visual style.

The older chapter scripts were written at different times and save images either
into `chapters/week_xx/images` directly or into `examples/output`.  This runner
executes all plotting example scripts under a CJK-safe Matplotlib style, then
copies regenerated files into the chapter image folders and Docusaurus mirror.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
DOCS_WEEKS = ROOT / "templates" / "docusaurus-site" / "site" / "docs" / "weeks"
LOG_DIR = ROOT / "audit" / "image-regeneration"

SKIP_SCRIPTS: set[Path] = set()


def existing_images_by_week() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for image in CHAPTERS.glob("week_*/images/*"):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
            continue
        result.setdefault(image.parents[1].name, set()).add(image.name)
    return result


def plotting_scripts() -> list[Path]:
    scripts: list[Path] = []
    for script in sorted(CHAPTERS.glob("week_*/examples/*.py")):
        if script in SKIP_SCRIPTS:
            continue
        text = script.read_text(encoding="utf-8", errors="ignore")
        if "savefig" in text:
            scripts.append(script)
    return scripts


def run_script(script: Path) -> tuple[bool, str]:
    rel = script.relative_to(ROOT)
    log_name = str(rel).replace("/", "__") + ".log"
    log_path = LOG_DIR / log_name
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(ROOT)

    # Most scripts are designed to run from repo root. A few older scripts use
    # relative paths intentionally; keep repo root as the standard cwd.
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0, log_name


def sync_example_outputs(week: str, known_names: set[str]) -> list[str]:
    """Copy matching files from a week's examples/output into week images."""
    copied: list[str] = []
    output_dir = CHAPTERS / week / "examples" / "output"
    image_dir = CHAPTERS / week / "images"
    if not output_dir.exists() or not image_dir.exists():
        return copied
    for image in output_dir.iterdir():
        if image.name in known_names and image.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            shutil.copy2(image, image_dir / image.name)
            copied.append(str((image_dir / image.name).relative_to(ROOT)))
    return copied


def sync_docs() -> list[str]:
    copied: list[str] = []
    for image_dir in sorted(CHAPTERS.glob("week_*/images")):
        week_num = image_dir.parent.name.removeprefix("week_")
        target_dir = DOCS_WEEKS / week_num / "images"
        if not target_dir.exists():
            continue
        for image in image_dir.iterdir():
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
                continue
            if (target_dir / image.name).exists():
                shutil.copy2(image, target_dir / image.name)
                copied.append(str((target_dir / image.name).relative_to(ROOT)))
    return copied


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    known = existing_images_by_week()
    scripts = plotting_scripts()

    failures: list[tuple[Path, str]] = []
    for idx, script in enumerate(scripts, 1):
        print(f"[{idx:02d}/{len(scripts):02d}] {script.relative_to(ROOT)}")
        try:
            ok, log_name = run_script(script)
        except subprocess.TimeoutExpired:
            log_name = str(script.relative_to(ROOT)).replace("/", "__") + ".timeout.log"
            (LOG_DIR / log_name).write_text("TIMEOUT after 240s\n", encoding="utf-8")
            ok = False
        if not ok:
            failures.append((script, log_name))
        week = script.parents[1].name
        sync_example_outputs(week, known.get(week, set()))

    docs = sync_docs()
    print(f"\nSynced {len(docs)} Docusaurus image copies.")

    if failures:
        print("\nFailures:")
        for script, log_name in failures:
            print(f"- {script.relative_to(ROOT)} -> audit/image-regeneration/{log_name}")
        return 1

    print("\nAll image scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
