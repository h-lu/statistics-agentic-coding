#!/usr/bin/env python3
"""Release gate for chapter-local assets.

Checks:
1. Any referenced ``images/*.png`` file must exist.
2. Unreferenced ``images/*.png`` files are reported as warnings unless
   documented in ``images/README.md``.
3. Every image should be followed by a short explanatory sentence.
4. Dense chapters (regression/classification/SHAP/PCA/clustering) must not
   end up with zero figure references.
5. Chapters with more than 10 code blocks must include an explicit exemption
   comment.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _common import add_error, repo_root


def _extract_image_targets(text: str) -> list[str]:
    """Return image targets from markdown image syntax.

    This parser is deliberately simple but robust enough for paths that
    themselves contain parentheses, which the common regex approach misses.
    """
    out: list[str] = []
    for line in text.splitlines():
        start = 0
        while True:
            bang = line.find("![", start)
            if bang == -1:
                break
            open_paren = line.find("](", bang)
            if open_paren == -1:
                break
            target = line[open_paren + 2 :]
            close_paren = target.rfind(")")
            if close_paren == -1:
                break
            target = target[:close_paren].strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            # Drop optional title text if present.
            if " " in target:
                target = target.split(" ", 1)[0]
            if target:
                out.append(target)
            start = open_paren + 2
    return out


def _image_followup_sentence_exists(lines: list[str], line_no: int) -> bool:
    """Return True if a short explanatory sentence appears after an image.

    We accept either a plain paragraph line or an emphasized caption line
    immediately after the image. Blank lines are skipped.
    """
    for idx in range(line_no, min(len(lines), line_no + 4)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped.startswith("```") or stripped.startswith("<!--"):
            return True
        if stripped.startswith("*") or stripped.startswith("-") or stripped.startswith(">"):
            return True
        if "。" in stripped or "." in stripped:
            return True
        return False
    return False


def check_chapter_assets(errors: list[str], week_dir: Path, root: Path | None = None) -> None:
    if root is None:
        root = repo_root()

    chapter_path = week_dir / "CHAPTER.md"
    if not chapter_path.is_file():
        add_error(errors, f"missing required file: {chapter_path.relative_to(root)}")
        return

    text = chapter_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    image_targets = _extract_image_targets(text)
    image_targets = [t for t in image_targets if t.startswith("images/")]

    # Missing files.
    for target in image_targets:
        if not (week_dir / target).is_file():
            add_error(errors, f"missing referenced image: {chapter_path.parent.name}/{target}")

    # Warnings for unreferenced local PNGs.
    images_dir = week_dir / "images"
    if images_dir.is_dir():
        referenced = {Path(t).name for t in image_targets}
        documented = set()
        readme_path = images_dir / "README.md"
        if readme_path.is_file():
            documented.update(re.findall(r"[\w.-]+\.png", readme_path.read_text(encoding="utf-8")))

        for img in sorted(images_dir.glob("*.png")):
            if img.name not in referenced and img.name not in documented:
                print(
                    f"[check-chapter-assets] WARNING: unreferenced image: "
                    f"{chapter_path.parent.name}/images/{img.name}",
                    file=sys.stderr,
                )

    # Dense chapters should have at least one figure.
    dense_weeks = {"week_09", "week_10", "week_11", "week_12", "week_13", "week_15"}
    if week_dir.name in dense_weeks and not image_targets:
        add_error(errors, f"dense chapter {week_dir.name} has no referenced images")

    # Each referenced image should be followed by a sentence or caption.
    if image_targets:
        for idx, line in enumerate(lines):
            if "![" not in line:
                continue
            if not _image_followup_sentence_exists(lines, idx + 1):
                add_error(
                    errors,
                    f"image at {week_dir.name}/CHAPTER.md:{idx+1} is missing a follow-up sentence",
                )

    # Code block budget.
    code_blocks = 0
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if not in_block:
                code_blocks += 1
                in_block = True
            else:
                in_block = False
    if code_blocks > 10 and "<!-- code_block_exemption:" not in text:
        add_error(
            errors,
            f"{week_dir.name} has {code_blocks} code blocks; add <!-- code_block_exemption: reason --> or reduce them",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check chapter-local assets.")
    parser.add_argument("--week", required=True)
    args = parser.parse_args()

    week_dir = repo_root() / "chapters" / args.week
    errors: list[str] = []
    check_chapter_assets(errors, week_dir)
    if errors:
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 2
    print(f"[check-chapter-assets] OK: {args.week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
