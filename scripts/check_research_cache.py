#!/usr/bin/env python3
"""Release gate for chapter research caches.

Checks:
1. If a chapter contains obvious time-sensitive/fact-check-triggering text,
   it must have a ``.research_cache.md`` file.
2. URLs used in AI sidebars / reference blocks must be present in the cache.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _common import add_error, repo_root


TRIGGER_WORDS = [
    "2025",
    "2026",
    "调查",
    "报告",
    "市场规模",
    "用户数",
    "法规",
    "研究显示",
]

_URL_RE = re.compile(r"https?://[^\s<>\"]+")


def _normalize_url(url: str) -> str:
    url = url.split("（", 1)[0]
    url = url.split("(", 1)[0]
    return url.rstrip(").,，；;：:】]）>")


def _extract_cache_urls(text: str) -> set[str]:
    return {_normalize_url(url) for url in _URL_RE.findall(text)}


def _extract_sidebar_urls(text: str) -> set[str]:
    urls: set[str] = set()
    in_sidebar = False
    for line in text.splitlines():
        stripped = line.strip()
        if "AI 小专栏" in stripped or "AI 时代小专栏" in stripped:
            in_sidebar = True
        elif in_sidebar and not stripped:
            in_sidebar = False
        if in_sidebar or stripped.startswith("> 参考") or stripped.startswith("> **参考"):
            urls.update(_normalize_url(url) for url in re.findall(r"\((https?://[^)]+)\)", line))
            urls.update(_normalize_url(url) for url in _URL_RE.findall(line))
    return urls


def check_research_cache(errors: list[str], week_dir: Path, root: Path | None = None) -> None:
    if root is None:
        root = repo_root()

    chapter_path = week_dir / "CHAPTER.md"
    if not chapter_path.is_file():
        add_error(errors, f"missing required file: {chapter_path.relative_to(root)}")
        return

    text = chapter_path.read_text(encoding="utf-8")
    cache_path = week_dir / ".research_cache.md"
    cache_exists = cache_path.is_file()

    needs_cache = any(word in text for word in TRIGGER_WORDS) or "AI 小专栏" in text or "AI 时代小专栏" in text
    if needs_cache and not cache_exists:
        add_error(errors, f"{week_dir.name} needs .research_cache.md because CHAPTER.md contains time-sensitive claims")
        return

    if not cache_exists:
        return

    cache_text = cache_path.read_text(encoding="utf-8")
    cache_urls = _extract_cache_urls(cache_text)
    sidebar_urls = _extract_sidebar_urls(text)

    missing = sorted(url for url in sidebar_urls if url not in cache_urls)
    for url in missing:
        add_error(errors, f"{week_dir.name} sidebar URL missing from research cache: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check chapter research cache coverage.")
    parser.add_argument("--week", required=True)
    args = parser.parse_args()

    week_dir = repo_root() / "chapters" / args.week
    errors: list[str] = []
    check_research_cache(errors, week_dir)
    if errors:
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 2
    print(f"[check-research-cache] OK: {args.week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
