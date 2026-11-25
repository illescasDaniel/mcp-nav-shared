"""Directory names both servers skip when walking a workspace for candidate
files (codenav's `implementations` class scan, webnav's `web_index` file
scan): vendored/generated/virtualenv trees that are never a project's own
source, so scanning them wastes time at best and, for `implementations`,
can abort the whole call on the first non-UTF-8 file at worst (see
`docs/agent-tooling.md`)."""

from __future__ import annotations

import os
from pathlib import Path


EXCLUDED_DIR_NAMES = {
	"node_modules",
	".git",
	"vendor",
	"dist",
	"build",
	"__pycache__",
	".venv",
	"venv",
	".venv-bare",
	"env",
	".env",
	"site-packages",
	".tox",
	".mypy_cache",
	".pytest_cache",
	".ruff_cache",
	".pytest-testmon",
	".eggs",
}


def _env_names(variable: str) -> set[str]:
	return {part.strip() for part in os.environ.get(variable, "").split(",") if part.strip()}


def excluded_names(prefix: str = "") -> frozenset[str]:
	"""Directory names to skip: `EXCLUDED_DIR_NAMES` plus the names in env `<prefix>_EXCLUDE_DIRS`, minus
	the names in env `<prefix>_INCLUDE_DIRS` (comma-separated). An empty prefix reads no environment."""
	if not prefix:
		return frozenset(EXCLUDED_DIR_NAMES)
	return frozenset((EXCLUDED_DIR_NAMES | _env_names(f"{prefix}_EXCLUDE_DIRS")) - _env_names(f"{prefix}_INCLUDE_DIRS"))


def is_excluded(path: Path, root: Path, *, prefix: str = "") -> bool:
	"""Whether `path` (absolute or root-relative) sits under an excluded directory name
	(see `excluded_names`) anywhere below `root`."""
	try:
		rel_parts = path.relative_to(root).parts
	except ValueError:
		rel_parts = path.parts
	return not excluded_names(prefix).isdisjoint(rel_parts)


def is_inside_nested_repo(path: Path, root: Path) -> Path | None:
	"""The nearest directory below `root` that contains a `.git` entry and holds `path` (a nested
	checkout or linked worktree), or None. `root` itself never counts."""
	try:
		rel = path.relative_to(root)
	except ValueError:
		return None
	current = path if path.is_dir() else path.parent
	for _ in range(len(rel.parts)):
		if current == root:
			break
		if (current / ".git").exists():
			return current
		current = current.parent
	return None
