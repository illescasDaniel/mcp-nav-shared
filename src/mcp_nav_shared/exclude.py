"""Directory names both servers skip when walking a workspace for candidate
files (codenav's `implementations` class scan, webnav's `web_index` file
scan): vendored/generated/virtualenv trees that are never a project's own
source, so scanning them wastes time at best and, for `implementations`,
can abort the whole call on the first non-UTF-8 file at worst (see
`docs/agent-tooling.md`)."""

from __future__ import annotations

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


def is_excluded(path: Path, root: Path) -> bool:
	"""Whether `path` (absolute or root-relative) sits under a directory name
	in `EXCLUDED_DIR_NAMES` anywhere below `root`."""
	try:
		rel_parts = path.relative_to(root).parts
	except ValueError:
		rel_parts = path.parts
	return not EXCLUDED_DIR_NAMES.isdisjoint(rel_parts)
