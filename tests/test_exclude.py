"""Fast unit tests for `mcp_nav_shared.exclude`."""

from __future__ import annotations

from pathlib import Path

from mcp_nav_shared.exclude import is_excluded


def test_given_path_under_venv_when_is_excluded_then_true(tmp_path):
	# given
	path = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "foo.py"
	# when / then
	assert is_excluded(path, tmp_path) is True


def test_given_path_under_node_modules_when_is_excluded_then_true(tmp_path):
	# given
	path = tmp_path / "web" / "node_modules" / "pkg" / "index.js"
	# when / then
	assert is_excluded(path, tmp_path) is True


def test_given_ordinary_source_path_when_is_excluded_then_false(tmp_path):
	# given
	path = tmp_path / "src" / "pkg" / "module.py"
	# when / then
	assert is_excluded(path, tmp_path) is False


def test_given_path_outside_root_when_is_excluded_then_falls_back_to_absolute_parts(tmp_path):
	# given — a path that isn't actually under `root` (e.g. resolved via a
	# symlink elsewhere): relative_to raises, so exclusion falls back to
	# checking the path's own parts rather than crashing.
	other_root = Path("/some/other/place")
	path = other_root / ".venv" / "module.py"
	# when / then
	assert is_excluded(path, tmp_path) is True
