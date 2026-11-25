"""B5: shared exclusion policy (env include/exclude, nested repositories)."""

from __future__ import annotations

from mcp_nav_shared.exclude import EXCLUDED_DIR_NAMES, excluded_names, is_excluded, is_inside_nested_repo


def test_rc_002_excluded_names_default_has_no_env(monkeypatch):
	monkeypatch.setenv("X_EXCLUDE_DIRS", "gen")
	assert excluded_names() == frozenset(EXCLUDED_DIR_NAMES)


def test_rc_002_env_exclude_and_include_parsing(monkeypatch):
	monkeypatch.setenv("X_EXCLUDE_DIRS", " gen , out,,")
	monkeypatch.setenv("X_INCLUDE_DIRS", "build, vendor")
	names = excluded_names("X")
	assert {"gen", "out"} <= names
	assert "build" not in names and "vendor" not in names
	assert "node_modules" in names


def test_rc_002_is_excluded_honours_prefix(tmp_path, monkeypatch):
	monkeypatch.setenv("X_INCLUDE_DIRS", "build")
	path = tmp_path / "build" / "m.py"
	assert is_excluded(path, tmp_path) is True
	assert is_excluded(path, tmp_path, prefix="X") is False


def test_rc_040_nested_repo_detected(tmp_path):
	(tmp_path / ".git").mkdir()
	(tmp_path / "vend" / "sub").mkdir(parents=True)
	(tmp_path / "vend" / ".git").write_text("gitdir: elsewhere")
	(tmp_path / "plain").mkdir()
	assert is_inside_nested_repo(tmp_path / "vend" / "sub" / "m.py", tmp_path) == tmp_path / "vend"
	assert is_inside_nested_repo(tmp_path / "plain" / "m.py", tmp_path) is None
	assert is_inside_nested_repo(tmp_path / "m.py", tmp_path) is None
