"""QA block B13: workspace and source-root validation in mcp_nav_shared (RC-095, 096, 128, 155)."""

from __future__ import annotations

import pytest

from mcp_nav_shared.workspace import describe_workspace_root, resolve_extra_source_roots, resolve_source_root


def test_rc_095_given_filesystem_root_as_workspace_when_describe_then_refused(monkeypatch):
	monkeypatch.setenv("SOME_WORKSPACE", "/")

	with pytest.raises(ValueError, match=r"refusing to use / as the workspace; set SOME_WORKSPACE to your project"):
		describe_workspace_root("SOME_WORKSPACE")


def test_rc_155_given_pinned_workspace_that_does_not_exist_when_describe_then_refused(tmp_path, monkeypatch):
	monkeypatch.setenv("SOME_WORKSPACE", str(tmp_path / "qa-nope-proj"))

	with pytest.raises(ValueError, match="is not an existing directory"):
		describe_workspace_root("SOME_WORKSPACE")


def test_b13_guard_given_pinned_existing_workspace_when_describe_then_returned(tmp_path, monkeypatch):
	monkeypatch.setenv("SOME_WORKSPACE", str(tmp_path))

	root, source = describe_workspace_root("SOME_WORKSPACE")

	assert root == tmp_path.resolve()
	assert source == "SOME_WORKSPACE"


def test_rc_096_given_source_root_outside_workspace_when_resolve_then_refused(tmp_path, monkeypatch):
	workspace = tmp_path / "ws"
	workspace.mkdir()
	outside = tmp_path / "outside"
	outside.mkdir()
	monkeypatch.setenv("SOME_SOURCE_ROOT", str(outside))

	with pytest.raises(ValueError, match=f"SOME_SOURCE_ROOT={outside} is not a directory inside the workspace"):
		resolve_source_root("SOME_SOURCE_ROOT", workspace)


def test_rc_096_given_missing_source_root_when_resolve_then_refused(tmp_path, monkeypatch):
	monkeypatch.setenv("SOME_SOURCE_ROOT", "missing")

	with pytest.raises(ValueError, match="is not a directory inside the workspace"):
		resolve_source_root("SOME_SOURCE_ROOT", tmp_path)


def test_b13_guard_given_source_root_inside_workspace_when_resolve_then_returned(tmp_path, monkeypatch):
	(tmp_path / "lib").mkdir()
	monkeypatch.setenv("SOME_SOURCE_ROOT", "lib")

	assert resolve_source_root("SOME_SOURCE_ROOT", tmp_path) == (tmp_path / "lib").resolve()


def test_rc_096_given_extra_roots_outside_or_missing_when_resolve_then_only_inside_dirs_kept(tmp_path, monkeypatch):
	workspace = tmp_path / "ws"
	(workspace / "tests").mkdir(parents=True)
	outside = tmp_path / "outside"
	outside.mkdir()
	monkeypatch.setenv("SOME_EXTRA_ROOTS", f"{outside},tests,missing")

	roots = resolve_extra_source_roots("SOME_EXTRA_ROOTS", workspace)

	assert roots == [(workspace / "tests").resolve()]


def test_rc_096_given_skipped_extra_roots_when_ignored_then_listed_as_typed(tmp_path, monkeypatch):
	from mcp_nav_shared.workspace import ignored_extra_source_roots

	(tmp_path / "tests").mkdir()
	monkeypatch.setenv("SOME_EXTRA_ROOTS", "missing,tests,/definitely/not/inside")

	assert ignored_extra_source_roots("SOME_EXTRA_ROOTS", tmp_path) == ["missing", "/definitely/not/inside"]


def test_rc_128_given_src_layout_when_autodetect_then_src_is_source_root(tmp_path, monkeypatch):
	(tmp_path / "src" / "pkg").mkdir(parents=True)
	(tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
	monkeypatch.delenv("SOME_SOURCE_ROOT", raising=False)

	assert resolve_source_root("SOME_SOURCE_ROOT", tmp_path, autodetect_src=True) == (tmp_path / "src").resolve()


def test_b13_guard_given_src_layout_without_autodetect_when_resolve_then_workspace_root(tmp_path, monkeypatch):
	(tmp_path / "src" / "pkg").mkdir(parents=True)
	(tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
	monkeypatch.delenv("SOME_SOURCE_ROOT", raising=False)

	assert resolve_source_root("SOME_SOURCE_ROOT", tmp_path) == tmp_path


def test_rc_128_given_src_that_is_a_package_when_autodetect_then_workspace_root(tmp_path, monkeypatch):
	(tmp_path / "src" / "pkg").mkdir(parents=True)
	(tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
	monkeypatch.delenv("SOME_SOURCE_ROOT", raising=False)

	assert resolve_source_root("SOME_SOURCE_ROOT", tmp_path, autodetect_src=True) == tmp_path


def test_rc_128_given_empty_src_when_autodetect_then_workspace_root(tmp_path, monkeypatch):
	(tmp_path / "src").mkdir()
	monkeypatch.delenv("SOME_SOURCE_ROOT", raising=False)

	assert resolve_source_root("SOME_SOURCE_ROOT", tmp_path, autodetect_src=True) == tmp_path
