"""Fast unit tests for `mcp_nav_shared.workspace`."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_nav_shared.workspace import (
	Selection,
	WorkspaceSelector,
	resolve_extra_source_roots,
	resolve_source_root,
	resolve_workspace_root,
	same_repository,
)


def test_given_explicit_env_when_resolve_workspace_root_then_prefers_it(tmp_path, monkeypatch):
	# given
	override = tmp_path / "override"
	override.mkdir()
	fallback = tmp_path / "fallback"
	fallback.mkdir()
	monkeypatch.setenv("CODENAV_MCP_WORKSPACE", str(override))
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(fallback))
	# when
	root = resolve_workspace_root("CODENAV_MCP_WORKSPACE")
	# then
	assert root == override.resolve()


def test_given_claude_project_dir_when_no_explicit_env_then_uses_claude(tmp_path, monkeypatch):
	# given
	claude = tmp_path / "claude-root"
	claude.mkdir()
	monkeypatch.delenv("CODENAV_MCP_WORKSPACE", raising=False)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(claude))
	# when
	root = resolve_workspace_root("CODENAV_MCP_WORKSPACE")
	# then
	assert root == claude.resolve()


def test_given_no_env_when_resolve_workspace_root_then_uses_cwd(tmp_path, monkeypatch):
	# given — no explicit override and no host-injected project dir: the
	# server must fall back to wherever it was actually launched (its own
	# install location, e.g. a repo it ships from, is never the right guess
	# for a different project's checkout).
	monkeypatch.delenv("CODENAV_MCP_WORKSPACE", raising=False)
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	monkeypatch.chdir(tmp_path)
	# when
	root = resolve_workspace_root("CODENAV_MCP_WORKSPACE")
	# then
	assert root == tmp_path.resolve()


def test_given_explicit_source_root_env_when_resolve_source_root_then_used(tmp_path, monkeypatch):
	# given
	monkeypatch.setenv("SOME_SOURCE_ROOT", "src")
	# when
	root = resolve_source_root("SOME_SOURCE_ROOT", tmp_path)
	# then
	assert root == (tmp_path / "src").resolve()


def test_given_no_source_root_env_when_resolve_source_root_then_defaults_to_workspace_root(tmp_path, monkeypatch):
	# given
	monkeypatch.delenv("SOME_OTHER_SOURCE_ROOT", raising=False)
	# when
	root = resolve_source_root("SOME_OTHER_SOURCE_ROOT", tmp_path)
	# then
	assert root == tmp_path


def test_given_extra_roots_env_when_resolve_then_existing_dirs_only_deduped(tmp_path, monkeypatch):
	# given
	(tmp_path / "tests").mkdir()
	(tmp_path / "other").mkdir()
	monkeypatch.setenv("SOME_EXTRA_ROOTS", f"tests, missing ,other,tests,{tmp_path / 'other'},")
	# when
	roots = resolve_extra_source_roots("SOME_EXTRA_ROOTS", tmp_path)
	# then
	assert roots == [(tmp_path / "tests").resolve(), (tmp_path / "other").resolve()]


def test_given_no_extra_roots_env_when_resolve_then_empty(tmp_path, monkeypatch):
	# given
	monkeypatch.delenv("SOME_EXTRA_ROOTS", raising=False)
	# when / then
	assert resolve_extra_source_roots("SOME_EXTRA_ROOTS", tmp_path) == []


# --- WorkspaceSelector -------------------------------------------------------


def _run(cwd: Path, *args: str) -> None:
	subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture
def repo_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
	main = tmp_path / "main"
	main.mkdir()
	_run(main, "init", "-q", "-b", "trunk")
	_run(main, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
	wt = tmp_path / "linked"
	_run(main, "worktree", "add", "-q", "-b", "feature", str(wt))
	return main.resolve(), wt.resolve()


class _FakeSession:
	def __init__(self, roots: list[str] | None, *, supports_roots: bool = True, fail: bool = False) -> None:
		self._roots = roots or []
		self._fail = fail
		caps = SimpleNamespace(roots=SimpleNamespace() if supports_roots else None)
		self.client_params = SimpleNamespace(capabilities=caps)
		self.calls = 0

	async def list_roots(self) -> SimpleNamespace:
		self.calls += 1
		if self._fail:
			raise RuntimeError("boom")
		return SimpleNamespace(roots=[SimpleNamespace(uri=uri) for uri in self._roots])


def test_given_worktree_and_main_when_same_repository_then_true(repo_and_worktree):
	# given
	main, wt = repo_and_worktree
	# then
	assert same_repository(main, wt)
	assert same_repository(wt, main)


def test_given_unrelated_directories_when_same_repository_then_false(tmp_path):
	# given
	a = tmp_path / "a"
	b = tmp_path / "b"
	a.mkdir()
	b.mkdir()
	# then
	assert not same_repository(a, b)
	assert same_repository(a, a)


def test_given_worktree_root_when_select_then_switches_to_it(repo_and_worktree, monkeypatch):
	# given
	main, wt = repo_and_worktree
	monkeypatch.delenv("SEL_WORKSPACE", raising=False)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(main))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	# when
	selection = asyncio.run(selector.select(_FakeSession([wt.as_uri()])))
	# then
	assert selection == Selection(wt, "client roots")


def test_given_root_from_unrelated_project_when_select_then_ignored(repo_and_worktree, tmp_path, monkeypatch):
	# given
	main, _ = repo_and_worktree
	other = tmp_path / "elsewhere"
	other.mkdir()
	monkeypatch.delenv("SEL_WORKSPACE", raising=False)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(main))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	# when
	selection = asyncio.run(selector.select(_FakeSession([other.as_uri()])))
	# then
	assert selection == Selection(main, "CLAUDE_PROJECT_DIR")


def test_given_pinned_env_when_select_then_roots_not_consulted(repo_and_worktree, monkeypatch):
	# given
	main, wt = repo_and_worktree
	monkeypatch.setenv("SEL_WORKSPACE", str(main))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	session = _FakeSession([wt.as_uri()])
	# when
	selection = asyncio.run(selector.select(session))
	# then
	assert selection == Selection(main, "SEL_WORKSPACE")
	assert session.calls == 0


@pytest.mark.parametrize(
	"session",
	[None, _FakeSession(None, supports_roots=False), _FakeSession(["file:///x"], fail=True)],
	ids=["no-session", "no-roots-capability", "roots-request-fails"],
)
def test_given_no_usable_roots_when_select_then_keeps_configured_base(session, tmp_path, monkeypatch):
	# given
	monkeypatch.delenv("SEL_WORKSPACE", raising=False)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	# when
	selection = asyncio.run(selector.select(session))
	# then
	assert selection.root == tmp_path.resolve()


def test_given_non_file_root_when_select_then_skipped(tmp_path, monkeypatch):
	# given
	monkeypatch.delenv("SEL_WORKSPACE", raising=False)
	monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	# when
	selection = asyncio.run(selector.select(_FakeSession(["https://example.com/x"])))
	# then
	assert selection.root == tmp_path.resolve()


@pytest.mark.parametrize(
	("env", "source", "expected"),
	[
		("SEL_WORKSPACE", "SEL_WORKSPACE", "pinned by $SEL_WORKSPACE"),
		("CLAUDE_PROJECT_DIR", "CLAUDE_PROJECT_DIR", "$CLAUDE_PROJECT_DIR (default"),
		(None, "cwd", "working directory"),
		("CLAUDE_PROJECT_DIR", "client roots", "reported this checkout/worktree"),
	],
)
def test_given_source_when_explain_then_sentence_names_the_rule(env, source, expected, monkeypatch, tmp_path):
	# given
	monkeypatch.delenv("SEL_WORKSPACE", raising=False)
	monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
	if env:
		monkeypatch.setenv(env, str(tmp_path))
	selector = WorkspaceSelector("SEL_WORKSPACE")
	# then
	assert expected in selector.explain(source)
