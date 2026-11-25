"""QA block B31 (shared): a created directory is reported to the language server with its first file (BUG-DEPS-R10)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mcp_nav_shared.lsp_client import LspClient


class _FakeStdin:
	def write(self, data: bytes) -> None:
		return None

	async def drain(self) -> None:
		return None


class _FakeProc:
	stdin = _FakeStdin()


def _recording_client(tmp_path: Path, monkeypatch, **kwargs) -> tuple[LspClient, list[tuple[str, dict]]]:
	client = LspClient(workspace_root=tmp_path, command=["true"], language_id="python", **kwargs)
	client._started = True
	client._proc = _FakeProc()  # type: ignore[assignment]
	sent: list[tuple[str, dict]] = []
	monkeypatch.setattr(client, "_notify", lambda method, params: sent.append((method, params)))
	return client, sent


def _watch_changes(sent: list[tuple[str, dict]]) -> list[tuple[str, int]]:
	return [
		(Path(c["uri"]).name, c["type"])
		for method, params in sent
		if method == "workspace/didChangeWatchedFiles"
		for c in params["changes"]
	]


def test_bug_deps_r10_refresh_reports_a_new_directory_before_its_file(tmp_path, monkeypatch):
	# given
	(tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(tmp_path / "pkg" / "sub").mkdir(parents=True)
	(tmp_path / "pkg" / "sub" / "core.py").write_text("X = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then — outermost directory first, the file last
	assert _watch_changes(sent) == [("pkg", 1), ("sub", 1), ("core.py", 1)]


def test_bug_deps_r10_existing_directory_is_not_reported_again(tmp_path, monkeypatch):
	# given
	(tmp_path / "pkg").mkdir()
	(tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(tmp_path / "pkg" / "b.py").write_text("y = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert _watch_changes(sent) == [("b.py", 1)]


def test_bug_deps_r10_refresh_reports_a_removed_directory_after_its_file(tmp_path, monkeypatch):
	# given
	(tmp_path / "pkg").mkdir()
	(tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(tmp_path / "pkg" / "a.py").unlink()
	(tmp_path / "pkg").rmdir()
	asyncio.run(client.refresh())
	# then
	assert _watch_changes(sent) == [("a.py", 3), ("pkg", 3)]


def test_bug_deps_r10_overlay_reports_the_directories_it_creates_and_removes(tmp_path, monkeypatch):
	# given
	client, sent = _recording_client(tmp_path, monkeypatch)
	ghost = tmp_path / "newpkg" / "deeper" / "new.py"

	async def _run() -> None:
		async with client.overlay({ghost: "y = 1\n"}):
			pass

	# when
	asyncio.run(_run())
	# then
	changes = [
		(Path(c["uri"]).name, c["type"])
		for m, p in sent
		if m == "workspace/didChangeWatchedFiles"
		for c in p["changes"]
	]
	assert changes == [
		("newpkg", 1),
		("deeper", 1),
		("new.py", 1),
		("new.py", 3),
		("deeper", 3),
		("newpkg", 3),
	]


def test_bug_deps_r10_new_directory_is_not_opened_when_changed_files_are_opened(tmp_path, monkeypatch):
	# given — a client that `didOpen`s the watched files that appeared (the tsserver mode)
	(tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(
		tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}), open_watched_changes=True
	)
	asyncio.run(client.refresh())
	# when
	(tmp_path / "pkg").mkdir()
	(tmp_path / "pkg" / "new.py").write_text("X = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	opened = [Path(p["textDocument"]["uri"]).name for m, p in sent if m == "textDocument/didOpen"]
	assert opened == ["new.py"]
	assert _watch_changes(sent)[:2] == [("pkg", 1), ("new.py", 1)]
