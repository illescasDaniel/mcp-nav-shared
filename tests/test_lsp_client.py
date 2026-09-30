"""Fast unit tests for `mcp_nav_shared.lsp_client` (no live language servers)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from mcp_nav_shared import lsp_client
from mcp_nav_shared.lsp_client import LanguageServerExitedError, LspClient, LspRequestError


def test_given_message_when_lsp_request_error_then_exposes_method():
	# given / when
	err = LspRequestError("textDocument/hover", -32601, "Method not found")
	# then
	assert err.method == "textDocument/hover"
	assert err.code == -32601
	assert "Method not found" in str(err)


class _FakeStdin:
	def write(self, data: bytes) -> None:
		return None

	async def drain(self) -> None:
		return None


class _FakeProc:
	stdin = _FakeStdin()


def _started_client(tmp_path: Path, *, language_id: str = "python") -> LspClient:
	client = LspClient(workspace_root=tmp_path, command=["true"], language_id=language_id)
	client._started = True
	client._proc = _FakeProc()  # type: ignore[assignment]
	return client


def test_given_jsonrpc_error_when_request_then_raises_lsp_request_error(tmp_path):
	# given
	client = _started_client(tmp_path)

	async def _run() -> None:
		async def _respond() -> None:
			await asyncio.sleep(0)
			msg_id = next(iter(client._pending))
			client._dispatch(
				{
					"jsonrpc": "2.0",
					"id": msg_id,
					"error": {"code": -32601, "message": "Method not found"},
				}
			)

		task = asyncio.create_task(_respond())
		with pytest.raises(LspRequestError) as caught:
			await client._request("textDocument/hover", {})
		await task
		assert caught.value.method == "textDocument/hover"
		assert "Method not found" in str(caught.value)

	# when / then
	asyncio.run(_run())


def test_given_pull_unsupported_when_diagnostics_then_returns_push_cache(tmp_path, monkeypatch):
	# given
	src = tmp_path / "x.css"
	src.write_text("a {}\n", encoding="utf-8")
	client = _started_client(tmp_path, language_id="css")
	uri = src.resolve().as_uri()
	cached = [{"message": "from push", "range": {"start": {"line": 0, "character": 0}}}]
	client._diagnostics[uri] = cached

	async def _boom(_method: str, _params: dict, timeout: float = 20) -> dict:
		raise LspRequestError("textDocument/diagnostic", -32601, "Method not found")

	monkeypatch.setattr(client, "_request", _boom)
	monkeypatch.setattr(lsp_client, "PUSH_DIAGNOSTICS_TIMEOUT", 0.01)
	# when
	items = asyncio.run(client.diagnostics(str(src)))
	# then
	assert items == cached


def test_given_empty_full_pull_and_stale_push_cache_when_diagnostics_then_trusts_pull(tmp_path, monkeypatch):
	# given
	src = tmp_path / "x.html"
	src.write_text("<p></p>\n", encoding="utf-8")
	client = _started_client(tmp_path, language_id="html")
	uri = src.resolve().as_uri()
	client._diagnostics[uri] = [{"message": "pushed", "range": {"start": {"line": 0, "character": 0}}}]

	async def _empty(_method: str, _params: dict, timeout: float = 20) -> dict:
		return {"result": {"kind": "full", "items": []}}

	monkeypatch.setattr(client, "_request", _empty)
	monkeypatch.setattr(lsp_client, "PUSH_DIAGNOSTICS_TIMEOUT", 0.01)
	# when
	items = asyncio.run(client.diagnostics(str(src)))
	# then
	assert items == []


def test_given_fixed_file_when_resynced_then_pushed_diagnostics_are_dropped(tmp_path, monkeypatch):
	# given
	src = tmp_path / "x.py"
	src.write_text("x: int = 'a'\n", encoding="utf-8")
	client = _started_client(tmp_path)
	monkeypatch.setattr(client, "_notify", lambda *_a, **_k: None)
	uri = asyncio.run(client.ensure_open(str(src)))
	client._diagnostics[uri] = [{"message": "old error"}]
	# when
	src.write_text("x: int = 1\n\n", encoding="utf-8")
	asyncio.run(client.ensure_open(str(src)))
	# then
	assert uri not in client._diagnostics


class _RecordingStdin(_FakeStdin):
	def __init__(self) -> None:
		self.written: list[bytes] = []

	def write(self, data: bytes) -> None:
		self.written.append(data)


def _sent_bodies(stdin: _RecordingStdin) -> list[dict]:
	import json

	return [json.loads(chunk.split(b"\r\n\r\n", 1)[1]) for chunk in stdin.written]


def test_given_server_requests_when_dispatched_then_client_replies(tmp_path):
	# given
	client = _started_client(tmp_path)
	stdin = _RecordingStdin()
	client._proc.stdin = stdin  # type: ignore[union-attr]
	# when
	client._dispatch({"id": 7, "method": "workspace/configuration", "params": {"items": [{}, {}]}})
	client._dispatch({"id": 8, "method": "client/registerCapability", "params": {}})
	client._dispatch({"id": 9, "method": "some/unknown", "params": {}})
	# then
	replies = {body["id"]: body for body in _sent_bodies(stdin)}
	assert replies[7]["result"] == [None, None]
	assert replies[8]["result"] is None and "error" not in replies[8]
	assert replies[9]["error"]["code"] == -32601


def test_given_pending_request_when_server_exits_then_request_fails_fast(tmp_path):
	# given
	client = _started_client(tmp_path)

	async def _run() -> None:
		async def _exit() -> None:
			await asyncio.sleep(0)
			client._fail_pending()

		task = asyncio.create_task(_exit())
		# when / then — raises immediately instead of waiting out the timeout
		with pytest.raises(LanguageServerExitedError):
			await client._request("textDocument/hover", {}, timeout=5)
		await task

	asyncio.run(_run())


def test_given_server_dies_at_startup_when_start_then_error_quotes_its_stderr(tmp_path):
	# given — a "language server" that prints why it can't run, then exits
	script = "import sys; sys.stderr.write('Failed to spawn: ty\\n'); sys.exit(2)"
	client = LspClient(workspace_root=tmp_path, command=[sys.executable, "-c", script], language_id="python")
	# when
	with pytest.raises(LanguageServerExitedError) as caught:
		asyncio.run(client.start())
	# then — the agent sees the cause, not just "exited"
	assert "language server exited" in str(caught.value)
	assert "Failed to spawn: ty" in str(caught.value)


def test_given_never_started_when_is_alive_then_false(tmp_path):
	# given
	client = LspClient(workspace_root=tmp_path, command=["true"], language_id="python")
	# when / then
	assert client.is_alive is False


def _counting_symbol_request(client: LspClient, monkeypatch, calls: list[str]) -> None:
	async def _fake(method: str, _params: dict, timeout: float = 20) -> dict:
		calls.append(method)
		return {"result": [{"name": f"call{len(calls)}"}]}

	monkeypatch.setattr(client, "_request", _fake)
	monkeypatch.setattr(client, "_notify", lambda *_a, **_k: None)


def test_given_unchanged_file_when_document_symbol_twice_then_second_call_is_cached(tmp_path, monkeypatch):
	# given
	src = tmp_path / "a.py"
	src.write_text("x = 1\n", encoding="utf-8")
	client = _started_client(tmp_path)
	calls: list[str] = []
	_counting_symbol_request(client, monkeypatch, calls)
	# when
	first = asyncio.run(client.document_symbol(str(src)))
	second = asyncio.run(client.document_symbol(str(src)))
	# then
	assert first == second
	assert len(calls) == 1


def test_given_edited_file_when_document_symbol_then_cache_is_invalidated(tmp_path, monkeypatch):
	# given
	src = tmp_path / "a.py"
	src.write_text("x = 1\n", encoding="utf-8")
	client = _started_client(tmp_path)
	calls: list[str] = []
	_counting_symbol_request(client, monkeypatch, calls)
	first = asyncio.run(client.document_symbol(str(src)))
	# when
	src.write_text("x = 1\ny = 2\n", encoding="utf-8")
	second = asyncio.run(client.document_symbol(str(src)))
	# then
	assert len(calls) == 2
	assert first != second


def test_given_closed_scratch_document_when_reopened_then_symbols_are_not_stale(tmp_path, monkeypatch):
	# given
	client = _started_client(tmp_path)
	calls: list[str] = []
	_counting_symbol_request(client, monkeypatch, calls)
	uri = (tmp_path / "scratch.py").as_uri()
	asyncio.run(client.open_scratch_document(uri, "a = 1\n"))
	client._symbol_cache[uri] = (1, [{"name": "stale"}])
	# when
	asyncio.run(client.close_scratch_document(uri))
	# then
	assert uri not in client._symbol_cache


def test_given_ts_suffix_when_ensure_open_then_did_open_uses_mapped_language_id(tmp_path, monkeypatch):
	# given
	(tmp_path / "a.ts").write_text("export const x = 1;\n", encoding="utf-8")
	(tmp_path / "b.js").write_text("export const y = 1;\n", encoding="utf-8")
	client = _started_client(tmp_path, language_id="javascript")
	client.language_ids = {".ts": "typescript"}
	sent: list[dict] = []
	monkeypatch.setattr(client, "_notify", lambda _m, params: sent.append(params["textDocument"]))
	# when
	asyncio.run(client.ensure_open(str(tmp_path / "a.ts")))
	asyncio.run(client.ensure_open(str(tmp_path / "b.js")))
	# then
	assert [d["languageId"] for d in sent] == ["typescript", "javascript"]


def test_given_push_only_server_when_diagnostics_then_waits_for_push_after_sync(tmp_path, monkeypatch):
	# given
	src = tmp_path / "a.ts"
	src.write_text("x\n", encoding="utf-8")
	client = _started_client(tmp_path, language_id="typescript")
	monkeypatch.setattr(client, "_notify", lambda *_a, **_k: None)
	uri = src.resolve().as_uri()
	pushed = [{"message": "late push"}]

	async def _reject(_method: str, _params: dict, timeout: float = 20) -> dict:
		raise LspRequestError("textDocument/diagnostic", -32601, "Unhandled method")

	monkeypatch.setattr(client, "_request", _reject)

	async def _run() -> list[dict]:
		async def _push_later() -> None:
			await asyncio.sleep(0.05)
			client._dispatch(
				{"method": "textDocument/publishDiagnostics", "params": {"uri": uri, "diagnostics": pushed}}
			)

		task = asyncio.create_task(_push_later())
		items = await client.diagnostics(str(src))
		await task
		return items

	# when
	items = asyncio.run(_run())
	# then
	assert items == pushed


def _respond_with(client: LspClient, replies: list[dict]) -> asyncio.Future:
	async def _serve() -> None:
		for reply in replies:
			while not client._pending:
				await asyncio.sleep(0)
			client._dispatch({"jsonrpc": "2.0", "id": next(iter(client._pending)), **reply})
			await asyncio.sleep(0)

	return asyncio.ensure_future(_serve())


def _run_request(client: LspClient, replies: list[dict]) -> dict:
	async def _run() -> dict:
		serving = _respond_with(client, replies)
		try:
			return await client._request("textDocument/hover", {})
		finally:
			serving.cancel()

	return asyncio.run(_run())


@pytest.fixture
def _no_backoff(monkeypatch):
	monkeypatch.setattr(lsp_client, "_CONTENT_MODIFIED_BACKOFF", (0, 0, 0))


def test_given_content_modified_once_when_request_then_retried_and_succeeds(tmp_path, _no_backoff):
	# given
	client = _started_client(tmp_path)
	replies = [{"error": {"code": -32801, "message": "content modified"}}, {"result": {"ok": 1}}]
	# when
	resp = _run_request(client, replies)
	# then
	assert resp["result"] == {"ok": 1}


def test_given_persistent_content_modified_when_request_then_error_surfaces(tmp_path, _no_backoff):
	# given
	client = _started_client(tmp_path)
	replies = [{"error": {"code": -32801, "message": "content modified"}}] * 4
	# when / then
	with pytest.raises(LspRequestError) as caught:
		_run_request(client, replies)
	assert caught.value.code == -32801


def test_given_other_error_when_request_then_not_retried(tmp_path, _no_backoff):
	# given
	client = _started_client(tmp_path)
	replies = [{"error": {"code": -32601, "message": "nope"}}, {"result": {"ok": 1}}]
	# when / then
	with pytest.raises(LspRequestError) as caught:
		_run_request(client, replies)
	assert caught.value.code == -32601


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


def test_given_first_refresh_when_files_exist_then_reports_no_changes(tmp_path, monkeypatch):
	# given
	(tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	# when
	asyncio.run(client.refresh())
	# then
	assert sent == []


def test_given_created_edited_deleted_files_when_refresh_then_reports_each_change(tmp_path, monkeypatch):
	# given
	(tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
	(tmp_path / "gone.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(tmp_path / "new.py").write_text("y = 2\n", encoding="utf-8")
	(tmp_path / "keep.py").write_text("x = 100\n", encoding="utf-8")
	(tmp_path / "gone.py").unlink()
	(tmp_path / "notes.txt").write_text("ignored\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert sorted(_watch_changes(sent)) == [("gone.py", 3), ("keep.py", 2), ("new.py", 1)]


def test_given_unchanged_tree_when_refresh_twice_then_second_reports_nothing(tmp_path, monkeypatch):
	# given
	(tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	asyncio.run(client.refresh())
	# then
	assert sent == []


def test_given_excluded_dirs_and_ignored_paths_when_refresh_then_not_reported(tmp_path, monkeypatch):
	# given
	(tmp_path / ".venv").mkdir()
	(tmp_path / "gen").mkdir()
	client, sent = _recording_client(
		tmp_path,
		monkeypatch,
		watch_suffixes=frozenset({".py"}),
		watch_ignore=lambda p: "gen" in p.parts,
	)
	asyncio.run(client.refresh())
	# when
	(tmp_path / ".venv" / "lib.py").write_text("x = 1\n", encoding="utf-8")
	(tmp_path / "gen" / "out.py").write_text("x = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert sent == []


def test_given_nested_checkout_when_refresh_then_its_files_not_reported(tmp_path, monkeypatch):
	# given: a linked worktree nested inside the checkout (e.g. `.claude/worktrees/x`)
	nested = tmp_path / ".claude" / "worktrees" / "x"
	nested.mkdir(parents=True)
	(nested / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(nested / "other.py").write_text("x = 1\n", encoding="utf-8")
	(tmp_path / "own.py").write_text("x = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert _watch_changes(sent) == [("own.py", 1)]


def test_given_symlinked_workspace_root_when_file_created_then_uri_matches_ensure_open(tmp_path, monkeypatch):
	# given
	real = tmp_path / "real"
	real.mkdir()
	link = tmp_path / "link"
	link.symlink_to(real, target_is_directory=True)
	client, sent = _recording_client(link, monkeypatch, watch_suffixes=frozenset({".py"}))
	asyncio.run(client.refresh())
	# when
	(real / "new.py").write_text("x = 1\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	reported = [c["uri"] for m, p in sent if m == "workspace/didChangeWatchedFiles" for c in p["changes"]]
	assert reported == [client._to_uri("new.py").as_uri()]


def test_given_open_file_edited_on_disk_when_refresh_then_resynced_with_did_change(tmp_path, monkeypatch):
	# given
	src = tmp_path / "a.py"
	src.write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch)
	asyncio.run(client.ensure_open(str(src)))
	src.write_text("x = 12345\n", encoding="utf-8")
	sent.clear()
	# when
	asyncio.run(client.refresh())
	# then
	assert [m for m, _ in sent] == ["textDocument/didChange"]
	assert sent[0][1]["contentChanges"] == [{"text": "x = 12345\n"}]


def test_given_open_file_deleted_when_refresh_then_closed_and_caches_dropped(tmp_path, monkeypatch):
	# given
	src = tmp_path / "a.py"
	src.write_text("x = 1\n", encoding="utf-8")
	client, sent = _recording_client(tmp_path, monkeypatch)
	uri = asyncio.run(client.ensure_open(str(src)))
	client._diagnostics[uri] = [{"message": "old"}]
	client._symbol_cache[uri] = (1, [])
	src.unlink()
	sent.clear()
	# when
	asyncio.run(client.refresh())
	# then
	assert [m for m, _ in sent] == ["textDocument/didClose"]
	assert uri not in client._open_files
	assert uri not in client._diagnostics
	assert uri not in client._symbol_cache


def test_given_scratch_document_when_refresh_then_left_alone(tmp_path, monkeypatch):
	# given
	client, sent = _recording_client(tmp_path, monkeypatch)
	uri = (tmp_path / ".probe.py").as_uri()
	asyncio.run(client.open_scratch_document(uri, "x = 1\n"))
	sent.clear()
	# when
	asyncio.run(client.refresh())
	# then
	assert sent == []
	assert uri in client._open_files


def test_given_open_watched_changes_when_file_created_then_did_open(tmp_path, monkeypatch):
	# given
	client, sent = _recording_client(
		tmp_path, monkeypatch, watch_suffixes=frozenset({".py"}), open_watched_changes=True
	)
	asyncio.run(client.refresh())
	# when
	(tmp_path / "new.py").write_text("y = 2\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert [m for m, _ in sent] == ["workspace/didChangeWatchedFiles", "textDocument/didOpen"]


# -- position validation ---------------------------------------------------------


def _position_error(tmp_path: Path, text: str, line: int, column: int) -> str | None:
	(tmp_path / "a.py").write_text(text, encoding="utf-8")
	client = _started_client(tmp_path)
	try:
		client._check_position("a.py", line, column)
	except lsp_client.InvalidPositionError as exc:
		return str(exc)
	return None


def test_given_line_past_end_when_check_position_then_error_names_line_count(tmp_path):
	# given / when
	error = _position_error(tmp_path, "x = 1\ny = 2\n", 9, 1)
	# then: the trailing newline isn't counted as a line
	assert error is not None
	assert "line 9 is out of range" in error
	assert "has 2 line(s)" in error


def test_given_line_after_final_newline_when_check_position_then_rejected_like_the_count_says(tmp_path):
	# given / when
	error = _position_error(tmp_path, "x = 1\ny = 2\n", 3, 1)
	# then
	assert error is not None
	assert "has 2 line(s)" in error


def test_given_line_zero_when_check_position_then_error(tmp_path):
	# given / when
	error = _position_error(tmp_path, "x = 1\n", 0, 1)
	# then
	assert error is not None
	assert "line 0 is out of range" in error


def test_given_column_past_end_when_check_position_then_error_names_line_length(tmp_path):
	# given / when
	error = _position_error(tmp_path, "abc\n", 1, 50)
	# then
	assert error is not None
	assert "column 50 is out of range" in error
	assert "3 character(s) long" in error


def test_given_column_just_after_last_character_when_check_position_then_accepted(tmp_path):
	# given / when / then: the end-of-line position is valid in LSP
	assert _position_error(tmp_path, "abc\n", 1, 4) is None


def test_given_astral_character_when_check_position_then_column_counts_utf16_units(tmp_path):
	# given: "😀" is two UTF-16 code units, so "😀a" is 3 long and column 4 is its end
	text = "😀a\n"
	# when / then
	assert _position_error(tmp_path, text, 1, 4) is None
	assert _position_error(tmp_path, text, 1, 5) is not None


def test_given_bad_position_when_hover_then_raises_before_asking_the_server(tmp_path):
	# given
	(tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
	client = _started_client(tmp_path)
	# when / then
	with pytest.raises(lsp_client.InvalidPositionError):
		asyncio.run(client.hover("a.py", 99, 1))


# -- restart on config change ------------------------------------------------------


class _Restarts:
	def __init__(self) -> None:
		self.count = 0


def _config_client(tmp_path: Path, monkeypatch, **kwargs) -> tuple[LspClient, _Restarts, list[str], list]:
	notices: list[str] = []
	client, sent = _recording_client(
		tmp_path,
		monkeypatch,
		watch_suffixes=frozenset({".py"}),
		config_names=frozenset({"pyproject.toml"}),
		on_notice=notices.append,
		**kwargs,
	)
	restarts = _Restarts()

	async def _fake_restart() -> None:
		restarts.count += 1

	monkeypatch.setattr(client, "restart", _fake_restart)
	return client, restarts, notices, sent


def test_given_first_refresh_when_config_exists_then_no_restart(tmp_path, monkeypatch):
	# given
	(tmp_path / "pyproject.toml").write_text("[tool.ty]\n", encoding="utf-8")
	client, restarts, notices, _ = _config_client(tmp_path, monkeypatch)
	# when
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 0
	assert notices == []


def test_given_config_edited_when_refresh_then_restarts_once_and_names_the_file(tmp_path, monkeypatch):
	# given
	config = tmp_path / "pyproject.toml"
	config.write_text("[tool.ty]\n", encoding="utf-8")
	client, restarts, notices, _ = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when
	config.write_text("[tool.ty.environment]\npython-version = '3.12'\n", encoding="utf-8")
	asyncio.run(client.refresh())
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 1
	assert notices == ["restarted the language server because pyproject.toml changed"]


def test_given_nested_config_created_when_refresh_then_restarts(tmp_path, monkeypatch):
	# given
	(tmp_path / "web").mkdir()
	client, restarts, notices, _ = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when
	(tmp_path / "web" / "pyproject.toml").write_text("[tool.ty]\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 1
	assert notices == ["restarted the language server because pyproject.toml changed"]


def test_given_config_rewritten_with_same_text_when_refresh_then_no_restart(tmp_path, monkeypatch):
	# given
	config = tmp_path / "pyproject.toml"
	config.write_text("[tool.ty]\n", encoding="utf-8")
	client, restarts, notices, _ = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when: new mtime, identical content (touch, git checkout)
	stat = config.stat()
	os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 0
	assert notices == []


def test_given_config_deleted_when_refresh_then_restarts(tmp_path, monkeypatch):
	# given
	config = tmp_path / "pyproject.toml"
	config.write_text("[tool.ty]\n", encoding="utf-8")
	client, restarts, notices, _ = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when
	config.unlink()
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 1
	assert notices == ["restarted the language server because pyproject.toml changed"]


def test_given_only_source_edit_when_refresh_then_no_restart(tmp_path, monkeypatch):
	# given
	(tmp_path / "pyproject.toml").write_text("[tool.ty]\n", encoding="utf-8")
	(tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
	client, restarts, notices, sent = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when
	(tmp_path / "a.py").write_text("x = 22\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 0
	assert notices == []
	assert _watch_changes(sent) == [("a.py", 2)]


def test_given_config_and_source_changed_when_refresh_then_restart_replaces_change_report(tmp_path, monkeypatch):
	# given
	config = tmp_path / "pyproject.toml"
	config.write_text("[tool.ty]\n", encoding="utf-8")
	client, restarts, _, sent = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when: a fresh server reads the disk itself, so nothing needs reporting
	config.write_text("[tool.ty]\nx = 1\n", encoding="utf-8")
	(tmp_path / "new.py").write_text("y = 2\n", encoding="utf-8")
	asyncio.run(client.refresh())
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 1
	assert _watch_changes(sent) == []


def test_given_config_in_excluded_dir_when_edited_then_no_restart(tmp_path, monkeypatch):
	# given
	(tmp_path / "node_modules").mkdir()
	client, restarts, _, _ = _config_client(tmp_path, monkeypatch)
	asyncio.run(client.refresh())
	# when
	(tmp_path / "node_modules" / "pyproject.toml").write_text("x\n", encoding="utf-8")
	asyncio.run(client.refresh())
	# then
	assert restarts.count == 0


def test_given_started_client_when_restart_then_state_reset_and_on_restart_runs(tmp_path, monkeypatch):
	# given
	seen: list[LspClient] = []

	async def _on_restart(c: LspClient) -> None:
		seen.append(c)

	client = LspClient(workspace_root=tmp_path, command=["true"], language_id="python", on_restart=_on_restart)
	client._open_files["file:///x.py"] = lsp_client.OpenFile(uri="file:///x.py", version=3, mtime_ns=1, size=1)
	client._diagnostics["file:///x.py"] = [{"message": "old"}]
	client._symbol_cache["file:///x.py"] = (3, [])
	calls: list[str] = []

	async def _stop() -> None:
		calls.append("stop")

	async def _start() -> None:
		calls.append("start")

	monkeypatch.setattr(client, "stop", _stop)
	monkeypatch.setattr(client, "start", _start)

	async def _in_flight() -> str:
		fut = asyncio.get_running_loop().create_future()
		client._pending[1] = fut
		asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(client.restart()))
		try:
			await asyncio.wait_for(fut, timeout=5)
		except lsp_client.LanguageServerExitedError:
			return "failed fast"
		return "answered"

	# when: a request pending on the old server fails at restart instead of timing out
	assert asyncio.run(_in_flight()) == "failed fast"
	calls.clear()
	seen.clear()
	# when
	asyncio.run(client.restart())
	# then
	assert calls == ["stop", "start"]
	assert client._open_files == {}
	assert client._diagnostics == {}
	assert client._symbol_cache == {}
	assert seen == [client]
