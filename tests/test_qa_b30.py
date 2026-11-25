"""QA block B30: message wording (BUG-BATCH-R01, RC-050, BUG-MOVE-R02 hint, RC-034 log line)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.lsp_client import LspClient
from mcp_nav_shared.resolve import _same_file_hint
from mcp_nav_shared.transaction import EditJournal, WriteGuard


def _span(line: int, end: int | None = None) -> dict[str, Any]:
	return {"start": {"line": line, "character": 4}, "end": {"line": line if end is None else end, "character": 12}}


class _FakeClient:
	def __init__(self, tree: list[dict[str, Any]]) -> None:
		self.tree = tree

	async def document_symbol(self, rel: str) -> list[dict[str, Any]]:
		return self.tree


def _candidate(name: str, kind: int, line: int, uri: str) -> dict[str, Any]:
	return {"name": name, "kind": kind, "location": {"uri": uri, "range": _span(line)}}


def test_rc_b30_batch_r01_undo_without_id_mentions_eviction(tmp_path: Path) -> None:
	journal = EditJournal(WriteGuard(root=tmp_path, allowed_suffixes=frozenset({".py"})))
	with pytest.raises(ToolInputError) as info:
		journal.undo()
	text = str(info.value)
	assert "no edits were applied" not in text
	assert "older edits were evicted" in text
	assert "last 20 edits" in text
	assert "edit id from the original reply" in text


def test_rc_b30_move_r02_same_file_hint_names_the_qualified_candidates() -> None:
	uri = (Path("/ws") / "pkg" / "a.py").as_uri()
	tree = [
		{
			"name": "Cls",
			"kind": 5,
			"range": _span(0, 5),
			"selectionRange": _span(0),
			"children": [{"name": "run", "kind": 6, "range": _span(2), "selectionRange": _span(2), "children": []}],
		}
	]
	candidates = [_candidate("run", 6, 2, uri), _candidate("run", 12, 9, uri)]
	hint = asyncio.run(_same_file_hint(_FakeClient(tree), candidates, Path("/ws")))  # type: ignore[arg-type]
	assert "pkg/a.py" in hint
	assert "Cls.run (pkg/a.py:3)" in hint
	assert "run (pkg/a.py:10)" in hint


def test_rc_b30_move_r02_same_name_pair_asks_for_a_rename() -> None:
	uri = (Path("/ws") / "pkg" / "a.py").as_uri()
	candidates = [_candidate("helper", 12, 1, uri), _candidate("helper", 12, 7, uri)]
	hint = asyncio.run(_same_file_hint(_FakeClient([]), candidates, Path("/ws")))  # type: ignore[arg-type]
	assert "share a name" in hint
	assert "rename one of them first" in hint


def test_rc_034_lsp_exit_log_line_has_no_traceback_or_paths(caplog: pytest.LogCaptureFixture) -> None:
	client = LspClient.__new__(LspClient)
	client.command = ["/home/u/.venv/bin/ty", "server"]
	client._proc = None
	client._stderr_tail = [
		'  File "/home/u/.venv/lib/site.py", line 3, in <module>',
		"Traceback (most recent call last):",
	]
	loop = asyncio.new_event_loop()
	try:
		client._pending = {1: loop.create_future()}
		with caplog.at_level(logging.WARNING, logger="mcp_nav_shared.lsp_client"):
			client._fail_pending()
	finally:
		loop.close()
	logged = "\n".join(r.getMessage() for r in caplog.records)
	assert "exited unexpectedly" in logged
	assert "Traceback" not in logged
	assert "/home/u" not in logged
