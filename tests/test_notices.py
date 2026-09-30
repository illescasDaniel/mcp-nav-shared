"""Fast unit tests for `mcp_nav_shared.notices`."""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path

from mcp_nav_shared.notices import NoticeBoard, package_source_dirs


def _board(tmp_path: Path) -> tuple[NoticeBoard, Path]:
	source = tmp_path / "pkg" / "mod.py"
	source.parent.mkdir()
	source.write_text("x = 1\n", encoding="utf-8")
	return NoticeBoard("demo", [source.parent], recheck_seconds=0), source


def test_given_no_notices_and_fresh_code_when_annotate_then_text_unchanged(tmp_path):
	# given
	board, _ = _board(tmp_path)
	# when / then
	assert board.annotate("result") == "result"


def test_given_posted_notice_when_annotate_twice_then_shown_once(tmp_path):
	# given
	board, _ = _board(tmp_path)
	board.post("restarted the language server")
	# when
	first = board.annotate("result")
	second = board.annotate("result")
	# then
	assert first == "result\n\n[demo] restarted the language server"
	assert second == "result"


def test_given_same_notice_posted_twice_when_annotate_then_not_duplicated(tmp_path):
	# given
	board, _ = _board(tmp_path)
	board.post("once")
	board.post("once")
	# when / then
	assert board.annotate("r").count("once") == 1


def test_given_source_changed_since_start_when_annotate_then_stale_note_on_every_call(tmp_path):
	# given
	board, source = _board(tmp_path)
	source.write_text("x = 22222\n", encoding="utf-8")
	# when
	first = board.annotate("result")
	second = board.annotate("result")
	# then
	assert "restart the MCP servers" in first
	assert second == first


def test_given_new_source_file_since_start_when_code_is_stale_then_true(tmp_path):
	# given
	board, source = _board(tmp_path)
	(source.parent / "added.py").write_text("y = 1\n", encoding="utf-8")
	# when / then
	assert board.code_is_stale()


def test_given_source_touched_back_to_same_stamp_when_code_is_stale_then_false(tmp_path):
	# given
	board, source = _board(tmp_path)
	stamp = source.stat()
	source.write_text("x = 2\n", encoding="utf-8")  # same size, new mtime
	os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
	# when / then
	assert not board.code_is_stale()


def test_given_tool_when_decorated_then_signature_and_resolved_annotations_kept(tmp_path):
	# given: a postponed-annotation tool, like every real one
	board, _ = _board(tmp_path)

	async def sample(file_path: str, line: int = 1, ctx: Path | None = None) -> str:
		return f"{file_path}:{line}"

	# when
	wrapped = board.tool(sample)
	# then: the framework inspects these to build the schema and find `Context`
	assert list(inspect.signature(wrapped).parameters) == ["file_path", "line", "ctx"]
	assert wrapped.__annotations__["ctx"] == (Path | None)
	assert wrapped.__annotations__["file_path"] is str
	assert asyncio.run(wrapped("a.py", 3)) == "a.py:3"


def test_given_tool_and_pending_notice_when_called_then_result_annotated(tmp_path):
	# given
	board, _ = _board(tmp_path)

	async def sample() -> str:
		board.post("heads up")
		return "answer"

	# when
	result = asyncio.run(board.tool(sample)())
	# then
	assert result == "answer\n\n[demo] heads up"


def test_given_module_when_package_source_dirs_then_its_directory(tmp_path):
	# given / when
	import mcp_nav_shared

	dirs = package_source_dirs(mcp_nav_shared)
	# then
	assert dirs == [Path(mcp_nav_shared.__file__).resolve().parent]


def test_given_recent_check_when_source_changes_then_stale_verdict_reused_until_interval(tmp_path):
	# given
	source = tmp_path / "pkg" / "mod.py"
	source.parent.mkdir()
	source.write_text("x = 1\n", encoding="utf-8")
	board = NoticeBoard("demo", [source.parent], recheck_seconds=3600)
	# when
	source.write_text("x = 22222\n", encoding="utf-8")
	within_interval = board.code_is_stale()
	board._checked_at -= 3601
	after_interval = board.code_is_stale()
	# then
	assert not within_interval
	assert after_interval
