"""Fast unit tests for `mcp_nav_shared.errors` (no live language servers)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_nav_shared.errors import TOOL_ERRORS, ToolInputError, format_tool_error
from mcp_nav_shared.lsp_client import LspClient


class _FakeStdin:
	def write(self, _data: bytes) -> None:
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


def test_given_missing_file_when_ensure_open_then_tool_error_names_path(tmp_path):
	# given
	client = _started_client(tmp_path)
	# when
	with pytest.raises(TOOL_ERRORS) as caught:
		asyncio.run(client.ensure_open("nope/missing.py"))
	text = format_tool_error(caught.value)
	# then
	assert text.startswith("File not found: ")
	assert "missing.py" in text


def test_given_timeout_when_format_tool_error_then_mentions_retry():
	# given / when
	text = format_tool_error(TimeoutError())
	# then
	assert "timed out" in text
	assert "retry" in text


def test_given_unsupported_input_when_format_tool_error_then_passes_message_through():
	# given / when
	text = format_tool_error(ToolInputError("webnav has no language server for 'a.md'"))
	# then
	assert text == "webnav has no language server for 'a.md'"


def test_given_spawn_oserror_without_filename_when_format_then_mentions_language_server():
	# given — CreateProcess failures often set filename=None (WinError 193, missing npx, …)
	exc = OSError(193, "%1 is not a valid Win32 application")
	exc.filename = None
	# when
	text = format_tool_error(exc)
	# then
	assert text.startswith("Cannot start language server:")
	assert "Win32" in text or "valid" in text
