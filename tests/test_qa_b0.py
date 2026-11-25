from __future__ import annotations

import asyncio

import pytest

from mcp_nav_shared.notices import NoticeBoard
from mcp_nav_shared.output import cap_output, max_output_chars


def test_rc_049_cap_counts_and_boundaries() -> None:
	text = ("line of content\n" * 1000) + "tail"
	result = cap_output(text, 2000)
	kept, footer = result.rsplit("\n... ", 1)
	assert len(result) <= 2000
	assert f"({len(text) - len(kept)} characters)" in footer
	assert cap_output(text, len(text)) is text
	assert len(cap_output("x" * 100_000, 2000)) <= 2000


@pytest.mark.parametrize(
	("value", "expected"),
	[("", 30000), ("abc", 30000), ("-5", 30000), ("0", 30000), ("123456", 123456), ("1500", 2000)],
)
def test_rc_161_env(monkeypatch: pytest.MonkeyPatch, value: str, expected: int) -> None:
	monkeypatch.setenv("CODENAV_MCP_MAX_OUTPUT_CHARS", value)
	assert max_output_chars() == expected


def test_rc_161_notice_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CODENAV_MCP_MAX_OUTPUT_CHARS", "invalid")
	board = NoticeBoard("test", [])

	@board.tool
	async def tool() -> str:
		return "x" * 40000

	result = asyncio.run(tool())
	assert "more lines" in result
	assert "CODENAV_MCP_MAX_OUTPUT_CHARS=invalid is not a positive integer; using 30000." in result
	assert tool.__output_capped__


def test_rc_161_boundaries_and_unicode(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CODENAV_MCP_MAX_OUTPUT_CHARS", raising=False)
	assert max_output_chars() == 30000
	for size in range(1, 5001):
		text = ("é😀word\n" * 1000)[: size + 2000]
		result = cap_output(text, 2000)
		assert len(result) <= 2000
		if len(text) <= 2000:
			assert result is text
	text = "é" * 2001
	assert cap_output(text[:2000], 2000) == text[:2000]
	assert len(cap_output(text, 2000)) <= 2000
	text = "first\n" + "z" * 3000
	result = cap_output(text, 2000)
	assert result.startswith("first\n... ")
	assert "2 more lines (3001 characters)" in result


def test_rc_161_short_notice_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CODENAV_MCP_MAX_OUTPUT_CHARS", raising=False)
	board = NoticeBoard("test", [])

	@board.tool
	async def tool() -> str:
		return "short\ntext"

	assert asyncio.run(tool()) == "short\ntext"


def test_cap_tiny_limit_and_trailing_blank_lines() -> None:
	assert len(cap_output("x" * 5000, 50)) <= 50
	assert not cap_output("a\n\n" + "x" * 5000, 2000).startswith("a\n\n")
