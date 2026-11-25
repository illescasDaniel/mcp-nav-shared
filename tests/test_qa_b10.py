"""B10 (QA campaign): refusal text and name validation in the shared package."""

from __future__ import annotations

import asyncio

import pytest

from mcp_nav_shared.errors import FailedText, ToolInputError, format_tool_error
from mcp_nav_shared.notices import NoticeBoard
from mcp_nav_shared.params import resolve_name_query


def test_rc_032_file_not_found_does_not_echo_the_absolute_path():
	# given
	exc = FileNotFoundError(2, "No such file or directory", "/srv/proj/nope.py")
	# when
	text = format_tool_error(exc)
	# then
	assert isinstance(text, FailedText)
	assert "/srv" not in text
	assert text.startswith("File not found: nope.py (relative paths resolve against the workspace root)")


def test_rc_032_unreadable_file_does_not_echo_the_absolute_path():
	# given
	exc = PermissionError(13, "Permission denied", "/srv/proj/noperm.py")
	# when
	text = format_tool_error(exc)
	# then
	assert "/srv" not in text
	assert text == "Cannot read noperm.py: Permission denied."


def test_user_given_relative_path_is_kept_as_written():
	# given
	exc = FileNotFoundError(2, "No such file or directory", "../outside.py")
	# when
	text = format_tool_error(exc)
	# then
	assert text.startswith("File not found: ../outside.py ")


@pytest.mark.parametrize("bad", ["   ", "..", " . . ", "\x00total", ""])
def test_rc_146_blank_dots_or_nul_names_are_refused_up_front(bad):
	with pytest.raises(ToolInputError, match="^name must be a non-empty identifier or dotted path$"):
		resolve_name_query(name=bad, query=None)


def test_rc_146_blank_query_is_refused_for_query_preferred_tools():
	with pytest.raises(ToolInputError, match="^name must be a non-empty identifier or dotted path$"):
		resolve_name_query(preferred="query", example="create_user", query="  ", name=None)


def test_name_queries_still_resolve_alias_and_dotted_forms():
	assert resolve_name_query(name=None, query="Cart.total") == "Cart.total"
	assert resolve_name_query(name="", query="create_user") == "create_user"


def test_conflicting_name_and_query_is_refused_not_resolved_silently():
	with pytest.raises(ToolInputError, match="give either name or query, not both"):
		resolve_name_query(name="Cart", query="other")


def test_nothing_given_keeps_the_pass_hint():
	with pytest.raises(ToolInputError, match="Pass `name`"):
		resolve_name_query(name=None, query=None)


def test_failed_text_keeps_its_status_through_cap_and_annotate():
	async def refusal() -> str:
		return FailedText("File not found: x.py (relative paths resolve against the workspace root).")

	board = NoticeBoard("test", ())
	result = asyncio.run(board.tool(refusal)())
	assert isinstance(result, FailedText)
	assert result.startswith("File not found: x.py")
