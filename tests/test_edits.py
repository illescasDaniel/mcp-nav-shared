"""Fast unit tests for `mcp_nav_shared.edits`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_nav_shared.edits import (
	EditConflictError,
	EditPlan,
	FileChange,
	TextEdit,
	UnsupportedEditError,
	apply_text_edits,
	changes_from_edits,
	count_changed_lines,
	decode_source,
	detect_eol,
	encode_source,
	parse_workspace_edit,
	position_to_offset,
	replace_lines,
	unified_diff,
)


def _edit(sl: int, sc: int, el: int, ec: int, text: str) -> TextEdit:
	return TextEdit(sl, sc, el, ec, text)


def test_given_single_edit_when_apply_then_replaces_range():
	# given
	text = "def foo():\n\treturn 1\n"
	# when
	result = apply_text_edits(text, [_edit(0, 4, 0, 7, "bar")])
	# then
	assert result == "def bar():\n\treturn 1\n"


def test_given_edits_in_any_order_when_apply_then_positions_refer_to_original_text():
	# given
	text = "a = foo\nb = foo\n"
	edits = [_edit(1, 4, 1, 7, "bar"), _edit(0, 4, 0, 7, "bar")]
	# when
	result = apply_text_edits(text, edits)
	# then
	assert result == "a = bar\nb = bar\n"


def test_given_overlapping_edits_when_apply_then_conflict_error():
	# given
	text = "abcdef\n"
	# when / then
	with pytest.raises(EditConflictError):
		apply_text_edits(text, [_edit(0, 0, 0, 4, "x"), _edit(0, 2, 0, 5, "y")])


def test_given_two_insertions_at_same_point_when_apply_then_keep_given_order():
	# given
	text = "ab\n"
	# when
	result = apply_text_edits(text, [_edit(0, 1, 0, 1, "1"), _edit(0, 1, 0, 1, "2")])
	# then
	assert result == "a12b\n"


def test_given_non_bmp_characters_when_apply_then_columns_count_utf16_units():
	# given — the emoji takes two UTF-16 code units, so `x` starts at character 3
	text = "\U0001f600 x = 1\n"
	# when
	result = apply_text_edits(text, [_edit(0, 3, 0, 4, "y")])
	# then
	assert result == "\U0001f600 y = 1\n"


def test_given_crlf_file_when_apply_then_inserted_text_uses_crlf():
	# given
	text = "a\r\nb\r\n"
	# when
	result = apply_text_edits(text, [_edit(1, 1, 1, 1, "\nc")])
	# then
	assert result == "a\r\nb\r\nc\r\n"


def test_given_position_past_end_when_offset_then_clamped_to_text_end():
	# given
	text = "ab\ncd"
	# when / then
	assert position_to_offset(text, 9, 0) == len(text)
	assert position_to_offset(text, 0, 99) == 2  # clamped to the line, before its terminator


def test_given_mixed_endings_when_detect_eol_then_majority_wins():
	assert detect_eol("a\r\nb\r\nc\n") == "\r\n"
	assert detect_eol("a\nb\r\nc\n") == "\n"
	assert detect_eol("no newline") == "\n"


def test_given_bom_when_decode_then_bom_is_separated_and_round_trips():
	# given
	raw = b"\xef\xbb\xbfx = 1\r\n"
	# when
	text, bom = decode_source(raw)
	# then
	assert (text, bom) == ("x = 1\r\n", True)
	assert encode_source(text, bom) == raw


def test_given_line_range_when_replace_lines_then_whole_lines_swapped():
	# given
	text = "a\nb\nc\nd\n"
	# when / then
	assert replace_lines(text, 2, 3, "X\n") == "a\nX\nd\n"
	assert replace_lines(text, 2, 3, "") == "a\nd\n"
	assert replace_lines(text, 4, 4, "D") == "a\nb\nc\nD"


def test_given_range_outside_file_when_replace_lines_then_conflict_error():
	with pytest.raises(EditConflictError):
		replace_lines("a\n", 5, 6, "x")


def test_given_changes_and_document_changes_when_parse_then_edits_grouped_per_file(tmp_path):
	# given
	a, b = tmp_path / "a.py", tmp_path / "b.py"
	rng = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}
	workspace_edit = {
		"changes": {a.as_uri(): [{"range": rng, "newText": "x"}]},
		"documentChanges": [
			{"textDocument": {"uri": b.as_uri(), "version": 1}, "edits": [{"range": rng, "newText": "y"}]}
		],
	}
	# when
	parsed = parse_workspace_edit(workspace_edit)
	# then
	assert parsed == {a: [_edit(0, 0, 0, 1, "x")], b: [_edit(0, 0, 0, 1, "y")]}


def test_given_file_operation_when_parse_then_unsupported():
	with pytest.raises(UnsupportedEditError):
		parse_workspace_edit({"documentChanges": [{"kind": "rename", "oldUri": "file:///a", "newUri": "file:///b"}]})


def test_given_none_when_parse_then_empty():
	assert parse_workspace_edit(None) == {}


def test_given_edits_when_changes_from_edits_then_noops_are_dropped(tmp_path):
	# given
	(tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
	(tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
	edits = {tmp_path / "a.py": [_edit(0, 0, 0, 1, "z")], tmp_path / "b.py": [_edit(0, 0, 0, 1, "y")]}
	# when
	changes = changes_from_edits(edits)
	# then
	assert [c.path.name for c in changes] == ["a.py"]
	assert changes[0].new_text == "z = 1\n"


def test_given_file_with_bom_when_changes_from_edits_then_bom_is_kept(tmp_path):
	# given
	path = tmp_path / "a.py"
	path.write_bytes(b"\xef\xbb\xbfx = 1\n")
	# when
	[change] = changes_from_edits({path: [_edit(0, 0, 0, 1, "y")]})
	# then
	assert change.new_bytes() == b"\xef\xbb\xbfy = 1\n"


def test_given_change_when_inverse_then_swaps_old_and_new():
	# given
	change = FileChange(Path("/w/a.py"), "old", "new")
	# when / then
	assert change.inverse() == FileChange(Path("/w/a.py"), "new", "old")
	assert FileChange(Path("/w/a.py"), None, "x").kind == "create"
	assert FileChange(Path("/w/a.py"), "x", None).kind == "delete"


def test_given_modification_when_unified_diff_then_git_style_header(tmp_path):
	# given
	change = FileChange(tmp_path / "a.py", "x = 1\ny = 2\n", "x = 1\ny = 3\n")
	# when
	diff = unified_diff(change, tmp_path)
	# then
	assert diff.startswith("--- a/a.py\n+++ b/a.py\n")
	assert "-y = 2\n+y = 3\n" in diff


def test_given_missing_trailing_newline_when_diff_then_marker_is_emitted(tmp_path):
	# given
	change = FileChange(tmp_path / "a.py", "x", "y")
	# when / then
	assert "\\ No newline at end of file" in unified_diff(change, tmp_path)


def test_given_replacement_when_count_changed_lines_then_added_and_removed():
	assert count_changed_lines(FileChange(Path("a"), "a\nb\nc\n", "a\nB\nc\nd\n")) == (2, 1)


def test_given_plan_when_created_then_remembers_base_hashes_and_summarises(tmp_path):
	# given
	change = FileChange(tmp_path / "a.py", "x = 1\n", "x = 2\n")
	new_file = FileChange(tmp_path / "n.py", None, "z = 1\n")
	# when
	plan = EditPlan("demo", [change, new_file])
	# then
	assert plan.base_hashes[new_file.path] is None
	assert plan.base_hashes[change.path] is not None
	assert plan.summary(tmp_path) == "  a.py: +1 -1\n  n.py (new file): +1 -0"
	assert plan.inverse().changes == [new_file.inverse(), change.inverse()]


def test_given_long_diff_when_plan_diff_then_truncated(tmp_path):
	# given
	plan = EditPlan("big", [FileChange(tmp_path / "a.py", "", "".join(f"l{i}\n" for i in range(50)))])
	# when
	text = plan.diff(tmp_path, max_lines=10)
	# then
	assert "diff truncated" in text
