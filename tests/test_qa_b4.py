"""B4 regression tests (mcp-nav-shared): line splitting, BOM, EOL, file-list cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_nav_shared.edits import (
	EditConflictError,
	EditPlan,
	FileChange,
	TextEdit,
	apply_text_edits,
	changes_from_edits,
	convert_eol,
	count_changed_lines,
	encode_source,
	split_lines,
	unified_diff,
)


_BOM = b"\xef\xbb\xbf"


def test_rc_001_split_lines_ignores_form_feed_and_unicode_separators():
	# given — FF, FS/GS/RS, NEL, LS, PS are ordinary characters in Python source
	text = "a\x0cb\nc\x1cd\x1de\x1ef\x85g h i\r\nj\rk\n"
	# when / then
	assert split_lines(text) == ["a\x0cb", "c\x1cd\x1de\x1ef\x85g h i", "j", "k"]
	assert split_lines(text, keepends=True) == [
		"a\x0cb\n",
		"c\x1cd\x1de\x1ef\x85g h i\r\n",
		"j\r",
		"k\n",
	]
	assert len(text.splitlines()) > 4  # what the old code did


@pytest.mark.parametrize("text", ["", "a", "a\n", "a\n\n", "\n", "a\r\nb", "a\r\n\r\n", "a\rb\r", "\r\n"])
def test_rc_001_split_lines_matches_splitlines_on_real_line_ends(text):
	assert split_lines(text) == text.splitlines()
	assert split_lines(text, keepends=True) == text.splitlines(keepends=True)


def test_rc_001_diff_and_counts_keep_form_feed_lines_whole(tmp_path):
	# given — one changed line, a form feed line before it
	change = FileChange(
		tmp_path / "ff.py",
		"def a():\n\treturn 1\n\x0c\ndef b():\n\treturn 2\n",
		"def a():\n\treturn 1\n\x0c\ndef b():\n\treturn 3\n",
	)
	# when / then
	assert count_changed_lines(change) == (1, 1)
	diff = unified_diff(change, tmp_path)
	assert "-\treturn 2\n+\treturn 3\n" in diff
	assert "@@ -3,3 +3,3 @@" in diff


def test_rc_001_apply_text_edits_columns_survive_form_feed():
	# given — `\x0c` on line 3 must not shift the lines below it
	text = "def a():\n\treturn 1\n\x0c\ndef b():\n\treturn 2\n"
	# when
	result = apply_text_edits(text, [TextEdit(3, 4, 3, 5, "bb")])
	# then
	assert result == "def a():\n\treturn 1\n\x0c\ndef bb():\n\treturn 2\n"


def test_rc_008_bom_importer_first_line_rename_columns_counting_the_mark(tmp_path):
	# given — ty read this file from disk, so its columns on line 1 include the BOM (24 for `bomfn` at 23)
	path = tmp_path / "bomimp.py"
	path.write_bytes(_BOM + b"from pkg.bomdef import bomfn\n\n\ndef go():\n\treturn bomfn()\n")
	edits = {path: [TextEdit(0, 24, 0, 29, "bomfn2"), TextEdit(4, 8, 4, 13, "bomfn2")]}
	# when
	[change] = changes_from_edits(edits, bom_in_positions=True)
	# then
	assert change.new_bytes() == _BOM + b"from pkg.bomdef import bomfn2\n\n\ndef go():\n\treturn bomfn2()\n"


def test_rc_008_bom_file_opened_without_the_mark_keeps_its_columns(tmp_path):
	# given — the client opened the document without the BOM, so the columns are already right
	path = tmp_path / "lib.py"
	path.write_bytes(_BOM + b"def helper(x):\n\treturn x\n")
	# when
	[change] = changes_from_edits({path: [TextEdit(0, 4, 0, 10, "assist")]}, bom_in_positions=False)
	# then
	assert change.new_bytes() == _BOM + b"def assist(x):\n\treturn x\n"


def test_rc_008_bom_first_line_single_letter_name_is_shifted(tmp_path):
	# given — `a = a`: both readings sit on word boundaries, only the identifier one is a name
	path = tmp_path / "a.py"
	path.write_bytes(_BOM + b"a = a\n")
	# when
	[change] = changes_from_edits({path: [TextEdit(0, 1, 0, 2, "b"), TextEdit(0, 5, 0, 6, "b")]}, bom_in_positions=True)
	# then
	assert change.new_bytes() == _BOM + b"b = b\n"


def test_rc_030_file_change_without_flags_detects_the_bom_on_disk(tmp_path):
	# given — a write site that builds the change from BOM-stripped text and passes no flag
	path = tmp_path / "bom.py"
	path.write_bytes(_BOM + b"x = 1\n")
	# when
	change = FileChange(path, "x = 1\n", "x = 2\n")
	plan = EditPlan("edit", [change])
	# then
	assert change.bom is True
	assert change.old_bytes() == _BOM + b"x = 1\n"
	assert change.new_bytes() == _BOM + b"x = 2\n"
	assert plan.base_hashes[path] is not None
	assert (
		plan.base_hashes[path]
		== EditPlan("same", [FileChange(path, "x = 1\n", "x = 3\n", True, True)]).base_hashes[path]
	)
	assert change.inverse().new_bytes() == _BOM + b"x = 1\n"


def test_rc_030_file_without_bom_and_new_file_get_none(tmp_path):
	# given
	plain = tmp_path / "plain.py"
	plain.write_bytes(b"x = 1\n")
	# when / then
	assert FileChange(plain, "x = 1\n", "x = 2\n").new_bytes() == b"x = 2\n"
	assert FileChange(tmp_path / "new.py", None, "x = 2\n").new_bytes() == b"x = 2\n"
	assert FileChange(tmp_path / "gone.py", "x\n", None, True, False).old_bytes() == _BOM + b"x\n"


def test_rc_030_explicit_flags_win(tmp_path):
	# given
	path = tmp_path / "bom.py"
	path.write_bytes(_BOM + b"x = 1\n")
	# when / then — callers that know better can still drop the mark
	assert FileChange(path, "x = 1\n", "x = 2\n", True, False).new_bytes() == b"x = 2\n"


def test_rc_105_leading_bom_in_new_text_is_dropped_not_doubled(tmp_path):
	# given
	path = tmp_path / "bom.py"
	path.write_bytes(_BOM + b"x = 1\n")
	# when
	change = FileChange(path, "x = 1\n", "﻿x = 2\n")
	# then
	assert change.new_text == "x = 2\n"
	assert change.new_bytes() == _BOM + b"x = 2\n"
	assert encode_source("﻿x = 2\n", True) == _BOM + b"x = 2\n"


def test_rc_012_convert_eol_normalises_mixed_text():
	assert convert_eol("a\nb\r\nc\rd", "\r\n") == "a\r\nb\r\nc\r\nd"
	assert convert_eol("a\r\nb\n", "\n") == "a\nb\n"


def test_rc_012_inserted_text_in_crlf_file_stays_crlf():
	# given
	text = "def f():\r\n\treturn 1\r\n"
	# when
	result = apply_text_edits(text, [TextEdit(1, 0, 1, 0, "\tx = 1\n")])
	# then
	assert result == "def f():\r\n\tx = 1\r\n\treturn 1\r\n"
	assert "\n" not in result.replace("\r\n", "")


def test_rc_161_plan_summary_lists_at_most_limit_files(tmp_path):
	# given — 1,000 importers
	changes = [FileChange(tmp_path / f"m{i}.py", "a = 1\n", "a = 2\n") for i in range(1000)]
	plan = EditPlan("big", changes)
	# when
	summary = plan.summary(tmp_path)
	short = plan.summary(tmp_path, limit=3)
	# then
	lines = summary.splitlines()
	assert len(lines) == 61
	assert lines[-1] == "  ... and 940 more files"
	assert len(summary) < 4000
	assert short.splitlines()[-1] == "  ... and 997 more files"
	assert EditPlan("few", changes[:5]).summary(tmp_path).count("\n") == 4


def test_rc_161_plan_summary_below_limit_is_unchanged(tmp_path):
	# given
	plan = EditPlan(
		"two", [FileChange(tmp_path / "a.py", "a\n", "b\n"), FileChange(Path(tmp_path / "n.py"), None, "x\n")]
	)
	# when / then
	assert plan.summary(tmp_path) == "  a.py: +1 -1\n  n.py (new file): +1 -0"


def test_bom_positions_are_explicit_and_utf16_safe():
	text = "a😀bc\nnext\n"
	assert apply_text_edits(text, [TextEdit(0, 1, 0, 1, "X")], bom=True, bom_in_positions=False) == "aX😀bc\nnext\n"
	assert apply_text_edits(text, [TextEdit(0, 4, 1, 1, "X")], bom=True, bom_in_positions=True) == "a😀Xext\n"
	with pytest.raises(EditConflictError, match="coordinate convention is unknown"):
		apply_text_edits(text, [TextEdit(0, 1, 0, 1, "X")], bom=True)


def test_old_text_bom_is_detected_before_defaulting_new_bom(tmp_path):
	change = FileChange(tmp_path / "absent.py", "\ufeffx\n", "y\n")
	assert change.old_bytes() == _BOM + b"x\n"
	assert change.new_bytes() == _BOM + b"y\n"
	assert change.inverse().new_bytes() == _BOM + b"x\n"


def test_bom_conventions_are_selected_per_file(tmp_path):
	opened = tmp_path / "opened.py"
	unopened = tmp_path / "unopened.py"
	opened.write_bytes(_BOM + "a😀bc\n".encode())
	unopened.write_bytes(_BOM + "a😀bc\n".encode())
	changes = changes_from_edits(
		{opened: [TextEdit(0, 1, 0, 1, "X")], unopened: [TextEdit(0, 2, 0, 2, "Y")]},
		bom_in_positions={opened.resolve(): False, unopened.resolve(): True},
	)
	assert [c.new_bytes() for c in changes] == [
		_BOM + "aX😀bc\n".encode(),
		_BOM + "aY😀bc\n".encode(),
	]
