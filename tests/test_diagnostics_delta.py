"""Fast unit tests for `mcp_nav_shared.diagnostics_delta`."""

from __future__ import annotations

from mcp_nav_shared.diagnostics_delta import DiagEntry, diff_diagnostics, entries_from_lsp, format_delta


def _entry(path="a.py", line=1, code="rule", message="msg", source="x", severity=1) -> DiagEntry:
	return DiagEntry(path, line, 1, severity, code, message, source)


def _lsp(line: int, message: str, code: str = "rule", severity: int = 1) -> dict:
	return {"range": {"start": {"line": line, "character": 2}}, "message": message, "code": code, "severity": severity}


def test_given_lsp_items_when_entries_then_one_based_with_source_line():
	# given / when
	[entry] = entries_from_lsp("pkg/a.py", "x = 1\n  y = undefined\n", [_lsp(1, "bad name")])
	# then
	assert (entry.path, entry.line, entry.column, entry.code, entry.source_line) == (
		"pkg/a.py",
		2,
		3,
		"rule",
		"y = undefined",
	)


def test_given_missing_severity_and_code_when_entries_then_defaults_to_error():
	# given / when
	[entry] = entries_from_lsp("a.py", "x\n", [{"range": {"start": {"line": 0, "character": 0}}, "message": "m"}])
	# then
	assert entry.is_error and entry.code == ""


def test_given_same_diagnostics_shifted_down_when_diff_then_nothing_new():
	# given — an edit inserted lines above, moving the same problem from line 3 to line 7
	before = [_entry(line=3)]
	after = [_entry(line=7)]
	# when
	delta = diff_diagnostics(before, after)
	# then
	assert delta.new == [] and delta.fixed == [] and delta.unchanged == 1 and delta.is_clean


def test_given_new_problem_when_diff_then_reported_as_new_error():
	# given
	before = [_entry(message="old")]
	after = [_entry(message="old"), _entry(line=5, message="fresh", source="foo()")]
	# when
	delta = diff_diagnostics(before, after)
	# then
	assert [e.message for e in delta.new_errors] == ["fresh"]
	assert not delta.is_clean


def test_given_problem_gone_when_diff_then_reported_as_fixed():
	# given / when
	delta = diff_diagnostics([_entry(message="gone")], [])
	# then
	assert [e.message for e in delta.fixed_errors] == ["gone"] and delta.new == []


def test_given_same_problem_on_edited_line_when_diff_then_paired_by_message():
	# given — the offending line was reformatted but it is the same diagnostic
	before = [_entry(source="f(1)")]
	after = [_entry(source="f(1,)")]
	# when
	delta = diff_diagnostics(before, after)
	# then
	assert delta.new == [] and delta.fixed == []


def test_given_duplicate_diagnostics_when_one_removed_then_counts_matter():
	# given
	before = [_entry(), _entry()]
	after = [_entry()]
	# when
	delta = diff_diagnostics(before, after)
	# then
	assert len(delta.fixed) == 1 and delta.new == []


def test_given_new_warning_when_diff_then_not_counted_as_error():
	# given / when
	delta = diff_diagnostics([], [_entry(severity=2)])
	# then
	assert delta.is_clean and len(delta.new) == 1


def test_given_delta_when_format_then_headline_and_sections():
	# given
	delta = diff_diagnostics([_entry(message="gone")], [_entry(message="fresh", line=4)])
	delta.files_checked = 2
	delta.unchecked = ["z.py"]
	# when
	text = format_delta(delta)
	# then
	assert text.splitlines()[0] == "Diagnostics: 1 new error(s), 1 fixed (2 file(s) checked, 0 unchanged)"
	assert "not checked: z.py" in text
	assert "  + a.py:4:1 error [rule]: fresh" in text
	assert "  - a.py:1:1 error [rule]: gone" in text


def test_given_many_new_when_format_then_truncated_with_count():
	# given
	delta = diff_diagnostics([], [_entry(line=i, message=f"m{i}") for i in range(1, 30)])
	# when
	text = format_delta(delta, limit=5)
	# then
	assert "... 24 more" in text
