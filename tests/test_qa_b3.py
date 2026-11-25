from mcp_nav_shared.diagnostics_delta import DiagEntry, diff_diagnostics


def test_rc_035_rename() -> None:
	before = DiagEntry("old.py", 1, 1, 1, "error", "bad", "x")
	after = DiagEntry("new.py", 1, 1, 1, "error", "bad", "x")
	delta = diff_diagnostics([before], [after], renames={"old.py": "new.py"})
	assert not delta.new and not delta.fixed


def test_rc_035_moved_diagnostic_new_source_line() -> None:
	before = DiagEntry("old.py", 1, 1, 1, "error", "bad", "old line")
	after = DiagEntry("new.py", 1, 1, 1, "error", "bad", "new line")
	delta = diff_diagnostics([before], [after], renames={"old.py": "new.py"})
	assert delta.new == [after] and len(delta.fixed) == 1
