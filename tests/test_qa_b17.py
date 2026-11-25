"""B17: leftover line splitting (RC-001), write-limit wording (RC-033), and the notes of a diagnostics delta (RC-002)."""

from __future__ import annotations

import pytest

from mcp_nav_shared.diagnostics_delta import DiagnosticsDelta, entries_from_lsp, format_delta
from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.transaction import WriteGuard, WriteRefusedError


def test_rc_001_diagnostic_source_line_after_form_feed() -> None:
	# A form feed is an ordinary character: LSP line 1 is "foo = 2", not the empty piece str.splitlines() makes.
	text = "x = 1\x0c\nfoo = 2\n"
	items = [{"range": {"start": {"line": 1, "character": 0}}, "severity": 1, "message": "bad", "code": "E"}]
	(entry,) = entries_from_lsp("m.py", text, items)
	assert entry.line == 2
	assert entry.source_line == "foo = 2"


def test_rc_033_limit_message_says_how_to_apply(tmp_path) -> None:
	guard = WriteGuard(root=tmp_path, allowed_suffixes=frozenset({".py"}), max_files=1)
	plan = EditPlan(
		"Rename across files",
		[FileChange(tmp_path / "a.py", "a\n", "b\n"), FileChange(tmp_path / "b.py", "a\n", "b\n")],
	)
	with pytest.raises(WriteRefusedError) as refused:
		guard.check_plan(plan)
	assert "preview it with apply=false, then pass allow_large=true to apply it" in str(refused.value)


def test_rc_002_delta_report_prints_notes() -> None:
	delta = DiagnosticsDelta()
	delta.notes = ["Importers not checked (excluded directories): app/build/ (1 file)."]
	assert "Importers not checked (excluded directories): app/build/ (1 file)." in format_delta(delta)
