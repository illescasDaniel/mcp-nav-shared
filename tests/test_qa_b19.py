"""QA block B19: permission bits survive a move (a created file takes its new mode) and undo (a restored file gets
back the mode it had before the edit)."""

from __future__ import annotations

import os
import stat

from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.transaction import EditJournal, WriteGuard


def _guard(root) -> WriteGuard:
	return WriteGuard(root=root, allowed_suffixes=frozenset({".py", ".pyi"}))


def _mode(path) -> int:
	return stat.S_IMODE(path.stat().st_mode)


def test_rc_b19_created_file_takes_the_mode_it_is_given(tmp_path):
	journal = EditJournal(_guard(tmp_path))
	dest = tmp_path / "moved.py"
	plan = EditPlan("move", [FileChange(dest, None, "x = 1\n", new_mode=0o100755)])

	journal.apply(plan)

	assert _mode(dest) == 0o755


def test_rc_b19_undo_restores_a_deleted_file_with_its_original_mode(tmp_path):
	journal = EditJournal(_guard(tmp_path))
	script = tmp_path / "run.py"
	script.write_text("print(1)\n", encoding="utf-8")
	os.chmod(script, 0o755)  # noqa: S103 - the modes under test
	entry = journal.apply(EditPlan("delete", [FileChange(script, "print(1)\n", None)]))
	assert not script.exists()

	journal.undo(entry.id)

	assert script.read_text(encoding="utf-8") == "print(1)\n"
	assert _mode(script) == 0o755


def test_rc_b19_modified_file_keeps_its_mode_through_apply_and_undo(tmp_path):
	journal = EditJournal(_guard(tmp_path))
	script = tmp_path / "run.py"
	script.write_text("x = 1\n", encoding="utf-8")
	os.chmod(script, 0o750)  # noqa: S103 - the modes under test

	entry = journal.apply(EditPlan("edit", [FileChange(script, "x = 1\n", "x = 2\n")]))
	assert _mode(script) == 0o750

	journal.undo(entry.id)
	assert script.read_text(encoding="utf-8") == "x = 1\n"
	assert _mode(script) == 0o750
