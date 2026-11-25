"""QA block B14: write journal and guard (RC-002 guard, RC-036, RC-040, RC-048, RC-050, RC-059, RC-110)."""

from __future__ import annotations

import errno
import os
import signal
from pathlib import Path

import pytest

from mcp_nav_shared import transaction
from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.transaction import (
	ApplyFailedError,
	EditJournal,
	PlanStore,
	WriteGuard,
	WriteRefusedError,
)


def _guard(root: Path) -> WriteGuard:
	return WriteGuard(root=root, allowed_suffixes=frozenset({".py"}))


def _refusal(guard: WriteGuard, path: Path) -> str | None:
	"""The refusal message for `path`, or None when the guard accepts it."""
	try:
		guard.check_path(path)
	except WriteRefusedError as exc:
		return str(exc)
	return None


def _undo_refusal(journal: EditJournal, edit_id: str) -> str | None:
	try:
		journal.undo(edit_id)
	except ToolInputError as exc:
		return str(exc)
	return None


def _plan(tmp_path: Path, names: list[str], old: str = "old\n", new: str = "new\n") -> EditPlan:
	changes = []
	for name in names:
		path = tmp_path / name
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(old, encoding="utf-8")
		changes.append(FileChange(path, old, new))
	return EditPlan("test", changes)


def test_rc_002_build_directory_refusal_names_the_directory(tmp_path):
	# given the built-in exclusion list
	guard = _guard(tmp_path)
	# when / then
	assert _refusal(guard, tmp_path / "build" / "steps.py") == (
		"build/steps.py is inside build/, which the write tools never edit"
	)


def test_rc_002_git_directory_is_refused(tmp_path):
	# given
	guard = _guard(tmp_path)
	# when / then
	assert _refusal(guard, tmp_path / ".git" / "hooks" / "x.py") == (
		".git/hooks/x.py is inside .git/, which the write tools never edit"
	)


def test_rc_040_file_inside_nested_checkout_is_refused(tmp_path):
	# given a nested git checkout below the workspace (the workspace itself has no .git)
	(tmp_path / "vendor-repo" / ".git").mkdir(parents=True)
	(tmp_path / "vendor-repo" / "vend.py").write_text("x = 1\n", encoding="utf-8")
	(tmp_path / "pkg").mkdir()
	guard = _guard(tmp_path)
	# when / then
	assert _refusal(guard, tmp_path / "vendor-repo" / "vend.py") == (
		"vendor-repo/vend.py is inside the nested git repository vendor-repo/; "
		"the write tools do not touch other repositories"
	)
	assert _refusal(guard, tmp_path / "pkg" / "ok.py") is None


def test_rc_048_outside_refusal_never_prints_absolute_paths(tmp_path):
	# given a workspace one level below the outside target
	workspace = tmp_path / "ws"
	workspace.mkdir()
	guard = _guard(workspace)
	# when / then: absolute input shows no absolute path
	assert _refusal(guard, tmp_path / "elsewhere.py") == (
		"<outside workspace>/elsewhere.py is outside the workspace; give a path inside it"
	)
	# relative input is shown as given and resolved against the workspace, not the process directory
	assert _refusal(guard, Path("../x.py")) == "../x.py is outside the workspace; give a path inside it"


def test_rc_036_write_failure_names_the_file_and_reason_only(tmp_path, monkeypatch):
	# given a two-file plan whose second write fails with an OS error that carries a temporary path
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, ["a.py", "b.py"])
	real = transaction._write_change

	def failing(change):
		if change.path.name == "b.py":
			raise OSError(errno.EACCES, "Permission denied", str(tmp_path / ".b.py.1234abcd.tmp"))
		real(change)

	monkeypatch.setattr(transaction, "_write_change", failing)
	# when / then: the message has the relative name and the reason, no temp path, no errno number
	with pytest.raises(ApplyFailedError) as raised:
		journal.apply(plan)
	assert str(raised.value) == "could not write b.py: Permission denied; nothing was changed"
	assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old\n"


def test_rc_050_evicted_edit_says_it_is_out_of_the_undo_history(tmp_path):
	# given 21 applied edits: the first is older than the 20-entry history
	journal = EditJournal(_guard(tmp_path))
	ids = [journal.apply(_plan(tmp_path, [f"f{i}.py"])).id for i in range(21)]
	# when / then
	assert _undo_refusal(journal, ids[0]).startswith(
		f"edit {ids[0]} is not in the undo history (it was never applied, already undone, or older than the last 20)"
	)


def test_rc_050_evicted_preview_says_why(tmp_path):
	# given a preview store with room for two previews
	store = PlanStore(max_entries=2)
	ids = [store.add(_plan(tmp_path, [f"p{i}.py"])) for i in range(3)]
	# when / then
	with pytest.raises(ToolInputError) as raised:
		store.get(ids[0])
	assert str(raised.value) == (
		f"unknown or expired preview id '{ids[0]}' (pending: {ids[1]}, {ids[2]}); "
		"only the last 2 previews are kept, so re-run the edit if this one was evicted"
	)


def test_rc_059_given_id_with_empty_journal_is_reported_as_unknown(tmp_path):
	# when / then: a given id is looked up even when the journal is empty
	assert _undo_refusal(EditJournal(_guard(tmp_path)), "deadbeef").startswith(
		"edit deadbeef is not in the undo history (it was never applied, already undone, or older than the last 20)"
	)


def test_rc_110_blocked_undo_names_the_newer_edit(tmp_path):
	# given edit A changes x.py, then edit B changes it again
	journal = EditJournal(_guard(tmp_path))
	(tmp_path / "x.py").write_text("1\n", encoding="utf-8")
	first = journal.apply(EditPlan("A", [FileChange(tmp_path / "x.py", "1\n", "2\n")]))
	second = journal.apply(EditPlan("B", [FileChange(tmp_path / "x.py", "2\n", "3\n")]))
	# when / then
	assert _undo_refusal(journal, first.id) == (
		f"edit {first.id} cannot be undone while edit {second.id} (applied after it) is still applied; "
		f"undo {second.id} first"
	)
	# the newer edit can be undone, after which the older one can be too
	journal.undo(second.id)
	journal.undo(first.id)
	assert (tmp_path / "x.py").read_text(encoding="utf-8") == "1\n"


def test_undo_after_external_change_keeps_the_generic_refusal(tmp_path):
	# given an applied edit whose file was then changed outside the journal
	journal = EditJournal(_guard(tmp_path))
	entry = journal.apply(EditPlan("A", [FileChange(tmp_path / "x.py", None, "1\n")]))
	(tmp_path / "x.py").write_text("external\n", encoding="utf-8")
	# when / then
	assert _undo_refusal(journal, entry.id) == f"cannot undo {entry.id}: changed since it was applied: x.py"


def test_undo_interrupted_restores_applied_state_and_keeps_entry(tmp_path, monkeypatch):
	# given a two-file edit that is undone while a SIGTERM arrives after the first file
	journal = EditJournal(_guard(tmp_path))
	entry = journal.apply(_plan(tmp_path, ["a.py", "b.py"]))
	real = transaction._write_change
	calls = []

	def noisy(change):
		real(change)
		calls.append(change.path.name)
		if len(calls) == 1:
			os.kill(os.getpid(), signal.SIGTERM)

	monkeypatch.setattr(transaction, "_write_change", noisy)
	received = []
	previous = signal.signal(signal.SIGTERM, lambda signum, _frame: received.append(signum))
	try:
		# when / then: the undo is rolled back, the edit stays in the history, the signal reaches the handler
		with pytest.raises(ApplyFailedError, match="interrupted; nothing was changed"):
			journal.undo(entry.id)
		assert received == [signal.SIGTERM]
		assert [p.read_text(encoding="utf-8") for p in sorted(tmp_path.glob("*.py"))] == ["new\n", "new\n"]
		assert [e.id for e in journal.history()] == [entry.id]
	finally:
		signal.signal(signal.SIGTERM, previous)


def test_undo_restore_failure_is_reported_not_claimed_as_nothing_changed(tmp_path, monkeypatch):
	# given a two-file edit whose undo fails on a.py while b.py (already undone) cannot be put back
	journal = EditJournal(_guard(tmp_path))
	entry = journal.apply(_plan(tmp_path, ["a.py", "b.py"]))
	real = transaction._write_change

	def fail_on_a(change):
		if change.path.name == "a.py":
			raise OSError(errno.EIO, "I/O error")
		real(change)

	def failed_restore(change):
		raise OSError(errno.EIO, "I/O error")

	monkeypatch.setattr(transaction, "_write_change", fail_on_a)
	monkeypatch.setattr(transaction, "_restore", failed_restore)
	# when / then
	with pytest.raises(ApplyFailedError) as raised:
		journal.undo(entry.id)
	assert str(raised.value) == "could not write a.py: I/O error; COULD NOT RESTORE: b.py"
