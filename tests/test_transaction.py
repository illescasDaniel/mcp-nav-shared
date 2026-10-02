"""Fast unit tests for `mcp_nav_shared.transaction`."""

from __future__ import annotations

import os

import pytest

from mcp_nav_shared import transaction
from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.transaction import (
	ApplyFailedError,
	EditJournal,
	PlanStore,
	StaleFileError,
	WriteGuard,
	WriteRefusedError,
	read_only_from_env,
)


def _guard(root, **kwargs) -> WriteGuard:
	return WriteGuard(root=root, allowed_suffixes=frozenset({".py", ".pyi"}), **kwargs)


def _plan(tmp_path, files: dict[str, tuple[str | None, str | None]]) -> EditPlan:
	changes = []
	for name, (old, new) in files.items():
		path = tmp_path / name
		if old is not None:
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(old, encoding="utf-8")
		changes.append(FileChange(path, old, new))
	return EditPlan("test", changes)


def test_given_plan_when_apply_then_files_written_and_journaled(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"a.py": ("x = 1\n", "x = 2\n"), "b.py": (None, "y = 1\n")})
	# when
	entry = journal.apply(plan)
	# then
	assert (tmp_path / "a.py").read_text() == "x = 2\n"
	assert (tmp_path / "b.py").read_text() == "y = 1\n"
	assert journal.history()[0].id == entry.id


def test_given_file_changed_after_preview_when_apply_then_stale_and_nothing_written(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"a.py": ("x = 1\n", "x = 2\n"), "b.py": ("y = 1\n", "y = 2\n")})
	(tmp_path / "b.py").write_text("y = 99\n", encoding="utf-8")
	# when / then
	with pytest.raises(StaleFileError, match=r"b\.py"):
		journal.apply(plan)
	assert (tmp_path / "a.py").read_text() == "x = 1\n"


def test_given_created_file_exists_now_when_apply_then_stale(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = EditPlan("t", [FileChange(tmp_path / "n.py", None, "x\n")])
	(tmp_path / "n.py").write_text("someone else's\n", encoding="utf-8")
	# when / then
	with pytest.raises(StaleFileError):
		journal.apply(plan)
	assert (tmp_path / "n.py").read_text() == "someone else's\n"


def test_given_write_failure_midway_when_apply_then_earlier_files_restored(tmp_path, monkeypatch):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"a.py": ("a = 1\n", "a = 2\n"), "b.py": ("b = 1\n", "b = 2\n")})
	real = transaction._write_atomic

	def flaky(path, data):
		if path.name == "b.py" and data == b"b = 2\n":
			raise OSError("disk full")
		real(path, data)

	monkeypatch.setattr(transaction, "_write_atomic", flaky)
	# when
	with pytest.raises(ApplyFailedError, match="disk full"):
		journal.apply(plan)
	# then
	assert (tmp_path / "a.py").read_text() == "a = 1\n"
	assert (tmp_path / "b.py").read_text() == "b = 1\n"
	assert journal.history() == []


def test_given_applied_edit_when_undo_then_files_restored_and_entry_removed(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"a.py": ("x = 1\n", "x = 2\n"), "n.py": (None, "new\n")})
	entry = journal.apply(plan)
	# when
	journal.undo()
	# then
	assert (tmp_path / "a.py").read_text() == "x = 1\n"
	assert not (tmp_path / "n.py").exists()
	assert journal.history() == []
	assert entry.id not in [e.id for e in journal.history()]


def test_given_file_edited_after_apply_when_undo_then_refused(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	journal.apply(_plan(tmp_path, {"a.py": ("x = 1\n", "x = 2\n")}))
	(tmp_path / "a.py").write_text("x = 3\n", encoding="utf-8")
	# when / then
	with pytest.raises(StaleFileError):
		journal.undo()
	assert (tmp_path / "a.py").read_text() == "x = 3\n"


def test_given_deleted_file_when_undo_then_recreated(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	journal.apply(_plan(tmp_path, {"pkg/old.py": ("keep me\n", None)}))
	assert not (tmp_path / "pkg" / "old.py").exists()
	# when
	journal.undo()
	# then
	assert (tmp_path / "pkg" / "old.py").read_text() == "keep me\n"


def test_given_nothing_applied_when_undo_then_input_error(tmp_path):
	with pytest.raises(ToolInputError, match="nothing to undo"):
		EditJournal(_guard(tmp_path)).undo()


def test_given_unknown_id_when_undo_then_input_error(tmp_path):
	journal = EditJournal(_guard(tmp_path))
	journal.apply(_plan(tmp_path, {"a.py": ("x\n", "y\n")}))
	with pytest.raises(ToolInputError, match="unknown edit id"):
		journal.undo("nope")


def test_given_more_entries_than_limit_when_apply_then_oldest_forgotten(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path), max_entries=2)
	ids = []
	for index in range(3):
		ids.append(journal.apply(_plan(tmp_path, {f"f{index}.py": ("a\n", "b\n")})).id)
	# then
	assert [e.id for e in journal.history()] == [ids[2], ids[1]]


def test_given_read_only_when_apply_then_refused(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path, read_only=True))
	plan = _plan(tmp_path, {"a.py": ("x\n", "y\n")})
	# when / then
	with pytest.raises(WriteRefusedError, match="disabled"):
		journal.apply(plan)
	assert (tmp_path / "a.py").read_text() == "x\n"


def test_given_paths_outside_or_excluded_when_check_then_refused(tmp_path):
	# given
	guard = _guard(tmp_path)
	# when / then
	with pytest.raises(WriteRefusedError, match="outside the workspace"):
		guard.check_path(tmp_path.parent / "elsewhere.py")
	with pytest.raises(WriteRefusedError, match="never edit"):
		guard.check_path(tmp_path / ".venv" / "lib" / "x.py")
	with pytest.raises(WriteRefusedError, match="not a"):
		guard.check_path(tmp_path / "README.md")
	assert guard.check_path(tmp_path / "pkg" / "ok.py") == (tmp_path / "pkg" / "ok.py").resolve()


def test_given_symlink_escaping_workspace_when_check_then_refused(tmp_path):
	# given
	outside = tmp_path / "outside"
	outside.mkdir()
	(outside / "secret.py").write_text("x = 1\n", encoding="utf-8")
	root = tmp_path / "ws"
	root.mkdir()
	os.symlink(outside / "secret.py", root / "link.py")
	# when / then
	with pytest.raises(WriteRefusedError, match="outside the workspace"):
		_guard(root).check_path(root / "link.py")


def test_given_too_many_files_when_check_plan_then_refused_unless_allowed(tmp_path):
	# given
	guard = _guard(tmp_path, max_files=2)
	plan = EditPlan("t", [FileChange(tmp_path / f"f{i}.py", "a\n", "b\n") for i in range(3)])
	# when / then
	with pytest.raises(WriteRefusedError, match="allow_large"):
		guard.check_plan(plan)
	guard.check_plan(plan, allow_large=True)


def test_given_executable_file_when_apply_then_mode_preserved(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"run.py": ("#!/usr/bin/env python\n", "#!/usr/bin/env python3\n")})
	os.chmod(tmp_path / "run.py", 0o755)  # noqa: S103
	# when
	journal.apply(plan)
	# then
	assert os.stat(tmp_path / "run.py").st_mode & 0o777 == 0o755
	assert [p.name for p in tmp_path.iterdir()] == ["run.py"]  # no temp files left behind


def test_given_deleted_last_file_when_apply_then_empty_package_dir_removed(tmp_path):
	# given
	journal = EditJournal(_guard(tmp_path))
	plan = _plan(tmp_path, {"pkg/only.py": ("x\n", None)})
	# when
	journal.apply(plan)
	# then
	assert not (tmp_path / "pkg").exists()


def test_given_plan_store_when_add_get_then_roundtrip_and_eviction(tmp_path):
	# given
	store = PlanStore(max_entries=2)
	plans = [EditPlan(str(i), []) for i in range(3)]
	ids = [store.add(plan) for plan in plans]
	# then
	with pytest.raises(ToolInputError, match="unknown or expired"):
		store.get(ids[0])
	assert store.get(ids[2]) is plans[2]
	store.discard(ids[2])
	with pytest.raises(ToolInputError):
		store.get(ids[2])


def test_given_env_values_when_read_only_from_env_then_truthy_parsed(monkeypatch):
	for value, expected in [("1", True), ("true", True), ("YES", True), ("0", False), ("", False)]:
		monkeypatch.setenv("X_RO", value)
		assert read_only_from_env("X_RO") is expected
	monkeypatch.delenv("X_RO")
	assert read_only_from_env("X_RO") is False
