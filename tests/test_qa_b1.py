"""QA block B1: write apply safety (RC-011, RC-051, RC-151, RC-153)."""

from __future__ import annotations

import errno
import os
import signal

import pytest

from mcp_nav_shared import transaction
from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.transaction import (
	ApplyFailedError,
	EditJournal,
	WriteGuard,
	WriteRefusedError,
)


def _guard(root, **kwargs) -> WriteGuard:
	return WriteGuard(root=root, allowed_suffixes=frozenset({".py"}), **kwargs)


def _plan(tmp_path, names: list[str]) -> EditPlan:
	changes = []
	for name in names:
		path = tmp_path / name
		path.write_text("old\n", encoding="utf-8")
		changes.append(FileChange(path, "old\n", "new\n"))
	return EditPlan("test", changes)


def test_rc_011_sigterm_mid_apply_rolls_back(tmp_path, monkeypatch):
	# given a 3-file plan and a SIGTERM (handled by a test handler) after the first file
	plan = _plan(tmp_path, ["a.py", "b.py", "c.py"])
	real = transaction._write_change
	calls = []

	def noisy(change):
		real(change)
		calls.append(change.path.name)
		if len(calls) == 1:
			os.kill(os.getpid(), signal.SIGTERM)

	monkeypatch.setattr(transaction, "_write_change", noisy)
	received = []
	previous = signal.signal(signal.SIGTERM, lambda signum, _f: received.append(signum))
	try:
		# when
		with pytest.raises(ApplyFailedError, match="interrupted; nothing was changed"):
			EditJournal(_guard(tmp_path)).apply(plan)
		# then: restored, no temp files, previous handler got the re-raised signal and is back in place
		assert [p.read_text() for p in sorted(tmp_path.glob("*.py"))] == ["old\n"] * 3
		assert not list(tmp_path.glob(".*.tmp"))
		assert received == [signal.SIGTERM]
		assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
	finally:
		signal.signal(signal.SIGTERM, previous)


def test_rc_011_matching_user_temp_is_preserved(tmp_path):
	user_file = tmp_path / ".mod.py.deadbeef.tmp"
	user_file.write_text("valuable")
	EditJournal(_guard(tmp_path)).apply(_plan(tmp_path, ["a.py"]))
	assert user_file.read_text() == "valuable"


def test_rc_011_temp_collision_does_not_truncate_or_remove_user_file(tmp_path, monkeypatch):
	plan = _plan(tmp_path, ["a.py"])
	user_file = tmp_path / ".a.py.deadbeef.tmp"
	user_file.write_text("valuable")
	monkeypatch.setattr(transaction.uuid, "uuid4", lambda: type("FakeUUID", (), {"hex": "deadbeef"})())
	with pytest.raises(ApplyFailedError):
		EditJournal(_guard(tmp_path)).apply(plan)
	assert user_file.read_text() == "valuable"
	assert (tmp_path / "a.py").read_text() == "old\n"


def test_rc_011_failed_restore_is_reported(tmp_path, monkeypatch):
	plan = _plan(tmp_path, ["a.py", "b.py"])
	real = transaction._write_change

	def interrupted(change):
		real(change)
		os.kill(os.getpid(), signal.SIGTERM)

	def failed_restore(change):
		raise OSError("restore failed")

	monkeypatch.setattr(transaction, "_write_change", interrupted)
	monkeypatch.setattr(transaction, "_restore", failed_restore)
	previous = signal.signal(signal.SIGTERM, lambda *_: None)
	try:
		with pytest.raises(ApplyFailedError, match="COULD NOT RESTORE: a.py"):
			EditJournal(_guard(tmp_path)).apply(plan)
	finally:
		signal.signal(signal.SIGTERM, previous)


def test_rc_011_signal_during_error_rollback_reaches_prior_handler(tmp_path, monkeypatch):
	plan = _plan(tmp_path, ["a.py", "b.py"])
	real_write = transaction._write_change
	real_restore = transaction._restore

	def fail_second(change):
		if change.path.name == "b.py":
			raise OSError(errno.EIO, "write failed")
		real_write(change)

	def signal_during_restore(change):
		real_restore(change)
		os.kill(os.getpid(), signal.SIGTERM)

	monkeypatch.setattr(transaction, "_write_change", fail_second)
	monkeypatch.setattr(transaction, "_restore", signal_during_restore)
	received = []
	previous = signal.signal(signal.SIGTERM, lambda signum, _: received.append(signum))
	try:
		with pytest.raises(ApplyFailedError, match="could not write b.py: write failed; nothing was changed"):
			EditJournal(_guard(tmp_path)).apply(plan)
		assert received == [signal.SIGTERM]
		assert (tmp_path / "a.py").read_text() == "old\n"
	finally:
		signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize("interrupted", [False, True])
def test_rc_011_hard_link_created_during_apply_reports_restore_failure(tmp_path, monkeypatch, interrupted):
	plan = _plan(tmp_path, ["a.py", "b.py", "c.py"])
	real_write = transaction._write_change

	def write_then_link(change):
		if change.path.name == "c.py":
			raise OSError("write failed")
		real_write(change)
		if change.path.name == "b.py":
			os.link(change.path, tmp_path / "alias")
			if interrupted:
				os.kill(os.getpid(), signal.SIGTERM)

	monkeypatch.setattr(transaction, "_write_change", write_then_link)
	previous = signal.signal(signal.SIGTERM, lambda *_: None)
	try:
		with pytest.raises(ApplyFailedError, match=r"COULD NOT RESTORE: b\.py"):
			EditJournal(_guard(tmp_path)).apply(plan)
		assert (tmp_path / "a.py").read_text() == "old\n"
		assert (tmp_path / "b.py").read_text() == "new\n"
		assert (tmp_path / "c.py").read_text() == "old\n"
	finally:
		signal.signal(signal.SIGTERM, previous)


def test_rc_051_readonly_file_refused_then_allowed_keeps_mode(tmp_path):
	# given
	plan = _plan(tmp_path, ["ro.py"])
	target = tmp_path / "ro.py"
	target.chmod(0o444)
	# when / then refused with a relative message
	with pytest.raises(
		WriteRefusedError, match=r"ro\.py is read-only; make it writable or set CODENAV_MCP_ALLOW_READONLY_FILES=1"
	):
		EditJournal(_guard(tmp_path)).apply(plan)
	assert target.read_text() == "old\n"
	# with the flag it is written and the mode is kept
	EditJournal(_guard(tmp_path, allow_readonly=True)).apply(plan)
	assert target.read_text() == "new\n"
	assert target.stat().st_mode & 0o777 == 0o444


def test_rc_151_hard_link_refused_even_when_alias_outside_workspace(tmp_path):
	root = tmp_path / "workspace"
	root.mkdir()
	plan = _plan(root, ["a.py"])
	alias = tmp_path / "outside.py"
	os.link(root / "a.py", alias)
	with pytest.raises(WriteRefusedError, match="multiple hard links"):
		EditJournal(_guard(root)).apply(plan)
	assert alias.read_text() == "old\n"
	assert (root / "a.py").read_text() == "old\n"


def test_rc_153_case_collision_refused(tmp_path):
	# skip on case-insensitive filesystems
	probe = tmp_path / "Probe.txt"
	probe.write_text("x")
	if (tmp_path / "probe.txt").exists():
		pytest.skip("case-insensitive filesystem")
	(tmp_path / "case").mkdir()
	(tmp_path / "case" / "cart.py").write_text("x = 1\n")
	guard = _guard(tmp_path)
	with pytest.raises(
		WriteRefusedError,
		match=r"case/Cart\.py differs only in letter case from existing case/cart\.py; pick another name",
	):
		guard.check_path(tmp_path / "case" / "Cart.py")
	# the exact existing name and unrelated new names stay allowed
	guard.check_path(tmp_path / "case" / "cart.py")
	guard.check_path(tmp_path / "case" / "other.py")


def test_rc_011_late_signal_rolls_back_before_default_termination(tmp_path):
	import subprocess
	import sys

	target = tmp_path / "late.py"
	target.write_text("old\n")
	code = """
import os, signal, sys
from pathlib import Path
from mcp_nav_shared.edits import EditPlan, FileChange
from mcp_nav_shared.transaction import EditJournal, WriteGuard, _InterruptFlag
path = Path(sys.argv[1])
original = _InterruptFlag.release
def release(flag):
    os.kill(os.getpid(), signal.SIGTERM)
    original(flag)
_InterruptFlag.release = release
EditJournal(WriteGuard(path.parent, frozenset({".py"}))).apply(
    EditPlan("test", [FileChange(path, "old\\n", "new\\n")]))
"""
	result = subprocess.run([sys.executable, "-c", code, str(target)], check=False)
	assert result.returncode == -signal.SIGTERM
	assert target.read_text() == "old\n"


def test_rc_011_nested_apply_redelivers_signal_and_restores_both(tmp_path, monkeypatch):
	outer = _plan(tmp_path, ["outer.py"])
	inner = _plan(tmp_path, ["inner.py"])
	journal = EditJournal(_guard(tmp_path))
	real = transaction._write_change

	def nested(change):
		real(change)
		if change.path.name == "outer.py":
			journal.apply(inner)
		else:
			os.kill(os.getpid(), signal.SIGTERM)

	monkeypatch.setattr(transaction, "_write_change", nested)
	received = []
	previous = signal.signal(signal.SIGTERM, lambda sig, _: received.append(sig))
	try:
		with pytest.raises(ApplyFailedError):
			journal.apply(outer)
		assert received == [signal.SIGTERM]
		assert (tmp_path / "inner.py").read_text() == "old\n"
		assert (tmp_path / "outer.py").read_text() == "old\n"
	finally:
		signal.signal(signal.SIGTERM, previous)


def test_external_edit_during_apply_is_not_adopted_by_undo(tmp_path, monkeypatch):
	plan = _plan(tmp_path, ["a.py", "b.py"])
	journal = EditJournal(_guard(tmp_path))
	real = transaction._write_change

	def concurrent_write(change):
		real(change)
		if change.path.name == "b.py":
			(tmp_path / "a.py").write_text("external\n", encoding="utf-8")

	monkeypatch.setattr(transaction, "_write_change", concurrent_write)
	with pytest.raises(ApplyFailedError, match="COULD NOT RESTORE: a.py"):
		journal.apply(plan)
	assert (tmp_path / "a.py").read_text(encoding="utf-8") == "external\n"
	assert (tmp_path / "b.py").read_text(encoding="utf-8") == "old\n"
	assert journal.history() == []


def test_external_edit_before_later_write_is_preserved(tmp_path, monkeypatch):
	plan = _plan(tmp_path, ["a.py", "b.py"])
	real = transaction._write_change

	def concurrent_write(change):
		real(change)
		if change.path.name == "a.py":
			(tmp_path / "b.py").write_text("external\n", encoding="utf-8")

	monkeypatch.setattr(transaction, "_write_change", concurrent_write)
	journal = EditJournal(_guard(tmp_path))
	with pytest.raises(ApplyFailedError, match="changed while applying: b.py"):
		journal.apply(plan)
	assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old\n"
	assert (tmp_path / "b.py").read_text(encoding="utf-8") == "external\n"
	assert journal.history() == []


def test_unknown_native_signal_handler_is_not_replaced(monkeypatch) -> None:
	from mcp_nav_shared.transaction import _InterruptFlag

	installed = []
	monkeypatch.setattr(signal, "getsignal", lambda _signum: None)
	monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.append((signum, handler)))
	flag = _InterruptFlag()
	flag.install()
	flag.release()
	assert not installed
