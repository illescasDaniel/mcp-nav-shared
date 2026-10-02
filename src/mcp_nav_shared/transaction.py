"""Writing an `EditPlan` to disk safely: guards, atomic apply with rollback, undo.

A plan is only ever written after `WriteGuard.check_plan` accepted its paths
and `EditJournal.apply` verified that every file still has the bytes the plan
was made from. Files are replaced atomically one by one; if any write fails,
the ones already written are put back.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from mcp_nav_shared.edits import EditPlan, FileChange, hash_bytes, relative_name
from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.exclude import EXCLUDED_DIR_NAMES


# Not in EXCLUDED_DIR_NAMES because scans may still *read* them; writes never go there.
_NEVER_WRITE_DIR_NAMES = frozenset({"typeshed", "stubs", ".idea", ".vscode"})
DEFAULT_MAX_FILES = 40


class WriteRefusedError(ToolInputError):
	"""The edit was refused before anything was written (read-only mode, path outside the workspace, ...)."""


class StaleFileError(ToolInputError):
	"""A file changed after the plan was made, or after the edit that is being undone."""


class ApplyFailedError(ToolInputError):
	"""Writing failed part-way; the files already written were restored."""


def read_only_from_env(variable: str) -> bool:
	return os.environ.get(variable, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WriteGuard:
	"""What the write tools may touch."""

	root: Path
	allowed_suffixes: frozenset[str]
	read_only: bool = False
	max_files: int = DEFAULT_MAX_FILES

	def check_writes_enabled(self, variable_hint: str = "") -> None:
		if self.read_only:
			hint = f" ({variable_hint} is set)" if variable_hint else ""
			raise WriteRefusedError(f"write tools are disabled{hint}; previews are still available with apply=false")

	def check_path(self, path: Path) -> Path:
		"""The resolved path if writing it is allowed, else `WriteRefusedError`."""
		root = self.root.resolve()
		resolved = path.resolve()
		try:
			rel = resolved.relative_to(root)
		except ValueError:
			raise WriteRefusedError(f"{path} is outside the workspace {root}") from None
		blocked = (EXCLUDED_DIR_NAMES | _NEVER_WRITE_DIR_NAMES) & set(rel.parts[:-1])
		if blocked:
			raise WriteRefusedError(
				f"{rel.as_posix()} is inside {sorted(blocked)[0]!r}, which the write tools never edit"
			)
		if self.allowed_suffixes and resolved.suffix.lower() not in self.allowed_suffixes:
			raise WriteRefusedError(f"{rel.as_posix()} is not a {'/'.join(sorted(self.allowed_suffixes))} file")
		return resolved

	def check_plan(self, plan: EditPlan, *, allow_large: bool = False) -> None:
		"""Refuse a plan touching blocked paths, or more files than `max_files` unless `allow_large`."""
		for change in plan.changes:
			self.check_path(change.path)
		if len(plan.changes) > self.max_files and not allow_large:
			raise WriteRefusedError(
				f"the edit touches {len(plan.changes)} files (limit {self.max_files}); "
				"review the preview and pass allow_large=true to proceed"
			)


@dataclass
class AppliedEdit:
	id: str
	plan: EditPlan
	# Hash of each file right after the write (None: it no longer exists).
	applied_hashes: dict[Path, str | None] = field(default_factory=dict)


def _current_hash(path: Path) -> str | None:
	try:
		return hash_bytes(path.read_bytes())
	except FileNotFoundError:
		return None


def _write_atomic(path: Path, data: bytes) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
	try:
		with temp.open("wb") as handle:
			handle.write(data)
			handle.flush()
			os.fsync(handle.fileno())
		if path.exists():
			shutil.copymode(path, temp)
		os.replace(temp, path)
	except BaseException:
		with contextlib.suppress(OSError):
			temp.unlink()
		raise


def _write_change(change: FileChange) -> None:
	data = change.new_bytes()
	if data is None:
		change.path.unlink(missing_ok=True)
	else:
		_write_atomic(change.path, data)


def _restore(change: FileChange) -> None:
	"""Put a file back the way the plan found it (best effort, used on rollback)."""
	old = change.old_bytes()
	if old is None:
		with contextlib.suppress(OSError):
			change.path.unlink(missing_ok=True)
	else:
		_write_atomic(change.path, old)


def _prune_empty_dirs(path: Path, stop: Path) -> None:
	parent = path.parent
	while parent != stop and parent.is_dir():
		try:
			parent.rmdir()
		except OSError:
			return
		parent = parent.parent


class EditJournal:
	"""Applies plans and remembers the last few so they can be undone."""

	def __init__(self, guard: WriteGuard, max_entries: int = 20) -> None:
		self.guard = guard
		self._max_entries = max_entries
		self._entries: OrderedDict[str, AppliedEdit] = OrderedDict()

	def rebind(self, guard: WriteGuard) -> None:
		"""Re-target at another workspace; edits made in the old one can't be undone from here."""
		self.guard = guard
		self._entries.clear()

	def history(self) -> list[AppliedEdit]:
		return list(reversed(self._entries.values()))

	def check_fresh(self, plan: EditPlan) -> None:
		stale = [path for path, expected in plan.base_hashes.items() if _current_hash(path) != expected]
		if stale:
			names = ", ".join(relative_name(path, self.guard.root) for path in stale[:5])
			raise StaleFileError(
				f"changed since the preview was made: {names}"
				+ (f" (+{len(stale) - 5} more)" if len(stale) > 5 else "")
				+ ". Run the refactoring again."
			)

	def apply(self, plan: EditPlan, *, allow_large: bool = False) -> AppliedEdit:
		self.guard.check_writes_enabled()
		self.guard.check_plan(plan, allow_large=allow_large)
		self.check_fresh(plan)
		written: list[FileChange] = []
		failing = plan.changes[0] if plan.changes else None
		try:
			for failing in plan.changes:
				_write_change(failing)
				written.append(failing)
		except Exception as exc:
			stuck = []
			for done in reversed(written):
				try:
					_restore(done)
				except OSError:
					stuck.append(relative_name(done.path, self.guard.root))
			name = relative_name(failing.path, self.guard.root) if failing else "?"
			message = f"writing {name} failed ({exc}); "
			message += f"COULD NOT RESTORE: {', '.join(stuck)}" if stuck else "no changes were kept"
			raise ApplyFailedError(message) from exc
		for change in plan.changes:
			if change.kind == "delete":
				_prune_empty_dirs(change.path, self.guard.root.resolve())
		entry = AppliedEdit(
			id=uuid.uuid4().hex[:8],
			plan=plan,
			applied_hashes={change.path: _current_hash(change.path) for change in plan.changes},
		)
		self._entries[entry.id] = entry
		while len(self._entries) > self._max_entries:
			self._entries.popitem(last=False)
		return entry

	def undo(self, edit_id: str | None = None) -> AppliedEdit:
		"""Revert an applied edit (the latest by default). Refuses when any of its
		files was changed after it, so later work is never overwritten."""
		if not self._entries:
			raise ToolInputError("nothing to undo: no edits were applied by this server")
		if edit_id is None:
			edit_id = next(reversed(self._entries))
		entry = self._entries.get(edit_id)
		if entry is None:
			known = ", ".join(self._entries) or "none"
			raise ToolInputError(f"unknown edit id {edit_id!r} (known: {known})")
		stale = [path for path, expected in entry.applied_hashes.items() if _current_hash(path) != expected]
		if stale:
			names = ", ".join(relative_name(path, self.guard.root) for path in stale[:5])
			raise StaleFileError(f"cannot undo {edit_id}: changed since it was applied: {names}")
		inverse = entry.plan.inverse()
		self.guard.check_writes_enabled()
		self.guard.check_plan(inverse, allow_large=True)
		written: list[FileChange] = []
		try:
			for change in inverse.changes:
				_write_change(change)
				written.append(change)
		except Exception as exc:
			for change in reversed(written):
				with contextlib.suppress(OSError):
					_restore(change)
			raise ApplyFailedError(f"undo of {edit_id} failed ({exc}); nothing was reverted") from exc
		for change in inverse.changes:
			if change.kind == "delete":
				_prune_empty_dirs(change.path, self.guard.root.resolve())
		del self._entries[edit_id]
		return entry


class PlanStore:
	"""Previewed plans waiting to be applied by id."""

	def __init__(self, max_entries: int = 20) -> None:
		self._plans: OrderedDict[str, EditPlan] = OrderedDict()
		self._max_entries = max_entries

	def add(self, plan: EditPlan) -> str:
		plan_id = uuid.uuid4().hex[:8]
		self._plans[plan_id] = plan
		while len(self._plans) > self._max_entries:
			self._plans.popitem(last=False)
		return plan_id

	def get(self, plan_id: str) -> EditPlan:
		plan = self._plans.get(plan_id)
		if plan is None:
			known = ", ".join(self._plans) or "none"
			raise ToolInputError(f"unknown or expired preview id {plan_id!r} (pending: {known})")
		return plan

	def discard(self, plan_id: str) -> None:
		self._plans.pop(plan_id, None)

	def clear(self) -> None:
		self._plans.clear()
