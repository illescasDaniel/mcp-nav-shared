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
import signal
import stat
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mcp_nav_shared.edits import EditPlan, FileChange, hash_bytes, relative_name
from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.exclude import excluded_names, is_inside_nested_repo


# Not in the excluded names because scans may still *read* them; writes never go there.
_NEVER_WRITE_DIR_NAMES = frozenset({"typeshed", "stubs", ".idea", ".vscode"})
# Never writable, and never unlocked by `<PREFIX>_INCLUDE_DIRS`: repository metadata.
_ALWAYS_BLOCKED_DIR_NAMES = frozenset({".git"})
DEFAULT_MAX_FILES = 40


class WriteRefusedError(ToolInputError):
	"""The edit was refused before anything was written (read-only mode, path outside the workspace, ...)."""


class StaleFileError(ToolInputError):
	"""A file changed after the plan was made, or after the edit that is being undone."""


class ApplyFailedError(ToolInputError):
	"""Writing failed part-way; the files already written were restored."""


def read_only_from_env(variable: str) -> bool:
	return os.environ.get(variable, "").strip().lower() in {"1", "true", "yes", "on"}


def echo_text(value: str, limit: int = 40) -> str:
	"""A caller's input as it may be echoed in a reply: cut to `limit` characters, never the whole of a long input."""
	return value if len(value) <= limit else f"{value[: limit - 3]}..."


def refuse_home_path(file_path: str) -> None:
	"""Paths are never expanded: `~/x.py` would otherwise become a literal `~` folder in the workspace (WRX-020)."""
	if file_path.replace("\\", "/").split("/", 1)[0] == "~":
		raise WriteRefusedError(
			"paths starting with ~ are not expanded; give a path inside the workspace, such as src/x.py"
		)


@dataclass(frozen=True)
class WriteGuard:
	"""What the write tools may touch."""

	root: Path
	allowed_suffixes: frozenset[str]
	read_only: bool = False
	max_files: int = DEFAULT_MAX_FILES
	allow_readonly: bool = field(default=False, kw_only=True)
	# Env prefix (e.g. "CODENAV_MCP") whose `_EXCLUDE_DIRS` / `_INCLUDE_DIRS` variables shape the excluded
	# directory names: the same list the scans use. Empty: the built-in list only.
	exclude_prefix: str = field(default="", kw_only=True)

	def check_writes_enabled(self, variable_hint: str = "") -> None:
		if self.read_only:
			hint = f" ({variable_hint} is set)" if variable_hint else ""
			raise WriteRefusedError(f"write tools are disabled{hint}; previews are still available with apply=false")

	def check_path(self, path: Path) -> Path:
		"""The resolved path if writing it is allowed, else `WriteRefusedError`."""
		refuse_home_path(str(path))
		root = self.root.resolve()
		resolved = (path if path.is_absolute() else root / path).resolve()
		try:
			rel = resolved.relative_to(root)
		except ValueError:
			shown = self._shown_outside(path)
			raise WriteRefusedError(f"{shown} is outside the workspace; give a path inside it") from None
		self._check_directories(rel)
		nested = is_inside_nested_repo(resolved, root)
		if nested is not None:
			raise WriteRefusedError(
				f"{rel.as_posix()} is inside the nested git repository {nested.relative_to(root).as_posix()}/; "
				"the write tools do not touch other repositories"
			)
		if self.allowed_suffixes and resolved.suffix.lower() not in self.allowed_suffixes:
			raise WriteRefusedError(f"{rel.as_posix()} is not a {'/'.join(sorted(self.allowed_suffixes))} file")
		self._check_file_state(resolved, rel.as_posix())
		return resolved

	def _shown_outside(self, path: Path) -> str:
		"""The path as the caller gave it, never the absolute workspace or target."""
		if not path.is_absolute():
			return path.as_posix()
		try:
			return path.relative_to(self.root).as_posix()
		except ValueError:
			return f"<outside workspace>/{path.name}"

	def _check_directories(self, rel: Path) -> None:
		"""Refuse a file inside a directory the write tools never edit (the first such directory in path order)."""
		excluded = excluded_names(self.exclude_prefix)
		for part in rel.parts[:-1]:
			if part in _ALWAYS_BLOCKED_DIR_NAMES or part in _NEVER_WRITE_DIR_NAMES:
				raise WriteRefusedError(f"{rel.as_posix()} is inside {part}/, which the write tools never edit")
			if part in excluded:
				hint = f" (set {self.exclude_prefix}_INCLUDE_DIRS={part} to allow)" if self.exclude_prefix else ""
				raise WriteRefusedError(f"{rel.as_posix()} is inside {part}/, which the write tools never edit{hint}")

	def _check_file_state(self, resolved: Path, rel: str) -> None:
		"""Refuse read-only files (unless allowed) and new files that differ from a sibling only by letter case."""
		try:
			info = resolved.stat()
		except OSError:
			info = None
		if info is not None:
			if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
				raise WriteRefusedError(
					f"{rel} has multiple hard links, so writing it would also change its other names (some may be "
					"outside the workspace); edit it manually after checking all linked files"
				)
			if stat.S_ISREG(info.st_mode) and not self.allow_readonly:
				if not info.st_mode & 0o222 or not os.access(resolved, os.W_OK):
					raise WriteRefusedError(
						f"{rel} is read-only; make it writable or set CODENAV_MCP_ALLOW_READONLY_FILES=1"
					)
			return
		try:
			siblings = [entry.name for entry in resolved.parent.iterdir()]
		except OSError:
			return
		folded = resolved.name.casefold()
		for other in sorted(siblings):
			if other != resolved.name and other.casefold() == folded:
				rel_other = (Path(rel).parent / other).as_posix()
				raise WriteRefusedError(
					f"{rel} differs only in letter case from existing {rel_other}; pick another name"
				)

	def check_plan(self, plan: EditPlan, *, allow_large: bool = False) -> None:
		"""Refuse a plan touching blocked paths, or more files than `max_files` unless `allow_large`."""
		for change in plan.changes:
			self.check_path(change.path)
		if len(plan.changes) > self.max_files and not allow_large:
			raise WriteRefusedError(
				f"the edit touches {len(plan.changes)} files (limit {self.max_files}); "
				"preview it with apply=false, then pass allow_large=true to apply it"
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


def _write_atomic(path: Path, data: bytes, mode: int | None = None) -> None:
	"""Replace `path` with `data`. `mode` (permission bits) is set on the new file; without it an existing file
	keeps its mode, and a new file gets the default one."""
	path.parent.mkdir(parents=True, exist_ok=True)
	# Recheck immediately before writing: replacing a newly hard-linked file would
	# silently break its aliases, while writing it in place can modify outside roots.
	try:
		if path.stat().st_nlink > 1:
			raise WriteRefusedError(
				"the file has multiple hard links, so writing it would also change its other names; "
				"edit it manually after checking all linked files"
			)
	except FileNotFoundError:
		pass
	temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
	created = False
	try:
		with temp.open("xb") as handle:
			created = True
			handle.write(data)
			handle.flush()
			os.fsync(handle.fileno())
		if mode is not None:
			os.chmod(temp, stat.S_IMODE(mode))
		elif path.exists():
			shutil.copymode(path, temp)
		os.replace(temp, path)
	except BaseException:
		if created:
			with contextlib.suppress(OSError):
				temp.unlink()
		raise


def _write_change(change: FileChange) -> None:
	data = change.new_bytes()
	if data is None:
		change.path.unlink(missing_ok=True)
	else:
		_write_atomic(change.path, data, change.new_mode)


def _restore(change: FileChange) -> None:
	"""Put a file back the way the plan found it, mode included (best effort, used on rollback)."""
	old = change.old_bytes()
	if old is None:
		change.path.unlink(missing_ok=True)
	else:
		_write_atomic(change.path, old, change.old_mode)


def _roll_back(written: list[FileChange], root: Path) -> list[str]:
	"""Put back, newest first, the files a failed or interrupted write touched. A file that changed since our write
	is left alone. Returns the names that could not be restored."""
	stuck: list[str] = []
	for done in reversed(written):
		name = relative_name(done.path, root)
		try:
			current = _current_hash(done.path)
			old = done.old_bytes()
			new = done.new_bytes()
			if current == (hash_bytes(old) if old is not None else None):
				continue
			if current != (hash_bytes(new) if new is not None else None):
				stuck.append(name)
				continue
			_restore(done)
		except (OSError, WriteRefusedError):
			stuck.append(name)
	return stuck


def _failure_reason(exc: BaseException) -> str:
	"""A short reason for a failed write: never a path, a temporary name or an errno number."""
	if isinstance(exc, ToolInputError):
		return str(exc)
	if isinstance(exc, OSError) and exc.strerror:
		return exc.strerror
	return "unexpected error"


def _interrupted_message(stuck: list[str]) -> str:
	if stuck:
		return f"interrupted; COULD NOT RESTORE: {', '.join(stuck)}"
	return "interrupted; nothing was changed"


def _prune_empty_dirs(path: Path, stop: Path) -> None:
	parent = path.parent
	while parent != stop and parent.is_dir():
		try:
			parent.rmdir()
		except OSError:
			return
		parent = parent.parent


class _Interrupted(Exception):
	"""Internal: a termination signal arrived while a plan was being written."""


class _InterruptFlag:
	"""Temporary SIGTERM/SIGINT handlers (main thread only) that only record the signal."""

	_SIGNALS = (signal.SIGTERM, signal.SIGINT)

	def __init__(self) -> None:
		self.signum: int | None = None
		self._saved: dict[int, object] = {}

	def install(self) -> None:
		if threading.current_thread() is not threading.main_thread():
			return
		for signum in self._SIGNALS:
			try:
				previous = signal.getsignal(signum)
				if previous is None:
					continue  # Python cannot restore a handler installed outside Python.
				self._saved[signum] = previous
				signal.signal(signum, self._handle)
			except (ValueError, OSError):
				self._saved.pop(signum, None)

	def _handle(self, signum: int, _frame: object) -> None:
		self.signum = signum

	def release(self) -> None:
		for signum, previous in self._saved.items():
			with contextlib.suppress(ValueError, OSError, TypeError):
				signal.signal(signum, previous)  # ty: ignore[invalid-argument-type]
		self._saved.clear()

	def reraise(self) -> None:
		"""Deliver the recorded signal to the restored handler (so the server still terminates)."""
		if self.signum is not None:
			signal.raise_signal(self.signum)


def _write_changes(
	changes: list[FileChange],
	root: Path,
	*,
	check: Callable[[FileChange], None],
	verify: Callable[[], None],
	commit: Callable[[], None],
	uncommit: Callable[[], None],
) -> None:
	"""Write `changes` in order and record them with `commit`: all or nothing.

	`check` runs before each file and `verify` after the last one; then `commit` runs. A failure, or a termination
	signal recorded by `_InterruptFlag`, puts the written files back and raises `ApplyFailedError`. A signal that
	arrives after `commit` also runs `uncommit`; the recorded signal is always delivered to the restored handler.
	"""
	written: list[FileChange] = []
	failing: FileChange | None = None
	interrupt = _InterruptFlag()
	interrupt.install()
	committed = False
	rolled_back = False
	try:
		for failing in changes:
			if interrupt.signum is not None:
				raise _Interrupted
			check(failing)
			# Track before writing: a write that fails part-way is still put back (it is then a no-op on rollback).
			written.append(failing)
			_write_change(failing)
		if interrupt.signum is not None:
			raise _Interrupted
		verify()
		if interrupt.signum is not None:
			raise _Interrupted
		commit()
		committed = True
	except _Interrupted:
		stuck = _roll_back(written, root)
		rolled_back = True
		raise ApplyFailedError(_interrupted_message(stuck)) from None
	except Exception as exc:
		stuck = _roll_back(written, root)
		rolled_back = True
		name = relative_name(failing.path, root) if failing else "?"
		tail = f"COULD NOT RESTORE: {', '.join(stuck)}" if stuck else "nothing was changed"
		raise ApplyFailedError(f"could not write {name}: {_failure_reason(exc)}; {tail}") from exc
	finally:
		# Restore handlers first, then look at the recorded flag: a signal that arrived after the last check (or
		# during rollback) is handled here, before it is redelivered to the restored handler.
		interrupt.release()
		late: str | None = None
		if interrupt.signum is not None and not rolled_back:
			if committed:
				uncommit()
			late = _interrupted_message(_roll_back(written, root))
		interrupt.reraise()
		if late is not None:
			raise ApplyFailedError(late) from None


class EditJournal:
	"""Applies plans and remembers the last few so they can be undone."""

	def __init__(self, guard: WriteGuard, max_entries: int = 20) -> None:
		self.guard = guard
		self._max_entries = max_entries
		self._entries: OrderedDict[str, AppliedEdit] = OrderedDict()
		self._undone: OrderedDict[str, None] = OrderedDict()  # ids reverted in this session, for the message only

	def rebind(self, guard: WriteGuard) -> None:
		"""Re-target at another workspace; edits made in the old one can't be undone from here."""
		self.guard = guard
		self._entries.clear()
		self._undone.clear()

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

	def apply(self, plan: EditPlan, *, allow_large: bool = False, entry_id: str | None = None) -> AppliedEdit:
		"""Write `plan` and record it. `entry_id` (a previewed plan's id) becomes the applied edit's id, so the
		preview id can be undone; it is replaced by a fresh id when another edit already holds it."""
		self.guard.check_writes_enabled()
		self.guard.check_plan(plan, allow_large=allow_large)
		self.check_fresh(plan)
		applied_hashes = {
			change.path: hash_bytes(data) if (data := change.new_bytes()) is not None else None
			for change in plan.changes
		}
		new_id = entry_id if entry_id and entry_id not in self._entries else uuid.uuid4().hex[:8]
		entry = AppliedEdit(id=new_id, plan=plan, applied_hashes=applied_hashes)
		snapshot = list(self._entries.items())

		def check(change: FileChange) -> None:
			# Another process may have edited this file while earlier files were written.
			old = change.old_bytes()
			expected = plan.base_hashes.get(change.path, hash_bytes(old) if old is not None else None)
			if _current_hash(change.path) != expected:
				raise StaleFileError(f"changed while applying: {relative_name(change.path, self.guard.root)}")

		def verify() -> None:
			# Record the intended bytes, never an external edit made during apply.
			# Otherwise undo could mistake that edit for our own and overwrite it.
			changed = [path for path, expected in applied_hashes.items() if _current_hash(path) != expected]
			if changed:
				names = ", ".join(relative_name(path, self.guard.root) for path in changed)
				raise StaleFileError(f"changed while applying: {names}")

		def commit() -> None:
			# Register before the signal handlers are restored, so a completed edit has an entry.
			self._entries[entry.id] = entry
			while len(self._entries) > self._max_entries:
				self._entries.popitem(last=False)

		def uncommit() -> None:
			self._entries.clear()
			self._entries.update(snapshot)

		_write_changes(plan.changes, self.guard.root, check=check, verify=verify, commit=commit, uncommit=uncommit)
		for change in plan.changes:
			if change.kind == "delete":
				_prune_empty_dirs(change.path, self.guard.root.resolve())
		return entry

	def undo(self, edit_id: str | None = None) -> AppliedEdit:
		"""Revert an applied edit (the latest by default). Refuses when any of its
		files was changed after it, so later work is never overwritten."""
		if edit_id is None:
			if not self._entries:
				raise ToolInputError("nothing to undo: no edits were applied by this server")
			edit_id = next(reversed(self._entries))
		entry = self._entries.get(edit_id)
		if entry is None:
			shown = echo_text(edit_id)
			if edit_id in self._undone:
				raise ToolInputError(f"edit {shown} was already undone; nothing was changed")
			raise ToolInputError(
				f"edit {shown} is not in the undo history (it was never applied, already undone, "
				f"or older than the last {self._max_entries}). The history is kept in memory, per workspace, "
				"for this server session only: an edit from before a restart or a workspace switch is not in it."
			)
		stale = [path for path, expected in entry.applied_hashes.items() if _current_hash(path) != expected]
		if stale:
			blocker = self._blocking_edit(edit_id, stale)
			if blocker is not None:
				raise StaleFileError(
					f"edit {edit_id} cannot be undone while edit {blocker} (applied after it) is still applied; "
					f"undo {blocker} first"
				)
			names = ", ".join(relative_name(path, self.guard.root) for path in stale[:5])
			raise StaleFileError(f"cannot undo {edit_id}: changed since it was applied: {names}")
		inverse = entry.plan.inverse()
		self.guard.check_writes_enabled()
		self.guard.check_plan(inverse, allow_large=True)
		snapshot = list(self._entries.items())
		undone_snapshot = self._undone.copy()

		def commit() -> None:
			self._entries.pop(edit_id, None)
			self._undone[edit_id] = None
			while len(self._undone) > 100:
				self._undone.popitem(last=False)

		def uncommit() -> None:
			self._entries.clear()
			self._entries.update(snapshot)
			self._undone = undone_snapshot

		_write_changes(
			inverse.changes,
			self.guard.root,
			check=lambda _change: None,
			verify=lambda: None,
			commit=commit,
			uncommit=uncommit,
		)
		for change in inverse.changes:
			if change.kind == "delete":
				_prune_empty_dirs(change.path, self.guard.root.resolve())
		return entry

	def _blocking_edit(self, edit_id: str, stale: list[Path]) -> str | None:
		"""The newer journal edit that must be undone first. It must be the latest later edit on every stale file
		and still hold that file's bytes; otherwise the change came from outside the journal (None)."""
		ids = list(self._entries)
		latest: dict[Path, str] = {}
		for later_id in ids[ids.index(edit_id) + 1 :]:
			for path in self._entries[later_id].applied_hashes:
				if path in stale:
					latest[path] = later_id
		if set(latest) != set(stale):
			return None
		if any(
			_current_hash(path) != self._entries[later_id].applied_hashes[path] for path, later_id in latest.items()
		):
			return None
		return max(latest.values(), key=ids.index)


class PlanStore:
	"""Previewed plans waiting to be applied by id."""

	def __init__(self, max_entries: int = 20) -> None:
		self._plans: OrderedDict[str, EditPlan] = OrderedDict()
		self._applied: OrderedDict[str, None] = OrderedDict()  # ids whose preview was applied (for the message)
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
			shown = echo_text(plan_id)
			if plan_id in self._applied:
				raise ToolInputError(
					f"preview id {shown} was already applied; nothing was written again. "
					f'To revert that edit use undo_edit(id="{shown}") if it is still in the undo history.'
				)
			known = ", ".join(self._plans) or "none"
			raise ToolInputError(
				f"unknown or expired preview id {shown!r} (pending: {known}); only the last {self._max_entries} previews are kept, so re-run the edit if this one was evicted"
			)
		return plan

	def discard(self, plan_id: str) -> None:
		if self._plans.pop(plan_id, None) is not None:
			self._applied[plan_id] = None
			while len(self._applied) > 100:
				self._applied.popitem(last=False)

	def clear(self) -> None:
		self._plans.clear()
		self._applied.clear()
