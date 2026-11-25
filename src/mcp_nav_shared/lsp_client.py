"""Minimal async LSP client: generic JSON-RPC/LSP wire protocol plumbing.

Not a general-purpose LSP library: framing and the handful of requests used
here cover only what codenav/webnav's MCP tools need. Language-server-
specific bits (how to launch the server, its languageId, any non-default
initialize capabilities) are the caller's responsibility — pass a `command`
and `language_id` in.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import hashlib
import json
import logging
import os
import re
import signal
import stat as stat_module
import tempfile
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar
from urllib.parse import unquote, urlparse

from mcp_nav_shared.exclude import EXCLUDED_DIR_NAMES


logger = logging.getLogger(__name__)

# Max seconds to wait for a push-only server's publishDiagnostics after a sync.
PUSH_DIAGNOSTICS_TIMEOUT = 5.0

# Server->client requests that only need an acknowledgement.
_NULL_REPLY_METHODS = {
	"client/registerCapability",
	"client/unregisterCapability",
	"window/workDoneProgress/create",
	"workspace/diagnostic/refresh",
	"workspace/semanticTokens/refresh",
	"workspace/inlayHint/refresh",
	"workspace/codeLens/refresh",
}


# Last stderr lines kept to explain why a language server exited.
_STDERR_TAIL_LINES = 10
# Max seconds to wait, once stdout closes, for a dying server's last stderr lines.
_STDERR_FLUSH_TIMEOUT = 0.5


# LSP `ContentModified` (-32801) and `ServerCancelled` (-32802): both mean "ask again".
_RETRYABLE_CODES = frozenset({-32801, -32802})
_CONTENT_MODIFIED_BACKOFF = (0.1, 0.25, 0.5)


class NotStartedError(RuntimeError):
	pass


class LspRequestError(RuntimeError):
	"""JSON-RPC error response from the language server."""

	def __init__(self, method: str, code: Any, message: str) -> None:
		self.method = method
		self.code = code
		super().__init__(message)


class LanguageServerExitedError(RuntimeError):
	"""The language server process ended while a request was pending (or could not be started).

	`str()` is a short message that is safe to show to an agent; `detail` holds the technical
	part (the command line, the last stderr lines) for logs and debugging only.
	"""

	def __init__(self, message: str, detail: str = "") -> None:
		super().__init__(message)
		self.detail = detail


class InvalidPositionError(ValueError):
	"""A tool was asked about a line/column that doesn't exist in the file."""


_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")


def _read_document_text(path: Path) -> str:
	"""A file's text as an editor would hold it: a UTF-8 byte order mark is not part of the
	document, and counting it would shift every column on the first line by one."""
	text = path.read_text(encoding="utf-8")
	return text[1:] if text.startswith("\ufeff") else text


def _uri_to_path(uri: str) -> str:
	path = urlparse(uri).path
	return unquote(path.lstrip("/") if os.name == "nt" else path)


@dataclass
class OpenFile:
	uri: str
	version: int
	mtime_ns: int
	size: int
	# Change time: `touch -r` can restore an mtime after a same-size rewrite, but not the ctime.
	ctime_ns: int = 0


# `OpenFile.mtime_ns` of a document whose text came from `overlay()` rather than the disk: never
# equals a real mtime, so the next `ensure_open` after the overlay ends re-reads the file.
_OVERLAY_MTIME = -2

# LSP `FileChangeType`.
_FILE_CREATED, _FILE_CHANGED, _FILE_DELETED = 1, 2, 3


def _directory_events(directories: Iterable[Path], kind: int) -> list[dict[str, Any]]:
	"""`didChangeWatchedFiles` entries for directories: outermost first when created, innermost first when deleted.

	A file created in a directory the server has never seen is not enough for it: ty keeps resolving a new
	namespace package (a directory with no `__init__.py`) as missing until the directory itself is reported."""
	ordered = sorted(set(directories), key=lambda d: len(d.parts), reverse=kind == _FILE_DELETED)
	return [{"uri": directory.as_uri(), "type": kind} for directory in ordered]


_P = ParamSpec("_P")
_R = TypeVar("_R")

# Leases are inherited by child tasks, so each records its owning task as well as its lifetime.
# Only the owner may re-enter without acquiring a separate reader lease.
_OVERLAY_HOLDERS: contextvars.ContextVar[dict[int, list[Any]]] = contextvars.ContextVar("_OVERLAY_HOLDERS", default={})  # noqa: B039
_READING: contextvars.ContextVar[dict[int, list[Any]]] = contextvars.ContextVar("_READING", default={})  # noqa: B039

# Max seconds to wait for the server process (and its group) to go away after being told to.
_KILL_WAIT = 3.0
# Max seconds `start()` waits for the `initialize` answer.
_START_TIMEOUT = 20.0


def _reader(
	fn: Callable[Concatenate[LspClient, _P], Awaitable[_R]],
) -> Callable[Concatenate[LspClient, _P], Awaitable[_R]]:
	"""Run an LSP call that depends on document text as a reader: an `overlay()` waits for it to
	finish before showing the server unsaved text, and new readers wait for the overlay to end."""

	@functools.wraps(fn)
	async def wrapper(self: LspClient, *args: _P.args, **kwargs: _P.kwargs) -> _R:
		async with self._reading():
			return await fn(self, *args, **kwargs)

	return wrapper


def _pid_alive(pid: int) -> bool:
	if pid == os.getpid():
		return True
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except OSError:
		return True  # exists, but not ours to signal
	return True


def _sha1(data: bytes) -> str:
	return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def _set_event() -> asyncio.Event:
	event = asyncio.Event()
	event.set()
	return event


@dataclass
class LspClient:
	workspace_root: Path
	command: list[str]
	language_id: str
	# File suffixes whose on-disk changes `refresh()` reports to the server via
	# `workspace/didChangeWatchedFiles` (empty: only already-open files are re-synced).
	watch_suffixes: frozenset[str] = frozenset()
	# Paths `refresh()` must not report (e.g. generated output); directory names in
	# `EXCLUDED_DIR_NAMES` are always skipped.
	watch_ignore: Callable[[Path], bool] | None = None
	# Also `didOpen` watched files that appeared/changed, for servers (tsserver) that
	# only treat opened documents as part of the project deterministically.
	open_watched_changes: bool = False
	# File names (matched anywhere under the workspace) whose change alters how the
	# server resolves the project (pyproject.toml, tsconfig.json, ...): servers read
	# them once at startup, so `refresh()` restarts the server when one changes.
	config_names: frozenset[str] = frozenset()
	# Runs after every restart of the server, e.g. to eagerly open project files.
	on_restart: Callable[[LspClient], Awaitable[None]] | None = None
	# Told (in words) about anything the agent should know, e.g. a config-triggered restart.
	on_notice: Callable[[str], None] | None = None
	# Per-suffix override of `language_id` for servers that handle several
	# languages (e.g. typescript-go / tsc LSP: `.ts` -> "typescript").
	language_ids: dict[str, str] = field(default_factory=dict)
	_proc: asyncio.subprocess.Process | None = field(default=None, init=False)
	_next_id: int = field(default=0, init=False)
	_pending: dict[int, asyncio.Future] = field(default_factory=dict, init=False)
	_diagnostics: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False)
	_open_files: dict[str, OpenFile] = field(default_factory=dict, init=False)
	# uri -> event set by the first publishDiagnostics after the document was
	# last synced; lets push-only servers (no pull diagnostics) be awaited
	# instead of answered from a stale/empty cache.
	_diag_events: dict[str, asyncio.Event] = field(default_factory=dict, init=False)
	# uri -> (document version, documentSymbol result). documentSymbol depends
	# only on the one file's text, and `ensure_open` bumps the version exactly
	# when that text changes (stat mtime/size), so a version match means fresh.
	_symbol_cache: dict[str, tuple[int, list[dict[str, Any]]]] = field(default_factory=dict, init=False)
	# path -> (mtime_ns, size, ctime_ns) of every watched file at the last `refresh()`.
	_watch_snapshot: dict[Path, tuple[int, int, int]] | None = field(default=None, init=False)
	# Same, for `config_names` files, plus their content hashes: a `touch` or a
	# `git checkout` that rewrites identical text must not restart the server.
	_config_snapshot: dict[Path, tuple[int, int, int]] | None = field(default=None, init=False)
	_config_hashes: dict[Path, str] = field(default_factory=dict, init=False)
	_refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
	# uri -> text shown to the server instead of the disk's, while `overlay()` is active.
	_overlays: dict[str, str] = field(default_factory=dict, init=False)
	# Held for the whole of an `overlay()`: readers (every LSP call, `refresh()`) wait for it, and
	# the overlay waits for the readers that started before it (`_readers` / `_readers_idle`).
	_overlay_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
	_readers: int = field(default=0, init=False)
	_readers_idle: asyncio.Event = field(default_factory=lambda: _set_event(), init=False)
	# How many times `restart()` replaced the server (a config change, or a caller's recovery).
	restarts: int = field(default=0, init=False)
	# Seconds `start()` waits for the server's `initialize` answer.
	start_timeout: float = _START_TIMEOUT
	_manifest_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
	_reader_task: asyncio.Task | None = field(default=None, init=False)
	_stderr_task: asyncio.Task | None = field(default=None, init=False)
	_stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=_STDERR_TAIL_LINES), init=False)
	_started: bool = field(default=False, init=False)

	@property
	def _running_proc(self) -> asyncio.subprocess.Process:
		if self._proc is None:
			raise NotStartedError("LspClient.start() must be awaited before use")
		return self._proc

	@property
	def is_alive(self) -> bool:
		return self._proc is not None and self._proc.returncode is None

	async def start(self) -> None:
		if self._started:
			return
		self._stderr_tail.clear()
		self._sweep_materialized()
		try:
			self._proc = await asyncio.create_subprocess_exec(
				*self.command,
				cwd=str(self.workspace_root),
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				# Its own process group, so a hung `uvx` wrapper and the real server under it die together.
				start_new_session=os.name != "nt",
			)
		except FileNotFoundError as exc:
			if exc.filename is not None and Path(str(exc.filename)) == self.workspace_root:
				raise LanguageServerExitedError(
					"could not start the language server (the workspace folder does not exist)", detail=str(exc)
				) from exc
			argv0 = Path(self.command[0]).name if self.command else "?"
			raise LanguageServerExitedError(
				f"could not start the language server (command not found: {argv0}); install ty "
				"(for example `pip install ty`) or make sure uv/uvx is on PATH",
				detail=str(exc),
			) from exc
		self._reader_task = asyncio.create_task(self._read_loop())
		self._stderr_task = asyncio.create_task(self._drain_stderr())
		try:
			await self._request(
				"initialize",
				{
					"processId": None,
					"rootUri": self.workspace_root.as_uri(),
					"capabilities": {
						"textDocument": {
							"synchronization": {"didSave": True},
							"publishDiagnostics": {},
							"documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
							"callHierarchy": {},
							"typeHierarchy": {},
						},
						"workspace": {"workspaceFolders": True, "didChangeWatchedFiles": {"dynamicRegistration": True}},
					},
					"workspaceFolders": [{"uri": self.workspace_root.as_uri(), "name": self.workspace_root.name}],
				},
				timeout=self.start_timeout,
			)
		except BaseException:
			# A server that never answered (or died) must not stay behind, nor its children.
			await self._abort()
			raise
		self._notify("initialized", {})
		self._started = True

	def _signal_group(self, sig: int) -> None:
		"""Signal the server's whole process group (or just the process where groups don't exist)."""
		proc = self._proc
		if proc is None:
			return
		with contextlib.suppress(ProcessLookupError, PermissionError):
			if os.name == "nt":
				proc.terminate() if sig == signal.SIGTERM else proc.kill()
			else:
				os.killpg(proc.pid, sig)

	async def _end_process(self) -> None:
		"""Terminate the server and everything in its process group; escalate to SIGKILL, bounded."""
		proc = self._proc
		if proc is None:
			return
		self._signal_group(signal.SIGTERM)
		with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
			await asyncio.wait_for(proc.wait(), timeout=_KILL_WAIT)
		# Whatever is left (a wrapper's child that ignored SIGTERM) goes too.
		self._signal_group(signal.SIGKILL if os.name != "nt" else signal.SIGTERM)
		with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
			await asyncio.wait_for(proc.wait(), timeout=_KILL_WAIT)

	async def _cancel_tasks(self) -> None:
		tasks = [task for task in (self._reader_task, self._stderr_task) if task is not None]
		for task in tasks:
			task.cancel()
		# Let the reader's cleanup run now, not after a restart has created new pending requests.
		await asyncio.gather(*tasks, return_exceptions=True)

	async def _abort(self) -> None:
		"""Tear down a server that failed to start."""
		await self._cancel_tasks()
		await self._end_process()
		self._started = False
		self._proc = None

	async def stop(self) -> None:
		if self._proc is None:
			return
		try:
			await self._request("shutdown", {}, timeout=5)
			self._notify("exit", {})
		except Exception:
			logger.debug("language server didn't respond to shutdown in time; terminating it directly", exc_info=True)
		await self._cancel_tasks()
		await self._end_process()
		self._started = False
		self._proc = None

	async def restart(self) -> None:
		"""Stop the server and start it again with no memory of the old session."""
		self._fail_pending()  # requests in flight on the old server fail now, not after their timeout
		await self.stop()
		self._pending = {}
		self._diagnostics = {}
		self._open_files = {}
		self._diag_events = {}
		self._symbol_cache = {}
		self._overlays = {}
		self.restarts += 1
		await self.start()
		if self.on_restart is not None:
			await self.on_restart(self)

	# -- wire protocol -----------------------------------------------------

	async def _read_loop(self) -> None:
		try:
			await self._read_messages()
		finally:
			if self._stderr_task is not None and not self._stderr_task.done():
				# stdout closing usually means the process died; its last words are on stderr.
				with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
					await asyncio.wait_for(asyncio.shield(self._stderr_task), _STDERR_FLUSH_TIMEOUT)
			if self._proc is not None and self._proc.returncode is None and self._pending:
				with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
					await asyncio.wait_for(self._proc.wait(), _STDERR_FLUSH_TIMEOUT)  # for its exit code
			self._fail_pending()

	def _fail_pending(self) -> None:
		# Without this, requests in flight when the server dies wait out their full timeout.
		pending, self._pending = self._pending, {}
		if not pending:
			return
		message, detail = self._exit_message()
		logger.warning("%s\n%s", message, detail)
		for fut in pending.values():
			if not fut.done():
				fut.set_exception(LanguageServerExitedError(message, detail))

	def _exit_message(self) -> tuple[str, str]:
		"""(message for the agent, technical detail for logs): the command line and the stderr tail
		stay out of the message (they hold absolute paths and raw server output)."""
		code = getattr(self._proc, "returncode", None)
		message = "the language server exited unexpectedly"
		if code is not None:
			message += f" (exit code {code})"
		message += "; the next call restarts it"
		detail = f"command: {' '.join(self.command)}"
		if self._stderr_tail:
			detail += "\nlast stderr output:\n" + "\n".join(self._stderr_tail)
		return message, detail

	async def _read_messages(self) -> None:
		proc = self._running_proc
		if proc.stdout is None:
			raise NotStartedError("language server subprocess has no stdout pipe")
		stream = proc.stdout
		while True:
			header = b""
			while not header.endswith(b"\r\n\r\n"):
				chunk = await stream.read(1)
				if not chunk:
					return
				header += chunk
			length = 0
			for line in header.split(b"\r\n"):
				if line.lower().startswith(b"content-length"):
					length = int(line.split(b":")[1].strip())
			body = await stream.readexactly(length)
			try:
				msg = json.loads(body)
			except json.JSONDecodeError:
				continue
			self._dispatch(msg)

	async def _drain_stderr(self) -> None:
		proc = self._running_proc
		if proc.stderr is None:
			raise NotStartedError("language server subprocess has no stderr pipe")
		async for line in proc.stderr:
			# Only kept to explain an exit (see `_exit_message`); routine logging isn't surfaced.
			text = line.decode(errors="replace").rstrip()
			if text:
				self._stderr_tail.append(text)

	def _reply_to_server_request(self, msg: dict[str, Any]) -> None:
		"""Answer a server->client request so the server never waits on us
		(`workspace/configuration`, `client/registerCapability`, progress
		creation, ...). Unknown methods get MethodNotFound."""
		method = msg.get("method")
		if method == "workspace/configuration":
			items = (msg.get("params") or {}).get("items") or []
			reply: dict[str, Any] = {"result": [None] * len(items)}
		elif method in _NULL_REPLY_METHODS:
			reply = {"result": None}
		else:
			reply = {"error": {"code": -32601, "message": f"Method not found: {method}"}}
		try:
			self._send({"jsonrpc": "2.0", "id": msg["id"], **reply})
		except (NotStartedError, OSError, RuntimeError):
			logger.debug("could not reply to server request %s", method, exc_info=True)

	def _dispatch(self, msg: dict[str, Any]) -> None:
		if "id" in msg and "method" in msg:
			self._reply_to_server_request(msg)
			return
		if "id" in msg and "method" not in msg:
			fut = self._pending.pop(msg["id"], None)
			if fut is not None and not fut.done():
				fut.set_result(msg)
			return
		method = msg.get("method")
		if method == "textDocument/publishDiagnostics":
			params = msg.get("params", {})
			uri = params.get("uri")
			if uri:
				self._diagnostics[uri] = params.get("diagnostics", [])
				event = self._diag_events.get(uri)
				if event is not None:
					event.set()

	def _send(self, obj: dict[str, Any]) -> None:
		proc = self._running_proc
		if proc.stdin is None:
			raise NotStartedError("language server subprocess has no stdin pipe")
		body = json.dumps(obj).encode("utf-8")
		header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
		proc.stdin.write(header + body)

	async def _request(self, method: str, params: dict[str, Any], timeout: float = 20) -> dict[str, Any]:
		"""Send a request, retrying when the server reports the document changed mid-flight.

		LSP says a client should re-issue a request that failed with `ContentModified`
		(e.g. an edit landed while the server was still computing); other errors surface as-is.
		"""
		for delay in _CONTENT_MODIFIED_BACKOFF:
			try:
				return await self._request_once(method, params, timeout)
			except LspRequestError as exc:
				if exc.code not in _RETRYABLE_CODES:
					raise
				logger.debug("retrying %s after LSP error %s", method, exc.code)
			await asyncio.sleep(delay)
		return await self._request_once(method, params, timeout)

	async def _request_once(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
		self._next_id += 1
		msg_id = self._next_id
		fut: asyncio.Future = asyncio.get_running_loop().create_future()
		self._pending[msg_id] = fut
		self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
		stdin = self._running_proc.stdin
		if stdin is not None:
			await stdin.drain()
		try:
			resp = await asyncio.wait_for(fut, timeout=timeout)
		except TimeoutError:
			await self._abort()
			raise
		finally:
			self._pending.pop(msg_id, None)
		if "error" in resp:
			err = resp["error"] or {}
			raise LspRequestError(method, err.get("code"), str(err.get("message", err)))
		return resp

	def _notify(self, method: str, params: dict[str, Any]) -> None:
		self._send({"jsonrpc": "2.0", "method": method, "params": params})

	@contextlib.asynccontextmanager
	async def _reading(self) -> AsyncIterator[None]:
		"""Hold off `overlay()` while an LSP call that depends on document text is in flight, so the
		call is answered for the text it started with (the disk's), never for a preview's."""
		me = id(self)
		current = asyncio.current_task()
		overlay = _OVERLAY_HOLDERS.get().get(me, [False, None])
		reading = _READING.get().get(me, [False, None])
		if (overlay[0] and overlay[1] is current) or (reading[0] and reading[1] is current):
			yield
			return
		if overlay[0]:
			raise RuntimeError("a child task cannot read during its parent task overlay")
		if reading[0] and self._overlay_lock.locked():
			raise RuntimeError("a child task cannot start a read while an overlay waits for its parent")
		async with self._overlay_lock:
			self._readers += 1
			self._readers_idle.clear()
		lease = [True, asyncio.current_task()]
		token = _READING.set({**_READING.get(), me: lease})
		try:
			yield
		finally:
			lease[0] = False
			with contextlib.suppress(ValueError):
				_READING.reset(token)
			self._readers -= 1
			if self._readers <= 0:
				self._readers = 0
				self._readers_idle.set()

	# -- document sync -------------------------------------------------------

	def _to_uri(self, file_path: str) -> Path:
		p = Path(file_path)
		if not p.is_absolute():
			p = self.workspace_root / p
		return p.resolve()

	def _language_id_for(self, path: Path) -> str:
		return self.language_ids.get(path.suffix.lower(), self.language_id)

	async def ensure_open(self, file_path: str) -> str:
		abs_path = self._to_uri(file_path)
		uri = abs_path.as_uri()
		if uri in self._overlays:
			return uri
		stat = abs_path.stat()
		known = self._open_files.get(uri)
		if (
			known is not None
			and known.mtime_ns == stat.st_mtime_ns
			and known.size == stat.st_size
			and known.ctime_ns == stat.st_ctime_ns
		):
			return uri
		text = _read_document_text(abs_path)
		self._diag_events[uri] = asyncio.Event()
		# The previous version's pushed diagnostics describe text that no longer
		# exists; keeping them would resurface fixed errors as a "cache fallback".
		if known is not None:
			self._diagnostics.pop(uri, None)
		if known is None:
			self._notify(
				"textDocument/didOpen",
				{
					"textDocument": {
						"uri": uri,
						"languageId": self._language_id_for(abs_path),
						"version": 1,
						"text": text,
					}
				},
			)
			self._open_files[uri] = OpenFile(
				uri=uri, version=1, mtime_ns=stat.st_mtime_ns, size=stat.st_size, ctime_ns=stat.st_ctime_ns
			)
		else:
			new_version = known.version + 1
			self._notify(
				"textDocument/didChange",
				{
					"textDocument": {"uri": uri, "version": new_version},
					"contentChanges": [{"text": text}],
				},
			)
			self._open_files[uri] = OpenFile(
				uri=uri, version=new_version, mtime_ns=stat.st_mtime_ns, size=stat.st_size, ctime_ns=stat.st_ctime_ns
			)
		return uri

	async def close_document(self, uri: str) -> None:
		"""Tell the server a document is gone from disk / no longer ours, and drop its caches."""
		self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
		self._open_files.pop(uri, None)
		self._diagnostics.pop(uri, None)
		self._diag_events.pop(uri, None)
		# Reopening restarts versions at 1, which could falsely match an old entry.
		self._symbol_cache.pop(uri, None)

	def _scan_watched(self) -> tuple[dict[Path, tuple[int, int, int]], dict[Path, tuple[int, int, int]]]:
		"""(mtime_ns, size, ctime_ns) of every watched source file and every `config_names`
		file, each keyed by resolved path (as `ensure_open` keys its URIs). Runs on
		every tool call, so it resolves the root once instead of every file (only
		symlinked files need their own `resolve()`), and it skips nested checkouts:
		a directory holding a `.git` entry below the root is another repository or
		linked worktree (this repo creates worktrees under `.claude/worktrees/`),
		whose files are neither this workspace's nor cheap to stat."""
		sources: dict[Path, tuple[int, int, int]] = {}
		configs: dict[Path, tuple[int, int, int]] = {}
		if not self.watch_suffixes and not self.config_names:
			return sources, configs
		root = self.workspace_root.resolve()
		pending = [root]
		while pending:
			directory = pending.pop()
			try:
				with os.scandir(directory) as it:
					entries = list(it)
			except OSError:
				continue
			if directory != root and any(e.name == ".git" for e in entries):
				continue
			for entry in entries:
				try:
					if entry.is_dir(follow_symlinks=False):
						if entry.name not in EXCLUDED_DIR_NAMES:
							pending.append(Path(entry.path))
						continue
					is_config = entry.name in self.config_names
					if not is_config and os.path.splitext(entry.name)[1].lower() not in self.watch_suffixes:
						continue
					path = Path(entry.path)
					if not is_config and self.watch_ignore is not None and self.watch_ignore(path):
						continue
					st = entry.stat()
				except OSError:
					continue
				key = path.resolve() if entry.is_symlink() else path
				(configs if is_config else sources)[key] = (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
		return sources, configs

	def _changed_configs(
		self, configs: dict[Path, tuple[int, int, int]], previous: dict[Path, tuple[int, int, int]] | None
	) -> list[str]:
		"""Names of config files whose *content* differs from the last time we looked
		(stat-changed files are hashed; the remembered hashes are updated)."""
		touched: set[str] = set()
		hashes: dict[Path, str] = {}
		for path, stat in configs.items():
			known = self._config_hashes.get(path)
			if known is not None and previous is not None and previous.get(path) == stat:
				hashes[path] = known
				continue
			try:
				digest = hashlib.sha256(path.read_bytes()).hexdigest()
			except OSError:
				continue
			hashes[path] = digest
			if previous is not None and digest != known:
				touched.add(path.name)
		touched.update(path.name for path in self._config_hashes.keys() - hashes.keys())
		self._config_hashes = hashes
		return sorted(touched)

	async def refresh(self) -> None:
		"""Bring the language server's view of the disk up to date.

		Servers only see what a client tells them: a document opened with `didOpen`
		is frozen at that text, and files created/edited/deleted behind the
		server's back (by the agent's own Edit tool, git, a formatter) are
		invisible to workspace-wide answers. Call this before every tool call;
		it costs one stat walk (a few ms on this repo; see `_scan_watched`). Open
		documents are re-synced (or closed when deleted), and watched-suffix files
		that appeared/changed/vanished since the last call are reported via
		`workspace/didChangeWatchedFiles`. A change to a `config_names` file restarts
		the server instead (it only reads its project config at startup).
		"""
		# A simulation in progress shows the server unsaved text: wait for it (and let a simulation
		# that wants to start wait for this sync), so its state is never mixed with the disk's.
		async with self._reading(), self._refresh_lock:
			current, configs = await asyncio.to_thread(self._scan_watched)
			previous_configs = self._config_snapshot
			self._config_snapshot = configs
			if previous_configs is not None and configs != previous_configs:
				touched = await asyncio.to_thread(self._changed_configs, configs, previous_configs)
				if touched:
					# The server only reads its project config at startup: start a fresh one,
					# which also sees the disk as it is now, so nothing else needs reporting.
					self._watch_snapshot = current
					await self.restart()
					if self.on_notice is not None:
						self.on_notice(f"restarted the language server because {', '.join(touched)} changed")
					return
			elif previous_configs is None:
				await asyncio.to_thread(self._changed_configs, configs, None)
			changes: list[dict[str, Any]] = []
			created_directories: list[dict[str, Any]] = []
			deleted_directories: list[dict[str, Any]] = []
			previous = self._watch_snapshot
			if previous is not None:
				created_files = [path for path in current if path not in previous]
				deleted_files = [path for path in previous if path not in current]
				root = self.workspace_root.resolve()
				if created_files:
					known = {parent for path in previous for parent in path.parents}
					created_directories = _directory_events(
						{p for path in created_files for p in path.parents if p not in known and root in p.parents},
						_FILE_CREATED,
					)
				for path, stamp in current.items():
					if path not in previous:
						changes.append({"uri": path.as_uri(), "type": _FILE_CREATED})
					elif previous[path] != stamp:
						changes.append({"uri": path.as_uri(), "type": _FILE_CHANGED})
				changes.extend({"uri": path.as_uri(), "type": _FILE_DELETED} for path in deleted_files)
				if deleted_files:
					gone = {p for path in deleted_files for p in path.parents if root in p.parents and not p.exists()}
					deleted_directories = _directory_events(gone, _FILE_DELETED)
			self._watch_snapshot = current
			if changes:
				# Directories go with their files (a new one before them, a removed one after them), but only the
				# files are opened below.
				self._notify(
					"workspace/didChangeWatchedFiles",
					{"changes": [*created_directories, *changes, *deleted_directories]},
				)
			if self.open_watched_changes:
				for change in changes:
					if change["type"] != _FILE_DELETED:
						await self.ensure_open(_uri_to_path(change["uri"]))
			for uri, known in list(self._open_files.items()):
				if known.mtime_ns == -1:
					continue  # scratch document: never on disk
				path = Path(_uri_to_path(uri))
				if not path.exists():
					await self.close_document(uri)
				else:
					await self.ensure_open(str(path))

	# -- LSP calls used by the MCP tools --------------------------------------

	def _check_position(self, file_path: str, line: int, column: int) -> None:
		"""Reject a position outside the file, so a typo isn't answered with an
		empty result indistinguishable from "nothing there"."""
		abs_path = self._to_uri(file_path)
		lines = _LINE_BREAK_RE.split(_read_document_text(abs_path))
		if len(lines) > 1 and lines[-1] == "":
			lines.pop()  # the empty "line" after the final newline isn't a line anyone can point at
		if not 1 <= line <= len(lines):
			raise InvalidPositionError(
				f"line {line} is out of range: {file_path} has {len(lines)} line(s) (lines are 1-indexed)"
			)
		width = len(lines[line - 1].encode("utf-16-le")) // 2
		if not 1 <= column <= width + 1:
			raise InvalidPositionError(
				f"column {column} is out of range: line {line} of {file_path} is {width} character(s) long "
				"(columns are 1-indexed UTF-16 offsets; a tab counts as one)"
			)

	@_reader
	async def hover(self, file_path: str, line: int, column: int) -> dict[str, Any]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/hover",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or {}

	@_reader
	async def definition(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/definition",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		result = resp.get("result")
		if result is None:
			return []
		return result if isinstance(result, list) else [result]

	@_reader
	async def type_definition(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/typeDefinition",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		result = resp.get("result")
		if result is None:
			return []
		return result if isinstance(result, list) else [result]

	@_reader
	async def references(
		self, file_path: str, line: int, column: int, *, include_declaration: bool = True
	) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/references",
			{
				"textDocument": {"uri": uri},
				"position": {"line": line - 1, "character": column - 1},
				"context": {"includeDeclaration": include_declaration},
			},
		)
		return resp.get("result") or []

	@_reader
	async def workspace_symbol(self, query: str) -> list[dict[str, Any]]:
		resp = await self._request("workspace/symbol", {"query": query})
		return resp.get("result") or []

	@_reader
	async def diagnostics_with_open_files(
		self, file_paths: list[str], *, open_first: list[str]
	) -> dict[str, list[dict[str, Any]]]:
		"""Open prerequisites and collect diagnostics under one reader lease.

		Any failure aborts the whole snapshot: callers must not treat an incomplete result as clean.
		This orders LSP messages but is not a guarantee that a server has finished indexing.
		"""
		for path in open_first:
			await self.ensure_open(path)
		return {path: await self.diagnostics(path) for path in file_paths}

	@_reader
	async def diagnostics(self, file_path: str) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		cached = self._diagnostics.get(uri, [])
		try:
			resp = await self._request("textDocument/diagnostic", {"textDocument": {"uri": uri}})
		except LspRequestError:
			# HTML/CSS/TS servers often only push publishDiagnostics and reject
			# pull. If the document was just (re)synced, the push for this
			# version hasn't necessarily arrived yet: wait for it rather than
			# report the previous version's (or an empty) result.
			event = self._diag_events.get(uri)
			if event is not None and not event.is_set():
				with contextlib.suppress(TimeoutError):
					await asyncio.wait_for(event.wait(), timeout=PUSH_DIAGNOSTICS_TIMEOUT)
			return self._diagnostics.get(uri, [])
		result = resp.get("result") or {}
		if result.get("kind") == "unchanged":
			return cached
		items = result.get("items")
		if items is None:
			return cached
		# A pull answer for the current version is authoritative, empty included.
		return items

	@_reader
	async def document_symbol(self, file_path: str) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		version = self._open_files[uri].version
		hit = self._symbol_cache.get(uri)
		if hit is not None and hit[0] == version:
			return hit[1]
		resp = await self._request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
		result = resp.get("result") or []
		self._symbol_cache[uri] = (version, result)
		return result

	@_reader
	async def prepare_call_hierarchy(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		resp = await self._request(
			"textDocument/prepareCallHierarchy",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or []

	@_reader
	async def incoming_calls(self, item: dict[str, Any]) -> list[dict[str, Any]]:
		resp = await self._request("callHierarchy/incomingCalls", {"item": item})
		return resp.get("result") or []

	@_reader
	async def prepare_type_hierarchy(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		resp = await self._request(
			"textDocument/prepareTypeHierarchy",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or []

	@_reader
	async def supertypes(self, item: dict[str, Any]) -> list[dict[str, Any]]:
		resp = await self._request("typeHierarchy/supertypes", {"item": item})
		return resp.get("result") or []

	# -- write-side calls ------------------------------------------------------

	@_reader
	async def prepare_rename(self, file_path: str, line: int, column: int) -> dict[str, Any] | None:
		"""The range (and placeholder) a rename at the position would replace, or None
		when the symbol can't be renamed (builtins, keywords, library code)."""
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/prepareRename",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		result = resp.get("result")
		if not result:
			return None
		return result if "range" in result else {"range": result}

	@_reader
	async def rename(self, file_path: str, line: int, column: int, new_name: str) -> dict[str, Any] | None:
		"""The `WorkspaceEdit` renaming the symbol at the position (not applied anywhere)."""
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/rename",
			{
				"textDocument": {"uri": uri},
				"position": {"line": line - 1, "character": column - 1},
				"newName": new_name,
			},
		)
		return resp.get("result")

	@_reader
	async def code_actions(
		self,
		file_path: str,
		start: tuple[int, int],
		end: tuple[int, int],
		diagnostics: list[dict[str, Any]] | None = None,
		only: list[str] | None = None,
	) -> list[dict[str, Any]]:
		"""Code actions for a range; `start`/`end` are 1-based (line, column) like the other calls."""
		uri = await self.ensure_open(file_path)
		context: dict[str, Any] = {"diagnostics": diagnostics or []}
		if only:
			context["only"] = only
		resp = await self._request(
			"textDocument/codeAction",
			{
				"textDocument": {"uri": uri},
				"range": {
					"start": {"line": start[0] - 1, "character": start[1] - 1},
					"end": {"line": end[0] - 1, "character": end[1] - 1},
				},
				"context": context,
			},
		)
		return resp.get("result") or []

	@_reader
	async def signature_help(self, file_path: str, line: int, column: int) -> dict[str, Any]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/signatureHelp",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or {}

	@_reader
	async def subtypes(self, item: dict[str, Any]) -> list[dict[str, Any]]:
		resp = await self._request("typeHierarchy/subtypes", {"item": item})
		return resp.get("result") or []

	# -- simulation overlays ---------------------------------------------------

	def _show_text(self, uri: str, text: str) -> None:
		known = self._open_files.get(uri)
		self._diag_events[uri] = asyncio.Event()
		self._diagnostics.pop(uri, None)
		if known is None:
			version = 1
			self._notify(
				"textDocument/didOpen",
				{
					"textDocument": {
						"uri": uri,
						"languageId": self._language_id_for(Path(_uri_to_path(uri))),
						"version": version,
						"text": text,
					}
				},
			)
		else:
			version = known.version + 1
			self._notify(
				"textDocument/didChange",
				{"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
			)
		self._open_files[uri] = OpenFile(uri=uri, version=version, mtime_ns=_OVERLAY_MTIME, size=len(text))
		self._overlays[uri] = text

	def _manifest_prefix(self) -> str:
		key = _sha1(str(self.workspace_root.resolve()).encode("utf-8"))[:12]
		return f"mcp-nav-materialized-{key}-"

	def _manifest_dir(self) -> Path:
		uid = os.getuid() if hasattr(os, "getuid") else None
		base = Path(tempfile.gettempdir()) / f"mcp-nav-private-{uid}"
		base.mkdir(mode=0o700, exist_ok=True)
		info = base.lstat()
		if not stat_module.S_ISDIR(info.st_mode) or (uid is not None and (info.st_uid != uid or info.st_mode & 0o077)):
			raise OSError("unsafe manifest directory")
		return base

	def _manifest_path(self) -> Path:
		return self._manifest_dir() / f"{self._manifest_prefix()}{os.getpid()}-{self._manifest_id}.json"

	def _manifest_read(self, path: Path | None = None) -> list[dict[str, Any]]:
		try:
			data = json.loads((path or self._manifest_path()).read_text(encoding="utf-8"))
		except (OSError, ValueError):
			return []
		return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []

	def _manifest_write(self, entries: list[dict[str, Any]], path: Path | None = None) -> None:
		path = path or self._manifest_path()
		if not entries:
			path.unlink(missing_ok=True)
			return
		tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
		tmp.write_text(json.dumps(entries), encoding="utf-8")
		os.replace(tmp, path)

	def _manifest_add(self, target: Path, text: str, created_dirs: list[Path]) -> None:
		"""Persist before creating the preview. Failure must prevent materialisation."""
		entries = self._manifest_read()
		entries.append(
			{
				"path": str(target),
				"sha1": _sha1(text.encode("utf-8")),
				"pid": os.getpid(),
				"created_dirs": [str(d) for d in created_dirs],
			}
		)
		self._manifest_write(entries)

	def _manifest_drop(self, targets: list[Path]) -> None:
		paths = {str(t) for t in targets}
		entries = self._manifest_read()
		kept = [e for e in entries if e.get("path") not in paths]
		if len(kept) != len(entries):
			self._manifest_write(kept)

	def _sweep_materialized(self) -> None:
		"""Report stale previews once; a manifest cannot prove ownership of a current file."""
		if os.name == "nt":
			return
		root = self.workspace_root.resolve()
		for manifest in self._manifest_dir().glob(f"{self._manifest_prefix()}*.json"):
			entries = self._manifest_read(manifest)
			kept: list[dict[str, Any]] = []
			paths: list[str] = []
			for entry in entries:
				pid = entry.get("pid")
				if not isinstance(pid, int) or _pid_alive(pid):
					kept.append(entry)
					continue
				value = entry.get("path")
				if isinstance(value, str):
					try:
						path = Path(value)
						relative = path.relative_to(root)
						if ".." not in relative.parts and (path.exists() or path.is_symlink()):
							paths.append(relative.as_posix())
					except ValueError:
						pass
				# Directories the dead preview created and left empty: an empty directory is safe to
				# remove (rmdir refuses anything with content), unlike a file whose ownership is unknown.
				dirs = entry.get("created_dirs")
				if isinstance(dirs, list):
					for value in reversed(dirs):
						if not isinstance(value, str):
							continue
						try:
							directory = Path(value)
							directory.relative_to(root)
						except ValueError:
							continue
						with contextlib.suppress(OSError):
							directory.rmdir()
			if paths:
				message = "An interrupted preview may have left files; inspect these workspace paths: " + ", ".join(
					sorted(set(paths))
				)
				logger.warning(message)
				if self.on_notice is not None:
					self.on_notice(message)
			if len(kept) != len(entries):
				self._manifest_write(kept, manifest)

	def _free_sibling_name(self, target: Path) -> Path:
		"""A private, unused sibling name to move a preview file to before deciding its fate."""
		base = f".{target.name}.mcp-nav-preview-{os.getpid()}"
		candidate = target.with_name(base)
		number = 0
		while candidate.exists() or candidate.is_symlink():
			number += 1
			candidate = target.with_name(f"{base}-{number}")
		return candidate

	def _discard_if_ours(self, target: Path, identity: tuple[int, int, str]) -> bool:
		"""Remove the transient preview file iff the path still holds the exact bytes that were created.

		The file is renamed to a private sibling name first: that one atomic step takes ownership of whatever
		sits at the path, so a user file that replaces the target between the check and the deletion can never
		be unlinked. Content that turns out not to be ours is renamed back untouched (fail closed: the file is
		kept with a warning when it cannot be isolated or removed).
		"""
		dev, inode, sha = identity
		if target.is_symlink():
			return False
		stolen = self._free_sibling_name(target)
		try:
			os.rename(target, stolen)
		except OSError:
			logger.warning("could not isolate the preview file; it is left in place", exc_info=True)
			return False
		try:
			info = stolen.stat()
			if (info.st_dev, info.st_ino) == (dev, inode) and _sha1(stolen.read_bytes()) == sha:
				stolen.unlink()
				return True
			if target.exists() or target.is_symlink():
				# the user created something new at the path while we held the old file: keep both
				# (renaming back would replace their file — deletion must never rest on a guess)
				keep = self._free_sibling_name(target)
				os.rename(stolen, keep)
				logger.warning(
					"the preview path %s was recreated during cleanup; the transient is kept at %s",
					target.name,
					keep.name,
				)
				return False
			os.rename(stolen, target)  # not ours (any more): restore it for the user
			return False
		except FileNotFoundError:
			return True
		except OSError:
			logger.warning("could not remove the preview file; it is left at %s", stolen.name, exc_info=True)
			return False

	def _missing_parents(self, path: Path) -> list[Path]:
		"""Directories above `path` that don't exist yet, outermost first."""
		return [parent for parent in reversed(path.parents) if not parent.exists()]

	def _materialize(self, path: Path, text: str) -> tuple[int, int]:
		"""Create a transient file exclusively; preserve ambiguous partial files on failure."""
		path.parent.mkdir(parents=True, exist_ok=True)
		data = text.encode("utf-8")
		with path.open("xb") as output:
			identity = os.fstat(output.fileno())
			output.write(data)
			output.flush()

		return identity.st_dev, identity.st_ino

	@contextlib.asynccontextmanager
	async def overlay(self, texts: dict[Path, str]) -> AsyncIterator[None]:
		"""Show the server `texts` (path -> content) in place of the disk's, so `diagnostics()` and
		friends answer for an edit that hasn't been written. Existing files are never touched. A
		path that doesn't exist yet is written to disk for the duration only (the server resolves
		imports through the file system) and normally removed again with any directories it needed before the overlay ends.
		A failed partial write or SIGKILL may leave a file behind for manual inspection. The overlay waits for LSP calls already in flight, which are
		answered for the disk's text; other tool calls wait until it ends, so they never see
		simulated text or the transient files.
		"""
		async with self._overlay_lock:
			await self._readers_idle.wait()
			uris: list[str] = []
			transient: list[Path] = []
			owned: dict[Path, tuple[int, int, str]] = {}
			created_dirs: list[Path] = []
			lease = [True, asyncio.current_task()]
			holders_token = _OVERLAY_HOLDERS.set({**_OVERLAY_HOLDERS.get(), id(self): lease})
			try:
				for path, text in texts.items():
					target = self._to_uri(str(path))
					if not target.exists():
						missing = self._missing_parents(target)
						self._manifest_add(target, text, missing)
						created_dirs += missing
						try:
							device, inode = self._materialize(target, text)
						except FileExistsError:
							self._manifest_drop([target])
							raise
						except BaseException:
							# Keep the record if a partial file or replacement may need inspection.
							if not target.exists() and not target.is_symlink():
								with contextlib.suppress(OSError):
									self._manifest_drop([target])
							else:
								try:
									name = target.relative_to(self.workspace_root.resolve()).as_posix()
								except ValueError:
									name = "a file outside the workspace"
								message = f"Preview creation failed; inspect {name} for a partial file."
								logger.warning(message)
								if self.on_notice is not None:
									with contextlib.suppress(Exception):
										self.on_notice(message)
							raise
						transient.append(target)
						owned[target] = (device, inode, _sha1(text.encode("utf-8")))
					uri = target.as_uri()
					self._show_text(uri, text)
					uris.append(uri)
				if transient:
					self._notify(
						"workspace/didChangeWatchedFiles",
						{
							"changes": [
								*_directory_events(created_dirs, _FILE_CREATED),
								*({"uri": p.as_uri(), "type": _FILE_CREATED} for p in transient),
							]
						},
					)
				yield
			finally:
				try:
					for uri in uris:
						self._overlays.pop(uri, None)
					completed: list[Path] = []
					for target in transient:
						self._discard_if_ours(target, owned[target])
						completed.append(target)
					for directory in reversed(created_dirs):
						with contextlib.suppress(OSError):
							directory.rmdir()
					if completed:
						self._manifest_drop(completed)
					if transient and self._proc is not None:
						self._notify(
							"workspace/didChangeWatchedFiles",
							{
								"changes": [
									*({"uri": p.as_uri(), "type": _FILE_DELETED} for p in transient),
									*_directory_events([d for d in created_dirs if not d.exists()], _FILE_DELETED),
								]
							},
						)
					for uri in uris if self._proc is not None else []:
						path = Path(_uri_to_path(uri))
						if path.exists():
							await self.ensure_open(str(path))
						else:
							await self.close_document(uri)
				finally:
					lease[0] = False
					with contextlib.suppress(ValueError):
						_OVERLAY_HOLDERS.reset(holders_token)

	# -- scratch (in-memory-only) documents -----------------------------------
	#
	# For codenav's Protocol-conformance probe: a document that is never
	# written to disk, so it can't use `ensure_open`'s stat/read-based sync.

	async def open_scratch_document(self, uri: str, text: str) -> None:
		self._notify(
			"textDocument/didOpen",
			{"textDocument": {"uri": uri, "languageId": self.language_id, "version": 1, "text": text}},
		)
		self._open_files[uri] = OpenFile(uri=uri, version=1, mtime_ns=-1, size=len(text))

	async def change_scratch_document(self, uri: str, text: str) -> None:
		version = self._open_files[uri].version + 1
		self._notify(
			"textDocument/didChange",
			{"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
		)
		self._open_files[uri] = OpenFile(uri=uri, version=version, mtime_ns=-1, size=len(text))

	async def close_scratch_document(self, uri: str) -> None:
		await self.close_document(uri)

	@_reader
	async def pull_diagnostics(self, uri: str) -> list[dict[str, Any]]:
		"""Pull diagnostics for an already-open `uri` directly, with no cache
		fallback — used for the scratch-document probe above, where there is no
		prior `publishDiagnostics` push to fall back to."""
		resp = await self._request("textDocument/diagnostic", {"textDocument": {"uri": uri}})
		result = resp.get("result") or {}
		return result.get("items") or []
