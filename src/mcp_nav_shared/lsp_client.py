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
import hashlib
import json
import logging
import os
import re
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
	"""The language server process ended while a request was pending."""


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


# `OpenFile.mtime_ns` of a document whose text came from `overlay()` rather than the disk: never
# equals a real mtime, so the next `ensure_open` after the overlay ends re-reads the file.
_OVERLAY_MTIME = -2

# LSP `FileChangeType`.
_FILE_CREATED, _FILE_CHANGED, _FILE_DELETED = 1, 2, 3


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
	# path -> (mtime_ns, size) of every watched file at the last `refresh()`.
	_watch_snapshot: dict[Path, tuple[int, int]] | None = field(default=None, init=False)
	# Same, for `config_names` files, plus their content hashes: a `touch` or a
	# `git checkout` that rewrites identical text must not restart the server.
	_config_snapshot: dict[Path, tuple[int, int]] | None = field(default=None, init=False)
	_config_hashes: dict[Path, str] = field(default_factory=dict, init=False)
	_refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
	# uri -> text shown to the server instead of the disk's, while `overlay()` is active.
	_overlays: dict[str, str] = field(default_factory=dict, init=False)
	# Held for the whole of an `overlay()`: `refresh()` (every tool call) waits for it.
	_overlay_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
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
		self._proc = await asyncio.create_subprocess_exec(
			*self.command,
			cwd=str(self.workspace_root),
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		self._reader_task = asyncio.create_task(self._read_loop())
		self._stderr_task = asyncio.create_task(self._drain_stderr())
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
		)
		self._notify("initialized", {})
		self._started = True

	async def stop(self) -> None:
		if self._proc is None:
			return
		try:
			await self._request("shutdown", {}, timeout=5)
			self._notify("exit", {})
		except Exception:
			logger.debug("language server didn't respond to shutdown in time; terminating it directly", exc_info=True)
		tasks = [task for task in (self._reader_task, self._stderr_task) if task is not None]
		for task in tasks:
			task.cancel()
		# Let the reader's cleanup run now, not after a restart has created new pending requests.
		await asyncio.gather(*tasks, return_exceptions=True)
		with contextlib.suppress(ProcessLookupError):
			self._proc.terminate()
		with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
			await asyncio.wait_for(self._proc.wait(), timeout=3)
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
			self._fail_pending()

	def _fail_pending(self) -> None:
		# Without this, requests in flight when the server dies wait out their full timeout.
		pending, self._pending = self._pending, {}
		for fut in pending.values():
			if not fut.done():
				fut.set_exception(LanguageServerExitedError(self._exit_message()))

	def _exit_message(self) -> str:
		message = f"language server exited: {' '.join(self.command)}"
		if self._stderr_tail:
			message += "\nIts last stderr output:\n" + "\n".join(self._stderr_tail)
		return message

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
		finally:
			self._pending.pop(msg_id, None)
		if "error" in resp:
			err = resp["error"] or {}
			raise LspRequestError(method, err.get("code"), str(err.get("message", err)))
		return resp

	def _notify(self, method: str, params: dict[str, Any]) -> None:
		self._send({"jsonrpc": "2.0", "method": method, "params": params})

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
		if known is not None and known.mtime_ns == stat.st_mtime_ns and known.size == stat.st_size:
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
			self._open_files[uri] = OpenFile(uri=uri, version=1, mtime_ns=stat.st_mtime_ns, size=stat.st_size)
		else:
			new_version = known.version + 1
			self._notify(
				"textDocument/didChange",
				{
					"textDocument": {"uri": uri, "version": new_version},
					"contentChanges": [{"text": text}],
				},
			)
			self._open_files[uri] = OpenFile(uri=uri, version=new_version, mtime_ns=stat.st_mtime_ns, size=stat.st_size)
		return uri

	async def close_document(self, uri: str) -> None:
		"""Tell the server a document is gone from disk / no longer ours, and drop its caches."""
		self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
		self._open_files.pop(uri, None)
		self._diagnostics.pop(uri, None)
		self._diag_events.pop(uri, None)
		# Reopening restarts versions at 1, which could falsely match an old entry.
		self._symbol_cache.pop(uri, None)

	def _scan_watched(self) -> tuple[dict[Path, tuple[int, int]], dict[Path, tuple[int, int]]]:
		"""(mtime_ns, size) of every watched source file and every `config_names`
		file, each keyed by resolved path (as `ensure_open` keys its URIs). Runs on
		every tool call, so it resolves the root once instead of every file (only
		symlinked files need their own `resolve()`), and it skips nested checkouts:
		a directory holding a `.git` entry below the root is another repository or
		linked worktree (this repo creates worktrees under `.claude/worktrees/`),
		whose files are neither this workspace's nor cheap to stat."""
		sources: dict[Path, tuple[int, int]] = {}
		configs: dict[Path, tuple[int, int]] = {}
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
				(configs if is_config else sources)[key] = (st.st_mtime_ns, st.st_size)
		return sources, configs

	def _changed_configs(
		self, configs: dict[Path, tuple[int, int]], previous: dict[Path, tuple[int, int]] | None
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
		async with self._overlay_lock:
			pass  # a simulation in progress shows the server unsaved text: don't mix its state with the disk's
		async with self._refresh_lock:
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
			previous = self._watch_snapshot
			if previous is not None:
				for path, stamp in current.items():
					if path not in previous:
						changes.append({"uri": path.as_uri(), "type": _FILE_CREATED})
					elif previous[path] != stamp:
						changes.append({"uri": path.as_uri(), "type": _FILE_CHANGED})
				changes.extend(
					{"uri": path.as_uri(), "type": _FILE_DELETED} for path in previous if path not in current
				)
			self._watch_snapshot = current
			if changes:
				self._notify("workspace/didChangeWatchedFiles", {"changes": changes})
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

	async def hover(self, file_path: str, line: int, column: int) -> dict[str, Any]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/hover",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or {}

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

	async def workspace_symbol(self, query: str) -> list[dict[str, Any]]:
		resp = await self._request("workspace/symbol", {"query": query})
		return resp.get("result") or []

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

	async def prepare_call_hierarchy(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		resp = await self._request(
			"textDocument/prepareCallHierarchy",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or []

	async def incoming_calls(self, item: dict[str, Any]) -> list[dict[str, Any]]:
		resp = await self._request("callHierarchy/incomingCalls", {"item": item})
		return resp.get("result") or []

	async def prepare_type_hierarchy(self, file_path: str, line: int, column: int) -> list[dict[str, Any]]:
		uri = await self.ensure_open(file_path)
		resp = await self._request(
			"textDocument/prepareTypeHierarchy",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or []

	async def supertypes(self, item: dict[str, Any]) -> list[dict[str, Any]]:
		resp = await self._request("typeHierarchy/supertypes", {"item": item})
		return resp.get("result") or []

	# -- write-side calls ------------------------------------------------------

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

	async def signature_help(self, file_path: str, line: int, column: int) -> dict[str, Any]:
		uri = await self.ensure_open(file_path)
		self._check_position(file_path, line, column)
		resp = await self._request(
			"textDocument/signatureHelp",
			{"textDocument": {"uri": uri}, "position": {"line": line - 1, "character": column - 1}},
		)
		return resp.get("result") or {}

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

	def _materialize(self, path: Path, text: str) -> list[Path]:
		"""Write a file that doesn't exist yet (and any missing parent directories); returns the
		directories created, outermost first. Language servers find modules on disk, so an
		import of a new file only resolves once the file is really there."""
		created_dirs: list[Path] = []
		for parent in reversed(path.parents):
			if not parent.exists():
				created_dirs.append(parent)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(text.encode("utf-8"))
		return created_dirs

	@contextlib.asynccontextmanager
	async def overlay(self, texts: dict[Path, str]) -> AsyncIterator[None]:
		"""Show the server `texts` (path -> content) in place of the disk's, so `diagnostics()` and
		friends answer for an edit that hasn't been written. Existing files are never touched. A
		path that doesn't exist yet is written to disk for the duration only (the server resolves
		imports through the file system) and removed again, with any directories it needed, before
		the overlay ends, even on errors. Other tool calls wait in `refresh()` until then, so they
		never see simulated text or the transient files.
		"""
		async with self._overlay_lock:
			uris: list[str] = []
			transient: list[Path] = []
			created_dirs: list[Path] = []
			try:
				for path, text in texts.items():
					target = self._to_uri(str(path))
					if not target.exists():
						created_dirs += self._materialize(target, text)
						transient.append(target)
					uri = target.as_uri()
					self._show_text(uri, text)
					uris.append(uri)
				if transient:
					self._notify(
						"workspace/didChangeWatchedFiles",
						{"changes": [{"uri": p.as_uri(), "type": _FILE_CREATED} for p in transient]},
					)
				yield
			finally:
				for uri in uris:
					self._overlays.pop(uri, None)
				for target in transient:
					with contextlib.suppress(OSError):
						target.unlink()
				for directory in reversed(created_dirs):
					with contextlib.suppress(OSError):
						directory.rmdir()
				if transient:
					self._notify(
						"workspace/didChangeWatchedFiles",
						{"changes": [{"uri": p.as_uri(), "type": _FILE_DELETED} for p in transient]},
					)
				for uri in uris:
					path = Path(_uri_to_path(uri))
					if path.exists():
						await self.ensure_open(str(path))
					else:
						await self.close_document(uri)

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

	async def pull_diagnostics(self, uri: str) -> list[dict[str, Any]]:
		"""Pull diagnostics for an already-open `uri` directly, with no cache
		fallback — used for the scratch-document probe above, where there is no
		prior `publishDiagnostics` push to fall back to."""
		resp = await self._request("textDocument/diagnostic", {"textDocument": {"uri": uri}})
		result = resp.get("result") or {}
		return result.get("items") or []
