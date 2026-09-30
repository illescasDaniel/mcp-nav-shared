"""Out-of-band messages for the agent, appended to tool results.

Two kinds: one-shot notices posted while a call runs (e.g. "restarted the
language server because pyproject.toml changed") and a sticky one while the
server's own source files differ from what it started with. A stdio MCP
server can't reload itself (the client's `initialize` handshake happens
once), so the most it can do about stale code is say so on every call.
"""

from __future__ import annotations

import functools
import time
import typing
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any


_RECHECK_SECONDS = 2.0
_Stamp = tuple[tuple[str, int, int], ...]


def _source_stamp(dirs: Iterable[Path]) -> _Stamp:
	found: list[tuple[str, int, int]] = []
	for directory in dirs:
		for path in directory.rglob("*.py"):
			try:
				st = path.stat()
			except OSError:
				continue
			found.append((str(path), st.st_mtime_ns, st.st_size))
	return tuple(sorted(found))


class NoticeBoard:
	def __init__(
		self, server_name: str, source_dirs: Iterable[Path], *, recheck_seconds: float = _RECHECK_SECONDS
	) -> None:
		self.server_name = server_name
		self._recheck_seconds = recheck_seconds
		self._source_dirs = tuple(source_dirs)
		self._started_with = _source_stamp(self._source_dirs)
		self._pending: list[str] = []
		self._checked_at = time.monotonic()
		self._stale = False

	def post(self, message: str) -> None:
		if message not in self._pending:
			self._pending.append(message)

	def code_is_stale(self) -> bool:
		# A stat walk per tool call adds up; a few seconds' lag in noticing an edit is fine.
		now = time.monotonic()
		if now - self._checked_at >= self._recheck_seconds:
			self._checked_at = now
			self._stale = _source_stamp(self._source_dirs) != self._started_with
		return self._stale

	def drain(self) -> list[str]:
		"""One-shot notices posted since the last call, plus the sticky stale-code line."""
		notices, self._pending = self._pending, []
		if self.code_is_stale():
			notices.append(
				f"the {self.server_name} server's own code changed since it started; "
				"restart the MCP servers to use the new version."
			)
		return notices

	def annotate(self, text: str) -> str:
		notices = self.drain()
		if not notices:
			return text
		return text + "".join(f"\n\n[{self.server_name}] {notice}" for notice in notices)

	def tool(self, fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
		"""Decorator for an MCP tool that returns text: appends pending notices to its result.

		Apply it *below* `@mcp.tool()`. The wrapper keeps the tool's signature and
		resolves its (postponed) annotations in the tool's own module, since the
		framework inspects them to find the `Context` parameter.
		"""

		@functools.wraps(fn)
		async def wrapper(*args: Any, **kwargs: Any) -> str:
			return self.annotate(await fn(*args, **kwargs))

		wrapper.__annotations__ = typing.get_type_hints(fn)
		return wrapper


def package_source_dirs(*modules: Any) -> list[Path]:
	"""Directories of the given imported modules/packages (their `__file__`'s parent)."""
	return [Path(module.__file__).resolve().parent for module in modules]
