"""Resolve the workspace root for project MCP servers.

Hosts may spawn stdio MCP processes with a cwd that is not the repo
(e.g. Cursor using $HOME). Prefer an explicit override, then Claude Code's
injected project dir, then the process's current working directory — never
a path baked into this package's own install location, since that would
silently point every un-configured host at wherever these servers happen to
be installed from (e.g. this repo) instead of the project actually being
worked on.
"""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


logger = logging.getLogger(__name__)


def describe_workspace_root(explicit_env: str) -> tuple[Path, str]:
	"""The statically configured workspace root and where it came from
	(`explicit_env`, `CLAUDE_PROJECT_DIR` or `cwd`)."""
	for key in (explicit_env, "CLAUDE_PROJECT_DIR"):
		raw = os.environ.get(key)
		if raw:
			return Path(raw).expanduser().resolve(), key
	return Path.cwd().resolve(), "cwd"


def resolve_workspace_root(explicit_env: str) -> Path:
	return describe_workspace_root(explicit_env)[0]


@lru_cache(maxsize=256)
def _git_common_dir(path: Path) -> Path | None:
	"""The repository's shared `.git` directory for `path` (identical for a
	checkout and all its linked worktrees), or None outside a git repo."""
	try:
		out = subprocess.run(  # noqa: S603
			["git", "-C", str(path), "rev-parse", "--git-common-dir"],  # noqa: S607
			capture_output=True,
			text=True,
			timeout=5,
			check=False,
		)
	except (OSError, subprocess.SubprocessError):
		return None
	if out.returncode != 0 or not out.stdout.strip():
		return None
	common = Path(out.stdout.strip())
	return (common if common.is_absolute() else path / common).resolve()


def same_repository(a: Path, b: Path) -> bool:
	"""True when `a` and `b` are the same directory, or two checkouts/worktrees
	of one git repository. This is what makes a client-reported root safe to
	adopt: a session in some unrelated project must never redirect the server."""
	if a == b:
		return True
	common_a = _git_common_dir(a)
	return common_a is not None and common_a == _git_common_dir(b)


def _root_paths(uris: Iterable[str]) -> list[Path]:
	paths: list[Path] = []
	for uri in uris:
		parsed = urlparse(uri)
		if parsed.scheme == "file":
			path = unquote(parsed.path)
			if os.name == "nt" and path[:1] == "/" and path[2:3] == ":":
				path = path[1:]  # file:///C:/x -> C:/x
			paths.append(Path(path).resolve())
	return paths


@dataclass(frozen=True)
class Selection:
	root: Path
	source: str


class WorkspaceSelector:
	"""Decides which directory a navigation server operates on.

	Hosts start a project's MCP servers once, with the *main* checkout as
	`CLAUDE_PROJECT_DIR`/cwd, even when the session works in a git worktree —
	so a fixed root silently answers from the wrong tree. Order of authority:

	1. `explicit_env` set: pinned, never overridden (a host that knows the
		right folder, e.g. Cursor's `${workspaceFolder}`).
	2. The client's MCP roots (`roots/list`), first one that is the configured
		base or another worktree/checkout of the same git repository.
	3. The configured base (`CLAUDE_PROJECT_DIR`, else cwd).
	"""

	def __init__(self, explicit_env: str) -> None:
		self.base, self.base_source = describe_workspace_root(explicit_env)
		self._explicit_env = explicit_env
		self.pinned = bool(os.environ.get(explicit_env))

	def explain(self, source: str) -> str:
		"""One sentence an agent can act on for a `Selection.source` value."""
		if source == "client roots":
			return (
				"client roots (the MCP client reported this checkout/worktree of the same repository "
				f"as the configured base {self.base})"
			)
		if source == self._explicit_env:
			return f"pinned by ${self._explicit_env}; client roots are ignored"
		if source == "CLAUDE_PROJECT_DIR":
			return (
				"$CLAUDE_PROJECT_DIR (default; the client has not reported another checkout/worktree "
				"of this repository)"
			)
		return f"server working directory (${self._explicit_env} and $CLAUDE_PROJECT_DIR are unset)"

	async def select(self, session: Any) -> Selection:
		if self.pinned:
			return Selection(self.base, self.base_source)
		for root in await self._client_roots(session):
			if same_repository(root, self.base):
				return Selection(root, "client roots")
		return Selection(self.base, self.base_source)

	@staticmethod
	async def _client_roots(session: Any) -> list[Path]:
		if session is None:
			return []
		try:
			if session.client_params.capabilities.roots is None:
				return []
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")  # roots are deprecated in newer MCP revisions but still used
				result = await session.list_roots()
		except Exception:
			logger.debug("roots/list failed; keeping the configured workspace", exc_info=True)
			return []
		return _root_paths(str(root.uri) for root in result.roots)


def resolve_source_root(explicit_env: str, workspace_root: Path) -> Path:
	"""Root directory for import-path derivation and workspace-wide class
	scanning (codenav's `implementations`). Not every project keeps its
	source under `src/`, so default to the workspace root itself (scan
	everything) rather than assuming a layout; a project that wants a
	narrower/faster scan sets `explicit_env` (e.g. to `src`) in its MCP
	server config."""
	raw = os.environ.get(explicit_env)
	if raw:
		path = Path(raw)
		return path if path.is_absolute() else (workspace_root / path).resolve()
	return workspace_root


def resolve_extra_source_roots(explicit_env: str, workspace_root: Path) -> list[Path]:
	"""Additional directories (comma-separated in `explicit_env`, relative to the
	workspace root unless absolute) that `implementations` also scans, reporting
	their matches under a separate heading — typically a project's `tests`
	directory, so test doubles of a port are listed next to its real adapters.
	Import paths for these are derived from the workspace root (the usual
	`pythonpath = ["."]` layout: `tests.unit.fakes`). Missing directories are
	ignored; unset means none."""
	roots: list[Path] = []
	for part in os.environ.get(explicit_env, "").split(","):
		part = part.strip()
		if not part:
			continue
		path = Path(part)
		path = path if path.is_absolute() else (workspace_root / path)
		path = path.resolve()
		if path.is_dir() and path not in roots:
			roots.append(path)
	return roots
