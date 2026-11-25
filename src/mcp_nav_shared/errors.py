"""Expected tool failures for codenav/webnav, rendered as agent-readable text.

The MCP framework collapses any exception a tool raises into an opaque
"Error executing tool …", hiding actionable causes like a mistyped path.
Tools catch `TOOL_ERRORS` and return `format_tool_error(exc)` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mcp_nav_shared.lsp_client import InvalidPositionError, LanguageServerExitedError, LspRequestError


class FailedText(str):
	"""A tool refusal with a client-visible message and explicit failure status."""


class ToolInputError(ValueError):
	"""The caller asked for something the tool can't serve (e.g. unsupported file type)."""


# TimeoutError (asyncio's too, on 3.11+) is an OSError subclass; UnicodeDecodeError is a ValueError.
TOOL_ERRORS: tuple[type[Exception], ...] = (
	LspRequestError,
	LanguageServerExitedError,
	ToolInputError,
	InvalidPositionError,
	OSError,
	UnicodeDecodeError,
)


def _no_root() -> None:
	return None


_display_root: Callable[[], Path | None] = _no_root


def set_display_root(provider: Callable[[], Path | None] | None) -> None:
	"""Name the workspace root that error messages may show paths relative to (None: show bare file names)."""
	global _display_root
	_display_root = provider if provider is not None else _no_root


def format_tool_error(exc: Exception, root: Path | None = None) -> FailedText:
	"""Return the refusal text for `exc` with its explicit failure status.

	Absolute file paths are never echoed: they are shown relative to `root` (or the root set by
	`set_display_root`) when they lie inside it, otherwise as the bare file name.
	"""
	return FailedText(_format_tool_error(exc, root if root is not None else _display_root()))


def _shown_path(filename: object, root: Path | None) -> str:
	text = str(filename)
	path = Path(text)
	if not path.is_absolute():
		return text
	if root is not None:
		try:
			return str(path.relative_to(root))
		except ValueError:
			pass
	return path.name


def _format_tool_error(exc: Exception, root: Path | None) -> str:
	if isinstance(exc, LspRequestError):
		return f"LSP error on {exc.method}: {exc}"
	if isinstance(exc, TimeoutError):
		return "Language server timed out (it may still be indexing the workspace); retry shortly."
	if isinstance(exc, FileNotFoundError):
		if exc.filename:
			shown = _shown_path(exc.filename, root)
			return f"File not found: {shown} (relative paths resolve against the workspace root)."
		return f"Cannot start language server (file not found): {exc.strerror or exc}."
	if isinstance(exc, UnicodeDecodeError):
		return f"Cannot read file as UTF-8 text: {exc.reason}."
	if isinstance(exc, OSError):
		if exc.filename:
			return f"Cannot read {_shown_path(exc.filename, root)}: {exc.strerror or exc}."
		# Spawn failures (e.g. WinError 193 on an npm POSIX shim) often have no filename.
		return f"Cannot start language server: {exc.strerror or exc}."
	return str(exc)
