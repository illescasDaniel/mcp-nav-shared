"""Expected tool failures for codenav/webnav, rendered as agent-readable text.

The MCP framework collapses any exception a tool raises into an opaque
"Error executing tool …", hiding actionable causes like a mistyped path.
Tools catch `TOOL_ERRORS` and return `format_tool_error(exc)` instead.
"""

from __future__ import annotations

from mcp_nav_shared.lsp_client import InvalidPositionError, LanguageServerExitedError, LspRequestError


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


def format_tool_error(exc: Exception) -> str:
	if isinstance(exc, LspRequestError):
		return f"LSP error on {exc.method}: {exc}"
	if isinstance(exc, TimeoutError):
		return "Language server timed out (it may still be indexing the workspace); retry shortly."
	if isinstance(exc, FileNotFoundError):
		if exc.filename:
			return f"File not found: {exc.filename} (relative paths resolve against the workspace root)."
		return f"Cannot start language server (file not found): {exc.strerror or exc}."
	if isinstance(exc, UnicodeDecodeError):
		return f"Cannot read file as UTF-8 text: {exc.reason}."
	if isinstance(exc, OSError):
		if exc.filename:
			return f"Cannot read {exc.filename}: {exc.strerror or exc}."
		# Spawn failures (e.g. WinError 193 on an npm POSIX shim) often have no filename.
		return f"Cannot start language server: {exc.strerror or exc}."
	return str(exc)
