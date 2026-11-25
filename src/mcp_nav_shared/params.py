"""Param-name aliases for MCP tools — agents often guess `query` vs `name`.

Returning a `ToolInputError` (caught and rendered as tool text) beats the
framework's opaque pydantic "Field required" wall, which historically made
agents abandon the MCP after one wrong guess.
"""

from __future__ import annotations

from mcp_nav_shared.errors import ToolInputError


def resolve_name_query(
	*,
	preferred: str = "name",
	example: str = "Foo",
	**params: str | None,
) -> str:
	"""Return the first non-empty among `preferred` then the other kwargs.

	Example::

		resolve_name_query(
			preferred="name",
			example="UserService.create_user",
			name=name,
			query=query,
		)
		resolve_name_query(
			preferred="query",
			example="create_user",
			query=query,
			name=name,
		)
		resolve_name_query(
			preferred="port_name",
			example="FileSystemPort",
			port_name=port_name,
			name=name,
			query=query,
		)
	"""
	if preferred not in params:
		params = {preferred: None, **params}
	given = [key for key, val in params.items() if val]
	if len({params[key] for key in given}) > 1:
		# Two different names for one call: refuse rather than silently pick one.
		raise ToolInputError(f"give either {' or '.join(given)}, not both")
	value = params.get(preferred)
	if not value:
		for key, val in params.items():
			if key != preferred and val:
				value = val
				break
	if all(val is None for val in params.values()):
		others = [key for key in params if key != preferred]
		if not others:
			raise ToolInputError(f"Pass `{preferred}` (e.g. {preferred}={example!r}).")
		if len(others) == 1:
			alias_msg = f"`{others[0]}` is accepted as an alias"
		else:
			alias_msg = "aliases accepted: " + ", ".join(f"`{key}`" for key in others)
		raise ToolInputError(f"Pass `{preferred}` (e.g. {preferred}={example!r}); {alias_msg}.")
	if not value or not _is_name_like(value):
		raise ToolInputError("name must be a non-empty identifier or dotted path")
	return value


def _is_name_like(value: str) -> bool:
	"""False for text no symbol name can contain: NUL, blank, or only dots and blanks."""
	return "\x00" not in value and value.strip(" \t\r\n.") != ""
