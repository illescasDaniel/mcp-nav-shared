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
	value = params.get(preferred)
	if not value:
		for key, val in params.items():
			if key != preferred and val:
				value = val
				break
	if not value:
		others = [key for key in params if key != preferred]
		if not others:
			raise ToolInputError(f"Pass `{preferred}` (e.g. {preferred}={example!r}).")
		if len(others) == 1:
			alias_msg = f"`{others[0]}` is accepted as an alias"
		else:
			alias_msg = "aliases accepted: " + ", ".join(f"`{key}`" for key in others)
		raise ToolInputError(f"Pass `{preferred}` (e.g. {preferred}={example!r}); {alias_msg}.")
	return value
