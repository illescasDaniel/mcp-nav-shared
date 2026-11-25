"""Bound text returned by tools."""

from __future__ import annotations

import os


DEFAULT_MAX_OUTPUT_CHARS = 30_000
OMITTED_FOOTER = (
	"... {lines} more lines ({chars} characters) omitted. Narrow with path=, kind=, limit= or a more specific name."
)


def max_output_chars(variable: str = "CODENAV_MCP_MAX_OUTPUT_CHARS") -> int:
	value = os.environ.get(variable, "")
	try:
		number = int(value)
	except ValueError:
		return DEFAULT_MAX_OUTPUT_CHARS
	return max(number, 2000) if number > 0 else DEFAULT_MAX_OUTPUT_CHARS


def cap_output(text: str, limit: int | None = None) -> str:
	if limit is None:
		limit = max_output_chars()
	limit = max(0, limit)
	if len(text) <= limit:
		return text
	# Footer counts depend on the cut, so converge as the number of digits changes.
	cut = limit
	while True:
		removed = text[cut:]
		footer = OMITTED_FOOTER.format(lines=max(1, len(removed.split("\n"))), chars=len(removed))
		if len(footer) >= limit:
			return footer[:limit]
		available = max(0, limit - len(footer) - 1)
		boundary = text.rfind("\n", 0, available + 1)
		new_cut = boundary if boundary >= 0 else available
		while new_cut > 0 and text[new_cut - 1] == "\n":
			new_cut -= 1
		if new_cut == cut:
			break
		cut = new_cut
	return text[:cut] + "\n" + footer
