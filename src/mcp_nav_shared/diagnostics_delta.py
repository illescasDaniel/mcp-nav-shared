"""What an edit does to the diagnostics: new problems, fixed ones, the rest.

Diagnostics are compared by what they say and which source line they sit on,
not by line number, so an edit that shifts a file doesn't turn every existing
problem into "one fixed, one new".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


_SEVERITY_LABELS = {1: "error", 2: "warning", 3: "info", 4: "hint"}
ERROR_SEVERITY = 1
HINT_SEVERITY = 4


@dataclass(frozen=True)
class DiagEntry:
	path: str  # workspace-relative
	line: int  # 1-based
	column: int  # 1-based
	severity: int
	code: str
	message: str
	source_line: str  # stripped text of the line, part of the identity

	@property
	def is_error(self) -> bool:
		return self.severity == ERROR_SEVERITY

	@property
	def strict_key(self) -> tuple[str, str, int, str, str]:
		return (self.path, self.code, self.severity, self.message, self.source_line)

	@property
	def loose_key(self) -> tuple[str, str, int, str]:
		return (self.path, self.code, self.severity, self.message)

	def render(self) -> str:
		label = _SEVERITY_LABELS.get(self.severity, "error")
		code = f" [{self.code}]" if self.code else ""
		return f"{self.path}:{self.line}:{self.column} {label}{code}: {self.message.splitlines()[0] if self.message else ''}"


def entries_from_lsp(rel_path: str, text: str, items: list[dict[str, Any]]) -> list[DiagEntry]:
	lines = text.splitlines()
	entries = []
	for item in items:
		start = (item.get("range") or {}).get("start") or {}
		line0 = int(start.get("line", 0))
		source_line = lines[line0].strip() if 0 <= line0 < len(lines) else ""
		code = item.get("code")
		entries.append(
			DiagEntry(
				path=rel_path,
				line=line0 + 1,
				column=int(start.get("character", 0)) + 1,
				severity=int(item.get("severity") or ERROR_SEVERITY),
				code="" if code is None else str(code),
				message=str(item.get("message", "")),
				source_line=source_line,
			)
		)
	return entries


@dataclass
class DiagnosticsDelta:
	new: list[DiagEntry] = field(default_factory=list)
	fixed: list[DiagEntry] = field(default_factory=list)
	unchanged: int = 0
	# Files whose diagnostics were compared, for the report.
	files_checked: int = 0
	# Files that could not be checked (read/LSP failure), so the comparison is partial.
	unchecked: list[str] = field(default_factory=list)

	@property
	def new_errors(self) -> list[DiagEntry]:
		return [entry for entry in self.new if entry.is_error]

	@property
	def fixed_errors(self) -> list[DiagEntry]:
		return [entry for entry in self.fixed if entry.is_error]

	@property
	def is_clean(self) -> bool:
		return not self.new_errors


def _subtract(left: list[DiagEntry], right: list[DiagEntry], key: str) -> tuple[list[DiagEntry], list[DiagEntry]]:
	"""(left entries without a partner in right, right entries without a partner in left)."""
	pool: dict[tuple, list[DiagEntry]] = defaultdict(list)
	for entry in right:
		pool[getattr(entry, key)].append(entry)
	unmatched_left = []
	for entry in left:
		partners = pool.get(getattr(entry, key))
		if partners:
			partners.pop()
		else:
			unmatched_left.append(entry)
	unmatched_right = [entry for partners in pool.values() for entry in partners]
	return unmatched_left, unmatched_right


def diff_diagnostics(before: list[DiagEntry], after: list[DiagEntry]) -> DiagnosticsDelta:
	"""Pair identical diagnostics (same message on the same source line), then pair
	what is left by message alone (the same problem on an edited line)."""
	new, fixed = _subtract(after, before, "strict_key")
	new, fixed = _subtract(new, fixed, "loose_key")
	# `_subtract` returns the right-hand leftovers in dict order; keep reports stable.
	new.sort(key=lambda entry: (entry.path, entry.line, entry.column))
	fixed.sort(key=lambda entry: (entry.path, entry.line, entry.column))
	unchanged = len(after) - len(new)
	return DiagnosticsDelta(new=new, fixed=fixed, unchanged=unchanged)


def format_delta(delta: DiagnosticsDelta, *, limit: int = 15) -> str:
	"""Short report: verdict line, then the new problems, then what got fixed."""
	new_errors, fixed_errors = len(delta.new_errors), len(delta.fixed_errors)
	new_other = len(delta.new) - new_errors
	headline = f"Diagnostics: {new_errors} new error(s), {fixed_errors} fixed"
	if new_other:
		headline += f", {new_other} new warning(s)/hint(s)"
	headline += f" ({delta.files_checked} file(s) checked, {delta.unchanged} unchanged)"
	lines = [headline]
	if delta.unchecked:
		lines.append(f"  not checked: {', '.join(delta.unchecked[:5])}")
	shown = delta.new[:limit]
	if shown:
		lines.append("New:")
		lines.extend(f"  + {entry.render()}" for entry in shown)
		if len(delta.new) > limit:
			lines.append(f"  ... {len(delta.new) - limit} more")
	shown_fixed = delta.fixed[: max(3, limit // 3)]
	if shown_fixed:
		lines.append("Fixed:")
		lines.extend(f"  - {entry.render()}" for entry in shown_fixed)
		if len(delta.fixed) > len(shown_fixed):
			lines.append(f"  ... {len(delta.fixed) - len(shown_fixed)} more")
	return "\n".join(lines)
