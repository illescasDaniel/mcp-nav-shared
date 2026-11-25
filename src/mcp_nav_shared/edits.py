"""Text-edit primitives for the write tools: LSP `TextEdit`/`WorkspaceEdit`
handling, whole-file changes, diffs and edit plans.

Everything here is pure (no language server, no disk writes except the plain
readers): a refactoring produces an `EditPlan`, the plan is simulated and
reported, and only then handed to `mcp_nav_shared.transaction` to be written.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mcp_nav_shared.errors import ToolInputError


class EditConflictError(ToolInputError):
	"""Edits that overlap, or an edit that doesn't fit the text it targets."""


class UnsupportedEditError(ToolInputError):
	"""A workspace edit uses a feature the write tools don't carry out (file operations)."""


_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_TRAILING_BREAK_RE = re.compile(r"(?:\r\n|\r|\n)$")
_BOM = "\ufeff"
_BOM_BYTES = _BOM.encode("utf-8")


@dataclass(frozen=True)
class TextEdit:
	"""One replacement, positioned like LSP: 0-based line, UTF-16 `character`."""

	start_line: int
	start_character: int
	end_line: int
	end_character: int
	new_text: str

	@classmethod
	def from_lsp(cls, raw: dict[str, Any]) -> TextEdit:
		rng = raw["range"]
		return cls(
			start_line=int(rng["start"]["line"]),
			start_character=int(rng["start"]["character"]),
			end_line=int(rng["end"]["line"]),
			end_character=int(rng["end"]["character"]),
			new_text=str(raw.get("newText", "")),
		)


def decode_source(raw: bytes) -> tuple[str, bool]:
	"""Text and whether it started with a UTF-8 BOM. Line endings are kept as they are."""
	text = raw.decode("utf-8")
	if text.startswith(_BOM):
		return text[1:], True
	return text, False


def encode_source(text: str, bom: bool = False) -> bytes:
	"""UTF-8 bytes of `text`, with a byte-order mark first when `bom` (never two of them)."""
	return ((_BOM if bom and not text.startswith(_BOM) else "") + text).encode("utf-8")


def read_source(path: Path) -> tuple[str, bool]:
	return decode_source(path.read_bytes())


def hash_bytes(raw: bytes) -> str:
	return hashlib.sha256(raw).hexdigest()


def detect_eol(text: str) -> str:
	"""`\\r\\n` when the text mostly uses it, else `\\n`."""
	crlf = text.count("\r\n")
	lf = text.count("\n") - crlf
	return "\r\n" if crlf > lf else "\n"


def split_lines(text: str, keepends: bool = False) -> list[str]:
	"""Lines of `text`, split ONLY on `\\r\\n`, `\\r` and `\\n`.

	Unlike `str.splitlines` this does not split on form feed, `\\x1c`-`\\x1e`, NEL (U+0085) or
	U+2028/U+2029, which Python source (and the language server) treat as ordinary characters.
	"""
	if keepends:
		pieces: list[str] = []
		cursor = 0
		for match in _LINE_BREAK_RE.finditer(text):
			pieces.append(text[cursor : match.end()])
			cursor = match.end()
		if cursor < len(text):
			pieces.append(text[cursor:])
		return pieces
	lines = _LINE_BREAK_RE.split(text)
	if lines and lines[-1] == "":
		lines.pop()
	return lines


def convert_eol(text: str, eol: str) -> str:
	"""`text` with every line ending (`\\r\\n`, `\\r`, `\\n`) replaced by `eol`."""
	return _LINE_BREAK_RE.sub(eol, text)


def _line_starts(text: str) -> list[int]:
	starts = [0]
	starts.extend(match.end() for match in _LINE_BREAK_RE.finditer(text))
	return starts


def _column_to_index(line_text: str, character: int) -> int:
	"""Index into `line_text` for a UTF-16 code-unit offset (clamped to the line)."""
	if character <= 0:
		return 0
	if line_text.isascii():
		return min(character, len(line_text))
	units = 0
	for index, char in enumerate(line_text):
		if units >= character:
			return index
		units += 2 if ord(char) > 0xFFFF else 1
	return len(line_text)


def utf16_length(text: str) -> int:
	return len(text.encode("utf-16-le")) // 2


def position_to_offset(text: str, line: int, character: int, starts: list[int] | None = None) -> int:
	starts = starts if starts is not None else _line_starts(text)
	if line >= len(starts):
		return len(text)
	line_start = starts[line]
	line_end = starts[line + 1] if line + 1 < len(starts) else len(text)
	content = _TRAILING_BREAK_RE.sub("", text[line_start:line_end])
	return line_start + _column_to_index(content, character)


def _bom_adjusted(edit: TextEdit) -> TextEdit:
	"""Translate positions measured in a document that includes a leading BOM."""
	if (edit.start_line == 0 and edit.start_character < 1) or (edit.end_line == 0 and edit.end_character < 1):
		raise EditConflictError("edit targets the byte-order mark; cannot safely apply it")
	return TextEdit(
		start_line=edit.start_line,
		start_character=edit.start_character - (edit.start_line == 0),
		end_line=edit.end_line,
		end_character=edit.end_character - (edit.end_line == 0),
		new_text=edit.new_text,
	)


def apply_text_edits(
	text: str, edits: list[TextEdit], *, bom: bool = False, bom_in_positions: bool | None = None
) -> str:
	"""Apply LSP edits against the original `text` (positions all refer to it).

	Overlapping edits raise `EditConflictError`; several insertions at one point
	keep their given order. Inserted text uses the file's own line endings.
	`text` has no byte-order mark; `bom=True` says the file it came from had one, so first-line
	first-line positions require an explicit `bom_in_positions` convention.
	"""
	if not edits:
		return text
	eol = detect_eol(text)
	starts = _line_starts(text)
	spans: list[tuple[int, int, int, str]] = []
	for order, edit in enumerate(edits):
		if bom and (edit.start_line == 0 or edit.end_line == 0):
			if bom_in_positions is None:
				raise EditConflictError("cannot safely apply a first-line edit: BOM coordinate convention is unknown")
			if bom_in_positions:
				edit = _bom_adjusted(edit)
		start = position_to_offset(text, edit.start_line, edit.start_character, starts)
		end = position_to_offset(text, edit.end_line, edit.end_character, starts)
		if end < start:
			raise EditConflictError(f"edit range ends before it starts (line {edit.start_line + 1})")
		spans.append((start, end, order, convert_eol(edit.new_text, eol)))
	spans.sort(key=lambda span: (span[0], span[1], span[2]))
	pieces: list[str] = []
	cursor = 0
	for start, end, _order, new_text in spans:
		if start < cursor:
			line = text.count("\n", 0, start) + 1
			raise EditConflictError(f"overlapping edits near line {line}")
		pieces.append(text[cursor:start])
		pieces.append(new_text)
		cursor = end
	pieces.append(text[cursor:])
	return "".join(pieces)


def replace_lines(text: str, first_line: int, last_line: int, replacement: str) -> str:
	"""Replace whole lines `first_line`..`last_line` (1-based, inclusive, line
	terminators included) with `replacement`, which should end with a newline
	unless it is empty."""
	starts = _line_starts(text)
	if first_line < 1 or last_line < first_line or first_line > len(starts):
		raise EditConflictError(f"line range {first_line}-{last_line} is outside the file")
	begin = starts[first_line - 1]
	end = starts[last_line] if last_line < len(starts) else len(text)
	eol = detect_eol(text)
	return text[:begin] + convert_eol(replacement, eol) + text[end:]


def uri_to_file_path(uri: str) -> Path:
	parsed = urlparse(uri)
	if parsed.scheme != "file":
		raise UnsupportedEditError(f"edit targets a non-file document: {uri}")
	path = unquote(parsed.path)
	if len(path) > 2 and path[0] == "/" and path[2] == ":":  # file:///C:/...
		path = path[1:]
	return Path(path)


def parse_workspace_edit(edit: dict[str, Any] | None) -> dict[Path, list[TextEdit]]:
	"""Text edits per file from an LSP `WorkspaceEdit` (`changes` or `documentChanges`).

	File create/rename/delete operations are refused: the plans built from
	language-server edits must stay plain text changes.
	"""
	by_path: dict[Path, list[TextEdit]] = {}
	if not edit:
		return by_path
	for uri, raw_edits in (edit.get("changes") or {}).items():
		by_path.setdefault(uri_to_file_path(uri), []).extend(TextEdit.from_lsp(raw) for raw in raw_edits)
	for change in edit.get("documentChanges") or []:
		if "kind" in change:
			raise UnsupportedEditError(
				f"the language server asked for a file operation ({change['kind']}), not supported"
			)
		path = uri_to_file_path(change["textDocument"]["uri"])
		by_path.setdefault(path, []).extend(TextEdit.from_lsp(raw) for raw in change.get("edits") or [])
	return by_path


def _file_has_bom(path: Path) -> bool:
	try:
		with path.open("rb") as handle:
			return handle.read(len(_BOM_BYTES)) == _BOM_BYTES
	except OSError:
		return False


@dataclass(frozen=True)
class FileChange:
	"""The whole-file effect of an edit: `old_text` None creates the file, `new_text` None deletes it.

	Texts never carry a byte-order mark: it is the flags' job. Left at `None`, `old_bom` is read from
	the file as it is on disk (so every write site keeps a file's BOM) and `new_bom` follows `old_bom`.
	A leading U+FEFF on `new_text` is dropped.
	"""

	path: Path
	old_text: str | None
	new_text: str | None
	old_bom: bool | None = None
	new_bom: bool | None = None
	# Permission bits (`st_mode`). Left at `None`, `old_mode` is read from the file as it is on disk, and a
	# written file keeps the mode of the file it replaces; `new_mode` is for a file this change creates
	# (e.g. the destination of a move takes its source's mode, so +x scripts stay executable).
	old_mode: int | None = field(default=None, kw_only=True)
	new_mode: int | None = field(default=None, kw_only=True)

	def __post_init__(self) -> None:
		old_bom = self.old_bom
		if self.old_text is not None and self.old_text.startswith(_BOM):
			object.__setattr__(self, "old_text", self.old_text[1:])
			old_bom = True
		elif old_bom is None:
			old_bom = self.old_text is not None and _file_has_bom(self.path)
		object.__setattr__(self, "old_bom", old_bom)
		if self.new_bom is None:
			object.__setattr__(self, "new_bom", old_bom)
		if self.new_text is not None and self.new_text.startswith(_BOM):
			object.__setattr__(self, "new_text", self.new_text[1:])
		if self.old_mode is None and self.old_text is not None and self.path.exists():
			object.__setattr__(self, "old_mode", stat.S_IMODE(self.path.stat().st_mode))

	@property
	def bom(self) -> bool:
		"""Whether the file had a byte-order mark as read."""
		return bool(self.old_bom)

	@property
	def kind(self) -> str:
		if self.old_text is None:
			return "create"
		if self.new_text is None:
			return "delete"
		return "modify"

	@property
	def is_noop(self) -> bool:
		return self.old_text == self.new_text and self.old_bom == self.new_bom

	def old_bytes(self) -> bytes | None:
		return None if self.old_text is None else encode_source(self.old_text, bool(self.old_bom))

	def new_bytes(self) -> bytes | None:
		return None if self.new_text is None else encode_source(self.new_text, bool(self.new_bom))

	def inverse(self) -> FileChange:
		return FileChange(
			self.path,
			self.new_text,
			self.old_text,
			self.new_bom,
			self.old_bom,
			old_mode=self.new_mode,
			new_mode=self.old_mode,
		)


def changes_from_edits(
	edits_by_path: dict[Path, list[TextEdit]], *, bom_in_positions: bool | dict[Path, bool | None] | None = None
) -> list[FileChange]:
	"""Apply per-file edits to what is on disk now. No-op results are dropped."""
	changes: list[FileChange] = []
	for path in sorted(edits_by_path):
		old_text, bom = read_source(path)
		convention = bom_in_positions.get(path.resolve()) if isinstance(bom_in_positions, dict) else bom_in_positions
		new_text = apply_text_edits(old_text, edits_by_path[path], bom=bom, bom_in_positions=convention)
		change = FileChange(path, old_text, new_text, bom, bom)
		if not change.is_noop:
			changes.append(change)
	return changes


def relative_name(path: Path, root: Path) -> str:
	try:
		return path.relative_to(root).as_posix()
	except ValueError:
		try:
			return path.resolve().relative_to(root.resolve()).as_posix()
		except ValueError:
			return path.as_posix()


def unified_diff(change: FileChange, root: Path, *, context: int = 2) -> str:
	name = relative_name(change.path, root)
	old = split_lines(change.old_text or "", keepends=True)
	new = split_lines(change.new_text or "", keepends=True)
	label_old = "/dev/null" if change.old_text is None else f"a/{name}"
	label_new = "/dev/null" if change.new_text is None else f"b/{name}"
	lines = list(difflib.unified_diff(old, new, label_old, label_new, n=context))
	return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines)


def count_changed_lines(change: FileChange) -> tuple[int, int]:
	"""(added, removed) line counts."""
	old = split_lines(change.old_text or "")
	new = split_lines(change.new_text or "")
	added = removed = 0
	for opcode, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
		if opcode in ("replace", "delete"):
			removed += i2 - i1
		if opcode in ("replace", "insert"):
			added += j2 - j1
	return added, removed


@dataclass
class EditPlan:
	"""A set of whole-file changes that is previewed first and written later.

	`base_hashes` remembers the bytes each file had when the plan was made, so
	applying it can refuse when something else changed a file in between.
	"""

	description: str
	changes: list[FileChange]
	notes: list[str] = field(default_factory=list)
	base_hashes: dict[Path, str | None] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.base_hashes:
			for change in self.changes:
				old = change.old_bytes()
				self.base_hashes[change.path] = None if old is None else hash_bytes(old)

	@property
	def paths(self) -> list[Path]:
		return [change.path for change in self.changes]

	@property
	def is_empty(self) -> bool:
		return not self.changes

	def inverse(self) -> EditPlan:
		return EditPlan(f"undo: {self.description}", [change.inverse() for change in reversed(self.changes)])

	def summary(self, root: Path, *, limit: int = 60) -> str:
		"""One line per touched file; past `limit` files the rest is only counted."""
		lines = []
		for change in self.changes[:limit]:
			added, removed = count_changed_lines(change)
			name = relative_name(change.path, root)
			tag = {"create": " (new file)", "delete": " (deleted)"}.get(change.kind, "")
			lines.append(f"  {name}{tag}: +{added} -{removed}")
		if len(self.changes) > limit:
			lines.append(f"  ... and {len(self.changes) - limit} more files")
		return "\n".join(lines)

	def diff(self, root: Path, *, context: int = 2, max_lines: int = 300) -> str:
		text = "".join(unified_diff(change, root, context=context) for change in self.changes)
		lines = split_lines(text)
		if len(lines) <= max_lines:
			return text
		return "\n".join(lines[:max_lines]) + f"\n... diff truncated ({len(lines) - max_lines} more lines)\n"
