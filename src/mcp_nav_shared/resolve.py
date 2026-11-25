"""Name-based symbol resolution shared by codenav/webnav composite tools
(`symbol_info`, `callers`, `implementations`): a `workspace_symbol` lookup
with tiered ranking, dotted `Class.method` resolution via `documentSymbol`,
and disambiguation by `file_path` — so those tools take a name instead of a
hand-computed position.
"""

from __future__ import annotations

import ast
import difflib
import functools
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from mcp_nav_shared.errors import TOOL_ERRORS, ToolInputError
from mcp_nav_shared.format import (
	_match_tier,
	filter_workspace_symbols,
	is_hierarchical_document_symbols,
	rank_workspace_symbols,
	symbol_kind_label,
	to_symbol_tree,
	uri_to_path,
	uri_to_relative,
	workspace_symbol_position,
)
from mcp_nav_shared.lsp_client import LspClient


# LSP SymbolKind values that are class members, not top-level symbols — a
# candidate with one of these kinds gets qualified as `Class.member` in an
# ambiguity listing (see `_qualify_candidates`) since the bare name alone
# doesn't say which container it belongs to.
_MEMBER_KINDS = {6, 7, 9}  # Method, Property, Constructor

# How many near-miss names an unknown-name refusal suggests.
_NEAR_NAME_COUNT = 3

_T = TypeVar("_T")


class SymbolResolutionError(ToolInputError):
	"""No single confident match for a name-based symbol query: not found, or ambiguous."""


class SymbolNotFoundError(SymbolResolutionError):
	"""Nothing in the workspace matches the name (ambiguity is a plain `SymbolResolutionError`)."""


@dataclass
class ResolvedSymbol:
	name: str
	kind: Any
	uri: str
	line: int  # 0-based, aimed at the identifier
	column: int  # 0-based
	range_start_line: int  # 0-based, the symbol's full span (for dotted member lookup)
	range_end_line: int


def _nfkc(text: str) -> str:
	"""Python identifiers are NFKC-normalised by the parser (`ﬁ` is `fi`), so
	name comparisons happen on the normalised form."""
	return unicodedata.normalize("NFKC", text)


def _resolved_from_symbol(sym: dict[str, Any]) -> ResolvedSymbol:
	uri, line, column = workspace_symbol_position(sym)
	rng = (sym.get("location") or {}).get("range") or {}
	start_line = int((rng.get("start") or {}).get("line", line))
	end_line = int((rng.get("end") or {}).get("line", start_line))
	return ResolvedSymbol(
		name=str(sym.get("name") or "?"),
		kind=sym.get("kind"),
		uri=uri,
		line=line,
		column=column,
		range_start_line=start_line,
		range_end_line=end_line,
	)


def _member_start_line(sym: dict[str, Any]) -> int:
	rng = (sym.get("location") or {}).get("range") or {}
	return int((rng.get("start") or {}).get("line", 0))


def _member_span(sym: dict[str, Any]) -> tuple[int, int]:
	rng = (sym.get("location") or {}).get("range") or {}
	start = int((rng.get("start") or {}).get("line", 0))
	return start, int((rng.get("end") or {}).get("line", start))


# -- overloads -----------------------------------------------------------------
#
# ty lists every `@overload` stub next to the implementation, as separate
# symbols with the same name. A plain name lookup must not stop at the first
# stub (its body is `...`, so it is no place to edit or read the function).


def _is_overload_decorator(node: ast.expr) -> bool:
	return (isinstance(node, ast.Name) and node.id == "overload") or (
		isinstance(node, ast.Attribute) and node.attr == "overload"
	)


@functools.lru_cache(maxsize=256)
def _overload_name_lines_cached(path: str, mtime_ns: int, size: int) -> frozenset[int]:
	try:
		tree = ast.parse(Path(path).read_text(encoding="utf-8"))
	except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
		return frozenset()
	return frozenset(
		node.lineno - 1
		for node in ast.walk(tree)
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
		and any(_is_overload_decorator(dec) for dec in node.decorator_list)
	)


def _is_overload_at(uri: str, line: int) -> bool:
	"""True when the `def` whose name sits on 0-based `line` of `uri` is decorated with `@overload`."""
	path = uri_to_path(uri)
	try:
		stat = path.stat()
	except OSError:
		return False
	return line in _overload_name_lines_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _without_overload_stubs(items: list[_T], positions: list[tuple[str, int]]) -> list[_T]:
	"""Drop `@overload` stubs when the same file also has the implementation.
	A file whose definitions are all stubs keeps only its first one."""
	flags = [_is_overload_at(uri, line) for uri, line in positions]
	impl_files = {uri for (uri, _), stub in zip(positions, flags, strict=True) if not stub}
	kept: list[_T] = []
	stub_files_kept: set[str] = set()
	for item, (uri, _), stub in zip(items, positions, flags, strict=True):
		if not stub:
			kept.append(item)
		elif uri in impl_files:
			continue  # the file's implementation stands in for its stubs
		elif uri not in stub_files_kept:
			stub_files_kept.add(uri)
			kept.append(item)
	return kept


def _selection_line(node: dict[str, Any]) -> int:
	rng = node.get("selectionRange") or node.get("range") or {}
	return int((rng.get("start") or {}).get("line", -1))


def _pick_member(children: list[dict[str, Any]], name: str, uri: str) -> dict[str, Any] | None:
	matches = [child for child in children if _nfkc(str(child.get("name"))) == name]
	if len(matches) <= 1:
		return matches[0] if matches else None
	return _without_overload_stubs(matches, [(uri, _selection_line(child)) for child in matches])[0]


def _find_named_node(symbols: list[dict[str, Any]], name: str, line: int) -> dict[str, Any] | None:
	"""Hierarchical `DocumentSymbol` named `name` (any depth), preferring the
	one whose identifier sits on `line` so same-named classes don't get mixed up."""
	found: list[dict[str, Any]] = []

	def walk(nodes: list[dict[str, Any]]) -> None:
		for node in nodes:
			if str(node.get("name")) == name:
				found.append(node)
			walk(node.get("children") or [])

	walk(symbols)
	for node in found:
		sel = (node.get("selectionRange") or node.get("range") or {}).get("start") or {}
		if int(sel.get("line", -1)) == line:
			return node
	return found[0] if found else None


def _find_member_node(symbols: list[dict[str, Any]], container_name: str, member_name: str) -> dict[str, Any] | None:
	"""Search hierarchical `DocumentSymbol` results for `member_name` directly
	under a node named `container_name` (searches nested classes too, in case
	`container_name` isn't top-level)."""
	for node in symbols:
		if str(node.get("name")) == container_name:
			for child in node.get("children") or []:
				if str(child.get("name")) == member_name:
					return child
		found = _find_member_node(node.get("children") or [], container_name, member_name)
		if found is not None:
			return found
	return None


def _resolved_from_hierarchical_node(member_name: str, node: dict[str, Any], uri: str) -> ResolvedSymbol:
	sel_start = (node.get("selectionRange") or {}).get("start") or (node.get("range") or {}).get("start") or {}
	rng = node.get("range") or {}
	start_line = int((rng.get("start") or {}).get("line", 0))
	end_line = int((rng.get("end") or {}).get("line", start_line))
	return ResolvedSymbol(
		name=member_name,
		kind=node.get("kind"),
		uri=uri,
		line=int(sel_start.get("line", start_line)),
		column=int(sel_start.get("character", 0)),
		range_start_line=start_line,
		range_end_line=end_line,
	)


def _enclosing_class_name(nodes: list[dict[str, Any]], target_line: int) -> str | None:
	"""Walk a `to_symbol_tree` hierarchy for the Class(5) node whose range
	contains `target_line`, preferring the innermost match (nested classes)."""
	for node in nodes:
		start, end = node.get("start_line"), node.get("end_line")
		if start is None or end is None or not (start <= target_line <= end):
			continue
		inner = _enclosing_class_name(node.get("children") or [], target_line)
		if inner is not None:
			return inner
		return str(node.get("name")) if node.get("kind") == 5 else None
	return None


async def _qualify_candidates(
	client: LspClient, workspace_root: Path, candidates: list[dict[str, Any]]
) -> dict[int, str]:
	"""For Method/Property/Constructor-kind candidates, resolve their
	enclosing class so an ambiguity listing can show `Class.member` — ty's
	`workspace/symbol` results carry no `containerName`, so this costs one
	`document_symbol` call per distinct candidate file."""
	qualified: dict[int, str] = {}
	trees: dict[str, list[dict[str, Any]]] = {}
	for i, sym in enumerate(candidates):
		if sym.get("kind") not in _MEMBER_KINDS:
			continue
		uri = (sym.get("location") or {}).get("uri", "")
		rel = uri_to_relative(uri, workspace_root)
		if rel not in trees:
			try:
				members = await client.document_symbol(rel)
			except TOOL_ERRORS:
				trees[rel] = []
			else:
				trees[rel] = to_symbol_tree(members)
		cls_name = _enclosing_class_name(trees[rel], _member_start_line(sym))
		if cls_name:
			qualified[i] = cls_name
	return qualified


async def _format_candidates(client: LspClient, candidates: list[dict[str, Any]], workspace_root: Path) -> str:
	shown = candidates[:10]
	qualified = await _qualify_candidates(client, workspace_root, shown)
	lines = []
	for i, sym in enumerate(shown):
		name = str(sym.get("name") or "?")
		qualifier = qualified.get(i)
		display = f"{qualifier}.{name}" if qualifier else name
		kind = symbol_kind_label(sym.get("kind"))
		uri, line, col = workspace_symbol_position(sym)
		rel = uri_to_relative(uri, workspace_root) if uri else "?"
		lines.append(f"{display}  [{kind}]  ({rel}:{line + 1}:{col + 1})")
	return "\n".join(lines)


def _short_text(text: str, limit: int = 40) -> str:
	"""`repr` of `text`, cut to `limit` characters, so a long input is never echoed back whole."""
	shown = repr(text)
	return shown if len(shown) <= limit else f"{shown[: limit - 3]}..."


def symbol_name_problem(query: str) -> str | None:
	"""Why `query` cannot name a symbol, or None: every dot-separated part must be a Python identifier
	(`Cart.total`, `pkg.mod.func`, `café`). `search_symbol` takes free text and does not use this check."""
	if all(part.isidentifier() for part in query.split(".")):
		return None
	return (
		f"{_short_text(query)} is not a symbol name: use an identifier such as total, or a dotted name such as "
		"Cart.total or pkg.module.func (letters, digits and _ only)."
	)


def declares_type_parameter(path: Path, name: str) -> bool:
	"""True when the Python file at `path` declares `name` as a PEP 695 type parameter (`class Box[T]`,
	`def f[T]()`, `type Alias[T] = ...`). Unreadable or unparsable files answer False."""
	try:
		tree = ast.parse(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
		return False
	for node in ast.walk(tree):
		for param in getattr(node, "type_params", None) or []:
			if getattr(param, "name", None) == name:
				return True
	return False


def _not_found(query: str, near: list[str] | None = None) -> SymbolNotFoundError:
	message = f"No symbol found matching {query!r}."
	if near:
		message += f" Did you mean: {', '.join(near)}?"
	return SymbolNotFoundError(message)


def _near_names(query: str, symbols: list[dict[str, Any]]) -> list[str]:
	"""Up to three distinct names the lookup returned as fuzzy hits, closest first."""
	names = sorted({str(sym.get("name") or "") for sym in symbols} - {""})
	return difflib.get_close_matches(_nfkc(query), names, n=_NEAR_NAME_COUNT)


async def _ambiguous(
	client: LspClient,
	query: str,
	candidates: list[dict[str, Any]],
	workspace_root: Path,
) -> SymbolResolutionError:
	same_file = len({(c.get("location") or {}).get("uri", "") for c in candidates}) == 1
	if same_file:
		# Narrowing by file_path (or the candidates simply all live in one
		# file already) still leaves several same-named symbols there — e.g.
		# a module-level function and a same-named class method — so
		# file_path alone can't disambiguate further. Position tools take no
		# line/column in move/change_signature, so point at the qualified name.
		hint = "these all live in the same file; give file_path= and the qualified name (Class.member) instead"
	else:
		hint = "pass file_path to disambiguate"
	return SymbolResolutionError(
		f"{len(candidates)} symbols match {query!r}; {hint}:\n"
		f"{await _format_candidates(client, candidates, workspace_root)}"
	)


def _matches_file(sym: dict[str, Any], workspace_root: Path, file_path: str) -> bool:
	rel = uri_to_relative((sym.get("location") or {}).get("uri", ""), workspace_root)
	target = str(Path(file_path)).replace("\\", "/")
	return rel == target or rel.endswith("/" + target) or target.endswith("/" + rel)


async def _exact_candidates(
	client: LspClient, workspace_root: Path, name: str, *, file_path: str | None
) -> tuple[list[dict[str, Any]], list[str]]:
	"""Symbols named `name` (NFKC-compared), plus near-miss names for a refusal.
	The query is normalised before ty sees it, so `ﬁ` searches for `fi`."""
	query = _nfkc(name)
	symbols = await client.workspace_symbol(query)
	# Same filter as search_symbol: drop export-list Variable twins and
	# identical (name, kind, file) dupes so `symbol_info("foo")` isn't
	# ambiguous between `function foo` and `export { foo }`.
	ranked = filter_workspace_symbols(rank_workspace_symbols(symbols, query))
	exact = [s for s in ranked if _match_tier(_nfkc(str(s.get("name") or "")), query) <= 1]
	# Prefer a case-exact match (tier 0) over a merely case-insensitive one
	# (tier 1) when both exist, e.g. `repo_root` vs. a `REPO_ROOT` constant
	# elsewhere in the workspace — an agent asking for the lowercase name
	# almost always means the exact symbol, not an unrelated same-letters one.
	case_exact = [s for s in exact if _match_tier(_nfkc(str(s.get("name") or "")), query) == 0]
	if case_exact:
		exact = case_exact
	if not exact:
		return [], _near_names(query, symbols)
	if file_path is not None:
		narrowed = [s for s in exact if _matches_file(s, workspace_root, file_path)]
		if exact and not narrowed:
			# Silently answering with a symbol from a different file than the
			# one the caller named would be a confidently wrong result.
			raise SymbolResolutionError(
				f"No symbol {name!r} in {file_path!r}; {len(exact)} match(es) elsewhere:\n"
				f"{await _format_candidates(client, exact, workspace_root)}"
			)
		exact = narrowed
	return exact, []


async def _resolve_simple(
	client: LspClient, workspace_root: Path, query: str, *, file_path: str | None
) -> ResolvedSymbol:
	exact, near = await _exact_candidates(client, workspace_root, query, file_path=file_path)
	if not exact:
		raise _not_found(query, near)
	exact = _without_overload_stubs(exact, [workspace_symbol_position(s)[:2] for s in exact])
	if len(exact) > 1:
		raise await _ambiguous(client, query, exact, workspace_root)
	return _resolved_from_symbol(exact[0])


async def _resolve_dotted(
	client: LspClient, workspace_root: Path, query: str, *, file_path: str | None
) -> ResolvedSymbol:
	# `Outer.Inner.method`: the first segment is looked up workspace-wide, the
	# rest are walked down through that symbol's document-symbol tree.
	first, *path = (_nfkc(part) for part in query.split("."))
	member_name = path[-1]
	try:
		container = await _resolve_simple(client, workspace_root, first, file_path=file_path)
	except SymbolNotFoundError:
		if query.count(".") >= 2:
			# `pkg.mod.func` where `pkg` is no symbol: most likely a module path.
			module = query.rsplit(".", 1)[0]
			raise SymbolNotFoundError(
				f"No symbol found matching {query!r}; {module} looks like a module path; "
				"give file_path= and the bare name."
			) from None
		raise
	rel_path = uri_to_relative(container.uri, workspace_root)
	members = await client.document_symbol(rel_path)
	if is_hierarchical_document_symbols(members):
		found = _find_named_node(members, container.name, container.line)
		for part in path:
			found = _pick_member((found or {}).get("children") or [], part, container.uri)
		if found is not None:
			return _resolved_from_hierarchical_node(member_name, found, container.uri)
	else:
		# Flat `SymbolInformation`: no nesting, so narrow by range containment
		# one segment at a time.
		span = (container.range_start_line, container.range_end_line)
		match: dict[str, Any] | None = None
		for part in path:
			candidates = [
				m
				for m in members
				if _nfkc(str(m.get("name") or "")) == part
				and span[0] <= _member_start_line(m) <= span[1]
				and _member_span(m) != span
			]
			if len(candidates) > 1:
				candidates = _without_overload_stubs(candidates, [workspace_symbol_position(m)[:2] for m in candidates])
			if len(candidates) > 1:
				raise await _ambiguous(client, query, candidates, workspace_root)
			if not candidates:
				match = None
				break
			match = candidates[0]
			span = _member_span(match)
		if match is not None:
			return _resolved_from_symbol(match)
	# Inherited members only make sense for a plain `Class.member` query.
	inherited = await _resolve_inherited(client, workspace_root, container, member_name) if len(path) == 1 else None
	if inherited is None:
		raise _not_found(query)
	return inherited


async def _resolve_module_path(client: LspClient, workspace_root: Path, query: str) -> ResolvedSymbol | None:
	"""`pkg.mod.func` (or `pkg.mod.Class.method`): the longest leading part that
	names a module file (`pkg/mod.py` or `pkg/mod/__init__.py`) wins over the
	class-qualifier reading. `None` when no module file defines the rest."""
	parts = query.split(".")
	for cut in range(len(parts) - 1, 0, -1):
		head = _nfkc(parts[cut])
		module_path = "/".join(parts[:cut])
		suffixes = (module_path + ".py", module_path + "/__init__.py")
		symbols = await client.workspace_symbol(head)
		files = [
			rel
			for sym in symbols
			if _nfkc(str(sym.get("name") or "")) == head
			for rel in [uri_to_relative((sym.get("location") or {}).get("uri", ""), workspace_root)]
			if any(rel == suffix or rel.endswith("/" + suffix) for suffix in suffixes)
		]
		if not files:
			continue
		try:
			return await resolve_symbol(client, workspace_root, ".".join(parts[cut:]), file_path=files[0])
		except SymbolNotFoundError:
			continue
	return None


# Guards against pathological (or cyclic, if a server reports one) hierarchies;
# real class graphs are far smaller.
_MAX_SUPERTYPES_VISITED = 64


async def _resolve_inherited(
	client: LspClient, workspace_root: Path, container: ResolvedSymbol, member_name: str
) -> ResolvedSymbol | None:
	"""`Class.member` where `member` is inherited — e.g. an `AppServices`
	composed from mixins, the typed dependency call sites actually see.
	Walks `typeHierarchy/supertypes` breadth-first (MRO-like for the common
	case) and returns the first supertype that declares `member_name` directly.
	Needs hierarchical `documentSymbol` (a supertype item's range doesn't span
	its body, so the flat-shape range match can't be used), and returns `None`
	when the server doesn't support type hierarchy."""
	rel_path = uri_to_relative(container.uri, workspace_root)
	try:
		queue = list(await client.prepare_type_hierarchy(rel_path, container.line + 1, container.column + 1))
	except TOOL_ERRORS:
		return None
	seen: set[tuple[str, str]] = set()
	while queue and len(seen) < _MAX_SUPERTYPES_VISITED:
		item = queue.pop(0)
		try:
			supers = await client.supertypes(item)
		except TOOL_ERRORS:
			continue
		for sup in supers:
			uri, name = str(sup.get("uri") or ""), str(sup.get("name") or "")
			if (uri, name) in seen:
				continue
			seen.add((uri, name))
			queue.append(sup)
			try:
				members = await client.document_symbol(uri_to_relative(uri, workspace_root))
			except TOOL_ERRORS:
				# e.g. a stdlib/vendored base the server reports by a non-file URI.
				continue
			if not is_hierarchical_document_symbols(members):
				continue
			node = _find_member_node(members, name, member_name)
			if node is not None:
				return _resolved_from_hierarchical_node(member_name, node, uri)
	return None


async def resolve_symbol(
	client: LspClient, workspace_root: Path, query: str, *, file_path: str | None = None
) -> ResolvedSymbol:
	"""Resolve a name (or dotted `Class.method`, or `pkg.mod.name`) to a single symbol position.

	Raises `SymbolResolutionError` (a `ToolInputError`, so tools' existing
	`except TOOL_ERRORS` handles it) when nothing matches or several symbols
	tie on an exact name — callers should pass `file_path` to disambiguate
	rather than guess. `SymbolNotFoundError` is the no-match subclass.
	"""
	problem = symbol_name_problem(query)
	if problem is not None:
		raise ToolInputError(problem)
	if "." in query:
		if file_path is None:
			module_first = await _resolve_module_path(client, workspace_root, query)
			if module_first is not None:
				return module_first
		return await _resolve_dotted(client, workspace_root, query, file_path=file_path)
	try:
		return await _resolve_simple(client, workspace_root, query, file_path=file_path)
	except SymbolResolutionError:
		if file_path is not None:
			candidate = (workspace_root / file_path).resolve()
			if candidate.is_relative_to(workspace_root.resolve()) and declares_type_parameter(candidate, query):
				raise ToolInputError(
					f"{query} is a type parameter (PEP 695) in {file_path}, not a navigable symbol; "
					"edit its declaration and uses by hand."
				) from None
		raise
