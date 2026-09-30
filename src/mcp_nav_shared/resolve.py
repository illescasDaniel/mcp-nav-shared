"""Name-based symbol resolution shared by codenav/webnav composite tools
(`symbol_info`, `callers`, `implementations`): a `workspace_symbol` lookup
with tiered ranking, dotted `Class.method` resolution via `documentSymbol`,
and disambiguation by `file_path` — so those tools take a name instead of a
hand-computed position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_nav_shared.errors import TOOL_ERRORS, ToolInputError
from mcp_nav_shared.format import (
	_match_tier,
	filter_workspace_symbols,
	is_hierarchical_document_symbols,
	rank_workspace_symbols,
	symbol_kind_label,
	to_symbol_tree,
	uri_to_relative,
	workspace_symbol_position,
)
from mcp_nav_shared.lsp_client import LspClient


# LSP SymbolKind values that are class members, not top-level symbols — a
# candidate with one of these kinds gets qualified as `Class.member` in an
# ambiguity listing (see `_qualify_candidates`) since the bare name alone
# doesn't say which container it belongs to.
_MEMBER_KINDS = {6, 7, 9}  # Method, Property, Constructor


class SymbolResolutionError(ToolInputError):
	"""No single confident match for a name-based symbol query: not found, or ambiguous."""


@dataclass
class ResolvedSymbol:
	name: str
	kind: Any
	uri: str
	line: int  # 0-based, aimed at the identifier
	column: int  # 0-based
	range_start_line: int  # 0-based, the symbol's full span (for dotted member lookup)
	range_end_line: int


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


def _not_found(query: str) -> SymbolResolutionError:
	return SymbolResolutionError(f"No symbol found matching {query!r}.")


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
		# file_path alone can't disambiguate further; point at the
		# position-based escape hatch instead of repeating advice that won't help.
		hint = "these all live in the same file; use search_symbol to get exact line/column, then hover/definition/references with that position"
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
) -> list[dict[str, Any]]:
	symbols = await client.workspace_symbol(name)
	# Same filter as search_symbol: drop export-list Variable twins and
	# identical (name, kind, file) dupes so `symbol_info("foo")` isn't
	# ambiguous between `function foo` and `export { foo }`.
	ranked = filter_workspace_symbols(rank_workspace_symbols(symbols, name))
	exact = [s for s in ranked if _match_tier(str(s.get("name") or ""), name) <= 1]
	# Prefer a case-exact match (tier 0) over a merely case-insensitive one
	# (tier 1) when both exist, e.g. `repo_root` vs. a `REPO_ROOT` constant
	# elsewhere in the workspace — an agent asking for the lowercase name
	# almost always means the exact symbol, not an unrelated same-letters one.
	case_exact = [s for s in exact if _match_tier(str(s.get("name") or ""), name) == 0]
	if case_exact:
		exact = case_exact
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
	return exact


async def _resolve_simple(
	client: LspClient, workspace_root: Path, query: str, *, file_path: str | None
) -> ResolvedSymbol:
	exact = await _exact_candidates(client, workspace_root, query, file_path=file_path)
	if not exact:
		raise _not_found(query)
	if len(exact) > 1:
		raise await _ambiguous(client, query, exact, workspace_root)
	return _resolved_from_symbol(exact[0])


async def _resolve_dotted(
	client: LspClient, workspace_root: Path, query: str, *, file_path: str | None
) -> ResolvedSymbol:
	# `Outer.Inner.method`: the first segment is looked up workspace-wide, the
	# rest are walked down through that symbol's document-symbol tree.
	first, *path = query.split(".")
	member_name = path[-1]
	container = await _resolve_simple(client, workspace_root, first, file_path=file_path)
	rel_path = uri_to_relative(container.uri, workspace_root)
	members = await client.document_symbol(rel_path)
	if is_hierarchical_document_symbols(members):
		found = _find_named_node(members, container.name, container.line)
		for part in path:
			found = next((c for c in (found or {}).get("children") or [] if str(c.get("name")) == part), None)
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
				if str(m.get("name") or "") == part
				and span[0] <= _member_start_line(m) <= span[1]
				and _member_span(m) != span
			]
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
	"""Resolve a name (or dotted `Class.method`) to a single symbol position.

	Raises `SymbolResolutionError` (a `ToolInputError`, so tools' existing
	`except TOOL_ERRORS` handles it) when nothing matches or several symbols
	tie on an exact name — callers should pass `file_path` to disambiguate
	rather than guess.
	"""
	if "." in query:
		return await _resolve_dotted(client, workspace_root, query, file_path=file_path)
	return await _resolve_simple(client, workspace_root, query, file_path=file_path)
