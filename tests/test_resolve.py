"""Fast unit tests for `mcp_nav_shared.resolve` (no live language servers)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_nav_shared.resolve import SymbolResolutionError, resolve_symbol


class _FakeResolveClient:
	"""Duck-typed stand-in for LspClient: resolve_symbol only calls
	`workspace_symbol`/`document_symbol` (plus the type-hierarchy pair for
	inherited `Class.member` lookups), all trivial to fake for these tests.
	`document_symbols` is either one list for every file or a per-file dict;
	`supertypes` maps a class name to its direct supertype items."""

	def __init__(
		self,
		workspace_symbols: list[dict],
		document_symbols: list[dict] | dict[str, list[dict]] | None = None,
		supertypes: dict[str, list[dict]] | None = None,
	) -> None:
		self._workspace_symbols = workspace_symbols
		self._document_symbols = document_symbols or []
		self._supertypes = supertypes or {}

	async def workspace_symbol(self, query: str) -> list[dict]:
		return self._workspace_symbols

	async def document_symbol(self, file_path: str) -> list[dict]:
		if isinstance(self._document_symbols, dict):
			return self._document_symbols.get(file_path, [])
		return self._document_symbols

	async def prepare_type_hierarchy(self, file_path: str, line: int, column: int) -> list[dict]:
		for sym in self._workspace_symbols:
			if sym["location"]["uri"].endswith(file_path):
				return [{"name": sym["name"], "uri": sym["location"]["uri"]}]
		return []

	async def supertypes(self, item: dict) -> list[dict]:
		return self._supertypes.get(item["name"], [])


def test_given_single_exact_match_when_resolve_symbol_then_resolves_position(tmp_path):
	# given
	uri = (tmp_path / "pkg" / "mod.py").as_uri()
	sym = {
		"name": "target_fn",
		"kind": 12,
		"location": {"uri": uri, "range": {"start": {"line": 3, "character": 0}, "end": {"line": 3, "character": 9}}},
		"selectionRange": {"start": {"line": 3, "character": 4}, "end": {"line": 3, "character": 13}},
	}
	client = _FakeResolveClient([sym])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "target_fn"))
	# then
	assert resolved.name == "target_fn"
	assert resolved.uri == uri
	assert (resolved.line, resolved.column) == (3, 4)


def test_given_no_match_when_resolve_symbol_then_raises(tmp_path):
	# given
	client = _FakeResolveClient([])
	# when / then
	with pytest.raises(SymbolResolutionError, match="No symbol found"):
		asyncio.run(resolve_symbol(client, tmp_path, "missing"))


def test_given_two_exact_matches_when_resolve_symbol_then_ambiguous_lists_candidates(tmp_path):
	# given
	uri_a = (tmp_path / "a.py").as_uri()
	uri_b = (tmp_path / "b.py").as_uri()
	sym_a = {
		"name": "run",
		"kind": 12,
		"location": {"uri": uri_a, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}},
	}
	sym_b = {
		"name": "run",
		"kind": 12,
		"location": {"uri": uri_b, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}},
	}
	client = _FakeResolveClient([sym_a, sym_b])
	# when / then
	with pytest.raises(SymbolResolutionError) as caught:
		asyncio.run(resolve_symbol(client, tmp_path, "run"))
	text = str(caught.value)
	assert "2 symbols match 'run'" in text
	assert "a.py" in text
	assert "b.py" in text


def test_given_function_and_export_variable_same_file_when_resolve_then_picks_function(tmp_path):
	# given — tsserver lists `export { foo }` as Variable alongside Function
	uri = (tmp_path / "gallery-item.ts").as_uri()
	fn = {
		"name": "renderGalleryItemStage",
		"kind": 12,
		"location": {
			"uri": uri,
			"range": {"start": {"line": 134, "character": 9}, "end": {"line": 134, "character": 31}},
		},
	}
	export_var = {
		"name": "renderGalleryItemStage",
		"kind": 13,
		"location": {
			"uri": uri,
			"range": {"start": {"line": 522, "character": 1}, "end": {"line": 522, "character": 23}},
		},
	}
	client = _FakeResolveClient([fn, export_var])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "renderGalleryItemStage"))
	# then
	assert resolved.kind == 12
	assert resolved.line == 134


def test_given_ambiguous_methods_when_resolve_symbol_then_qualifies_with_class_name(tmp_path):
	# given — two classes in the same file each define a `run` method; the
	# bare name alone can't tell them apart, so the ambiguity listing should
	# show `ClassName.run` for each rather than two identical `run` lines.
	uri = (tmp_path / "svc.py").as_uri()
	range_a = {"start": {"line": 0, "character": 0}, "end": {"line": 5, "character": 0}}
	range_b = {"start": {"line": 10, "character": 0}, "end": {"line": 15, "character": 0}}
	method_a = {
		"name": "run",
		"kind": 6,
		"location": {"uri": uri, "range": {"start": {"line": 1, "character": 1}, "end": {"line": 1, "character": 4}}},
	}
	method_b = {
		"name": "run",
		"kind": 6,
		"location": {"uri": uri, "range": {"start": {"line": 11, "character": 1}, "end": {"line": 11, "character": 4}}},
	}
	class_a = {
		"name": "Alpha",
		"kind": 5,
		"range": range_a,
		"selectionRange": range_a,
		"children": [
			{
				"name": "run",
				"kind": 6,
				"range": method_a["location"]["range"],
				"selectionRange": method_a["location"]["range"],
				"children": [],
			}
		],
	}
	class_b = {
		"name": "Beta",
		"kind": 5,
		"range": range_b,
		"selectionRange": range_b,
		"children": [
			{
				"name": "run",
				"kind": 6,
				"range": method_b["location"]["range"],
				"selectionRange": method_b["location"]["range"],
				"children": [],
			}
		],
	}
	client = _FakeResolveClient([method_a, method_b], [class_a, class_b])
	# when / then
	with pytest.raises(SymbolResolutionError) as caught:
		asyncio.run(resolve_symbol(client, tmp_path, "run"))
	text = str(caught.value)
	assert "Alpha.run" in text
	assert "Beta.run" in text
	assert "same file" in text
	assert "search_symbol" in text


def test_given_file_path_when_two_exact_matches_then_narrows_to_match(tmp_path):
	# given
	uri_a = (tmp_path / "a.py").as_uri()
	uri_b = (tmp_path / "b.py").as_uri()
	range_ = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}
	sym_a = {"name": "run", "kind": 12, "location": {"uri": uri_a, "range": range_}, "selectionRange": range_}
	sym_b = {"name": "run", "kind": 12, "location": {"uri": uri_b, "range": range_}, "selectionRange": range_}
	client = _FakeResolveClient([sym_a, sym_b])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "run", file_path="b.py"))
	# then
	assert resolved.uri == uri_b


def test_given_case_exact_and_case_insensitive_match_when_resolve_symbol_then_prefers_exact_case(tmp_path):
	# given: `repo_root` (a function) and `REPO_ROOT` (a constant elsewhere)
	# both match case-insensitively; asking for the lowercase name should
	# resolve straight to the case-exact one instead of raising ambiguous.
	uri_exact = (tmp_path / "a.py").as_uri()
	uri_other = (tmp_path / "b.py").as_uri()
	range_ = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 9}}
	exact = {"name": "repo_root", "kind": 12, "location": {"uri": uri_exact, "range": range_}, "selectionRange": range_}
	other_case = {
		"name": "REPO_ROOT",
		"kind": 13,
		"location": {"uri": uri_other, "range": range_},
		"selectionRange": range_,
	}
	client = _FakeResolveClient([exact, other_case])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "repo_root"))
	# then
	assert resolved.name == "repo_root"
	assert resolved.uri == uri_exact


def test_given_dotted_query_when_hierarchical_members_then_finds_child_node(tmp_path):
	# given
	uri = (tmp_path / "svc.py").as_uri()
	container_sym = {
		"name": "AppServices",
		"kind": 5,
		"location": {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}}},
		"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 17}},
	}
	members = [
		{
			"name": "AppServices",
			"kind": 5,
			"range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}},
			"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 17}},
			"children": [
				{
					"name": "enter_module",
					"kind": 6,
					"range": {"start": {"line": 5, "character": 1}, "end": {"line": 7, "character": 0}},
					"selectionRange": {"start": {"line": 5, "character": 5}, "end": {"line": 5, "character": 17}},
					"children": [],
				}
			],
		}
	]
	client = _FakeResolveClient([container_sym], members)
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "AppServices.enter_module"))
	# then
	assert resolved.name == "enter_module"
	assert (resolved.line, resolved.column) == (5, 5)


def test_given_dotted_query_when_flat_members_then_matches_within_container_range(tmp_path):
	# given — no hierarchicalDocumentSymbolSupport: fall back to range containment
	uri = (tmp_path / "svc.py").as_uri()
	container_sym = {
		"name": "AppServices",
		"kind": 5,
		"location": {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}}},
		"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 17}},
	}
	member_sym = {
		"name": "enter_module",
		"kind": 6,
		"location": {"uri": uri, "range": {"start": {"line": 5, "character": 1}, "end": {"line": 7, "character": 0}}},
		"selectionRange": {"start": {"line": 5, "character": 5}, "end": {"line": 5, "character": 17}},
	}
	client = _FakeResolveClient([container_sym], [container_sym, member_sym])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "AppServices.enter_module"))
	# then
	assert resolved.name == "enter_module"
	assert (resolved.line, resolved.column) == (5, 5)


def test_given_dotted_query_when_member_missing_then_raises(tmp_path):
	# given
	uri = (tmp_path / "svc.py").as_uri()
	container_sym = {
		"name": "AppServices",
		"kind": 5,
		"location": {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}}},
		"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 17}},
	}
	client = _FakeResolveClient([container_sym], [])
	# when / then
	with pytest.raises(SymbolResolutionError):
		asyncio.run(resolve_symbol(client, tmp_path, "AppServices.missing_method"))


def _class_node(name: str, children: list[dict]) -> dict:
	return {
		"name": name,
		"kind": 5,
		"range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}},
		"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 6 + len(name)}},
		"children": children,
	}


def test_given_member_defined_on_grandparent_mixin_when_dotted_query_then_resolves_through_supertypes(tmp_path):
	# given — `AppServices(JobsMixin)`, `JobsMixin(BaseMixin)`; the method lives on
	# BaseMixin, so it's only reachable via the type hierarchy, two levels up.
	svc_uri = (tmp_path / "svc.py").as_uri()
	jobs_uri = (tmp_path / "jobs.py").as_uri()
	base_uri = (tmp_path / "base.py").as_uri()
	container_sym = {
		"name": "AppServices",
		"kind": 5,
		"location": {
			"uri": svc_uri,
			"range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}},
		},
	}
	method = {
		"name": "start_convert",
		"kind": 6,
		"range": {"start": {"line": 9, "character": 1}, "end": {"line": 12, "character": 0}},
		"selectionRange": {"start": {"line": 9, "character": 5}, "end": {"line": 9, "character": 18}},
		"children": [],
	}
	client = _FakeResolveClient(
		[container_sym],
		{
			"svc.py": [_class_node("AppServices", [])],
			"jobs.py": [_class_node("JobsMixin", [])],
			"base.py": [_class_node("BaseMixin", [method])],
		},
		supertypes={
			"AppServices": [{"name": "JobsMixin", "uri": jobs_uri}],
			"JobsMixin": [{"name": "BaseMixin", "uri": base_uri}],
		},
	)
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "AppServices.start_convert"))
	# then
	assert resolved.uri == base_uri
	assert (resolved.line, resolved.column) == (9, 5)


def test_given_member_on_no_supertype_when_dotted_query_then_raises_not_found(tmp_path):
	# given
	svc_uri = (tmp_path / "svc.py").as_uri()
	jobs_uri = (tmp_path / "jobs.py").as_uri()
	container_sym = {
		"name": "AppServices",
		"kind": 5,
		"location": {
			"uri": svc_uri,
			"range": {"start": {"line": 0, "character": 0}, "end": {"line": 20, "character": 0}},
		},
	}
	client = _FakeResolveClient(
		[container_sym],
		{"svc.py": [_class_node("AppServices", [])], "jobs.py": [_class_node("JobsMixin", [])]},
		supertypes={"AppServices": [{"name": "JobsMixin", "uri": jobs_uri}]},
	)
	# when / then
	with pytest.raises(SymbolResolutionError, match="No symbol found"):
		asyncio.run(resolve_symbol(client, tmp_path, "AppServices.nope"))


def _outer_symbol(uri: str) -> dict:
	return {
		"name": "Outer",
		"kind": 5,
		"location": {"uri": uri, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 9, "character": 0}}},
		"selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 11}},
	}


def test_given_nested_dotted_query_when_hierarchical_members_then_walks_each_segment(tmp_path):
	# given
	uri = (tmp_path / "m.py").as_uri()
	deep = {
		"name": "deep",
		"kind": 6,
		"range": {"start": {"line": 2, "character": 2}, "end": {"line": 3, "character": 0}},
		"selectionRange": {"start": {"line": 2, "character": 6}, "end": {"line": 2, "character": 10}},
		"children": [],
	}
	inner = {
		"name": "Inner",
		"kind": 5,
		"range": {"start": {"line": 1, "character": 1}, "end": {"line": 3, "character": 0}},
		"selectionRange": {"start": {"line": 1, "character": 7}, "end": {"line": 1, "character": 12}},
		"children": [deep],
	}
	outer = _class_node("Outer", [inner])
	client = _FakeResolveClient([_outer_symbol(uri)], [outer])
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "Outer.Inner.deep"))
	# then
	assert resolved.name == "deep"
	assert (resolved.line, resolved.column) == (2, 6)


def test_given_nested_dotted_query_when_flat_members_then_narrows_by_range_per_segment(tmp_path):
	# given — flat shape; a same-named `deep` outside Inner must not match
	uri = (tmp_path / "m.py").as_uri()

	def flat(name: str, kind: int, start: int, end: int) -> dict:
		return {
			"name": name,
			"kind": kind,
			"location": {
				"uri": uri,
				"range": {"start": {"line": start, "character": 0}, "end": {"line": end, "character": 0}},
			},
			"selectionRange": {"start": {"line": start, "character": 4}, "end": {"line": start, "character": 8}},
		}

	members = [
		flat("Outer", 5, 0, 9),
		flat("Inner", 5, 1, 4),
		flat("deep", 6, 2, 3),
		flat("deep", 6, 6, 7),
	]
	client = _FakeResolveClient([_outer_symbol(uri)], members)
	# when
	resolved = asyncio.run(resolve_symbol(client, tmp_path, "Outer.Inner.deep"))
	# then
	assert resolved.line == 2


def test_given_file_path_matching_no_candidate_when_resolve_then_errors_instead_of_using_other_file(tmp_path):
	# given
	range_ = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}
	sym = {
		"name": "run",
		"kind": 12,
		"location": {"uri": (tmp_path / "a.py").as_uri(), "range": range_},
		"selectionRange": range_,
	}
	client = _FakeResolveClient([sym])
	# when / then
	with pytest.raises(SymbolResolutionError, match=r"(?s)No symbol 'run' in 'other.py'.*a\.py"):
		asyncio.run(resolve_symbol(client, tmp_path, "run", file_path="other.py"))
