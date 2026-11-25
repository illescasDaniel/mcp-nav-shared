"""B12a (QA campaign): name resolution and name/query params in the shared package (stub clients, no ty)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.params import resolve_name_query
from mcp_nav_shared.resolve import SymbolResolutionError, resolve_symbol


class _FakeClient:
	"""Duck-typed LspClient: resolve_symbol only needs the workspace and document symbol calls here."""

	def __init__(self, workspace_symbols: list[dict], document_symbols: list[dict] | None = None) -> None:
		self._workspace_symbols = workspace_symbols
		self._document_symbols = document_symbols or []

	async def workspace_symbol(self, query: str) -> list[dict]:
		return self._workspace_symbols

	async def document_symbol(self, file_path: str) -> list[dict]:
		return self._document_symbols

	async def prepare_type_hierarchy(self, file_path: str, line: int, column: int) -> list[dict]:
		return []

	async def supertypes(self, item: dict) -> list[dict]:
		return []


def _outcome(client, root, query):
	"""The resolved symbol, or the refusal it raised, so a failing test reads as an assertion."""
	try:
		return asyncio.run(resolve_symbol(client, root, query))
	except SymbolResolutionError as exc:
		return exc


def _range(line: int, start: int, end: int) -> dict:
	return {"start": {"line": line, "character": start}, "end": {"line": line, "character": end}}


def _sym(name: str, uri: str, line: int, kind: int = 12, col: int = 4) -> dict:
	"""A workspace symbol whose identifier sits on `line` (0-based) at character `col`."""
	return {
		"name": name,
		"kind": kind,
		"location": {"uri": uri, "range": _range(line, 0, 1)},
		"selectionRange": _range(line, col, col + len(name)),
	}


def test_rc_099_overload_stubs_defer_to_the_implementation(tmp_path):
	# given: three `ov` definitions in one file: two @overload stubs (0-based lines 4 and 6), then the implementation (7)
	path = tmp_path / "ov.py"
	path.write_text(
		"from typing import overload\n\n\n@overload\ndef ov(x: int) -> int: ...\n"
		"@overload\ndef ov(x: str) -> str: ...\ndef ov(x):\n    return x\n"
	)
	uri = path.as_uri()
	client = _FakeClient([_sym("ov", uri, 4), _sym("ov", uri, 6), _sym("ov", uri, 7)])
	# when
	resolved = _outcome(client, tmp_path, "ov")
	assert not isinstance(resolved, Exception), str(resolved)
	# then
	assert resolved.line == 7


def test_rc_099_file_with_only_overload_stubs_keeps_the_first_stub(tmp_path):
	# given: a stub file where every definition is an @overload (no implementation)
	path = tmp_path / "ov.pyi"
	path.write_text("@overload\ndef ov(x: int) -> int: ...\n@overload\ndef ov(x: str) -> str: ...\n")
	uri = path.as_uri()
	client = _FakeClient([_sym("ov", uri, 1), _sym("ov", uri, 3)])
	# when
	resolved = _outcome(client, tmp_path, "ov")
	assert not isinstance(resolved, Exception), str(resolved)
	# then
	assert resolved.line == 1


def test_rc_090_nfkc_query_finds_the_normalised_name(tmp_path):
	# given: the identifier is stored as `func`; the caller types fullwidth `ｆｕｎｃ` (casefold does not fold it)
	path = tmp_path / "fw.py"
	path.write_text("func = 1\n")
	client = _FakeClient([_sym("func", path.as_uri(), 0, kind=13, col=0)])
	# when
	resolved = _outcome(client, tmp_path, "\uff46\uff55\uff4e\uff43")
	assert not isinstance(resolved, Exception), str(resolved)
	# then
	assert resolved.line == 0
	assert resolved.name == "func"


def test_rc_101_not_found_offers_a_did_you_mean_hint(tmp_path):
	# given
	path = tmp_path / "tgt.py"
	path.write_text("def target():\n    return 1\n")
	client = _FakeClient([_sym("target", path.as_uri(), 0, col=4)])
	# when / then
	with pytest.raises(SymbolResolutionError) as caught:
		asyncio.run(resolve_symbol(client, tmp_path, "targt"))
	assert "No symbol found matching 'targt'. Did you mean: target?" in str(caught.value)


def test_rc_102_same_file_ambiguity_points_to_file_path_and_qualified_name(tmp_path):
	# given: `run` is a method of two classes in one file
	path = tmp_path / "pkg" / "mod.py"
	path.parent.mkdir()
	path.write_text(
		"class Alpha:\n    def run(self):\n        pass\n\n\n"  # lines 0-4: Alpha.run is on line 1
		"class Beta:\n    pass\n\n\n    def run(self):\n        pass\n"  # lines 5-10: Beta.run is on line 9
	)
	uri = path.as_uri()
	method_a = _sym("run", uri, 1, kind=6, col=8)
	method_b = _sym("run", uri, 9, kind=6, col=8)
	class_a = {
		"name": "Alpha",
		"kind": 5,
		"range": _range(0, 0, 1),
		"selectionRange": _range(0, 6, 11),
		"children": [
			{"name": "run", "kind": 6, "range": _range(1, 0, 1), "selectionRange": _range(1, 8, 11), "children": []}
		],
	}
	class_b = {
		"name": "Beta",
		"kind": 5,
		"range": _range(5, 0, 1),
		"selectionRange": _range(5, 6, 10),
		"children": [
			{"name": "run", "kind": 6, "range": _range(9, 0, 1), "selectionRange": _range(9, 8, 11), "children": []}
		],
	}
	client = _FakeClient([method_a, method_b], [class_a, class_b])
	# when / then
	with pytest.raises(SymbolResolutionError) as caught:
		asyncio.run(resolve_symbol(client, tmp_path, "run"))
	text = str(caught.value)
	assert "these all live in the same file; give file_path= and the qualified name (Class.member) instead" in text
	assert "search_symbol" not in text


def test_rc_080_dotted_module_path_resolves_the_function_in_that_module(tmp_path):
	# given: pkg/mod.py defines `func`; `pkg.mod` is a module, not a class
	path = tmp_path / "pkg" / "mod.py"
	path.parent.mkdir()
	path.write_text("def func():\n    return 1\n")
	client = _FakeClient([_sym("func", path.as_uri(), 0)])
	# when
	resolved = _outcome(client, tmp_path, "pkg.mod.func")
	assert not isinstance(resolved, Exception), str(resolved)
	# then
	assert resolved.name == "func"
	assert resolved.uri.endswith("pkg/mod.py")


def test_rc_103_conflicting_name_and_query_is_refused(tmp_path):
	# when / then: two different names for one call are refused, not resolved by precedence
	with pytest.raises(ToolInputError, match="give either name or query, not both"):
		resolve_name_query(preferred="name", example="Cart", name="Cart", query="other")


def test_rc_080_missing_module_path_says_it_is_probably_a_module(tmp_path):
	# given: no symbol named `pkg` anywhere in the workspace
	client = _FakeClient([])
	# when / then
	with pytest.raises(SymbolResolutionError) as caught:
		asyncio.run(resolve_symbol(client, tmp_path, "pkg.nomod.func"))
	assert "pkg.nomod looks like a module path; give file_path= and the bare name." in str(caught.value)
