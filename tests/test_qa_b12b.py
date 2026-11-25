"""QA block B12b: list reads, paging and ordering in `mcp_nav_shared.format` (RC-017, 043, 049, 082, 085, 087, 142, 157).

New names are reached through the module object and checked first, so each test fails on the base code by assertion.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any


fmt = importlib.import_module("mcp_nav_shared.format")


def _has_param(func: Any, name: str) -> bool:
	return name in inspect.signature(func).parameters


def _sym(uri: str, name: str, line: int = 0) -> dict[str, Any]:
	return {
		"name": name,
		"kind": 12,
		"location": {
			"uri": uri,
			"range": {"start": {"line": line, "character": 4}, "end": {"line": line, "character": 10}},
		},
	}


def _loc(root: Path, rel: str, line: int = 0) -> dict[str, Any]:
	return {
		"uri": (root / rel).as_uri(),
		"range": {"start": {"line": line, "character": 0}, "end": {"line": line, "character": 4}},
	}


def _write(root: Path, rel: str, text: str = "x = 1\n" * 12) -> None:
	path = root / rel
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _call(root: Path, rel: str, line: int) -> dict[str, Any]:
	uri = (root / rel).as_uri()
	start = {"line": line, "character": 0}
	return {
		"from": {"name": f"caller{line}", "kind": 12, "uri": uri, "selectionRange": {"start": start}},
		"fromRanges": [{"start": start}],
	}


def test_rc_017_hits_outside_the_workspace_are_split_off(tmp_path):
	# given
	assert hasattr(fmt, "split_inside_workspace") and hasattr(fmt, "outside_workspace_note")
	ws = tmp_path / "ws"
	ws.mkdir()
	inside = {"uri": (ws / "a.py").as_uri()}
	outside = {"uri": (tmp_path / "lib" / "site.py").as_uri()}
	# when
	kept, dropped = fmt.split_inside_workspace([inside, outside], ws, lambda item: item["uri"])
	# then
	assert kept == [inside]
	assert dropped == 1
	assert fmt.outside_workspace_note(dropped) == "1 result(s) outside the workspace were not shown."
	assert fmt.outside_workspace_note(0) == ""


def test_rc_043_same_rank_symbols_come_out_in_path_order(tmp_path):
	# given
	b = _sym((tmp_path / "pkg" / "b.py").as_uri(), "helper", line=5)
	a = _sym((tmp_path / "pkg" / "a.py").as_uri(), "helper", line=9)
	# when
	ranked = fmt.rank_workspace_symbols([b, a], "helper")
	# then
	assert ranked == [a, b]


def test_rc_049_references_state_the_exact_total_even_for_few_hits(tmp_path):
	# given
	assert _has_param(fmt.format_references, "limit")
	_write(tmp_path, "a.py")
	locs = [_loc(tmp_path, "a.py", 0), _loc(tmp_path, "a.py", 1)]
	# when
	text = fmt.format_references(locs, tmp_path)
	capped = fmt.format_references([_loc(tmp_path, "a.py", i) for i in range(3)], tmp_path, limit=2)
	# then
	assert text.splitlines()[0] == "2 reference(s) in 1 file(s):"
	assert capped.splitlines()[0] == "3 reference(s) in 1 file(s):"
	assert capped.splitlines()[-1] == "… 1 more omitted; raise limit= to see them"


def test_rc_049_callers_state_the_exact_count_and_cap_the_list(tmp_path):
	# given
	assert _has_param(fmt.format_callers, "limit")
	calls = [_call(tmp_path, "a.py", i) for i in range(3)]
	# when
	text = fmt.format_callers(calls, tmp_path, limit=2)
	# then
	assert text.splitlines()[0] == "3 caller(s) in 1 file(s):"
	assert "1 more omitted; raise limit= to see them" in text


def test_rc_082_static_callers_note_is_a_shared_constant():
	# given
	assert hasattr(fmt, "STATIC_CALLERS_NOTE")
	# then
	assert fmt.STATIC_CALLERS_NOTE == "Static callers only; overriding methods are not included."


def test_rc_085_fuzzy_only_hits_say_there_is_no_exact_match(tmp_path):
	# given
	assert hasattr(fmt, "NO_EXACT_MATCH_NOTE")
	symbols = [_sym((tmp_path / "a.py").as_uri(), "unrelated")]
	# when
	text = fmt.format_workspace_symbols(symbols, tmp_path, query="helper")
	# then
	assert "No exact match" in text and "fuzzy=true" in text


def test_rc_087_diagnostic_lines_name_their_file():
	# given
	assert _has_param(fmt.format_diagnostics, "path")
	item = {
		"range": {"start": {"line": 11, "character": 4}, "end": {"line": 11, "character": 9}},
		"severity": 1,
		"code": "unresolved-reference",
		"message": "boom",
	}
	# when
	text = fmt.format_diagnostics([item], path="pkg/a.py")
	# then
	assert text.splitlines()[0].startswith("pkg/a.py:12:5 ")


def test_rc_142_test_files_are_listed_after_production_files(tmp_path):
	# given
	_write(tmp_path, "tests/test_x.py")
	_write(tmp_path, "src/x.py")
	locs = [_loc(tmp_path, "tests/test_x.py"), _loc(tmp_path, "src/x.py")]
	# when
	text = fmt.format_references(locs, tmp_path)
	ranked = fmt.rank_workspace_symbols(
		[
			_sym((tmp_path / "tests" / "test_x.py").as_uri(), "helper"),
			_sym((tmp_path / "src" / "x.py").as_uri(), "helper"),
		],
		"helper",
	)
	# then
	assert text.index("src/x.py") < text.index("tests/test_x.py")
	assert [s["location"]["uri"].rsplit("/", 2)[-2] for s in ranked] == ["src", "tests"]


def test_rc_157_outline_max_depth_one_lists_only_top_level_definitions():
	# given
	assert _has_param(fmt.format_outline, "max_depth")
	rng = {"start": {"line": 0, "character": 0}, "end": {"line": 3, "character": 0}}
	member = {"name": "method_0", "kind": 6, "range": rng, "selectionRange": rng, "children": []}
	tree = [{"name": "Big", "kind": 5, "range": rng, "selectionRange": rng, "children": [member]}]
	# when
	text = fmt.format_outline(tree, max_depth=1)
	# then
	assert "Big" in text
	assert "method_0" not in text
