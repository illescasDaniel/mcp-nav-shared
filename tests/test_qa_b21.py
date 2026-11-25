"""QA block B21 (shared package): name validation, fuzzy listing, undo and preview messages, id echo cap."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_nav_shared import format as fmt, resolve
from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.transaction import EditJournal, PlanStore, WriteGuard


ROOT = Path("/ws")


def _symbol(name: str, path: str = "a.py") -> dict:
	return {
		"name": name,
		"kind": 5,
		"location": {
			"uri": f"file://{ROOT}/{path}",
			"range": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 10}},
		},
	}


# NAVN-037/038: a name with spaces, a digit start, a trailing dot or empty dot-parts is refused with a reason.


@pytest.mark.parametrize("bad", ["a b", "1x", "Cart.", "..total", "Cart..total", "a-b", "x y.z"])
def test_rc_b21_invalid_symbol_name_gets_validation_message(bad):
	problem = resolve.symbol_name_problem(bad)

	assert problem is not None
	assert "not a symbol name" in problem
	assert repr(bad) in problem


@pytest.mark.parametrize("good", ["Cart", "Cart.total", "pkg.mod.func", "_private", "café"])
def test_rc_b21_valid_symbol_name_has_no_problem(good):
	assert resolve.symbol_name_problem(good) is None


def test_rc_b21_resolve_symbol_refuses_bad_name_before_any_lookup():
	with pytest.raises(ToolInputError, match="not a symbol name"):
		asyncio.run(resolve.resolve_symbol(None, ROOT, "a b", file_path=None))  # type: ignore[arg-type]


def test_rc_b21_long_bad_name_is_cut_short():
	problem = resolve.symbol_name_problem("x" * 5000 + " y")

	assert problem is not None
	assert len(problem) < 300


# SRCH-009: fuzzy=false never lists the fuzzy hits in full; fuzzy=true marks them.


def test_rc_b21_fuzzy_hits_are_hidden_without_fuzzy_flag():
	text = fmt.format_workspace_symbols([_symbol("Cart")], ROOT, query="crt", fuzzy=False)

	assert "Cart" not in text
	assert "fuzzy=true" in text


def test_rc_b21_fuzzy_hits_are_marked_with_fuzzy_flag():
	text = fmt.format_workspace_symbols([_symbol("Cart")], ROOT, query="crt", fuzzy=True)

	assert "Cart" in text
	assert "(fuzzy)" in text


def test_rc_b21_real_hits_are_not_marked_fuzzy():
	text = fmt.format_workspace_symbols([_symbol("Cart"), _symbol("crt_helper")], ROOT, query="Cart", fuzzy=True)

	assert "Cart" in text
	assert "(fuzzy)" not in text.split("\n")[0]


# UNDO-021: an unknown id is echoed in at most ~40 characters.


def test_rc_b21_unknown_preview_id_echo_is_capped():
	store = PlanStore()

	with pytest.raises(ToolInputError) as info:
		store.get("A" * 1000)

	message = str(info.value)
	assert len(message) < 400
	assert "A" * 100 not in message


# UNDO-007: a preview that was already applied says so.


def test_rc_b21_applied_preview_id_says_already_applied():
	store = PlanStore()
	plan_id = store.add(object())  # type: ignore[arg-type]
	store.discard(plan_id)

	with pytest.raises(ToolInputError, match="already applied"):
		store.get(plan_id)


# UNDO-010 / UNDO-011: the not-in-history message says the history is in memory, per workspace and session.


def test_rc_b21_undo_unknown_id_explains_in_memory_history(tmp_path):
	journal = EditJournal(WriteGuard(root=tmp_path, allowed_suffixes=frozenset({".py"})))

	with pytest.raises(ToolInputError) as info:
		journal.undo("deadbeef")

	message = str(info.value)
	assert "in memory" in message
	assert "this server session" in message
	assert "deadbeef" in message


def test_type_parameter_does_not_hide_a_real_same_named_symbol(tmp_path, monkeypatch):
	"""A type parameter in one declaration must not mask a separate navigable symbol."""
	path = tmp_path / "a.py"
	path.write_text("class Box[T]:\n    pass\n\nT = 1\n", encoding="utf-8")
	found = object()

	async def resolve_simple(*args, **kwargs):
		return found

	monkeypatch.setattr(resolve, "_resolve_simple", resolve_simple)
	assert asyncio.run(resolve.resolve_symbol(None, tmp_path, "T", file_path="a.py")) is found
