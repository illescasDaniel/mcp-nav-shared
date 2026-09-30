"""Fast unit tests for `mcp_nav_shared.params`."""

from __future__ import annotations

import pytest

from mcp_nav_shared.errors import ToolInputError
from mcp_nav_shared.params import resolve_name_query


def test_given_name_when_resolve_then_returns_name():
	assert resolve_name_query(preferred="name", example="Foo", name="Foo", query=None) == "Foo"


def test_given_only_query_when_preferred_name_then_returns_query_as_alias():
	assert resolve_name_query(preferred="name", example="Foo", name=None, query="Foo") == "Foo"


def test_given_only_name_when_preferred_query_then_returns_name_as_alias():
	assert resolve_name_query(preferred="query", example="Foo", query=None, name="Foo") == "Foo"


def test_given_both_when_resolve_then_prefers_preferred_param():
	assert resolve_name_query(preferred="name", example="A", name="from-name", query="from-query") == "from-name"
	assert resolve_name_query(preferred="query", example="A", name="from-name", query="from-query") == "from-query"


def test_given_neither_when_resolve_then_tool_input_error_mentions_alias():
	with pytest.raises(ToolInputError, match="query.*alias") as caught:
		resolve_name_query(preferred="name", example=".my-class", name=None, query=None)
	assert "name='.my-class'" in str(caught.value)


def test_given_port_name_aliases_when_resolve_then_accepts_name_or_query():
	assert (
		resolve_name_query(
			preferred="port_name",
			example="FileSystemPort",
			port_name=None,
			name="FileSystemPort",
			query=None,
		)
		== "FileSystemPort"
	)
	assert (
		resolve_name_query(
			preferred="port_name",
			example="FileSystemPort",
			port_name=None,
			name=None,
			query="FileSystemPort",
		)
		== "FileSystemPort"
	)


def test_given_neither_port_alias_when_resolve_then_lists_aliases():
	with pytest.raises(ToolInputError, match="aliases accepted") as caught:
		resolve_name_query(
			preferred="port_name",
			example="FileSystemPort",
			port_name=None,
			name=None,
			query=None,
		)
	assert "port_name='FileSystemPort'" in str(caught.value)
