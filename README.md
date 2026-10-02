# mcp-nav-shared

Shared helpers for the [`codenav-mcp`](https://pypi.org/project/codenav-mcp/)
and [`webnav-mcp`](https://pypi.org/project/webnav-mcp/) MCP servers. This
package is **not** an MCP server itself. It's installed automatically as a
dependency of both servers, so you don't need to install it yourself.

The API follows the servers' needs and may change between minor versions;
the servers pin a compatible range.

## What it provides

| Module | Role |
|--------|------|
| `mcp_nav_shared.lsp_client` | Async JSON-RPC/LSP subprocess client (`LspClient`): framing, request dispatch, document sync, file-change refresh, scratch documents, `overlay()` (show the server unsaved text to type-check an edit before writing it), rename / code-action / signature-help calls |
| `mcp_nav_shared.resolve` | Name-based symbol resolution (`resolve_symbol`, dotted `Class.method`, ambiguity errors) |
| `mcp_nav_shared.format` | Location headers (`path:line:col`), snippets, compact references, diagnostics and outline lists |
| `mcp_nav_shared.workspace` | Workspace-root selection (`WorkspaceSelector`: env pin, client MCP roots in the same git repository, `CLAUDE_PROJECT_DIR`, working directory) |
| `mcp_nav_shared.errors` | Tool-facing error text (missing file, timeout, LSP errors, …) |
| `mcp_nav_shared.params` | `name`/`query` parameter aliases with a helpful hint when both are missing |
| `mcp_nav_shared.notices` | Notices appended to tool results (e.g. a config change restarted the language server, or the server's own code changed since it started) |
| `mcp_nav_shared.edits` | Write-side primitives: LSP `TextEdit`/`WorkspaceEdit` application (UTF-16 columns, CRLF, BOM), whole-file `FileChange`s, diffs, `EditPlan` |
| `mcp_nav_shared.transaction` | `WriteGuard` (workspace/path/read-only rules), `EditJournal` (stale check, atomic apply with rollback, undo), `PlanStore` for previews |
| `mcp_nav_shared.diagnostics_delta` | Before/after diagnostics comparison keyed on message + source line, so shifted lines don't look like new errors |
| `mcp_nav_shared.exclude` | Directory names every scan skips (`.git`, `node_modules`, `.venv`, …) |

## Development

Source: [github.com/illescasDaniel/mcp-nav-shared](https://github.com/illescasDaniel/mcp-nav-shared). Design
notes: [docs/agent-tooling.md](https://github.com/illescasDaniel/SpaceMaker/blob/main/docs/agent-tooling.md).

## License

MIT. See [LICENSE](https://github.com/illescasDaniel/mcp-nav-shared/blob/main/LICENSE).
