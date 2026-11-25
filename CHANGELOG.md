# Changelog

## 0.3.0

Hardening release from the codenav-mcp v1.0 black-box QA campaign (24 reviewed fix blocks). Test suite: 219 → 395 tests.

### Added
- `LspClient.diagnostics_with_open_files(...)`: opens created documents and gathers diagnostics under one reader
  lease, so a write's new files are checked consistently; a synchronization failure marks all affected files
  unchecked (additive API).
- `mcp_nav_shared.output`: output-cap helpers behind codenav-mcp's global output budget.
- `errors.FailedText`: a `str` subclass marking explicit tool failures, so clients and `batch` classify results
  correctly; `notices` and strict `params` validation support it.
- Interrupt-safe apply: `SIGTERM`/`SIGINT` during a multi-file apply roll back and report; unknown native signal
  handlers are left untouched (Python cannot restore them). `SIGKILL` mid-apply remains a documented limit.
- `EditJournal.undo` reports failed restores instead of claiming nothing was reverted; the plan store keeps a plan
  for explicit approval after a preview refusal.

### Changed
- Preview cleanup is atomic steal-and-decide: transient preview files are renamed to a private sibling before
  verification and unlinked only when dev+inode+sha1 still match; user content is restored or kept and never
  deleted on a guess. The startup sweep removes only the empty directories an interrupted preview created and
  warns about files.
- Text model: lines split only on `\n`, `\r\n` and `\r` (form feed, VT, NEL, U+2028 and U+0085 stay inside the
  line, matching the Python parser and ty); EOL style and BOMs are preserved on writes; per-file BOM convention
  is inferred from identifiers and ambiguous edits are refused.
- Excluded directory names are one documented policy for scans and write guards, overridable via
  `CODENAV_MCP_EXCLUDE_DIRS` / `CODENAV_MCP_INCLUDE_DIRS`; skipped importers are named, never silently ignored.
- Position resolution, list reads, paging and ordering follow the documented contract (relative paths only,
  stable order, explicit counts).
- Workspace and environment validation: warns and drops results outside the workspace instead of guessing.
- Diagnostics after a write are honest: a dead language server is never reported as "no errors".
