Use `lsp_rename` to compute a rename refactoring across the whole workspace.

- **Never use `edit`/`sed` directly to rename a symbol** — text edits miss re-exports, aliased imports, and dynamic references
- When bonsai is available, prefer `pyrename`/`tsrename` — broader AST coverage

**Workflow:**
1. `lsp_references` — confirm blast radius (or `pyfindrefs` if you only have a name)
2. `lsp_rename` — returns a `workspace_edit` dict with all required changes
3. Apply each file's edits with the `edit` tool (user sees and approves each)

The tool does NOT auto-apply edits — it returns the edit plan for you to execute.

- Returns null if the symbol is not renameable at that position (e.g., a keyword)
- `line` is 1-indexed; `character` is 0-indexed
