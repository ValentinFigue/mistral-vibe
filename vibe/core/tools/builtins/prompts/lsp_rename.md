Use `lsp_rename` to compute a rename refactoring across the whole workspace.

**Workflow:**
1. Use `lsp_references` first to confirm the blast radius
2. Call `lsp_rename` — it returns a `workspace_edit` dict with all required changes
3. Apply each change using the `edit` tool (the user sees and approves each file)

The tool does NOT auto-apply edits — it returns the edit plan for you to execute.
This preserves the normal edit approval flow and keeps the rename auditable.

- Returns null if the symbol cannot be renamed at that position (e.g., a keyword)
- `line` is 1-indexed; `character` is 0-indexed
