Use `lsp_references` to find all usages of a symbol across the workspace.

- Always call this before renaming to understand the blast radius
- Set `include_declaration: true` to include the definition site in results
- Results include file path, line (1-indexed), and character (0-indexed)
- Prefer this over grep for symbol-level searches — it's scope-aware and handles re-exports
