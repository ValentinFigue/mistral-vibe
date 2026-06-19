Use `lsp_references` to find all usages of a symbol — scope-aware, handles re-exports and aliased imports.

- **Prefer over `grep`** for symbol-level searches — grep matches raw text and misses dynamic uses, falsely matches comments and strings
- Call before renaming or deleting a symbol to confirm blast radius
- Requires a position (file + line + char). If you only have a name, locate it first with `lsp_document_symbols` or grep, then call here. For name-only queries, `pyfindrefs` (bonsai) is often faster.
- Set `include_declaration: true` to include the definition site in results
- Results include file path, line (1-indexed), and character (0-indexed)
