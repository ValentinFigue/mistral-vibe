Use `lsp_definition` to jump to where a symbol is defined.

- **Prefer over grep** when locating where a function, class, or variable is defined
- Requires a position (file + line + char). If you only have a name, get the line number first with `lsp_document_symbols` or grep, then call here.
- Returns a list of locations (file, line, character); most symbols have exactly one
- `line` is 1-indexed; `character` is 0-indexed
