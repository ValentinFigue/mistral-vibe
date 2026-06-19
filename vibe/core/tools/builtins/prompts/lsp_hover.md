Use `lsp_hover` to get the type signature, documentation, or inferred type of a symbol.

- Use when reading unfamiliar code and a type or return value is unclear — not on symbols you already understand
- Use before renaming to confirm which symbol is at the cursor position
- `line` is 1-indexed; `character` is 0-indexed
- Returns null if the server has no information for that position
