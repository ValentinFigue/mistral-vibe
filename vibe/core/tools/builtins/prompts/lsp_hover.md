Use `lsp_hover` to get the type signature, documentation, or inferred type of the symbol at a position.

- Useful before renaming to confirm what symbol you're targeting
- Use when a type is unclear from reading the code alone
- `line` is 1-indexed; `character` is 0-indexed
- Returns null contents if the server has no info for that position
