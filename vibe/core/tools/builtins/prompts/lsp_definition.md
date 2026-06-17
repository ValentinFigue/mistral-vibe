Use `lsp_definition` to jump to where a symbol is defined.

- Use when you need to read the implementation of a function/class called in the current file
- Returns a list of locations (file, line, character); most symbols have exactly one
- `line` is 1-indexed; `character` is 0-indexed
