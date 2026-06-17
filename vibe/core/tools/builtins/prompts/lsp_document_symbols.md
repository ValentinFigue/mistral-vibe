Use `lsp_document_symbols` to get a structural outline of a file — all classes, functions, variables, and other named symbols.

- **Use before reading a large or unfamiliar file** — get the class/function map first, then read only the relevant section. For short files, just read them directly.
- Use to get a line number when you need a position to pass to `lsp_definition`, `lsp_hover`, or `lsp_references`
- More precise than grep for finding where a specific type or function is defined within a file
- Returns symbols in document order with kind (Class, Function, Variable, etc.) and line number
- Nested symbols (methods inside a class) appear with `depth > 0`
