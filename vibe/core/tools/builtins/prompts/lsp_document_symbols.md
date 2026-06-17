Use `lsp_document_symbols` to get a structural outline of a file — all classes, functions, variables, and other named symbols.

- Useful as a first step when exploring an unfamiliar file
- Returns symbols in document order with their kind (Class, Function, Variable, etc.) and line number
- Nested symbols (methods inside a class) appear with `depth > 0`
- More precise than grep for finding where specific types/functions are defined
