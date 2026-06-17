Use `lsp_diagnostics` to get type errors, syntax errors, and warnings from the language server for a file.

- In **auto/strict** mode: diagnostics are surfaced automatically after edits — your job is to react to errors shown and fix them before continuing, not to re-invoke this tool
- In **manual** mode: call this after every edit to a `.py` or `.ts` file before moving on
- Also call before declaring a task complete when there is any doubt
- Errors (severity 1) indicate a broken build; warnings (severity 2) are advisory
- Line numbers in results are 1-indexed
