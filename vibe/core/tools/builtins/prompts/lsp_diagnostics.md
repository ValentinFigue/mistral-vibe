Use `lsp_diagnostics` to get type errors, syntax errors, and warnings from the language server for a file.

- Call after editing Python/TypeScript files to verify no new errors were introduced
- Use before declaring a task complete when `lsp.mode` is `manual`
- Errors (severity 1) block a clean build; warnings (severity 2) are advisory
- Line numbers in results are 1-indexed
- Only available when `lsp.servers` are configured and `lsp.mode != off`
