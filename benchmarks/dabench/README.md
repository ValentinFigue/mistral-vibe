# InfiAgent-DABench (DA-Agent) evaluation harness

Runs the `analyst` agent (via a headless variant, `analyst-headless`) against
[InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent),
a 257-question closed-form data-analysis benchmark, and scores the results with
InfiAgent's own evaluation script.

## Setup

```bash
./benchmarks/dabench/setup.sh
```

Fetches the validation set into `benchmarks/dabench/vendor/` (gitignored —
third-party data/scripts, not committed): questions, labels, the 68 CSV tables,
and `eval_closed_form.py`. Re-running is a no-op if `vendor/` is already populated;
delete it first to re-fetch.

## The `analyst-headless` agent

`.vibe/agents/analyst-headless.toml` is a copy of the builtin `analyst` agent profile
with `bypass_tool_permissions = true`, so `run_pipeline`/`graph_save_block` calls
execute without an interactive approval prompt. This is required for any non-interactive
run — without it, `vibe -p ... --agent analyst` silently skips every tool call
("Tool execution not permitted") when run headlessly.

**This profile is benchmark-only.** It auto-approves every tool call for the analyst's
tool set. Don't use `--agent analyst-headless` for interactive/exploratory work — use
`--agent analyst` instead, which keeps the normal approval prompts.

**Agent discovery vs. `--workdir`:** vibe resolves project-scoped custom agents
(`.vibe/agents/`) relative to whatever `--workdir` you pass, not the shell's cwd — so
once `run_dabench.py` points `--workdir` at a question's isolated temp dir, `vibe`
would no longer find `analyst-headless.toml` under this repo's `.vibe/agents/`
("Agent 'analyst-headless' not found"). `run_dabench.py` works around this by setting
`VIBE_AGENT_PATHS` (a JSON-encoded list, per vibe's `agent_paths` config field) to this
repo's `.vibe/agents/` directory as an absolute path on every invocation
(`vibe_env()`), so discovery works regardless of `--workdir`.

## Running

Preconditions: `vibe` must already have a working model/API key configured (e.g.
`MISTRAL_API_KEY` set, matching your `~/.vibe/config.toml`). `run_dabench.py` runs a
trivial probe question against `analyst-headless` before starting and fails fast with
a clear error if that doesn't come back correctly — skip it with
`--skip-precondition-check` if you've already verified this.

Smoke test first (cheap, catches harness bugs before spending on the full set):

```bash
uv run python benchmarks/dabench/run_dabench.py --limit 10
```

Full run (257 questions, concurrency 4 by default):

```bash
uv run python benchmarks/dabench/run_dabench.py
```

Each question runs in its own temp workdir (only that question's CSV is copied in, so
runs don't clobber each other or leave stray chart/export files behind) and shells out to:

```
uv run vibe -p "<question + constraints + exact @name[value] format instructions>" \
  --agent analyst-headless --output text --workdir <tmp> --trust --max-turns 30 --max-price 1.0
```

`--max-price` caps spend per question (default $1, `0` disables the cap) — check the
smoke test's actual cost before running the full 257-question set unbounded.

Progress and per-question failures are logged to stderr. Results are written
incrementally to `benchmarks/dabench/responses.jsonl` (gitignored) as
`{"id": .., "response": ".."}`, one line per question, so a crash mid-run doesn't lose
completed work.

Useful flags: `--limit N`, `--concurrency N`, `--max-turns N`, `--max-price USD`,
`--timeout SECONDS` (per-question subprocess timeout), `--keep-tmp` (don't delete
temp workdirs, for debugging a specific question).

## Scoring

```bash
python3 benchmarks/dabench/vendor/eval_closed_form.py \
  --questions_file_path benchmarks/dabench/vendor/data/da-dev-questions.jsonl \
  --labels_file_path benchmarks/dabench/vendor/data/da-dev-labels.jsonl \
  --responses_file_path benchmarks/dabench/responses.jsonl
```

Prints accuracy by question, by sub-question, and by concept, and writes a full
breakdown to `eval_outputs/<responses>_evaluation_analysis.json` (relative to wherever
you ran the command from).
