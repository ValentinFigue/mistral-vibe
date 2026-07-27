"""Drive the `analyst-headless` vibe agent over the InfiAgent-DABench (DA-Agent) validation set.

Run `./setup.sh` first to fetch the benchmark data into vendor/, then:

    uv run python benchmarks/dabench/run_dabench.py --limit 10   # smoke test
    uv run python benchmarks/dabench/run_dabench.py              # full run

Score the resulting responses.jsonl with vendor/eval_closed_form.py (see README.md).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
DEFAULT_QUESTIONS = BENCH_DIR / "vendor" / "data" / "da-dev-questions.jsonl"
DEFAULT_TABLES_DIR = BENCH_DIR / "vendor" / "data" / "da-dev-tables"
DEFAULT_OUT = BENCH_DIR / "responses.jsonl"
AGENT_NAME = "analyst-headless"
AGENT_DIR = REPO_ROOT / ".vibe" / "agents"


def vibe_env() -> dict[str, str]:
    # --workdir points vibe at a per-question temp dir, which makes it treat that temp
    # dir as the project root for discovering custom agents — so analyst-headless.toml
    # under this repo's .vibe/agents/ would otherwise silently fail to resolve.
    return {**os.environ, "VIBE_AGENT_PATHS": json.dumps([str(AGENT_DIR)])}


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_responses(path: Path) -> dict[int, dict]:
    """Load a prior responses.jsonl into an id -> record map.

    Returns {} if the file doesn't exist. A malformed line is skipped (with a stderr
    warning) rather than raising, so a corrupted/hand-edited file doesn't crash a resume.
    """
    if not path.exists():
        return {}
    responses: dict[int, dict] = {}
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                qid = record["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"[dabench] WARNING: skipping malformed line {lineno} in {path}: {exc}", file=sys.stderr)
                continue
            responses[qid] = record
    return responses


def is_successful(record: dict | None) -> bool:
    """A record only counts as "already available" if it has a non-empty response."""
    return bool(record) and bool(record.get("response"))


def already_done(qid: int, existing: dict[int, dict], skip_failed: bool) -> bool:
    if skip_failed:
        return qid in existing
    return is_successful(existing.get(qid))


def select_questions_to_run(
    questions: list[dict], existing: dict[int, dict], overwrite: bool, skip_failed: bool
) -> list[dict]:
    if overwrite:
        return list(questions)
    return [q for q in questions if not already_done(q["id"], existing, skip_failed)]


def write_responses_atomic(path: Path, responses: dict[int, dict]) -> None:
    """Rewrite the whole output file from `responses`, atomically.

    Only ~257 small JSON rows ever exist, so a full rewrite per completion is cheap;
    correctness (exactly one line per id, no partial file on a crash) matters far more
    than incremental-append throughput here.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            for qid in sorted(responses):
                f.write(json.dumps(responses[qid]) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        os.unlink(tmp_name)
        raise


def build_prompt(question: dict) -> str:
    parts = [question["question"]]
    if question.get("constraints"):
        parts.append(f"Constraints:\n{question['constraints']}")
    parts.append(
        f'The dataset file is available in your current working directory as "{question["file_name"]}".\n\n'
        "IMPORTANT: your final reply MUST end with the answer expressed EXACTLY in this "
        "format (substitute the computed value(s) for the placeholder text, keep the "
        f"@name[...] syntax verbatim):\n{question['format']}"
    )
    return "\n\n".join(parts)


def vibe_command(prompt: str, workdir: Path, max_turns: int, max_price: float | None) -> list[str]:
    cmd = [
        "uv",
        "run",
        "vibe",
        "-p",
        prompt,
        "--agent",
        AGENT_NAME,
        "--output",
        "text",
        "--workdir",
        str(workdir),
        "--trust",
        "--max-turns",
        str(max_turns),
    ]
    if max_price is not None:
        cmd += ["--max-price", str(max_price)]
    return cmd


def run_one(
    question: dict,
    tables_dir: Path,
    max_turns: int,
    max_price: float | None,
    timeout: int,
    keep_tmp: bool,
) -> dict:
    qid = question["id"]
    start = time.monotonic()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"dabench-{qid}-"))
    try:
        shutil.copy(tables_dir / question["file_name"], tmp_dir / question["file_name"])
        cmd = vibe_command(build_prompt(question), tmp_dir, max_turns, max_price)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=vibe_env())
        elapsed = time.monotonic() - start
        if result.returncode != 0:
            print(f"[dabench] id={qid} FAILED ({elapsed:.1f}s): {result.stderr.strip()[-500:]}", file=sys.stderr)
            return {"id": qid, "response": ""}
        print(f"[dabench] id={qid} ok ({elapsed:.1f}s)", file=sys.stderr)
        return {"id": qid, "response": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        print(f"[dabench] id={qid} TIMEOUT after {timeout}s", file=sys.stderr)
        return {"id": qid, "response": ""}
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def check_preconditions() -> None:
    cmd = [
        "uv",
        "run",
        "vibe",
        "-p",
        "Reply with exactly: OK",
        "--agent",
        AGENT_NAME,
        "--output",
        "text",
        "--trust",
        "--max-turns",
        "3",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=vibe_env())
    except subprocess.TimeoutExpired:
        raise SystemExit(
            f"Precondition check timed out running `{' '.join(cmd)}`. "
            "Verify vibe and its model/API key are configured before running the benchmark."
        )
    if result.returncode != 0 or "OK" not in result.stdout:
        raise SystemExit(
            f"Precondition check failed for agent '{AGENT_NAME}'.\n"
            f"stdout:\n{result.stdout[-1000:]}\nstderr:\n{result.stderr[-1000:]}\n"
            "Verify your model/API key is configured (e.g. MISTRAL_API_KEY) and that "
            ".vibe/agents/analyst-headless.toml is discovered by vibe."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--max-price",
        type=float,
        default=1.0,
        help="Per-question cost ceiling passed to `vibe --max-price` (USD). Use 0 to disable.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Per-question subprocess timeout, seconds")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep per-question temp workdirs for debugging")
    parser.add_argument("--skip-precondition-check", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing --out file and recompute every selected question from scratch.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="On resume, also skip ids that previously came back empty (default: retry them). No effect with --overwrite.",
    )
    args = parser.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found — run ./setup.sh first.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    existing = {} if args.overwrite else load_existing_responses(args.out)
    to_run = select_questions_to_run(questions, existing, args.overwrite, args.skip_failed)
    all_responses = {} if args.overwrite else dict(existing)

    skipped = len(questions) - len(to_run)
    if skipped:
        print(
            f"[dabench] resuming: {skipped} already-completed question(s) will be skipped "
            "(use --overwrite to force a full re-run)",
            file=sys.stderr,
        )

    if not to_run:
        write_responses_atomic(args.out, all_responses)
        print(f"[dabench] nothing to do — {args.out} already has all {len(questions)} requested responses", file=sys.stderr)
        return

    if not args.skip_precondition_check:
        check_preconditions()

    max_price = args.max_price if args.max_price > 0 else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, q, args.tables_dir, args.max_turns, max_price, args.timeout, args.keep_tmp)
            for q in to_run
        ]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            all_responses[record["id"]] = record
            write_responses_atomic(args.out, all_responses)
            done += 1
            print(
                f"[dabench] progress {done}/{len(to_run)} (file has {len(all_responses)}/{len(questions)} total)",
                file=sys.stderr,
            )

    print(f"[dabench] wrote {len(all_responses)} total responses ({done} new) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
