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
    args = parser.parse_args()

    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found — run ./setup.sh first.")

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if not args.skip_precondition_check:
        check_preconditions()

    max_price = args.max_price if args.max_price > 0 else None
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as out_f, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, q, args.tables_dir, args.max_turns, max_price, args.timeout, args.keep_tmp)
            for q in questions
        ]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            done += 1
            print(f"[dabench] progress {done}/{len(questions)}", file=sys.stderr)

    print(f"[dabench] wrote {len(questions)} responses to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
