"""Categorize a DABench run's responses.jsonl: empty/missing-placeholder/correct/partial/wrong,
broken down by question `level` and `concepts`, replacing the one-off analysis previously done by
hand each run.

    uv run python benchmarks/dabench/analyze_results.py
    uv run python benchmarks/dabench/analyze_results.py --out-detail non_correct.jsonl

Reimplements (rather than imports) vendor/eval_closed_form.py's scoring logic — that script's
`from utils.utils import read_jsonl` assumes it's run from inside vendor/, not imported as a
package module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import run_dabench

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = BENCH_DIR / "vendor" / "data" / "da-dev-labels.jsonl"

_PLACEHOLDER_RE = re.compile(r"@(\w+)\[(.*?)\]")
_FLOAT_TOLERANCE = 1e-6


def extract_answers(response: str) -> dict[str, str]:
    """Pull every ``@name[value]`` placeholder out of a response, same regex as InfiAgent's
    ``eval_closed_form.py``.
    """
    return dict(_PLACEHOLDER_RE.findall(response))


def values_equal(predicted: str | None, expected: str) -> bool:
    """Exact string match, else float-tolerant (matches ``eval_closed_form.py``'s ``is_equal``)."""
    if predicted == expected:
        return True
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(expected)) < _FLOAT_TOLERANCE
    except ValueError:
        return False


def categorize(question: dict, label: dict, record: dict | None) -> dict:
    """Classify one question's outcome: empty / missing_placeholder / correct / partial / wrong_value."""
    result = {
        "id": question["id"],
        "level": question.get("level"),
        "concepts": question.get("concepts", []),
    }
    response = (record or {}).get("response", "")
    if not response:
        result["status"] = "empty"
        result["error"] = (record or {}).get("error", "missing")
        return result

    expected = dict(label.get("common_answers", []))
    predicted = extract_answers(response)
    result["expected"] = expected
    result["predicted"] = predicted
    if not predicted:
        result["status"] = "missing_placeholder"
        return result

    correctness = {name: values_equal(predicted.get(name), exp) for name, exp in expected.items()}
    result["correctness"] = correctness
    if all(correctness.values()):
        result["status"] = "correct"
    elif any(correctness.values()):
        result["status"] = "partial"
    else:
        result["status"] = "wrong_value"
    return result


def analyze(responses_path: Path, questions_path: Path, labels_path: Path) -> list[dict]:
    """Join questions/labels/responses by id and categorize each question's outcome."""
    questions = {q["id"]: q for q in run_dabench.load_questions(questions_path)}
    labels = {label["id"]: label for label in run_dabench.load_questions(labels_path)}
    responses = run_dabench.load_existing_responses(responses_path)
    rows = [categorize(q, labels[qid], responses.get(qid)) for qid, q in questions.items() if qid in labels]
    return sorted(rows, key=lambda r: r["id"])


def _accuracy(counts: Counter) -> float:
    total = sum(counts.values())
    return round(counts.get("correct", 0) / total, 4) if total else 0.0


def summarize(rows: list[dict]) -> dict:
    """Overall + per-level + per-concept breakdown, plus a breakdown of empty responses by error."""
    by_status: Counter = Counter(r["status"] for r in rows)
    by_level: dict[str, Counter] = defaultdict(Counter)
    by_concept: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_level[r["level"]][r["status"]] += 1
        for concept in r["concepts"]:
            by_concept[concept][r["status"]] += 1
    empty_by_error = Counter(r.get("error", "missing") for r in rows if r["status"] == "empty")

    return {
        "total": len(rows),
        "counts": dict(by_status),
        "accuracy": _accuracy(by_status),
        "by_level": {lvl: {"counts": dict(c), "accuracy": _accuracy(c)} for lvl, c in sorted(by_level.items())},
        "by_concept": {
            c: {"counts": dict(counts), "accuracy": _accuracy(counts)}
            for c, counts in sorted(by_concept.items(), key=lambda kv: _accuracy(kv[1]))
        },
        "empty_by_error": dict(empty_by_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=run_dabench.DEFAULT_OUT)
    parser.add_argument("--questions", type=Path, default=run_dabench.DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-detail", type=Path, default=None, help="Write every non-correct row as JSONL here")
    args = parser.parse_args()

    rows = analyze(args.responses, args.questions, args.labels)
    print(json.dumps(summarize(rows), indent=2))

    if args.out_detail:
        non_correct = [r for r in rows if r["status"] != "correct"]
        with args.out_detail.open("w") as f:
            for r in non_correct:
                f.write(json.dumps(r) + "\n")
        print(f"[analyze] wrote {len(non_correct)} non-correct row(s) to {args.out_detail}", file=sys.stderr)


if __name__ == "__main__":
    main()
