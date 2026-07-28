from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_results as R

QUESTION = {"id": 1, "level": "easy", "concepts": ["Summary Statistics"], "question": "q", "format": "f"}
LABEL = {"id": 1, "common_answers": [["mean_fare", "34.65"]]}


def test_extract_answers_parses_placeholders():
    assert R.extract_answers("prose @mean_fare[34.65] more prose @x[1, 2]") == {"mean_fare": "34.65", "x": "1, 2"}


def test_extract_answers_empty_when_no_placeholder():
    assert R.extract_answers("no placeholders here") == {}


def test_values_equal_exact_and_float_tolerant():
    assert R.values_equal("34.65", "34.65") is True
    assert R.values_equal("34.66", "34.65") is False
    assert R.values_equal(str(34.65 + 1e-9), "34.65") is True
    assert R.values_equal(None, "34.65") is False
    assert R.values_equal("author,x", "author, x") is False  # exact-string mismatch, not numeric


def test_categorize_empty_response_carries_error_reason():
    row = R.categorize(QUESTION, LABEL, {"id": 1, "response": "", "error": "turn_limit_exceeded"})
    assert row["status"] == "empty"
    assert row["error"] == "turn_limit_exceeded"


def test_categorize_empty_response_defaults_error_to_missing_when_absent():
    row = R.categorize(QUESTION, LABEL, None)
    assert row["status"] == "empty" and row["error"] == "missing"


def test_categorize_missing_placeholder():
    row = R.categorize(QUESTION, LABEL, {"id": 1, "response": "The answer is thirty-four sixty-five."})
    assert row["status"] == "missing_placeholder"


def test_categorize_correct():
    row = R.categorize(QUESTION, LABEL, {"id": 1, "response": "@mean_fare[34.65]"})
    assert row["status"] == "correct"


def test_categorize_wrong_value():
    row = R.categorize(QUESTION, LABEL, {"id": 1, "response": "@mean_fare[99.99]"})
    assert row["status"] == "wrong_value"


def test_categorize_partial_when_some_subanswers_match():
    label = {"id": 1, "common_answers": [["a", "1"], ["b", "2"]]}
    row = R.categorize(QUESTION, label, {"id": 1, "response": "@a[1] @b[999]"})
    assert row["status"] == "partial"


def test_analyze_joins_questions_labels_responses(tmp_path):
    questions_path = tmp_path / "q.jsonl"
    labels_path = tmp_path / "l.jsonl"
    responses_path = tmp_path / "r.jsonl"
    questions_path.write_text(
        json.dumps({**QUESTION, "id": 1}) + "\n" + json.dumps({**QUESTION, "id": 2, "level": "hard"}) + "\n"
    )
    labels_path.write_text(
        json.dumps({"id": 1, "common_answers": [["mean_fare", "34.65"]]}) + "\n"
        + json.dumps({"id": 2, "common_answers": [["mean_fare", "1.00"]]}) + "\n"
    )
    responses_path.write_text(
        json.dumps({"id": 1, "response": "@mean_fare[34.65]"}) + "\n" + json.dumps({"id": 2, "response": ""}) + "\n"
    )

    rows = R.analyze(responses_path, questions_path, labels_path)
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["status"] == "correct"
    assert rows[1]["status"] == "empty"


def test_summarize_breaks_down_by_level_and_concept_and_error():
    rows = [
        {"id": 1, "level": "easy", "concepts": ["A"], "status": "correct"},
        {"id": 2, "level": "hard", "concepts": ["A", "B"], "status": "wrong_value"},
        {"id": 3, "level": "hard", "concepts": ["B"], "status": "empty", "error": "timeout"},
    ]
    summary = R.summarize(rows)
    assert summary["total"] == 3
    assert summary["counts"] == {"correct": 1, "wrong_value": 1, "empty": 1}
    assert summary["accuracy"] == round(1 / 3, 4)
    assert summary["by_level"]["easy"]["accuracy"] == 1.0
    assert summary["by_level"]["hard"]["accuracy"] == 0.0
    assert summary["by_concept"]["A"]["counts"] == {"correct": 1, "wrong_value": 1}
    assert summary["by_concept"]["B"]["counts"] == {"wrong_value": 1, "empty": 1}
    assert summary["empty_by_error"] == {"timeout": 1}
