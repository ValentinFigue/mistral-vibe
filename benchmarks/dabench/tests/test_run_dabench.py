from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_dabench

QUESTION = {
    "id": 5,
    "question": "Calculate the mean fare.",
    "concepts": ["Summary Statistics"],
    "constraints": "Round to two decimal places.",
    "format": '@mean_fare[mean_fare_value] where "mean_fare_value" is a float rounded to 2 decimal places.',
    "file_name": "test_ave.csv",
    "level": "easy",
}


def test_build_prompt_includes_question_constraints_and_format():
    prompt = run_dabench.build_prompt(QUESTION)
    assert QUESTION["question"] in prompt
    assert QUESTION["constraints"] in prompt
    assert QUESTION["format"] in prompt
    assert QUESTION["file_name"] in prompt


def test_build_prompt_omits_constraints_block_when_absent():
    question = {**QUESTION, "constraints": ""}
    prompt = run_dabench.build_prompt(question)
    assert "Constraints:" not in prompt


def test_vibe_command_includes_agent_and_max_price():
    cmd = run_dabench.vibe_command("hello", Path("/tmp/x"), max_turns=10, max_price=2.5)
    assert "--agent" in cmd
    assert cmd[cmd.index("--agent") + 1] == run_dabench.AGENT_NAME
    assert "--max-price" in cmd
    assert cmd[cmd.index("--max-price") + 1] == "2.5"


def test_vibe_command_omits_max_price_when_none():
    cmd = run_dabench.vibe_command("hello", Path("/tmp/x"), max_turns=10, max_price=None)
    assert "--max-price" not in cmd


def test_load_questions_parses_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(QUESTION) + "\n\n" + json.dumps({**QUESTION, "id": 6}) + "\n")
    questions = run_dabench.load_questions(path)
    assert [q["id"] for q in questions] == [5, 6]


def test_load_existing_responses_returns_empty_dict_when_file_missing(tmp_path):
    assert run_dabench.load_existing_responses(tmp_path / "missing.jsonl") == {}


def test_load_existing_responses_parses_valid_lines(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps({"id": 1, "response": "a"}) + "\n" + json.dumps({"id": 2, "response": "b"}) + "\n")
    responses = run_dabench.load_existing_responses(path)
    assert responses == {1: {"id": 1, "response": "a"}, 2: {"id": 2, "response": "b"}}


def test_load_existing_responses_skips_malformed_line_and_warns(tmp_path, capsys):
    path = tmp_path / "responses.jsonl"
    path.write_text("not json\n" + json.dumps({"id": 1, "response": "a"}) + "\n")
    responses = run_dabench.load_existing_responses(path)
    assert responses == {1: {"id": 1, "response": "a"}}
    assert "malformed line 1" in capsys.readouterr().err


def test_load_existing_responses_last_line_wins_on_duplicate_id(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps({"id": 1, "response": "old"}) + "\n" + json.dumps({"id": 1, "response": "new"}) + "\n")
    responses = run_dabench.load_existing_responses(path)
    assert responses[1]["response"] == "new"


def test_is_successful():
    assert run_dabench.is_successful({"id": 1, "response": "x"}) is True
    assert run_dabench.is_successful({"id": 1, "response": ""}) is False
    assert run_dabench.is_successful({"id": 1}) is False
    assert run_dabench.is_successful(None) is False


def test_select_questions_to_run_skips_successful_and_retries_empty_by_default():
    questions = [{**QUESTION, "id": 1}, {**QUESTION, "id": 2}, {**QUESTION, "id": 3}]
    existing = {1: {"id": 1, "response": "ok"}, 2: {"id": 2, "response": ""}}
    to_run = run_dabench.select_questions_to_run(questions, existing, overwrite=False, skip_failed=False)
    assert [q["id"] for q in to_run] == [2, 3]


def test_select_questions_to_run_with_overwrite_returns_all():
    questions = [{**QUESTION, "id": 1}, {**QUESTION, "id": 2}]
    existing = {1: {"id": 1, "response": "ok"}}
    to_run = run_dabench.select_questions_to_run(questions, existing, overwrite=True, skip_failed=False)
    assert [q["id"] for q in to_run] == [1, 2]


def test_select_questions_to_run_with_skip_failed_does_not_retry_empty():
    questions = [{**QUESTION, "id": 1}, {**QUESTION, "id": 2}, {**QUESTION, "id": 3}]
    existing = {1: {"id": 1, "response": "ok"}, 2: {"id": 2, "response": ""}}
    to_run = run_dabench.select_questions_to_run(questions, existing, overwrite=False, skip_failed=True)
    assert [q["id"] for q in to_run] == [3]


def test_write_responses_atomic_writes_one_sorted_line_per_id(tmp_path):
    path = tmp_path / "responses.jsonl"
    run_dabench.write_responses_atomic(path, {3: {"id": 3, "response": "c"}, 1: {"id": 1, "response": "a"}})
    lines = path.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [1, 3]
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_responses_atomic_replaces_stale_content(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps({"id": 99, "response": "stale"}) + "\n")
    run_dabench.write_responses_atomic(path, {1: {"id": 1, "response": "fresh"}})
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"id": 1, "response": "fresh"}


def test_write_responses_atomic_leaves_original_untouched_on_failure(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps({"id": 1, "response": "original"}) + "\n")

    with patch("run_dabench.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            run_dabench.write_responses_atomic(path, {1: {"id": 1, "response": "new"}})

    assert path.read_text() == json.dumps({"id": 1, "response": "original"}) + "\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_run_one_writes_response_and_cleans_up_tmp_dir(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    captured_workdir = {}

    def fake_run(cmd, capture_output, text, timeout, env):
        captured_workdir["path"] = Path(cmd[cmd.index("--workdir") + 1])
        assert (captured_workdir["path"] / "test_ave.csv").exists()
        assert json.loads(env["VIBE_AGENT_PATHS"]) == [str(run_dabench.AGENT_DIR)]

        class Result:
            returncode = 0
            stdout = "The answer is @mean_fare[34.65]"
            stderr = ""

        return Result()

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=60, keep_tmp=False)

    assert result == {"id": 5, "response": "The answer is @mean_fare[34.65]"}
    assert not captured_workdir["path"].exists()


def test_run_one_returns_empty_response_on_nonzero_exit(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    def fake_run(cmd, capture_output, text, timeout, env):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "boom"

        return Result()

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=60, keep_tmp=False)

    assert result == {"id": 5, "response": "", "error": "nonzero_exit"}


def test_run_one_returns_empty_response_on_timeout(tmp_path):
    import subprocess

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    def fake_run(cmd, capture_output, text, timeout, env):
        raise subprocess.TimeoutExpired(cmd, timeout)

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=1, keep_tmp=False)

    assert result == {"id": 5, "response": "", "error": "timeout"}


def test_run_one_classifies_turn_limit_exceeded(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    def fake_run(cmd, capture_output, text, timeout, env):
        class Result:
            returncode = 1
            stdout = "<vibe_stop_event>Turn limit of 30 reached</vibe_stop_event>"
            stderr = ""

        return Result()

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=60, keep_tmp=False)

    assert result == {"id": 5, "response": "", "error": "turn_limit_exceeded"}


def test_run_one_classifies_price_limit_exceeded(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    def fake_run(cmd, capture_output, text, timeout, env):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "<vibe_stop_event>Price limit exceeded: $1.2000 > $1.00</vibe_stop_event>"

        return Result()

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=60, keep_tmp=False)

    assert result == {"id": 5, "response": "", "error": "price_limit_exceeded"}


def test_classify_failure():
    assert run_dabench.classify_failure("Turn limit of 30 reached", "") == "turn_limit_exceeded"
    assert run_dabench.classify_failure("", "Price limit exceeded: $2 > $1") == "price_limit_exceeded"
    assert run_dabench.classify_failure("some other error", "boom") == "nonzero_exit"


def _main_argv(questions_path, tables_dir, out_path, *extra):
    return [
        "run_dabench.py",
        "--questions",
        str(questions_path),
        "--tables-dir",
        str(tables_dir),
        "--out",
        str(out_path),
        "--skip-precondition-check",
        *extra,
    ]


def _write_questions(path, ids):
    lines = [json.dumps({**QUESTION, "id": qid}) for qid in ids]
    path.write_text("\n".join(lines) + "\n")


def test_main_resume_skips_completed_and_retries_failed(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    out_path = tmp_path / "responses.jsonl"

    _write_questions(questions_path, [1, 2, 3])
    out_path.write_text(
        json.dumps({"id": 1, "response": "@mean_fare[34.65]"}) + "\n" + json.dumps({"id": 2, "response": ""}) + "\n"
    )

    called_ids = []

    def fake_run_one(question, tables_dir_arg, max_turns, max_price, timeout, keep_tmp):
        called_ids.append(question["id"])
        return {"id": question["id"], "response": f"@mean_fare[{question['id']}.00]"}

    with (
        patch("run_dabench.run_one", side_effect=fake_run_one),
        patch("sys.argv", _main_argv(questions_path, tables_dir, out_path)),
    ):
        run_dabench.main()

    assert sorted(called_ids) == [2, 3]
    final = run_dabench.load_existing_responses(out_path)
    assert set(final) == {1, 2, 3}
    assert final[1]["response"] == "@mean_fare[34.65]"
    assert final[2]["response"] == "@mean_fare[2.00]"
    assert final[3]["response"] == "@mean_fare[3.00]"


def test_main_overwrite_reruns_everything(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    out_path = tmp_path / "responses.jsonl"

    _write_questions(questions_path, [1, 2])
    out_path.write_text(json.dumps({"id": 1, "response": "@mean_fare[34.65]"}) + "\n")

    called_ids = []

    def fake_run_one(question, tables_dir_arg, max_turns, max_price, timeout, keep_tmp):
        called_ids.append(question["id"])
        return {"id": question["id"], "response": f"@mean_fare[{question['id']}.00]"}

    with (
        patch("run_dabench.run_one", side_effect=fake_run_one),
        patch("sys.argv", _main_argv(questions_path, tables_dir, out_path, "--overwrite")),
    ):
        run_dabench.main()

    assert sorted(called_ids) == [1, 2]
    final = run_dabench.load_existing_responses(out_path)
    assert final[1]["response"] == "@mean_fare[1.00]"


def test_main_skip_failed_does_not_retry_empty(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    out_path = tmp_path / "responses.jsonl"

    _write_questions(questions_path, [1, 2])
    out_path.write_text(json.dumps({"id": 1, "response": ""}) + "\n")

    called_ids = []

    def fake_run_one(question, tables_dir_arg, max_turns, max_price, timeout, keep_tmp):
        called_ids.append(question["id"])
        return {"id": question["id"], "response": "@mean_fare[9.99]"}

    with (
        patch("run_dabench.run_one", side_effect=fake_run_one),
        patch("sys.argv", _main_argv(questions_path, tables_dir, out_path, "--skip-failed")),
    ):
        run_dabench.main()

    assert called_ids == [2]


def test_main_nothing_to_do_skips_precondition_check(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    out_path = tmp_path / "responses.jsonl"

    _write_questions(questions_path, [1])
    out_path.write_text(json.dumps({"id": 1, "response": "@mean_fare[34.65]"}) + "\n")

    argv_without_skip_flag = [
        "run_dabench.py",
        "--questions",
        str(questions_path),
        "--tables-dir",
        str(tables_dir),
        "--out",
        str(out_path),
    ]
    with (
        patch("run_dabench.check_preconditions") as fake_check,
        patch("run_dabench.run_one") as fake_run_one,
        patch("sys.argv", argv_without_skip_flag),
    ):
        run_dabench.main()

    fake_check.assert_not_called()
    fake_run_one.assert_not_called()
