from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

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

    assert result == {"id": 5, "response": ""}


def test_run_one_returns_empty_response_on_timeout(tmp_path):
    import subprocess

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "test_ave.csv").write_text("a,b\n1,2\n")

    def fake_run(cmd, capture_output, text, timeout, env):
        raise subprocess.TimeoutExpired(cmd, timeout)

    with patch("run_dabench.subprocess.run", side_effect=fake_run):
        result = run_dabench.run_one(QUESTION, tables_dir, max_turns=10, max_price=None, timeout=1, keep_tmp=False)

    assert result == {"id": 5, "response": ""}
