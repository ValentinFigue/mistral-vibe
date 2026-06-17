from __future__ import annotations

import pytest

from vibe.core.lsp._diagnostics import (
    DiagnosticDelta,
    compute_delta,
    format_code_frame,
    format_diagnostic_line,
    severity_name,
)


def _diag(line: int, char: int, msg: str, sev: int = 1, code: str = "") -> dict:
    return {
        "severity": sev,
        "range": {"start": {"line": line, "character": char}, "end": {"character": char + 1}},
        "message": msg,
        "code": code,
    }


# ── severity_name ─────────────────────────────────────────────────────────────

def test_severity_name_known_codes() -> None:
    assert severity_name(1) == "error"
    assert severity_name(2) == "warning"
    assert severity_name(3) == "information"
    assert severity_name(4) == "hint"


def test_severity_name_unknown_returns_unknown() -> None:
    assert severity_name(99) == "unknown"


# ── compute_delta ─────────────────────────────────────────────────────────────

def test_compute_delta_empty_before_and_after_is_clean() -> None:
    delta = compute_delta([], [])
    assert delta.is_clean
    assert not delta.has_changes
    assert delta.summary == "no changes"


def test_compute_delta_new_error_detected() -> None:
    after = [_diag(5, 3, "undefined variable", sev=1, code="E001")]
    delta = compute_delta([], after)
    assert len(delta.new_errors) == 1
    assert delta.new_errors[0]["message"] == "undefined variable"
    assert not delta.fixed_errors
    assert delta.summary == "introduced 1 error"


def test_compute_delta_new_warning_detected() -> None:
    after = [_diag(10, 0, "unused import", sev=2)]
    delta = compute_delta([], after)
    assert len(delta.new_warnings) == 1
    assert delta.summary == "1 warning"


def test_compute_delta_fixed_error_detected() -> None:
    before = [_diag(5, 3, "undefined variable", sev=1)]
    delta = compute_delta(before, [])
    assert len(delta.fixed_errors) == 1
    assert delta.summary == "fixed 1 error"


def test_compute_delta_unchanged_diag_is_not_reported() -> None:
    diag = _diag(5, 3, "undefined variable", sev=1)
    delta = compute_delta([diag], [diag])
    assert not delta.has_changes


def test_compute_delta_mixed_new_and_fixed() -> None:
    before = [_diag(5, 3, "old error", sev=1)]
    after = [_diag(10, 0, "new error", sev=1), _diag(20, 4, "new warning", sev=2)]
    delta = compute_delta(before, after)
    assert len(delta.new_errors) == 1
    assert len(delta.new_warnings) == 1
    assert len(delta.fixed_errors) == 1
    assert "introduced 1 error" in delta.summary
    assert "fixed 1 error" in delta.summary


def test_compute_delta_min_severity_error_filters_warnings() -> None:
    after = [_diag(1, 0, "warn", sev=2), _diag(2, 0, "err", sev=1)]
    delta = compute_delta([], after, min_severity="error")
    assert len(delta.new_errors) == 1
    assert len(delta.new_warnings) == 0


def test_compute_delta_plural_summary() -> None:
    after = [_diag(i, 0, f"err {i}", sev=1) for i in range(3)]
    delta = compute_delta([], after)
    assert delta.summary == "introduced 3 errors"


# ── DiagnosticDelta.is_clean ──────────────────────────────────────────────────

def test_diagnostic_delta_is_clean_with_only_fixed() -> None:
    delta = DiagnosticDelta(fixed_errors=[_diag(1, 0, "x")])
    assert delta.is_clean


def test_diagnostic_delta_not_clean_with_new_errors() -> None:
    delta = DiagnosticDelta(new_errors=[_diag(1, 0, "x")])
    assert not delta.is_clean


# ── format_diagnostic_line ────────────────────────────────────────────────────

def test_format_diagnostic_line_error_icon() -> None:
    diag = _diag(40, 11, "undefined", sev=1, code="E001")
    line = format_diagnostic_line(diag)
    assert "✗" in line
    assert "41:11" in line  # 0-indexed line → 1-indexed display
    assert "undefined" in line
    assert "(E001)" in line


def test_format_diagnostic_line_warning_icon() -> None:
    diag = _diag(7, 0, "unused", sev=2)
    line = format_diagnostic_line(diag)
    assert "⚠" in line


def test_format_diagnostic_line_with_file_path() -> None:
    diag = _diag(0, 0, "msg", sev=1)
    line = format_diagnostic_line(diag, file_path="foo.py")
    assert "foo.py:1:0" in line


# ── format_code_frame ─────────────────────────────────────────────────────────

def test_format_code_frame_caret_length() -> None:
    diag = {
        "severity": 1,
        "range": {"start": {"line": 0, "character": 4}, "end": {"character": 9}},
        "message": "bad name",
        "code": "",
    }
    frame = format_code_frame("    timout()", diag, line_number=41)
    lines = frame.split("\n")
    assert len(lines) == 2
    assert "41 │" in lines[0]
    assert "timout()" in lines[0]
    assert "^^^^^" in lines[1]
    assert "bad name" in lines[1]


def test_format_code_frame_minimum_one_caret() -> None:
    diag = {
        "severity": 1,
        "range": {"start": {"line": 0, "character": 5}, "end": {"character": 5}},
        "message": "oops",
        "code": "",
    }
    frame = format_code_frame("hello world", diag, line_number=1)
    assert "^" in frame


def test_format_code_frame_with_code() -> None:
    diag = {
        "severity": 1,
        "range": {"start": {"line": 0, "character": 0}, "end": {"character": 3}},
        "message": "err",
        "code": "E99",
    }
    frame = format_code_frame("foo", diag, line_number=1)
    assert "E99" in frame
