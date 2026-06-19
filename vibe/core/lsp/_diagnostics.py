from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_SEVERITY_NAMES = {1: "error", 2: "warning", 3: "information", 4: "hint"}
_MIN_SEVERITY_THRESHOLDS = {
    "error": 1,
    "warning": 2,
    "information": 3,
    "hint": 4,
}


def severity_name(code: int) -> str:
    return _SEVERITY_NAMES.get(code, "unknown")


def _diag_key(diag: dict[str, Any]) -> tuple[Any, ...]:
    r = diag.get("range", {})
    start = r.get("start", {})
    return (
        start.get("line"),
        start.get("character"),
        diag.get("message", ""),
        diag.get("code", ""),
    )


@dataclass
class DiagnosticDelta:
    new_errors: list[dict[str, Any]] = field(default_factory=list)
    fixed_errors: list[dict[str, Any]] = field(default_factory=list)
    new_warnings: list[dict[str, Any]] = field(default_factory=list)
    fixed_warnings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.new_errors or self.fixed_errors or self.new_warnings or self.fixed_warnings
        )

    @property
    def is_clean(self) -> bool:
        return not self.new_errors and not self.new_warnings

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.new_errors:
            parts.append(f"introduced {len(self.new_errors)} error{'s' if len(self.new_errors) != 1 else ''}")
        if self.new_warnings:
            parts.append(f"{len(self.new_warnings)} warning{'s' if len(self.new_warnings) != 1 else ''}")
        if self.fixed_errors:
            parts.append(f"fixed {len(self.fixed_errors)} error{'s' if len(self.fixed_errors) != 1 else ''}")
        if self.fixed_warnings:
            parts.append(f"fixed {len(self.fixed_warnings)} warning{'s' if len(self.fixed_warnings) != 1 else ''}")
        if not parts:
            return "no changes"
        return ", ".join(parts)


def compute_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    min_severity: str = "warning",
) -> DiagnosticDelta:
    threshold = _MIN_SEVERITY_THRESHOLDS.get(min_severity, 2)

    def _filter(diags: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [d for d in diags if d.get("severity", 2) <= threshold]

    before_filtered = _filter(before)
    after_filtered = _filter(after)

    before_keys = {_diag_key(d): d for d in before_filtered}
    after_keys = {_diag_key(d): d for d in after_filtered}

    new_diags = [d for k, d in after_keys.items() if k not in before_keys]
    fixed_diags = [d for k, d in before_keys.items() if k not in after_keys]

    delta = DiagnosticDelta()
    for d in new_diags:
        if d.get("severity", 2) == 1:
            delta.new_errors.append(d)
        else:
            delta.new_warnings.append(d)
    for d in fixed_diags:
        if d.get("severity", 2) == 1:
            delta.fixed_errors.append(d)
        else:
            delta.fixed_warnings.append(d)

    return delta


def format_diagnostic_line(diag: dict[str, Any], *, file_path: str = "") -> str:
    r = diag.get("range", {})
    start = r.get("start", {})
    line = start.get("line", 0) + 1
    char = start.get("character", 0)
    sev = severity_name(diag.get("severity", 2))
    icon = "✗" if sev == "error" else "⚠"
    msg = diag.get("message", "").replace("\n", " ")
    code = diag.get("code", "")
    code_str = f"  ({code})" if code else ""
    loc = f"{file_path}:{line}:{char}" if file_path else f"{line}:{char}"
    return f"  {icon} {loc}  {sev:<8} {msg}{code_str}"


def format_code_frame(
    source_line: str,
    diag: dict[str, Any],
    line_number: int,
) -> str:
    r = diag.get("range", {})
    start_char = r.get("start", {}).get("character", 0)
    end_char = r.get("end", {}).get("character", start_char + 1)
    length = max(1, end_char - start_char)
    carets = "^" * length
    msg = diag.get("message", "").replace("\n", " ")
    code = diag.get("code", "")
    annotation = f"{msg} ({code})" if code else msg
    line_prefix = f"  {line_number} │ "
    padding = " " * (len(line_prefix) + start_char)
    return (
        f"{line_prefix}{source_line}\n"
        f"{padding}{carets} {annotation}"
    )
